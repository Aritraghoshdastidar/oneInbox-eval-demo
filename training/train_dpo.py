"""DPO Training Pipeline with LoRA for OneInbox Real-Estate Agent.

Fine-tunes Qwen2.5-0.5B-Instruct using Direct Preference Optimization (DPO)
on audited behavioral preference pairs derived from evaluator failure cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from app.agent.prompts import SYSTEM_PROMPT


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_and_format_dataset(
    jsonl_path: Path,
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[Dataset, list[dict[str, Any]]]:
    """Load JSONL dataset and prepare it for DPO training.

    Ensures the system prompt is present, applies the chat template to format
    the prompt prefix, and prepares chosen/rejected completion texts.
    """
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    eos_token = tokenizer.eos_token or "<|im_end|>"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(item)

            raw_prompt = item.get("prompt") or item.get("prompt_or_context") or []

            # Prepend system prompt if not already present
            conversation_history = []
            if not raw_prompt or raw_prompt[0].get("role") != "system":
                conversation_history.append({"role": "system", "content": system_prompt})
            conversation_history.extend(raw_prompt)

            # Apply Qwen chat template to format the prompt prefix
            # add_generation_prompt=True adds the trailing '<|im_start|>assistant\n'
            formatted_prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )

            # The completions: ensure they end with eos_token
            chosen_text = item["chosen"]
            if not chosen_text.endswith(eos_token):
                chosen_text = chosen_text + eos_token

            rejected_text = item["rejected"]
            if not rejected_text.endswith(eos_token):
                rejected_text = rejected_text + eos_token

            prompts.append(formatted_prompt)
            chosens.append(chosen_text)
            rejecteds.append(rejected_text)

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })

    return dataset, records


def run_dpo_training(
    train_file: str = "data/preferences/training_dedup_v2.jsonl",
    val_file: str = "data/preferences/val_dedup_v2.jsonl",
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "experiments/dpo_qwen05b_v1",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 768,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute LoRA + DPO training."""
    print("=" * 70, flush=True)
    print("STARTING DPO TRAINING FOR ONEINBOX AGENT", flush=True)
    print("=" * 70, flush=True)

    train_path = Path(train_file)
    val_path = Path(val_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    adapter_path = out_path / "adapter"
    tokenizer_path = out_path / "tokenizer"

    train_sha = compute_file_sha256(train_path)
    val_sha = compute_file_sha256(val_path)

    print(f"Base Model: {base_model_name}", flush=True)
    print(f"Train dataset: {train_path} (SHA256: {train_sha[:12]}...)", flush=True)
    print(f"Val dataset:   {val_path} (SHA256: {val_sha[:12]}...)", flush=True)
    print(f"Output dir:    {out_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device}, Precision: {dtype}", flush=True)

    # 1. Load Tokenizer
    print("\n--- [1/6] Loading Tokenizer ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Prepare Datasets
    print("\n--- [2/6] Formatting Datasets for DPO ---", flush=True)
    train_dataset, train_records = load_and_format_dataset(train_path, tokenizer)
    val_dataset, val_records = load_and_format_dataset(val_path, tokenizer)
    print(f"Formatted {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.", flush=True)

    # 3. Load Model
    print("\n--- [3/6] Loading Pretrained Base Model ---", flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time()-t_load:.2f}s", flush=True)
    if device == "cuda":
        print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB", flush=True)

    # 4. LoRA Configuration
    print("\n--- [4/6] Configuring LoRA ---", flush=True)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. DPO Config & Trainer
    print("\n--- [5/6] Initializing DPOTrainer ---", flush=True)
    training_args = DPOConfig(
        output_dir=str(out_path / "checkpoints"),
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",  # We save final adapter explicitly
        max_length=max_length,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        seed=seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Automatically handled by PEFT (reference = base model with adapters disabled)
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print("\n--- [6/6] Executing DPO Training ---", flush=True)
    t_train_start = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - t_train_start
    print(f"Training completed in {train_runtime:.2f}s!", flush=True)

    # Save artifacts FIRST to guarantee checkpoint persistence
    print(f"\nSaving LoRA adapter to {adapter_path}...", flush=True)
    trainer.model.save_pretrained(str(adapter_path))
    print(f"Saving tokenizer to {tokenizer_path}...", flush=True)
    tokenizer.save_pretrained(str(tokenizer_path))

    # Run evaluation on validation set
    print("\nRunning post-training evaluation on validation set...", flush=True)
    eval_metrics: dict[str, Any] = {}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_metrics = trainer.evaluate()
        print("Eval metrics:", eval_metrics, flush=True)
    except Exception as exc:
        print(f"Validation evaluate notice: {exc}", flush=True)
        eval_metrics = {"eval_note": str(exc)}

    # Clean up checkpoints folder if any
    ckpt_dir = out_path / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Collect metadata
    train_fingerprints = [r.get("lesson_fingerprint") or r.get("pair_id") for r in train_records]
    val_fingerprints = [r.get("lesson_fingerprint") or r.get("pair_id") for r in val_records]

    experiment_metadata = {
        "experiment_id": out_path.name,
        "base_model": base_model_name,
        "training_time_seconds": round(train_runtime, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "precision": str(dtype),
        "dataset": {
            "train_file": str(train_path),
            "train_sha256": train_sha,
            "train_pair_count": len(train_records),
            "train_lesson_fingerprints": train_fingerprints,
            "val_file": str(val_path),
            "val_sha256": val_sha,
            "val_pair_count": len(val_records),
            "val_lesson_fingerprints": val_fingerprints,
        },
        "hyperparameters": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": list(peft_config.target_modules) if isinstance(peft_config.target_modules, (set, list)) else list(peft_config.target_modules),
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
            "beta": beta,
            "max_length": max_length,
            "seed": seed,
        },
        "training_metrics": {
            "train_runtime": round(train_runtime, 2),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
            "total_flos": train_result.metrics.get("total_flos"),
            "train_loss": train_result.metrics.get("train_loss"),
            "epoch": train_result.metrics.get("epoch"),
        },
        "eval_metrics": eval_metrics,
        "artifact_paths": {
            "adapter_dir": str(adapter_path),
            "tokenizer_dir": str(tokenizer_path),
        },
    }

    metadata_path = out_path / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2)

    metrics_path = out_path / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**train_result.metrics, **eval_metrics}, f, indent=2)

    # Write config.json
    config_path = out_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata["hyperparameters"], f, indent=2)

    # Copy dataset manifest if present
    manifest_src = train_path.parent / "manifest_dpo_v2.json"
    if manifest_src.exists():
        shutil.copy(manifest_src, out_path / "dataset_manifest.json")

    # Write README.md in experiment directory
    readme_path = out_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# DPO Experiment: {out_path.name}

- **Base Model**: `{base_model_name}`
- **Adapter Type**: LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})
- **Target Modules**: `{', '.join(peft_config.target_modules)}`
- **Training Samples**: {len(train_records)} deduplicated pairs
- **Validation Samples**: {len(val_records)} deduplicated pairs
- **Epochs**: {num_epochs}
- **Beta**: {beta}
- **Final Train Loss**: {train_result.metrics.get('train_loss', 'N/A')}
- **Final Eval Loss**: {eval_metrics.get('eval_loss', 'N/A')}
- **Rewards Margin**: {eval_metrics.get('eval_rewards/margins', 'N/A')}
""")

    print(f"\nMetadata and artifacts written to {out_path}!", flush=True)
    print("=" * 70, flush=True)
    return experiment_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO LoRA Adapter")
    parser.add_argument("--train_file", default="data/preferences/training_dedup_v2.jsonl")
    parser.add_argument("--val_file", default="data/preferences/val_dedup_v2.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="experiments/dpo_qwen05b_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    run_dpo_training(
        train_file=args.train_file,
        val_file=args.val_file,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
    )
