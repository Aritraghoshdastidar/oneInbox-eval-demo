"""DPO Experiment V5: Tool-Routing Discrimination + Escalation Preservation.

THIS IS THE FINAL DPO TRAINING EXPERIMENT.

Fine-tunes Qwen2.5-0.5B-Instruct using Direct Preference Optimization (DPO)
on a routing-focused trajectory preference dataset.

Strict Holdouts:
- RE-004, AMB-001, and RE-005 (and any same-lesson fingerprints) are held out.

Hyperparameters are IDENTICAL to V4A/V4B unless technically necessary.
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
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_formatted_dataset(jsonl_path: Path) -> tuple[Dataset, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(item)
            prompts.append(item["prompt"])
            chosens.append(item["chosen"])
            rejecteds.append(item["rejected"])

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })
    return dataset, records


def run_dpo_training_v5(
    train_file: str = "data/preferences/dpo_train_formatted_v5.jsonl",
    val_file: str = "data/preferences/dpo_val_formatted_v5.jsonl",
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "experiments/dpo_qwen05b_v5",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    beta: float = 0.1,
    max_length: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    print("=" * 70, flush=True)
    print("STARTING DPO EXPERIMENT V5: TOOL-ROUTING DISCRIMINATION + ESCALATION PRESERVATION", flush=True)
    print("THIS IS THE FINAL TRAINING EXPERIMENT.", flush=True)
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
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device}, Precision: {dtype}", flush=True)

    # 1. Load Tokenizer
    print("\n--- [1/6] Loading Tokenizer ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Prepare Datasets
    print("\n--- [2/6] Loading Formatted Datasets for DPO V5 ---", flush=True)
    train_dataset, train_records = load_formatted_dataset(train_path)
    val_dataset, val_records = load_formatted_dataset(val_path)
    print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.", flush=True)

    # Strict holdout checks before training
    train_sids = {r.get("scenario_id") for r in train_records}
    assert "RE-004" not in train_sids, "FATAL: RE-004 found in training dataset!"
    assert "AMB-001" not in train_sids, "FATAL: AMB-001 found in training dataset!"
    assert "RE-005" not in train_sids, "FATAL: RE-005 found in training dataset!"
    print("Holdout check passed: RE-004, AMB-001, and RE-005 are 100% held out!", flush=True)

    # Fingerprint overlap check
    train_fps = {r.get("lesson_fingerprint") for r in train_records}
    val_fps = {r.get("lesson_fingerprint") for r in val_records}
    fp_overlap = train_fps & val_fps
    assert len(fp_overlap) == 0, f"FATAL: Fingerprint overlap detected: {fp_overlap}"
    print(f"Fingerprint overlap check passed: 0 overlaps.", flush=True)

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

    # 4. LoRA Configuration — IDENTICAL to V4A/V4B
    print("\n--- [4/6] Configuring LoRA ---", flush=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    print(f"LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}", flush=True)
    print(f"Target modules: {target_modules}", flush=True)

    # 5. DPO Config & Trainer
    print("\n--- [5/6] Initializing DPOTrainer ---", flush=True)
    training_args = DPOConfig(
        output_dir=str(out_path / "checkpoints"),
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        max_length=max_length,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        seed=seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print("\n--- [6/6] Executing DPO V5 Training ---", flush=True)
    t_train_start = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - t_train_start
    print(f"Training completed in {train_runtime:.2f}s!", flush=True)

    # Save artifacts
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
    train_decision_types = {}
    for r in train_records:
        dt = r.get("decision_type", "unknown")
        train_decision_types[dt] = train_decision_types.get(dt, 0) + 1

    experiment_metadata = {
        "experiment_id": out_path.name,
        "experiment_name": "V5: Tool-Routing Discrimination + Escalation Preservation",
        "is_final_experiment": True,
        "base_model": base_model_name,
        "training_time_seconds": round(train_runtime, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "precision": str(dtype),
        "dataset": {
            "train_file": str(train_path),
            "train_sha256": train_sha,
            "train_pair_count": len(train_records),
            "train_scenarios": sorted(list(train_sids)),
            "train_decision_type_distribution": train_decision_types,
            "train_lesson_fingerprints": train_fingerprints,
            "val_file": str(val_path),
            "val_sha256": val_sha,
            "val_pair_count": len(val_records),
            "val_scenarios": sorted(list({r.get("scenario_id") for r in val_records})),
            "val_lesson_fingerprints": val_fingerprints,
            "holdout_verified": "RE-004" not in train_sids and "AMB-001" not in train_sids and "RE-005" not in train_sids,
            "fingerprint_overlap": 0,
        },
        "hyperparameters": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
            "beta": beta,
            "max_length": max_length,
            "seed": seed,
            "changes_from_v4b": "NONE — identical hyperparameters",
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

    config_path = out_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata["hyperparameters"], f, indent=2)

    manifest_src = Path("data/preferences/manifest_v5.json")
    if manifest_src.exists():
        shutil.copy(manifest_src, out_path / "dataset_manifest.json")

    readme_path = out_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# DPO Experiment V5: {out_path.name}

**THIS IS THE FINAL TRAINING EXPERIMENT.**

- **Paradigm**: Tool-Routing Discrimination + Escalation Preservation
- **Base Model**: `{base_model_name}`
- **Adapter Type**: LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})
- **Target Modules**: `{', '.join(target_modules)}`
- **Training Samples**: {len(train_records)} trajectory pairs
- **Validation Samples**: {len(val_records)} trajectory pairs (Strict Holdouts: RE-004, AMB-001, RE-005)
- **Epochs**: {num_epochs}
- **Beta**: {beta}
- **Final Train Loss**: {train_result.metrics.get('train_loss', 'N/A')}
- **Final Eval Loss**: {eval_metrics.get('eval_loss', 'N/A')}
- **Rewards Margin**: {eval_metrics.get('eval_rewards/margins', 'N/A')}
- **Rewards Accuracy**: {eval_metrics.get('eval_rewards/accuracies', 'N/A')}
""")

    print(f"\nMetadata and artifacts written to {out_path}!", flush=True)
    print("=" * 70, flush=True)
    return experiment_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO V5 LoRA Adapter (FINAL)")
    parser.add_argument("--train_file", default="data/preferences/dpo_train_formatted_v5.jsonl")
    parser.add_argument("--val_file", default="data/preferences/dpo_val_formatted_v5.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="experiments/dpo_qwen05b_v5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    run_dpo_training_v5(
        train_file=args.train_file,
        val_file=args.val_file,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
    )
