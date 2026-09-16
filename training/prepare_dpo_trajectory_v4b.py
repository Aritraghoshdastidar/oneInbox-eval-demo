"""Prepare Trajectory DPO Dataset V4B for Qwen2.5-0.5B-Instruct.

Applies Qwen's native chat template with `tools=TOOL_SCHEMAS` to the conversation context,
and formats chosen and rejected actions with explicit `<tool_call>` XML or natural text.

Outputs:
- data/preferences/dpo_train_formatted_v4b.jsonl
- data/preferences/dpo_val_formatted_v4b.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from app.learning.trajectory_models import TrajectoryAction, TrajectoryToolCall
from app.tools.registry import TOOL_SCHEMAS


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_INPUT = Path("data/preferences/dpo_training_v4b.jsonl")
VAL_INPUT = Path("data/preferences/dpo_validation_v4b.jsonl")

TRAIN_OUTPUT = Path("data/preferences/dpo_train_formatted_v4b.jsonl")
VAL_OUTPUT = Path("data/preferences/dpo_val_formatted_v4b.jsonl")


def format_action_completion(action_data: dict[str, Any], eos_token: str = "<|im_end|>") -> str:
    """Reconstruct TrajectoryAction and format completion."""
    raw_calls = action_data.get("tool_calls", [])
    tool_calls = [
        TrajectoryToolCall(
            name=tc.get("name") or tc.get("tool_name", ""),
            arguments=tc.get("arguments", {}),
        )
        for tc in raw_calls
    ]
    content = action_data.get("content")
    action = TrajectoryAction(content=content, tool_calls=tool_calls)
    return action.format_completion(eos_token=eos_token)


def process_dataset(
    input_file: Path,
    output_file: Path,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    formatted_records: list[dict[str, Any]] = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            context = item["context"]

            # Apply Qwen chat template with tool schemas
            prompt_str = tokenizer.apply_chat_template(
                context,
                tools=TOOL_SCHEMAS,
                tokenize=False,
                add_generation_prompt=True,
            )

            chosen_completion = format_action_completion(item["chosen"])
            rejected_completion = format_action_completion(item["rejected"])

            record = {
                "pair_id": item["pair_id"],
                "scenario_id": item["scenario_id"],
                "decision_type": item["decision_type"],
                "learning_target": item["learning_target"],
                "prompt": prompt_str,
                "chosen": chosen_completion,
                "rejected": rejected_completion,
                "failure_category": item["failure_category"],
                "lesson_fingerprint": item.get("lesson_fingerprint", ""),
            }
            formatted_records.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in formatted_records:
            f.write(json.dumps(rec) + "\n")

    return formatted_records


def main() -> None:
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Processing training dataset from {TRAIN_INPUT}...")
    train_records = process_dataset(TRAIN_INPUT, TRAIN_OUTPUT, tokenizer)
    print(f"Saved {len(train_records)} formatted train records to {TRAIN_OUTPUT}")

    print(f"Processing validation dataset from {VAL_INPUT}...")
    val_records = process_dataset(VAL_INPUT, VAL_OUTPUT, tokenizer)
    print(f"Saved {len(val_records)} formatted val records to {VAL_OUTPUT}")

    # Inspect token lengths
    prompt_lengths = [len(tokenizer.encode(r["prompt"])) for r in train_records]
    chosen_lengths = [len(tokenizer.encode(r["chosen"])) for r in train_records]
    rejected_lengths = [len(tokenizer.encode(r["rejected"])) for r in train_records]

    print("\n--- Token Length Statistics (Train V4B) ---")
    print(f"Prompt tokens: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.1f}")
    print(f"Chosen tokens: min={min(chosen_lengths)}, max={max(chosen_lengths)}, avg={sum(chosen_lengths)/len(chosen_lengths):.1f}")
    print(f"Rejected tokens: min={min(rejected_lengths)}, max={max(rejected_lengths)}, avg={sum(rejected_lengths)/len(rejected_lengths):.1f}")

    print("\n=== SAMPLE: TARGETED ESCALATION EXAMPLE (RE-005 Root Cause Fix) ===")
    esc_sample = next(r for r in train_records if "INSIST-01" in r["pair_id"])
    print(f"Pair ID: {esc_sample['pair_id']} | Scenario: {esc_sample['scenario_id']}")
    print(f"Prompt (tail 350 chars):\n...{esc_sample['prompt'][-350:]}")
    print(f"Chosen:\n{esc_sample['chosen']}")
    print(f"Rejected:\n{esc_sample['rejected']}")

    # Check holdouts in train
    train_sids = {r["scenario_id"] for r in train_records}
    assert "RE-004" not in train_sids, "FATAL: RE-004 found in formatted train dataset!"
    assert "AMB-001" not in train_sids, "FATAL: AMB-001 found in formatted train dataset!"
    assert "RE-005" not in train_sids, "FATAL: RE-005 found in formatted train dataset!"
    print("\nStrict holdout assertion PASSED: RE-004, AMB-001, and RE-005 are 100% excluded from train!")


if __name__ == "__main__":
    main()
