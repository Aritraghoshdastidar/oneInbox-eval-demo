"""Prepare deduplicated DPO dataset from preferences_v2.jsonl / train_v2.jsonl.

Creates a deduplicated training view where multiple failure labels on the same
failed run (e.g., wrong_tool + constraint_violation) are collapsed into a single
unique behavioral lesson with merged metadata.

Outputs:
    data/preferences/training_dedup_v2.jsonl
    data/preferences/val_dedup_v2.jsonl
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path("data/preferences")
PREFERENCES_V2_PATH = DATA_DIR / "preferences_v2.jsonl"
TRAIN_V2_PATH = DATA_DIR / "train_v2.jsonl"
VAL_V2_PATH = DATA_DIR / "val_v2.jsonl"
MANIFEST_V2_PATH = DATA_DIR / "manifest_v2.json"

TRAIN_DEDUP_PATH = DATA_DIR / "training_dedup_v2.jsonl"
VAL_DEDUP_PATH = DATA_DIR / "val_dedup_v2.jsonl"


def _normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for semantic fingerprinting."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _compute_lesson_fingerprint(record: dict[str, Any]) -> str:
    """Compute fingerprint based on scenario + prompt + chosen + rejected."""
    scenario_id = record.get("scenario_id", "")
    prompt_str = " ".join(
        m.get("content", "") for m in record.get("prompt_or_context", [])
    )
    chosen = record.get("chosen_response", "")
    rejected = record.get("rejected_response", "")

    raw = f"{scenario_id}|{_normalize_text(prompt_str)}|{_normalize_text(chosen)}|{_normalize_text(rejected)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate_dataset(
    input_path: Path, output_path: Path
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate preference pairs sharing the same underlying behavioral lesson."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    seen_lessons: dict[str, dict[str, Any]] = {}
    duplicates_count = 0

    for r in records:
        fp = _compute_lesson_fingerprint(r)
        if fp in seen_lessons:
            duplicates_count += 1
            # Merge categories in existing entry
            existing = seen_lessons[fp]
            existing_cats = existing.get("all_categories", [existing["failure_category"]])
            cat = r.get("failure_category")
            if cat and cat not in existing_cats:
                existing_cats.append(cat)
            existing["all_categories"] = existing_cats
            # Track merged pair_ids
            existing_pair_ids = existing.get("merged_pair_ids", [existing["pair_id"]])
            if r["pair_id"] not in existing_pair_ids:
                existing_pair_ids.append(r["pair_id"])
            existing["merged_pair_ids"] = existing_pair_ids
        else:
            r_copy = dict(r)
            r_copy["all_categories"] = [r.get("failure_category")]
            r_copy["merged_pair_ids"] = [r.get("pair_id")]
            r_copy["lesson_fingerprint"] = fp

            # Format specifically for standard DPO
            # prompt: list of message dicts (compatible with conversational DPO)
            # chosen: customer-facing chosen string
            # rejected: verbatim customer-facing rejected string
            r_copy["prompt"] = r.get("prompt_or_context", [])
            r_copy["chosen"] = r.get("chosen_response", "")
            r_copy["rejected"] = r.get("rejected_response", "")

            # Metadata preserved cleanly
            r_copy["metadata"] = {
                "pair_id": r.get("pair_id"),
                "scenario_id": r.get("scenario_id"),
                "primary_category": r.get("failure_category"),
                "all_categories": r_copy["all_categories"],
                "severity": r.get("severity"),
                "provenance": r.get("provenance"),
                "confidence": r.get("confidence"),
                "construction_method": r.get("construction_method"),
            }

            seen_lessons[fp] = r_copy

    deduped_records = list(seen_lessons.values())

    with open(output_path, "w", encoding="utf-8") as f:
        for item in deduped_records:
            f.write(json.dumps(item) + "\n")

    return deduped_records, duplicates_count


def prepare_dpo_splits() -> dict[str, Any]:
    print("=" * 76)
    print("  DPO DATASET PREPARATION & DEDUPLICATION AUDIT")
    print("=" * 76)

    # 1. Deduplicate Training Split
    train_dedup, train_dups = deduplicate_dataset(TRAIN_V2_PATH, TRAIN_DEDUP_PATH)

    # 2. Deduplicate Validation Split
    val_dedup, val_dups = deduplicate_dataset(VAL_V2_PATH, VAL_DEDUP_PATH)

    # 3. Inspect Total V2
    total_records = []
    with open(PREFERENCES_V2_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_records.append(json.loads(line))

    # Categories represented in deduplicated training view
    from collections import Counter
    train_cats = Counter()
    for item in train_dedup:
        for c in item.get("all_categories", []):
            train_cats[c] += 1

    val_cats = Counter()
    for item in val_dedup:
        for c in item.get("all_categories", []):
            val_cats[c] += 1

    print("\n[Audit Summary]")
    print(f"  Original pairs in preferences_v2.jsonl : {len(total_records)}")
    print(f"  Original training pairs (train_v2)     : {len(train_dedup) + train_dups}")
    print(f"  Duplicates removed from training       : {train_dups}")
    print(f"  Unique behavioral training pairs       : {len(train_dedup)}")
    print(f"  Original validation pairs (val_v2)     : {len(val_dedup) + val_dups}")
    print(f"  Duplicates removed from validation     : {val_dups}")
    print(f"  Unique behavioral validation pairs     : {len(val_dedup)}")
    print(f"  Total unique behavioral pairs          : {len(train_dedup) + len(val_dedup)}")

    print("\n[Categories Represented in Deduplicated Training Set]")
    for cat, count in sorted(train_cats.items()):
        print(f"    {cat:<26}: {count} lessons")

    print("\n[Categories Represented in Validation Set]")
    for cat, count in sorted(val_cats.items()):
        print(f"    {cat:<26}: {count} lessons")

    print(f"\n[Generated Files]")
    print(f"  Deduplicated Training : {TRAIN_DEDUP_PATH}")
    print(f"  Deduplicated Val      : {VAL_DEDUP_PATH}")

    # Compute hashes
    def _file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    train_hash = _file_sha256(TRAIN_DEDUP_PATH)
    val_hash = _file_sha256(VAL_DEDUP_PATH)
    print(f"  Train Dedup SHA256    : {train_hash}")
    print(f"  Val Dedup SHA256      : {val_hash}")

    print("\n" + "=" * 76)
    print("  DEDUPLICATED DATASET PREPARATION COMPLETE")
    print("=" * 76)

    return {
        "original_pairs": len(total_records),
        "train_original": len(train_dedup) + train_dups,
        "train_dedup": len(train_dedup),
        "train_duplicates_removed": train_dups,
        "val_original": len(val_dedup) + val_dups,
        "val_dedup": len(val_dedup),
        "val_duplicates_removed": val_dups,
        "train_categories": dict(train_cats),
        "val_categories": dict(val_cats),
        "train_dedup_hash": train_hash,
        "val_dedup_hash": val_hash,
    }


if __name__ == "__main__":
    prepare_dpo_splits()
