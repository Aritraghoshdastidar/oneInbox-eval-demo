"""Dataset writer — produce versioned JSONL preference dataset + manifest.

Writes validated preference pairs to:
    data/preferences/preferences_v1.jsonl
    data/preferences/manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.learning.models import DatasetManifest, PreferencePair

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("data/preferences")


def write_dataset(
    pairs: list[PreferencePair],
    *,
    total_candidates: int = 0,
    training_eligible: int = 0,
    duplicates_removed: int = 0,
    scenario_file_hash: str = "",
    source_agent_versions: list[str] | None = None,
    held_out_scenarios: list[str] | None = None,
    dataset_version: str = "v1",
    output_dir: Path | None = None,
    val_ratio: float = 0.25,
) -> tuple[Path, Path]:
    """Write preference pairs to JSONL, create train/val splits, and create a manifest.

    Only pairs with validation_status == "validated" are written to the
    JSONL files.  Rejected pairs are counted in the manifest.

    Args:
        pairs: all preference pairs (validated and rejected).
        total_candidates: total mined candidates (for manifest).
        training_eligible: eligible candidates (for manifest).
        duplicates_removed: count of duplicate pairs filtered.
        scenario_file_hash: SHA256 of scenarios.json.
        source_agent_versions: versions that contributed data.
        held_out_scenarios: scenarios excluded from training.
        dataset_version: version label for the dataset ("v1" or "v2").
        output_dir: where to write (default: data/preferences/).
        val_ratio: fraction of validated pairs allocated to validation split.

    Returns:
        (jsonl_path, manifest_path).
    """
    out = output_dir or _DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / f"preferences_{dataset_version}.jsonl"
    manifest_path = out / f"manifest_{dataset_version}.json" if dataset_version != "v1" else out / "manifest.json"

    # Separate validated from rejected
    validated = [p for p in pairs if p.validation_status == "validated"]
    rejected = [p for p in pairs if p.validation_status == "rejected"]

    # Write full JSONL — one validated pair per line
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in validated:
            line = pair.model_dump(mode="json")
            f.write(json.dumps(line, default=str) + "\n")

    dataset_hash = _file_sha256(jsonl_path)

    # Train / Val Split (stratified by failure category & scenario)
    train_pairs: list[PreferencePair] = []
    val_pairs: list[PreferencePair] = []

    if dataset_version != "v1" and len(validated) > 1:
        # Group pairs by category for stratified partition
        by_category: dict[str, list[PreferencePair]] = {}
        for p in validated:
            by_category.setdefault(p.failure_category, []).append(p)

        for cat, cat_pairs in by_category.items():
            # Stable sort by pair_id
            sorted_cat = sorted(cat_pairs, key=lambda x: x.pair_id)
            if len(sorted_cat) == 1:
                train_pairs.append(sorted_cat[0])
            else:
                num_val = max(1, round(len(sorted_cat) * val_ratio))
                # Ensure train has at least 1
                if num_val >= len(sorted_cat):
                    num_val = len(sorted_cat) - 1
                val_pairs.extend(sorted_cat[:num_val])
                train_pairs.extend(sorted_cat[num_val:])

        # Write train and val jsonl files
        train_path = out / f"train_{dataset_version}.jsonl"
        val_path = out / f"val_{dataset_version}.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for pair in train_pairs:
                line = pair.model_dump(mode="json")
                f.write(json.dumps(line, default=str) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for pair in val_pairs:
                line = pair.model_dump(mode="json")
                f.write(json.dumps(line, default=str) + "\n")

        train_hash = _file_sha256(train_path)
        val_hash = _file_sha256(val_path)
        logger.info(f"Train split written: {train_path} ({len(train_pairs)} pairs)")
        logger.info(f"Val split written: {val_path} ({len(val_pairs)} pairs)")
    else:
        train_pairs = list(validated)
        train_hash = dataset_hash
        val_hash = ""

    # Collect metadata
    included_scenarios = sorted(set(p.scenario_id for p in validated))
    failure_categories = sorted(set(p.failure_category for p in validated))

    limitations = [
        "Dataset derived from 8-scenario synthetic benchmark — not production data.",
        "Covers 6 controlled behavioral failure variants.",
        "Evaluator strictly validates customer-facing responses against deterministic domain rules.",
        "Chosen responses prioritize clean baseline v1.0 passing traces without internal policy leak.",
    ] if dataset_version == "v2" else [
        "Dataset derived from 8-scenario synthetic benchmark — not production data.",
        "First version focuses on one failure family (ambiguous/clarification).",
        "Benchmark is too small for a rigorous train/validation split.",
        "Chosen responses from passing traces may contain provider-specific phrasing.",
    ]

    manifest = DatasetManifest(
        dataset_version=dataset_version,
        scenario_file_hash=scenario_file_hash,
        source_agent_versions=source_agent_versions or [],
        included_scenario_ids=included_scenarios,
        failure_categories=failure_categories,
        total_candidates=total_candidates,
        training_eligible=training_eligible,
        pairs_generated=len(pairs),
        pairs_validated=len(validated),
        pairs_rejected=len(rejected),
        duplicates_removed=duplicates_removed,
        train_pairs=len(train_pairs),
        val_pairs=len(val_pairs),
        train_hash=train_hash,
        val_hash=val_hash,
        held_out_scenarios=held_out_scenarios or [],
        dataset_hash=dataset_hash,
        limitations=limitations,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Also keep default manifest.json updated
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Dataset written: {jsonl_path} ({len(validated)} pairs)")
    logger.info(f"Manifest written: {manifest_path}")

    return jsonl_path, manifest_path


def _file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
