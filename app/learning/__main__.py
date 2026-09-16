"""CLI entry point — run the full failure mining pipeline.

Usage:
    python -m app.learning

Pipeline:
    1. Load stored runs and evaluations from SQLite
    2. Mine failures → FailureCandidate objects
    3. Rank by candidate_score
    4. Build preference pairs for eligible candidates
    5. Validate chosen responses deterministically
    6. Write JSONL dataset + manifest
    7. Print summary
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from app.learning.dataset import write_dataset
from app.learning.miner import mine_failures
from app.learning.preference_builder import build_preference_pairs
from app.models import TestScenario
from app.storage.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _scenario_file_hash() -> str:
    """Compute SHA256 of scenarios/scenarios.json."""
    path = Path("scenarios/scenarios.json")
    if not path.exists():
        return "unknown"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_scenarios() -> dict[str, TestScenario]:
    """Load scenario definitions."""
    path = Path("scenarios/scenarios.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: TestScenario(**s) for s in data["scenarios"]}


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


async def run_pipeline(
    dataset_version: str = "v2",
    agent_versions: list[str] | None = None,
) -> None:
    """Execute the full failure mining → preference dataset pipeline."""
    target_versions = agent_versions or [
        "v1.0",
        "v1.1",
        "v1.2_premature_action",
        "v1.3_wrong_tool",
        "v1.4_missed_escalation",
        "v1.5_multi_intent",
        "v1.6_constraint_violation",
    ]

    print(_hr("═"))
    print(f"  FAILURE MINING PIPELINE (Target: preferences_{dataset_version}.jsonl)")
    print(f"  Target Agent Versions: {', '.join(target_versions)}")
    print(_hr("═"))

    # --- Setup ---
    db = Database()
    await db.initialize()
    scenarios = _load_scenarios()
    scenario_hash = _scenario_file_hash()

    print(f"\n  Scenarios loaded: {len(scenarios)}")
    print(f"  Scenario file hash: {scenario_hash[:16]}...")

    # --- Step 1: Mine failures ---
    print(f"\n{_hr()}")
    print("  STEP 1: Mining failures from stored evaluations")
    print(_hr())

    candidates = await mine_failures(
        db, scenarios, agent_versions=target_versions
    )

    # Categorize
    behavioral = [c for c in candidates if c.failure_origin == "behavioral"]
    operational = [c for c in candidates if c.failure_origin == "operational"]
    infrastructure = [c for c in candidates if c.failure_origin == "infrastructure"]
    eligible = [c for c in candidates if c.training_eligible]

    print(f"\n  Target versions: {len(target_versions)}")
    print(f"  Total failure candidates mined: {len(candidates)}")
    print(f"    Behavioral:      {len(behavioral)}")
    print(f"    Operational:     {len(operational)}")
    print(f"    Infrastructure:  {len(infrastructure)}")
    print(f"  Training-eligible: {len(eligible)}")

    # --- Step 2: Show ranked candidates ---
    print(f"\n{_hr()}")
    print("  STEP 2: Ranked candidates (Top 20)")
    print(_hr())

    print(f"\n  {'#':<4} {'Scenario':<10} {'Version':<26} {'Severity':<10} "
          f"{'Category':<24} {'Origin':<15} {'Elig':<6} {'Score':<6}")
    print(f"  {'─'*4} {'─'*10} {'─'*26} {'─'*10} {'─'*24} {'─'*15} {'─'*6} {'─'*6}")

    for i, c in enumerate(candidates[:20], 1):
        elig = "YES" if c.training_eligible else "no"
        print(f"  {i:<4} {c.scenario_id:<10} {c.agent_version:<26} {c.severity:<10} "
              f"{c.failure_category:<24} {c.failure_origin:<15} {elig:<6} {c.candidate_score:<6.2f}")

    # --- Step 3: Build preference pairs ---
    print(f"\n{_hr()}")
    print("  STEP 3: Building preference pairs (with semantic deduplication)")
    print(_hr())

    pairs, duplicates_removed = await build_preference_pairs(eligible, scenarios, db)

    validated_pairs = [p for p in pairs if p.validation_status == "validated"]
    rejected_pairs = [p for p in pairs if p.validation_status == "rejected"]

    print(f"\n  Preference pairs:")
    print(f"    Generated:          {len(pairs)}")
    print(f"    Duplicates removed: {duplicates_removed}")
    print(f"    Validated:          {len(validated_pairs)}")
    print(f"    Rejected:           {len(rejected_pairs)}")

    # Distribution
    from collections import Counter
    cat_counts = Counter(p.failure_category for p in validated_pairs)
    sev_counts = Counter(p.severity for p in validated_pairs)
    conf_counts = Counter(round(p.confidence, 2) for p in validated_pairs)

    print("\n  Validated pairs by Failure Category:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat:<25}: {cnt}")

    print("\n  Validated pairs by Severity:")
    for sev, cnt in sorted(sev_counts.items()):
        print(f"    {sev:<12}: {cnt}")

    print("\n  Validated pairs by Confidence:")
    for conf, cnt in sorted(conf_counts.items()):
        print(f"    {conf:<12}: {cnt}")

    # --- Step 4: Show example pair content ---
    if validated_pairs:
        print(f"\n{_hr()}")
        print("  STEP 4: Representative preference pair")
        print(_hr())

        p = validated_pairs[0]
        print(f"\n  Scenario: {p.scenario_id} | Category: {p.failure_category}")
        print(f"  Pair ID:  {p.pair_id}")
        print(f"  Method:   {p.construction_method}")

        print(f"\n  Context:")
        for msg in p.prompt_or_context:
            role = msg["role"]
            content = msg["content"][:120]
            print(f"    [{role}] {content}")

        print(f"\n  Chosen response (first 200 chars):")
        print(f"    {p.chosen_response[:200]}")

        print(f"\n  Rejected response (first 200 chars):")
        print(f"    {p.rejected_response[:200]}")

        print(f"\n  Provenance:")
        for k, v in p.provenance.items():
            print(f"    {k}: {v}")

    # --- Step 5: Write dataset ---
    print(f"\n{_hr()}")
    print("  STEP 5: Writing dataset & train/val splits")
    print(_hr())

    training_scenarios = {p.scenario_id for p in validated_pairs}
    held_out = sorted(set(scenarios.keys()) - training_scenarios)

    jsonl_path, manifest_path = write_dataset(
        pairs,
        total_candidates=len(candidates),
        training_eligible=len(eligible),
        duplicates_removed=duplicates_removed,
        scenario_file_hash=scenario_hash,
        source_agent_versions=target_versions,
        held_out_scenarios=held_out,
        dataset_version=dataset_version,
    )

    print(f"\n  Dataset:  {jsonl_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Training scenarios:  {sorted(training_scenarios)}")
    print(f"  Held-out scenarios:  {held_out}")

    # --- Summary ---
    print(f"\n{_hr('═')}")
    print("  FAILURE MINING COMPLETE")
    print(_hr("═"))

    print(f"""
  Agent versions:       {len(target_versions)}
  Behavioral failures:  {len(behavioral)}
  Infrastructure failures: {len(infrastructure)}
  Operational failures: {len(operational)}
  Training-eligible:    {len(eligible)}
  Duplicates removed:   {duplicates_removed}

  Preference pairs:
    Generated: {len(pairs)}
    Validated: {len(validated_pairs)}
    Rejected:  {len(rejected_pairs)}

  Files produced:
    {jsonl_path}
    {manifest_path}
""")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Failure mining & preference dataset builder")
    parser.add_argument("--version", default="v2", help="Dataset version (default: v2)")
    parser.add_argument("--agent-versions", nargs="+", default=None, help="Agent versions to mine")
    args = parser.parse_args()

    asyncio.run(run_pipeline(dataset_version=args.version, agent_versions=args.agent_versions))


if __name__ == "__main__":
    main()

