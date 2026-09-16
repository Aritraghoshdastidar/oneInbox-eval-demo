"""Run all controlled behavioral failure variants against the benchmark.

Usage:
    python run_variants.py                # Run all controlled variants
    python run_variants.py --variant v1.2_premature_action  # Run a specific variant
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from app.agent.variants import VARIANT_METADATA
from app.storage.database import Database
from app.testing.runner import TestRunner


def _hr(char: str = "─", width: int = 76) -> str:
    return char * width


async def run_controlled_variants(
    selected_variants: list[str] | None = None,
) -> dict[str, dict]:
    """Execute selected or all controlled variants against the 8 benchmark scenarios."""
    db = Database()
    runner = TestRunner(db=db)
    await runner.initialize()

    variants_to_run = selected_variants or [
        k for k in VARIANT_METADATA.keys() if k != "v1.1"  # v1.1 already has runs, but can run if specified
    ]

    print(_hr("═"))
    print("  CONTROLLED BEHAVIORAL FAILURE RUNNER")
    print(f"  Total variants to execute: {len(variants_to_run)}")
    print(_hr("═"))

    all_results: dict[str, dict] = {}

    for idx, v_id in enumerate(variants_to_run, 1):
        meta = VARIANT_METADATA.get(v_id, {})
        name = meta.get("name", v_id)
        family = meta.get("failure_family", "unknown")

        print(f"\n[{idx}/{len(variants_to_run)}] Variant: {v_id} ({name})")
        print(f"    Intended Failure Family: {family}")
        print(f"    Description: {meta.get('description', '')}")

        res = await runner.run_all(version_override=v_id)
        all_results[v_id] = res

        print(f"    Results: {res['passed']}/{res['total_scenarios']} passed "
              f"({res['failed']} failed) | Total failures recorded: {res['total_failures']}")
        for r in res["results"]:
            status = "✓ PASS" if r.get("overall_passed") else "✗ FAIL"
            cats = r.get("failure_categories", [])
            cat_str = f" [{', '.join(cats)}]" if cats else ""
            print(f"      {r['scenario_id']}: {status}{cat_str} ({r.get('latency_ms', 0):.0f}ms)")

    # Save summary report
    summary_path = Path("data/variants_execution_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "variants_executed": variants_to_run,
                "summary": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{_hr('═')}")
    print(f"  All variant runs complete. Summary saved to: {summary_path}")
    print(_hr("═"))

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled failure variants")
    parser.add_argument("--variant", help="Specific variant ID to run", default=None)
    args = parser.parse_args()

    variants = [args.variant] if args.variant else None
    asyncio.run(run_controlled_variants(variants))


if __name__ == "__main__":
    main()
