"""Benchmark CLI — run V1 → V1.1 regression demo end-to-end.

Usage:
    python run_benchmark.py                 # Run both V1 and V1.1, then compare
    python run_benchmark.py --v1-only       # Run only V1
    python run_benchmark.py --v1.1-only     # Run only V1.1
    python run_benchmark.py --compare-only  # Compare existing results (no new runs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows with line buffering
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from app.regression.comparator import compare_versions
from app.regression.trace_diff import build_trace_diff
from app.storage.database import Database
from app.testing.runner import TestRunner


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 80) -> str:
    return char * width


def _print_suite_summary(result: dict) -> None:
    """Print a compact suite-run summary."""
    print(f"\n  Agent Version : {result['agent_version']}")
    print(f"  Scenarios     : {result['total_scenarios']}")
    print(f"  Passed        : {result['passed']}")
    print(f"  Failed        : {result['failed']}")
    print(f"  Pass Rate     : {result['pass_rate']:.1%}")
    print(f"  Avg Score     : {result['average_score']:.3f}")
    print(f"  Failures      : {result['total_failures']}")
    print()
    for r in result["results"]:
        status = "✓ PASS" if r.get("overall_passed") else "✗ FAIL"
        score = r.get("overall_score", 0)
        lat = r.get("latency_ms", 0)
        cats = r.get("failure_categories", [])
        cat_str = f" [{', '.join(cats)}]" if cats else ""
        print(f"    {r['scenario_id']}: {status}  score={score:.3f}  "
              f"latency={lat:.0f}ms{cat_str}")
    print()


def _print_comparison_table(result: dict) -> None:
    """Print the scenario-by-scenario comparison table."""
    comparisons = result.get("scenario_comparisons", [])
    if not comparisons:
        print("  No comparisons available.\n")
        return

    # Header
    print(f"\n  {'Scenario':<10} {'V1 Pass':<10} {'V1.1 Pass':<10} "
          f"{'V1 Score':<10} {'V1.1 Score':<10} "
          f"{'Classification':<16} {'Severity':<10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*16} {'─'*10}")

    for c in comparisons:
        base_pass = "PASS" if c["base_overall_passed"] else "FAIL"
        comp_pass = "PASS" if c["compare_overall_passed"] else "FAIL"
        classification = c["classification"].upper()
        severity = (c.get("severity") or "—").upper()
        print(f"  {c['scenario_id']:<10} {base_pass:<10} {comp_pass:<10} "
              f"{c['base_overall_score']:<10.3f} {c['compare_overall_score']:<10.3f} "
              f"{classification:<16} {severity:<10}")

        # Print change reasons
        for reason in c.get("change_reasons", []):
            print(f"  {'':>10} └─ {reason}")

    print()


def _print_evaluation_vectors(result: dict) -> None:
    """Print side-by-side evaluation vectors."""
    bv = result.get("base_vector", {})
    cv = result.get("compare_vector", {})

    print(f"\n  {'Metric':<30} {'V1 (base)':<20} {'V1.1 (compare)':<20}")
    print(f"  {'─'*30} {'─'*20} {'─'*20}")
    print(f"  {'Task Success Rate':<30} {bv.get('task_success_rate', 0):<20.1%} "
          f"{cv.get('task_success_rate', 0):<20.1%}")
    print(f"  {'Tool Correctness Avg':<30} {bv.get('tool_correctness_avg', 0):<20.3f} "
          f"{cv.get('tool_correctness_avg', 0):<20.3f}")
    print(f"  {'Constraint Adherence Rate':<30} {bv.get('constraint_adherence_rate', 0):<20.1%} "
          f"{cv.get('constraint_adherence_rate', 0):<20.1%}")
    print(f"  {'P95 Latency (ms)':<30} {bv.get('p95_latency_ms', 0):<20.1f} "
          f"{cv.get('p95_latency_ms', 0):<20.1f}")
    print(f"  {'Mean Latency (ms)':<30} {bv.get('mean_latency_ms', 0):<20.1f} "
          f"{cv.get('mean_latency_ms', 0):<20.1f}")
    print(f"  {'Infrastructure Failures':<30} {bv.get('infrastructure_failures', 0):<20} "
          f"{cv.get('infrastructure_failures', 0):<20}")
    print(f"  {'Total Retry Delay (ms)':<30} {bv.get('total_retry_delay_ms', 0):<20.1f} "
          f"{cv.get('total_retry_delay_ms', 0):<20.1f}")
    print(f"  {'Scenarios Passed/Run':<30} "
          f"{bv.get('scenarios_passed', 0)}/{bv.get('scenarios_run', 0):<14} "
          f"{cv.get('scenarios_passed', 0)}/{cv.get('scenarios_run', 0):<14}")
    print()


def _print_deployment_gate(result: dict) -> None:
    """Print deployment recommendation."""
    rec = result.get("deployment_recommendation", "unknown").upper().replace("_", " ")
    reasons = result.get("deployment_reasons", [])

    # Color-coded output
    icon = {
        "SAFE TO DEPLOY": "✅",
        "DEPLOY WITH REVIEW": "⚠️",
        "BLOCK DEPLOYMENT": "🚫",
        "INCONCLUSIVE": "❓",
    }.get(rec, "❓")

    print(f"\n  {icon}  Deployment Recommendation: {rec}")
    for reason in reasons:
        print(f"     └─ {reason}")
    print()


def _print_trace_diff(diff: dict) -> None:
    """Print a trace diff for one scenario."""
    print(f"\n  Scenario: {diff['scenario_id']}")
    print(f"  {diff['base_version']} → {diff['compare_version']}")
    print()

    # Metric transitions
    print("  Metric Transitions:")
    for m in diff.get("metric_transitions", []):
        if m["changed"]:
            print(f"    ⚡ {m['metric']}: {m['transition']}  "
                  f"(score: {m['base_score']:.3f} → {m['compare_score']:.3f})")
        else:
            status = "pass" if m["base_passed"] else "fail"
            print(f"    ─  {m['metric']}: {status}  (score: {m['base_score']:.3f})")

    # Tool call diff
    tool_diff = diff.get("tool_call_diff", [])
    if tool_diff:
        print("\n  Tool Call Differences:")
        for td in tool_diff:
            change = td["change"]
            if change == "identical":
                print(f"    =  [{td['index']}] {td['tool']}")
            elif change == "args_changed":
                print(f"    ~  [{td['index']}] {td['tool']}")
                print(f"       base args:    {td['base_args']}")
                print(f"       compare args: {td['compare_args']}")
            elif change == "different_tool":
                print(f"    ✗  [{td['index']}] {td['base_tool']} → {td['compare_tool']}")
            elif change == "added_in_compare":
                print(f"    +  [{td['index']}] {td['compare_tool']}({td.get('compare_args', {})})")
            elif change == "removed_in_compare":
                print(f"    -  [{td['index']}] {td['base_tool']}")

    # Failures
    base_f = diff.get("base_failures", [])
    comp_f = diff.get("compare_failures", [])
    if base_f or comp_f:
        print(f"\n  Failures (base: {len(base_f)}, compare: {len(comp_f)}):")
        for f in comp_f:
            print(f"    V1.1: [{f['severity']}] {f['category']}: {f['explanation'][:100]}")
        for f in base_f:
            print(f"    V1:   [{f['severity']}] {f['category']}: {f['explanation'][:100]}")

    print()


# ---------------------------------------------------------------------------
# Main benchmark flow
# ---------------------------------------------------------------------------

async def run_benchmark(
    run_v1: bool = True,
    run_v1_1: bool = True,
    compare_only: bool = False,
) -> dict:
    """Execute the full V1 → V1.1 regression benchmark."""
    db = Database()
    runner = TestRunner(db=db)
    await runner.initialize()

    v1_result = None
    v1_1_result = None

    if not compare_only:
        if run_v1:
            print(_hr("═"))
            print("  RUNNING V1.0 BENCHMARK")
            print(_hr("═"))
            v1_result = await runner.run_all(version_override="v1.0")
            _print_suite_summary(v1_result)

        if run_v1_1:
            print(_hr("═"))
            print("  RUNNING V1.1 BENCHMARK")
            print(_hr("═"))
            v1_1_result = await runner.run_all(version_override="v1.1")
            _print_suite_summary(v1_1_result)

    # --- Comparison ---
    print(_hr("═"))
    print("  REGRESSION COMPARISON: V1.0 → V1.1")
    print(_hr("═"))

    comparison = await compare_versions("v1.0", "v1.1", db)
    comp_dict = comparison.model_dump(mode="json")

    print(f"\n  Total scenarios compared: {comparison.total_scenarios}")
    print(f"  Improved:    {len(comparison.improved)} {comparison.improved}")
    print(f"  Regressed:   {len(comparison.regressed)} {comparison.regressed}")
    print(f"  Unchanged:   {len(comparison.unchanged)} {comparison.unchanged}")
    print(f"  Inconclusive:{len(comparison.inconclusive)} {comparison.inconclusive}")

    # Per-scenario table
    print(_hr("─"))
    print("  SCENARIO-BY-SCENARIO COMPARISON")
    print(_hr("─"))
    _print_comparison_table(comp_dict)

    # Evaluation vectors
    print(_hr("─"))
    print("  EVALUATION VECTORS")
    print(_hr("─"))
    _print_evaluation_vectors(comp_dict)

    # Deployment gate
    print(_hr("─"))
    print("  DEPLOYMENT GATE")
    print(_hr("─"))
    _print_deployment_gate(comp_dict)

    # Trace diffs for changed scenarios
    changed_scenarios = comparison.improved + comparison.regressed
    if changed_scenarios:
        print(_hr("─"))
        print("  TRACE DIFFS (changed scenarios)")
        print(_hr("─"))

        for sid in changed_scenarios:
            try:
                base_runs = await db.list_runs(
                    agent_version="v1.0", scenario_id=sid, limit=1
                )
                comp_runs = await db.list_runs(
                    agent_version="v1.1", scenario_id=sid, limit=1
                )
                if base_runs and comp_runs:
                    base_eval = await db.get_evaluation(base_runs[0].run_id)
                    comp_eval = await db.get_evaluation(comp_runs[0].run_id)
                    if base_eval and comp_eval:
                        diff = build_trace_diff(
                            base_runs[0], comp_runs[0], base_eval, comp_eval
                        )
                        _print_trace_diff(diff)
            except Exception as exc:
                print(f"  Error building trace diff for {sid}: {exc}")

    # Save report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v1_summary": v1_result,
        "v1_1_summary": v1_1_result,
        "comparison": comp_dict,
    }

    report_path = Path("data/regression_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path.absolute()}")

    print(_hr("═"))
    print("  BENCHMARK COMPLETE")
    print(_hr("═"))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OneInbox Agent Regression Benchmark")
    parser.add_argument("--v1-only", action="store_true", help="Run only V1 benchmark")
    parser.add_argument("--v1.1-only", action="store_true",
                        dest="v1_1_only", help="Run only V1.1 benchmark")
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare existing results without running new benchmarks")
    args = parser.parse_args()

    if args.compare_only:
        asyncio.run(run_benchmark(run_v1=False, run_v1_1=False, compare_only=True))
    elif args.v1_only:
        asyncio.run(run_benchmark(run_v1=True, run_v1_1=False))
    elif args.v1_1_only:
        asyncio.run(run_benchmark(run_v1=False, run_v1_1=True))
    else:
        asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
