"""End-to-End Evaluation & Regression Analysis for Open Models (Base vs DPO).

Benchmarks Qwen2.5-0.5B Base against the DPO LoRA Fine-Tuned Model across
all 8 Real-Estate Scenarios, stores traces in SQLite, executes the
deterministic regression comparator, and generates the final deployment decision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.agent.agent import RealtorAgent
from app.agent.llm_adapter import LocalHuggingFaceAdapter
from app.config import settings
from app.models import (
    DeploymentRecommendation,
    EvaluationResult,
    RegressionResult,
    ScenarioClassification,
)
from app.regression.comparator import compare_versions
from app.storage.database import Database
from app.testing.runner import TestRunner


async def run_evaluation_suite(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "experiments/dpo_qwen05b_v1/adapter",
    output_dir: str = "reports",
    scenarios_path: str | None = None,
    skip_runs: bool = False,
) -> dict[str, Any]:
    """Execute complete Base vs DPO benchmark evaluation."""
    print("=" * 80, flush=True)
    print("ONEINBOX AGENT EVALUATION & REGRESSION SUITE: BASE vs DPO", flush=True)
    print("=" * 80, flush=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    db = Database()
    await db.initialize()

    base_version = "qwen2.5_0.5b_base"
    dpo_version = "qwen2.5_0.5b_dpo"

    if not skip_runs:
        # Load scenarios
        runner_helper = TestRunner(db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await runner_helper.initialize()
        scenarios = runner_helper.list_scenarios()
        print(f"Loaded {len(scenarios)} benchmark scenarios.", flush=True)

        # -----------------------------------------------------------------------
        # Phase 1: Evaluate Base Open-Source Model
        # -----------------------------------------------------------------------
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 1: BENCHMARKING BASE MODEL [{base_version}]", flush=True)
        print(f"Base Model: {base_model_name}", flush=True)
        print("#" * 70, flush=True)

        base_llm = LocalHuggingFaceAdapter(
            base_model_name=base_model_name,
            adapter_path=None,
        )
        base_agent = RealtorAgent(llm=base_llm, agent_version=base_version)
        base_runner = TestRunner(agent=base_agent, db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await base_runner.initialize()

        for i, s in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] Running Scenario {s.id}: {s.name} ...", flush=True)
            t0 = time.perf_counter()
            res = await base_runner.run_scenario(s.id)
            elapsed = time.perf_counter() - t0
            eval_dict = res.get("evaluation", {})
            passed = eval_dict.get("passed", False)
            print(
                f"   -> Result: {'PASS' if passed else 'FAIL'} "
                f"(Total latency: {elapsed:.2f}s, Score: {eval_dict.get('overall_score', 0):.2f})",
                flush=True,
            )

        # Free base model from VRAM if possible before loading DPO model
        del base_runner
        del base_agent
        del base_llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------------
        # Phase 2: Evaluate DPO Fine-Tuned Model
        # -----------------------------------------------------------------------
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 2: BENCHMARKING DPO MODEL [{dpo_version}]", flush=True)
        print(f"Base Model: {base_model_name} + Adapter: {adapter_path}", flush=True)
        print("#" * 70, flush=True)

        dpo_llm = LocalHuggingFaceAdapter(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
        )
        dpo_agent = RealtorAgent(llm=dpo_llm, agent_version=dpo_version)
        dpo_runner = TestRunner(agent=dpo_agent, db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await dpo_runner.initialize()

        for i, s in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] Running Scenario {s.id}: {s.name} ...", flush=True)
            t0 = time.perf_counter()
            res = await dpo_runner.run_scenario(s.id)
            elapsed = time.perf_counter() - t0
            eval_dict = res.get("evaluation", {})
            passed = eval_dict.get("passed", False)
            print(
                f"   -> Result: {'PASS' if passed else 'FAIL'} "
                f"(Total latency: {elapsed:.2f}s, Score: {eval_dict.get('overall_score', 0):.2f})",
                flush=True,
            )
    else:
        print("\nSkipping live model runs (using existing completed runs from database).", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Regression Engine Comparison
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70, flush=True)
    print(f"PHASE 3: RUNNING REGRESSION ENGINE [{base_version} vs {dpo_version}]", flush=True)
    print("#" * 70, flush=True)

    comparison: RegressionResult = await compare_versions(
        base_version=base_version,
        compare_version=dpo_version,
        db=db,
    )

    # Print regression matrix
    print("\n" + "=" * 95, flush=True)
    print(f"{'SCENARIO':<10} | {'BASE':<6} | {'DPO':<6} | {'STATUS':<14} | {'SEVERITY':<10} | {'LATENCY DELTA':<14} | {'REASON'}", flush=True)
    print("-" * 95, flush=True)

    comparison_items: list[dict[str, Any]] = []
    for sc in comparison.scenario_comparisons:
        base_status = "PASS" if sc.base_overall_passed else "FAIL"
        comp_status = "PASS" if sc.compare_overall_passed else "FAIL"
        class_str = sc.classification.value.upper()
        sev_str = sc.severity.value.upper() if sc.severity else "NONE"
        delta_lat = sc.compare_latency.end_to_end_ms - sc.base_latency.end_to_end_ms
        lat_delta_str = f"{delta_lat:+.0f} ms"
        reason = "; ".join(sc.change_reasons) if sc.change_reasons else "N/A"
        print(f"{sc.scenario_id:<10} | {base_status:<6} | {comp_status:<6} | {class_str:<14} | {sev_str:<10} | {lat_delta_str:<14} | {reason}", flush=True)

        comparison_items.append({
            "scenario_id": sc.scenario_id,
            "scenario_name": sc.scenario_name,
            "base_passed": sc.base_overall_passed,
            "compare_passed": sc.compare_overall_passed,
            "base_score": sc.base_overall_score,
            "compare_score": sc.compare_overall_score,
            "classification": sc.classification.value,
            "severity": sc.severity.value if sc.severity else "none",
            "base_latency_ms": sc.base_latency.end_to_end_ms,
            "compare_latency_ms": sc.compare_latency.end_to_end_ms,
            "delta_latency_ms": delta_lat,
            "reasons": sc.change_reasons,
        })
    print("=" * 95, flush=True)

    # Gate decision
    gate_decision = comparison.deployment_recommendation.value
    gate_reasons = comparison.deployment_reasons
    print(f"\nDEPLOYMENT GATE RECOMMENDATION: {gate_decision.upper()}", flush=True)
    for r in gate_reasons:
        print(f"  - {r}", flush=True)

    # Aggregate metric scores
    base_evals = await db.get_evaluations_by_version(base_version)
    dpo_evals = await db.get_evaluations_by_version(dpo_version)

    def calc_stats(eval_list: list[EvaluationResult]) -> dict[str, Any]:
        if not eval_list:
            return {}
        by_s: dict[str, EvaluationResult] = {}
        for e in eval_list:
            if e.scenario_id not in by_s:
                by_s[e.scenario_id] = e
        evals = list(by_s.values())
        n = len(evals)
        pass_count = sum(1 for e in evals if e.overall_passed)
        avg_task = sum(e.task_success.score for e in evals) / n
        avg_tool = sum(e.tool_correctness.score for e in evals) / n
        avg_constraint = sum(e.constraint_adherence.score for e in evals) / n
        avg_latency_ms = sum(
            float((e.latency.evidence or {}).get("total_ms", e.latency.score))
            for e in evals
        ) / n
        return {
            "total": n,
            "passed": pass_count,
            "pass_rate": round(pass_count / n, 4),
            "avg_task_success": round(avg_task, 4),
            "avg_tool_correctness": round(avg_tool, 4),
            "avg_constraint_adherence": round(avg_constraint, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
        }

    base_stats = calc_stats(base_evals)
    dpo_stats = calc_stats(dpo_evals)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model_name,
        "base_version": base_version,
        "dpo_version": dpo_version,
        "adapter_path": str(adapter_path),
        "base_stats": base_stats,
        "dpo_stats": dpo_stats,
        "regression_summary": {
            "total_scenarios": len(comparison.scenario_comparisons),
            "improved": comparison.improved,
            "regressed": comparison.regressed,
            "unchanged": comparison.unchanged,
            "inconclusive": comparison.inconclusive,
            "score_deltas": {
                "task_success_delta": comparison.task_success_delta,
                "tool_correctness_delta": comparison.tool_correctness_delta,
                "constraint_adherence_delta": comparison.constraint_adherence_delta,
                "latency_delta_ms": comparison.latency_delta_ms,
            },
        },
        "deployment_gate": {
            "recommendation": gate_decision,
            "reasons": gate_reasons,
        },
        "scenario_matrix": comparison_items,
    }

    # Save JSON report
    json_report_file = out_path / "dpo_experiment_report.json"
    with open(json_report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved structured report to {json_report_file}", flush=True)

    # Save Markdown report
    md_report_file = out_path / "dpo_experiment_report.md"
    with open(md_report_file, "w", encoding="utf-8") as f:
        f.write(f"""# DPO Experiment & Regression Report

- **Date**: {report_data['timestamp']}
- **Base Model**: `{base_model_name}` (`{base_version}`)
- **DPO Model**: `{base_model_name}` + LoRA (`{dpo_version}`)
- **Deployment Recommendation**: **`{gate_decision.upper()}`**

---

## Executive Summary

| Metric | Base Model (`{base_version}`) | DPO Model (`{dpo_version}`) | Delta |
| :--- | :--- | :--- | :--- |
| **Pass Rate** | {base_stats.get('passed', 0)}/{base_stats.get('total', 0)} ({base_stats.get('pass_rate', 0):.1%}) | {dpo_stats.get('passed', 0)}/{dpo_stats.get('total', 0)} ({dpo_stats.get('pass_rate', 0):.1%}) | {dpo_stats.get('pass_rate', 0) - base_stats.get('pass_rate', 0):+.1%} |
| **Task Success** | {base_stats.get('avg_task_success', 0):.2f} | {dpo_stats.get('avg_task_success', 0):.2f} | {dpo_stats.get('avg_task_success', 0) - base_stats.get('avg_task_success', 0):+.2f} |
| **Tool Correctness** | {base_stats.get('avg_tool_correctness', 0):.2f} | {dpo_stats.get('avg_tool_correctness', 0):.2f} | {dpo_stats.get('avg_tool_correctness', 0) - base_stats.get('avg_tool_correctness', 0):+.2f} |
| **Constraint Adherence** | {base_stats.get('avg_constraint_adherence', 0):.2f} | {dpo_stats.get('avg_constraint_adherence', 0):.2f} | {dpo_stats.get('avg_constraint_adherence', 0) - base_stats.get('avg_constraint_adherence', 0):+.2f} |
| **Average Latency** | {base_stats.get('avg_latency_ms', 0):.0f} ms | {dpo_stats.get('avg_latency_ms', 0):.0f} ms | {dpo_stats.get('avg_latency_ms', 0) - base_stats.get('avg_latency_ms', 0):+.0f} ms |

---

## Scenario Regression Matrix

| Scenario | Base Outcome | DPO Outcome | Classification | Severity | Latency Delta | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""")
        for item in comparison_items:
            b_p = "PASS" if item['base_passed'] else "FAIL"
            d_p = "PASS" if item['compare_passed'] else "FAIL"
            reasons_txt = "<br>".join(item['reasons']) if item['reasons'] else "No regressions"
            f.write(f"| `{item['scenario_id']}` | {b_p} | {d_p} | **{item['classification'].upper()}** | {item['severity'].upper()} | {item['delta_latency_ms']:+.0f} ms | {reasons_txt} |\n")

        f.write(f"""
---

## Deployment Gate Decision

**Recommendation**: `{gate_decision.upper()}`

**Justification**:
""")
        for r in gate_reasons:
            f.write(f"- {r}\n")

    print(f"Saved markdown report to {md_report_file}", flush=True)
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Base vs DPO Model Benchmark Evals")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default="experiments/dpo_qwen05b_v1/adapter")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--scenarios_path", default=None)
    parser.add_argument("--skip_runs", action="store_true", help="Skip live scenario execution and use existing DB runs")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation_suite(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            scenarios_path=args.scenarios_path,
            skip_runs=args.skip_runs,
        )
    )
