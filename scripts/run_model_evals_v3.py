"""End-to-End Evaluation & Regression Analysis for Experiment 3 (Base V3 vs DPO V3).

Strictly controlled evaluation:
- Identical scenario definitions (8 real-estate scenarios RE-001 to RE-008)
- Identical system prompts and tool definitions
- Identical decoding parameters (temperature=0.0)
- Identical hardware/runtime configuration
- Evaluates NO-TOOL holdout accuracy on RE-004 and AMB-001

Produces:
- reports/dpo_experiment_v3_report.md
- reports/dpo_experiment_v3_report.json
- experiments/dpo_qwen05b_v3/evaluation_results.json
- experiments/dpo_qwen05b_v3/regression_report.json
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


async def run_evaluation_suite_v3(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "experiments/dpo_qwen05b_v3/adapter",
    output_dir: str = "reports",
    scenarios_path: str | None = None,
    skip_runs: bool = False,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("ONEINBOX EXPERIMENT 3 BENCHMARK: BASE V3 vs DPO V3 (TRAJECTORY PREFERENCES)", flush=True)
    print("=" * 80, flush=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_dir = Path("experiments/dpo_qwen05b_v3")
    exp_dir.mkdir(parents=True, exist_ok=True)

    db = Database()
    await db.initialize()

    base_version = "qwen2.5_0.5b_base_v3"
    dpo_version = "qwen2.5_0.5b_dpo_v3"

    if not skip_runs:
        # Load canonical 8 scenarios
        runner_helper = TestRunner(db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await runner_helper.initialize()
        scenarios = runner_helper.list_scenarios()
        print(f"Loaded {len(scenarios)} benchmark scenarios.", flush=True)

        # -----------------------------------------------------------------------
        # Phase 1: Benchmark Base Model
        # -----------------------------------------------------------------------
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 1: BENCHMARKING BASE MODEL [{base_version}]", flush=True)
        print(f"Base Model: {base_model_name} (temperature=0.0)", flush=True)
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

        # Free base model from VRAM before loading DPO V3
        del base_runner
        del base_agent
        del base_llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------------
        # Phase 2: Benchmark DPO V3 Model
        # -----------------------------------------------------------------------
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 2: BENCHMARKING DPO V3 MODEL [{dpo_version}]", flush=True)
        print(f"Base Model: {base_model_name} + Adapter: {adapter_path} (temperature=0.0)", flush=True)
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
    print(f"{'SCENARIO':<10} | {'BASE':<6} | {'DPO V3':<6} | {'STATUS':<14} | {'SEVERITY':<10} | {'LATENCY DELTA':<14} | {'REASON'}", flush=True)
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

    # Fetch raw runs for detailed analysis (tool calls, messages)
    base_runs = await db.list_runs(agent_version=base_version)
    dpo_runs = await db.list_runs(agent_version=dpo_version)
    base_evals = await db.get_evaluations_by_version(base_version)
    dpo_evals = await db.get_evaluations_by_version(dpo_version)


    base_eval_map = {e.scenario_id: e for e in base_evals}
    dpo_eval_map = {e.scenario_id: e for e in dpo_evals}
    base_run_map = {r.scenario_id: r for r in base_runs}
    dpo_run_map = {r.scenario_id: r for r in dpo_runs}

    # Focus on RE-004 behavioral inspection
    re004_base_run = base_run_map.get("RE-004")
    re004_dpo_run = dpo_run_map.get("RE-004")
    re004_base_eval = base_eval_map.get("RE-004")
    re004_dpo_eval = dpo_eval_map.get("RE-004")

    base_pass_rate = sum(1 for e in base_evals if e.overall_passed) / max(len(base_evals), 1)
    dpo_pass_rate = sum(1 for e in dpo_evals if e.overall_passed) / max(len(dpo_evals), 1)

    # Construct Evaluation Results Document
    eval_results = {
        "benchmark": "OneInbox Real-Estate Suite (8 Scenarios)",
        "experiment": "DPO Experiment V3: Trajectory-Aware / Structured Tool-Use Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": {
            "version": base_version,
            "model_name": base_model_name,
            "adapter": None,
            "overall_pass_rate": base_pass_rate,
            "passed_count": sum(1 for e in base_evals if e.overall_passed),
            "total_count": len(base_evals),
        },
        "dpo_model": {
            "version": dpo_version,
            "model_name": base_model_name,
            "adapter": adapter_path,
            "overall_pass_rate": dpo_pass_rate,
            "passed_count": sum(1 for e in dpo_evals if e.overall_passed),
            "total_count": len(dpo_evals),
        },
        "comparison_matrix": comparison_items,
        "regression_summary": {
            "regressions_count": len(comparison.regressed),
            "improvements_count": len(comparison.improved),
            "neutral_count": len(comparison.unchanged),
            "inconclusive_count": len(comparison.inconclusive),
            "deployment_recommendation": gate_decision,
            "deployment_reasons": gate_reasons,
        },
        "re004_analysis": {
            "base_passed": re004_base_eval.overall_passed if re004_base_eval else False,
            "dpo_passed": re004_dpo_eval.overall_passed if re004_dpo_eval else False,
            "base_tools_called": [tc.tool_name for tc in (re004_base_run.actual_tool_calls if re004_base_run else [])],
            "dpo_tools_called": [tc.tool_name for tc in (re004_dpo_run.actual_tool_calls if re004_dpo_run else [])],
            "base_actual_outcome": re004_base_run.actual_outcome if re004_base_run else None,
            "dpo_actual_outcome": re004_dpo_run.actual_outcome if re004_dpo_run else None,
        },
    }

    # Save JSON files
    with open(exp_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    with open(exp_dir / "regression_report.json", "w", encoding="utf-8") as f:
        json.dump(comparison.model_dump(), f, indent=2, default=str)

    with open(out_path / "dpo_experiment_v3_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # -----------------------------------------------------------------------
    # Generate Markdown Report
    # -----------------------------------------------------------------------
    md_lines = [
        "# DPO Experiment V3 Evaluation Report: Trajectory-Aware Preference Learning",
        "",
        "## Executive Summary",
        "",
        f"- **Experiment ID**: `dpo_qwen05b_v3`",
        f"- **Paradigm**: Trajectory-Aware / Structured Tool-Use Preference Optimization (`<tool_call>` vs no-tool)",
        f"- **Base Model**: `{base_model_name}` (`{base_version}`)",
        f"- **DPO V3 Model**: `{base_model_name}` + LoRA (`{dpo_version}`)",
        f"- **Deployment Gate**: **{gate_decision.upper()}**",
        "",
        "### Benchmark Score Comparison",
        "",
        f"| Version | Scenarios Evaluated | Passed | Pass Rate | Average Latency |",
        f"|---|---|---|---|---|",
        f"| **Base Model V3** | {len(base_evals)} | {sum(1 for e in base_evals if e.overall_passed)} | {base_pass_rate * 100:.1f}% | {sum(r.total_latency_ms for r in base_runs)/max(len(base_runs),1):.0f} ms |",
        f"| **DPO Model V3** | {len(dpo_evals)} | {sum(1 for e in dpo_evals if e.overall_passed)} | {dpo_pass_rate * 100:.1f}% | {sum(r.total_latency_ms for r in dpo_runs)/max(len(dpo_runs),1):.0f} ms |",
        "",
        "## Scenario Comparison Matrix",
        "",
        "| Scenario ID | Name | Base Status | DPO V3 Status | Classification | Severity | Latency Delta |",
        "|---|---|---|---|---|---|---|",
    ]


    for item in comparison_items:
        b_st = "✅ PASS" if item["base_passed"] else "❌ FAIL"
        d_st = "✅ PASS" if item["compare_passed"] else "❌ FAIL"
        cls_st = item["classification"].upper()
        sev = item["severity"].upper()
        d_lat = f"{item['delta_latency_ms']:+.0f} ms"
        md_lines.append(f"| `{item['scenario_id']}` | {item['scenario_name']} | {b_st} | {d_st} | `{cls_st}` | `{sev}` | {d_lat} |")

    md_lines.extend([
        "",
        "## Target Scenario Deep Dive: RE-004 (Ambiguous Inquiry / No-Tool Decision)",
        "",
        f"- **Scenario Expectation**: Customer provides vague description without ID/address. Expected outcome is `clarification_requested` with **0 tool calls** (`expected_tool_calls: []`).",
        f"- **Base Model V3 Behavior**: Outcome = `{eval_results['re004_analysis']['base_actual_outcome']}`, Tools Called = `{eval_results['re004_analysis']['base_tools_called']}`",
        f"- **DPO Model V3 Behavior**: Outcome = `{eval_results['re004_analysis']['dpo_actual_outcome']}`, Tools Called = `{eval_results['re004_analysis']['dpo_tools_called']}`",
        "",
        "## Deployment Gate Recommendation",
        "",
        f"> **Recommendation**: `{gate_decision.upper()}`",
        "",
        "### Decision Reasons:",
    ])
    for r in gate_reasons:
        md_lines.append(f"- {r}")

    report_md_path = out_path / "dpo_experiment_v3_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nGenerated evaluation reports in {report_md_path} and {exp_dir}!", flush=True)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Experiment V3")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running scenarios and use existing DB records")
    args = parser.parse_args()

    asyncio.run(run_evaluation_suite_v3(skip_runs=args.skip_runs))
