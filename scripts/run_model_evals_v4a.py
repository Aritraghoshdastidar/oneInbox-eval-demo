"""End-to-End Evaluation & Regression Analysis for Experiment 4A.

Evaluates Base V4A vs DPO V4A on:
- Canonical 8 real-estate scenarios (RE-001 to RE-008)
- AMB-001 holdout scenario
- Conditional Tool Decision Accuracy (Precision, Recall)
- Tool Argument & Order Accuracy
- Full Regression Analysis and Deployment Gate

Produces:
- reports/dpo_experiment_v4a_report.md
- reports/dpo_experiment_v4a_report.json
- experiments/dpo_qwen05b_v4a/evaluation_results.json
- experiments/dpo_qwen05b_v4a/regression_report.json
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
    AgentRun,
    DeploymentRecommendation,
    EvaluationResult,
    RegressionResult,
    ScenarioClassification,
)
from app.regression.comparator import compare_versions
from app.storage.database import Database
from app.testing.runner import TestRunner


async def run_evaluation_suite_v4a(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "experiments/dpo_qwen05b_v4a/adapter",
    output_dir: str = "reports",
    scenarios_path: str | None = None,
    skip_runs: bool = False,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("ONEINBOX EXPERIMENT 4A BENCHMARK: BASE V4A vs DPO V4A (BALANCED DATASET)", flush=True)
    print("=" * 80, flush=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_dir = Path("experiments/dpo_qwen05b_v4a")
    exp_dir.mkdir(parents=True, exist_ok=True)

    db = Database()
    await db.initialize()

    base_version = "qwen2.5_0.5b_base_v4a"
    dpo_version = "qwen2.5_0.5b_dpo_v4a"

    if not skip_runs:
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
            passed = res.get("overall_passed", False)
            score = res.get("overall_score", 0.0)
            print(f"   -> Result: {'PASS' if passed else 'FAIL'} (Latency: {elapsed:.2f}s, Score: {score:.2f})", flush=True)

        # Also evaluate AMB-001 holdout on base
        amb_runner_base = TestRunner(agent=base_agent, db=db, scenarios_path="scenarios/ambiguity_scenarios.json")
        await amb_runner_base.initialize()
        print("\nRunning Holdout Scenario AMB-001 on Base Model ...", flush=True)
        await amb_runner_base.run_scenario("AMB-001", version_override=base_version + "_amb")

        # Free base model from VRAM before loading DPO V4A
        del base_runner, amb_runner_base, base_agent, base_llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------------
        # Phase 2: Benchmark DPO V4A Model
        # -----------------------------------------------------------------------
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 2: BENCHMARKING DPO V4A MODEL [{dpo_version}]", flush=True)
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
            passed = res.get("overall_passed", False)
            score = res.get("overall_score", 0.0)
            print(f"   -> Result: {'PASS' if passed else 'FAIL'} (Latency: {elapsed:.2f}s, Score: {score:.2f})", flush=True)

        # Also evaluate AMB-001 holdout on DPO V4A
        amb_runner_dpo = TestRunner(agent=dpo_agent, db=db, scenarios_path="scenarios/ambiguity_scenarios.json")
        await amb_runner_dpo.initialize()
        print("\nRunning Holdout Scenario AMB-001 on DPO V4A Model ...", flush=True)
        await amb_runner_dpo.run_scenario("AMB-001", version_override=dpo_version + "_amb")

        del dpo_runner, amb_runner_dpo, dpo_agent, dpo_llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    print(f"{'SCENARIO':<10} | {'BASE':<6} | {'DPO V4A':<7} | {'STATUS':<14} | {'SEVERITY':<10} | {'LATENCY DELTA':<14} | {'REASON'}", flush=True)
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
        print(f"{sc.scenario_id:<10} | {base_status:<6} | {comp_status:<7} | {class_str:<14} | {sev_str:<10} | {lat_delta_str:<14} | {reason}", flush=True)

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

    # Fetch AMB-001 holdout runs
    base_amb_runs = await db.list_runs(agent_version=base_version + "_amb")
    dpo_amb_runs = await db.list_runs(agent_version=dpo_version + "_amb")
    base_amb_evals = await db.get_evaluations_by_version(base_version + "_amb")
    dpo_amb_evals = await db.get_evaluations_by_version(dpo_version + "_amb")

    base_amb001_run = base_amb_runs[0] if base_amb_runs else None
    dpo_amb001_run = dpo_amb_runs[0] if dpo_amb_runs else None
    base_amb001_eval = base_amb_evals[0] if base_amb_evals else None
    dpo_amb001_eval = dpo_amb_evals[0] if dpo_amb_evals else None

    # Calculate Conditional Tool Decision Accuracy
    # Load scenarios to know expected tool calls
    with open("scenarios/scenarios.json", "r", encoding="utf-8") as f:
        scenarios_data = json.load(f)["scenarios"]
    sc_map = {s["id"]: s for s in scenarios_data}

    def compute_tool_metrics(runs: list[AgentRun]) -> dict[str, Any]:
        tp_tool = 0  # Tool required & called
        fp_tool = 0  # Tool not required but called
        tn_tool = 0  # Tool not required & not called
        fn_tool = 0  # Tool required but not called
        arg_correct = 0
        arg_total = 0
        order_correct = 0
        order_total = 0

        for r in runs:
            sc = sc_map.get(r.scenario_id)
            if not sc:
                continue
            exp_tools = sc.get("expected_tool_calls", [])
            act_tools = r.actual_tool_calls

            if len(exp_tools) == 0:
                # No tool scenario (RE-004)
                if len(act_tools) == 0:
                    tn_tool += 1
                else:
                    fp_tool += 1
            else:
                # Required tool scenario
                if len(act_tools) > 0:
                    tp_tool += 1
                    # Check arguments for expected tools
                    for et in exp_tools:
                        arg_total += 1
                        matching = [at for at in act_tools if at.tool_name == et["tool_name"]]
                        if matching:
                            # check args subset match
                            match_found = False
                            for m in matching:
                                args_match = all(str(m.arguments.get(k)) == str(v) for k, v in et.get("arguments", {}).items())
                                if args_match:
                                    match_found = True
                                    break
                            if match_found:
                                arg_correct += 1
                else:
                    fn_tool += 1

                # Check ordering if both check_availability and book_appointment expected
                exp_names = [et["tool_name"] for et in exp_tools]
                if "check_availability" in exp_names and "book_appointment" in exp_names:
                    order_total += 1
                    act_names = [at.tool_name for at in act_tools]
                    if "check_availability" in act_names and "book_appointment" in act_names:
                        idx_check = act_names.index("check_availability")
                        idx_book = act_names.index("book_appointment")
                        if idx_check < idx_book:
                            order_correct += 1

        total_decisions = tp_tool + fp_tool + tn_tool + fn_tool
        accuracy = (tp_tool + tn_tool) / max(total_decisions, 1)
        no_tool_prec = tn_tool / max(tn_tool + fn_tool, 1)
        no_tool_rec = tn_tool / max(tn_tool + fp_tool, 1)
        req_tool_prec = tp_tool / max(tp_tool + fp_tool, 1)
        req_tool_rec = tp_tool / max(tp_tool + fn_tool, 1)
        arg_acc = arg_correct / max(arg_total, 1)
        order_acc = order_correct / max(order_total, 1)

        return {
            "total_decisions": total_decisions,
            "accuracy": round(accuracy, 4),
            "no_tool_precision": round(no_tool_prec, 4),
            "no_tool_recall": round(no_tool_rec, 4),
            "required_tool_precision": round(req_tool_prec, 4),
            "required_tool_recall": round(req_tool_rec, 4),
            "argument_accuracy": round(arg_acc, 4),
            "argument_correct": arg_correct,
            "argument_total": arg_total,
            "order_accuracy": round(order_acc, 4),
            "order_correct": order_correct,
            "order_total": order_total,
        }

    base_tool_metrics = compute_tool_metrics(base_runs)
    dpo_tool_metrics = compute_tool_metrics(dpo_runs)

    base_pass_rate = sum(1 for e in base_evals if e.overall_passed) / max(len(base_evals), 1)
    dpo_pass_rate = sum(1 for e in dpo_evals if e.overall_passed) / max(len(dpo_evals), 1)

    eval_results = {
        "benchmark": "OneInbox Real-Estate Suite (8 Scenarios)",
        "experiment": "DPO Experiment V4A: Balanced Trajectory-Aware Preference Learning",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": {
            "version": base_version,
            "model_name": base_model_name,
            "adapter": None,
            "overall_pass_rate": base_pass_rate,
            "passed_count": sum(1 for e in base_evals if e.overall_passed),
            "total_count": len(base_evals),
            "conditional_tool_metrics": base_tool_metrics,
        },
        "dpo_model": {
            "version": dpo_version,
            "model_name": base_model_name,
            "adapter": adapter_path,
            "overall_pass_rate": dpo_pass_rate,
            "passed_count": sum(1 for e in dpo_evals if e.overall_passed),
            "total_count": len(dpo_evals),
            "conditional_tool_metrics": dpo_tool_metrics,
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
            "base_passed": base_eval_map["RE-004"].overall_passed if "RE-004" in base_eval_map else False,
            "dpo_passed": dpo_eval_map["RE-004"].overall_passed if "RE-004" in dpo_eval_map else False,
            "base_tools_called": [tc.tool_name for tc in (base_run_map["RE-004"].actual_tool_calls if "RE-004" in base_run_map else [])],
            "dpo_tools_called": [tc.tool_name for tc in (dpo_run_map["RE-004"].actual_tool_calls if "RE-004" in dpo_run_map else [])],
        },
        "amb001_holdout_analysis": {
            "base_passed": base_amb001_eval.overall_passed if base_amb001_eval else False,
            "dpo_passed": dpo_amb001_eval.overall_passed if dpo_amb001_eval else False,
            "base_tools_called": [tc.tool_name for tc in (base_amb001_run.actual_tool_calls if base_amb001_run else [])],
            "dpo_tools_called": [tc.tool_name for tc in (dpo_amb001_run.actual_tool_calls if dpo_amb001_run else [])],
        },
    }

    # Save JSON files
    with open(exp_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    with open(exp_dir / "regression_report.json", "w", encoding="utf-8") as f:
        json.dump(comparison.model_dump(), f, indent=2, default=str)

    with open(out_path / "dpo_experiment_v4a_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation reports saved to {exp_dir} and {out_path}!", flush=True)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Experiment V4A")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running scenarios and use existing DB records")
    args = parser.parse_args()

    asyncio.run(run_evaluation_suite_v4a(skip_runs=args.skip_runs))
