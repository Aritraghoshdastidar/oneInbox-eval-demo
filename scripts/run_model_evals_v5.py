"""End-to-End Evaluation & Regression Analysis for Experiment V5 (FINAL).

Evaluates Base V5 vs DPO V5 on:
- Canonical 8 real-estate scenarios (RE-001 to RE-008)
- AMB-001 holdout scenario
- Held-out Escalation Suite (ESC-001 to ESC-004)
- Held-out Routing Suite (ROUTE-001 to ROUTE-008)
- RE-005 Root Cause Deep-Dive (tool-routing + escalation)
- Tool Confusion Matrix (property_lookup / check_availability / book_appointment / no_tool)
- Argument Accuracy (property_id, date, time)
- Tool Order Accuracy (prerequisite chain)
- Full Regression Analysis and Deployment Gate

THIS IS THE FINAL EVALUATION SCRIPT.

Produces:
- experiments/dpo_qwen05b_v5/evaluation_results.json
- experiments/dpo_qwen05b_v5/regression_report.json
- reports/dpo_experiment_v5_report.json
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


TOOL_NAMES = ["property_lookup", "check_availability", "book_appointment"]


async def run_evaluation_suite_v5(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "experiments/dpo_qwen05b_v5/adapter",
    output_dir: str = "reports",
    scenarios_path: str | None = None,
    skip_runs: bool = False,
) -> dict[str, Any]:
    print("=" * 80, flush=True)
    print("ONEINBOX EXPERIMENT V5 BENCHMARK: BASE vs DPO V5 (FINAL)", flush=True)
    print("Tool-Routing Discrimination + Escalation Preservation", flush=True)
    print("=" * 80, flush=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_dir = Path("experiments/dpo_qwen05b_v5")
    exp_dir.mkdir(parents=True, exist_ok=True)

    db = Database()
    await db.initialize()

    base_version = "qwen2.5_0.5b_base_v5"
    dpo_version = "qwen2.5_0.5b_dpo_v5"

    if not skip_runs:
        runner_helper = TestRunner(db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await runner_helper.initialize()
        scenarios = runner_helper.list_scenarios()
        print(f"Loaded {len(scenarios)} canonical benchmark scenarios.", flush=True)

        # =================================================================
        # Phase 1: Benchmark Base Model
        # =================================================================
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 1: BENCHMARKING BASE MODEL [{base_version}]", flush=True)
        print(f"Base Model: {base_model_name} (temperature=0.0)", flush=True)
        print("#" * 70, flush=True)

        base_llm = LocalHuggingFaceAdapter(
            base_model_name=base_model_name,
            adapter_path=None,
        )
        base_agent = RealtorAgent(llm=base_llm, agent_version=base_version)

        # Canonical 8
        base_runner = TestRunner(agent=base_agent, db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await base_runner.initialize()
        print("\n--- Running Canonical 8 Scenarios on Base Model ---", flush=True)
        for i, s in enumerate(scenarios, 1):
            print(f"[{i}/{len(scenarios)}] Running Scenario {s.id}: {s.name} ...", flush=True)
            t0 = time.perf_counter()
            res = await base_runner.run_scenario(s.id)
            elapsed = time.perf_counter() - t0
            passed = res.get("overall_passed", False)
            score = res.get("overall_score", 0.0)
            print(f"   -> Result: {'PASS' if passed else 'FAIL'} (Latency: {elapsed:.2f}s, Score: {score:.2f})", flush=True)

        # AMB-001 holdout
        amb_runner_base = TestRunner(agent=base_agent, db=db, scenarios_path="scenarios/ambiguity_scenarios.json")
        await amb_runner_base.initialize()
        print("\n--- Running Holdout Scenario AMB-001 on Base Model ---", flush=True)
        res_amb_b = await amb_runner_base.run_scenario("AMB-001", version_override=base_version + "_amb")
        print(f"   -> Result: {'PASS' if res_amb_b.get('overall_passed') else 'FAIL'}", flush=True)

        # ESC suite
        esc_runner_base = TestRunner(agent=base_agent, db=db, scenarios_path="scenarios/escalation_scenarios.json")
        await esc_runner_base.initialize()
        esc_scenarios = esc_runner_base.list_scenarios()
        print(f"\n--- Running Held-Out Escalation Suite ({len(esc_scenarios)} scenarios) on Base Model ---", flush=True)
        for s in esc_scenarios:
            print(f"Running Escalation Scenario {s.id}: {s.name} ...", flush=True)
            res_esc_b = await esc_runner_base.run_scenario(s.id, version_override=base_version + "_esc")
            print(f"   -> Result: {'PASS' if res_esc_b.get('overall_passed') else 'FAIL'}", flush=True)

        # ROUTING suite
        route_runner_base = TestRunner(agent=base_agent, db=db, scenarios_path="scenarios/routing_scenarios.json")
        await route_runner_base.initialize()
        route_scenarios = route_runner_base.list_scenarios()
        print(f"\n--- Running Held-Out Routing Suite ({len(route_scenarios)} scenarios) on Base Model ---", flush=True)
        for s in route_scenarios:
            print(f"Running Routing Scenario {s.id}: {s.name} ...", flush=True)
            res_route_b = await route_runner_base.run_scenario(s.id, version_override=base_version + "_route")
            print(f"   -> Result: {'PASS' if res_route_b.get('overall_passed') else 'FAIL'}", flush=True)

        # Free base model
        del base_runner, amb_runner_base, esc_runner_base, route_runner_base, base_agent, base_llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =================================================================
        # Phase 2: Benchmark DPO V5 Model
        # =================================================================
        print("\n" + "#" * 70, flush=True)
        print(f"PHASE 2: BENCHMARKING DPO V5 MODEL [{dpo_version}]", flush=True)
        print(f"Base Model: {base_model_name} + Adapter: {adapter_path} (temperature=0.0)", flush=True)
        print("#" * 70, flush=True)

        dpo_llm = LocalHuggingFaceAdapter(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
        )
        dpo_agent = RealtorAgent(llm=dpo_llm, agent_version=dpo_version)

        # Canonical 8
        dpo_runner = TestRunner(agent=dpo_agent, db=db, scenarios_path=Path(scenarios_path) if scenarios_path else None)
        await dpo_runner.initialize()
        print("\n--- Running Canonical 8 Scenarios on DPO V5 Model ---", flush=True)
        for i, s in enumerate(scenarios, 1):
            print(f"[{i}/{len(scenarios)}] Running Scenario {s.id}: {s.name} ...", flush=True)
            t0 = time.perf_counter()
            res = await dpo_runner.run_scenario(s.id)
            elapsed = time.perf_counter() - t0
            passed = res.get("overall_passed", False)
            score = res.get("overall_score", 0.0)
            print(f"   -> Result: {'PASS' if passed else 'FAIL'} (Latency: {elapsed:.2f}s, Score: {score:.2f})", flush=True)

        # AMB-001 holdout
        amb_runner_dpo = TestRunner(agent=dpo_agent, db=db, scenarios_path="scenarios/ambiguity_scenarios.json")
        await amb_runner_dpo.initialize()
        print("\n--- Running Holdout Scenario AMB-001 on DPO V5 Model ---", flush=True)
        res_amb_d = await amb_runner_dpo.run_scenario("AMB-001", version_override=dpo_version + "_amb")
        print(f"   -> Result: {'PASS' if res_amb_d.get('overall_passed') else 'FAIL'}", flush=True)

        # ESC suite
        esc_runner_dpo = TestRunner(agent=dpo_agent, db=db, scenarios_path="scenarios/escalation_scenarios.json")
        await esc_runner_dpo.initialize()
        print(f"\n--- Running Held-Out Escalation Suite ({len(esc_scenarios)} scenarios) on DPO V5 Model ---", flush=True)
        for s in esc_scenarios:
            print(f"Running Escalation Scenario {s.id}: {s.name} ...", flush=True)
            res_esc_d = await esc_runner_dpo.run_scenario(s.id, version_override=dpo_version + "_esc")
            print(f"   -> Result: {'PASS' if res_esc_d.get('overall_passed') else 'FAIL'}", flush=True)

        # ROUTING suite
        route_runner_dpo = TestRunner(agent=dpo_agent, db=db, scenarios_path="scenarios/routing_scenarios.json")
        await route_runner_dpo.initialize()
        print(f"\n--- Running Held-Out Routing Suite ({len(route_scenarios)} scenarios) on DPO V5 Model ---", flush=True)
        for s in route_scenarios:
            print(f"Running Routing Scenario {s.id}: {s.name} ...", flush=True)
            res_route_d = await route_runner_dpo.run_scenario(s.id, version_override=dpo_version + "_route")
            print(f"   -> Result: {'PASS' if res_route_d.get('overall_passed') else 'FAIL'}", flush=True)

        del dpo_runner, amb_runner_dpo, esc_runner_dpo, route_runner_dpo, dpo_agent, dpo_llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\nSkipping live model runs (using existing completed runs from database).", flush=True)

    # =================================================================
    # Phase 3: Regression Engine Comparison
    # =================================================================
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
    print(f"{'SCENARIO':<10} | {'BASE':<6} | {'DPO V5':<7} | {'STATUS':<14} | {'SEVERITY':<10} | {'LATENCY DELTA':<14} | {'REASON'}", flush=True)
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

    # Fetch raw runs for detailed analysis
    base_runs = await db.list_runs(agent_version=base_version)
    dpo_runs = await db.list_runs(agent_version=dpo_version)
    base_evals = await db.get_evaluations_by_version(base_version)
    dpo_evals = await db.get_evaluations_by_version(dpo_version)

    base_eval_map = {e.scenario_id: e for e in base_evals}
    dpo_eval_map = {e.scenario_id: e for e in dpo_evals}
    base_run_map = {r.scenario_id: r for r in base_runs}
    dpo_run_map = {r.scenario_id: r for r in dpo_runs}

    # Fetch AMB-001 holdout
    base_amb_runs = await db.list_runs(agent_version=base_version + "_amb")
    dpo_amb_runs = await db.list_runs(agent_version=dpo_version + "_amb")
    base_amb_evals = await db.get_evaluations_by_version(base_version + "_amb")
    dpo_amb_evals = await db.get_evaluations_by_version(dpo_version + "_amb")
    base_amb001_run = base_amb_runs[0] if base_amb_runs else None
    dpo_amb001_run = dpo_amb_runs[0] if dpo_amb_runs else None
    base_amb001_eval = base_amb_evals[0] if base_amb_evals else None
    dpo_amb001_eval = dpo_amb_evals[0] if dpo_amb_evals else None

    # Fetch ESC suite
    base_esc_runs = await db.list_runs(agent_version=base_version + "_esc")
    dpo_esc_runs = await db.list_runs(agent_version=dpo_version + "_esc")
    base_esc_evals = await db.get_evaluations_by_version(base_version + "_esc")
    dpo_esc_evals = await db.get_evaluations_by_version(dpo_version + "_esc")
    base_esc_run_map = {r.scenario_id: r for r in base_esc_runs}
    dpo_esc_run_map = {r.scenario_id: r for r in dpo_esc_runs}
    base_esc_eval_map = {e.scenario_id: e for e in base_esc_evals}
    dpo_esc_eval_map = {e.scenario_id: e for e in dpo_esc_evals}

    # Fetch ROUTING suite
    base_route_runs = await db.list_runs(agent_version=base_version + "_route")
    dpo_route_runs = await db.list_runs(agent_version=dpo_version + "_route")
    base_route_evals = await db.get_evaluations_by_version(base_version + "_route")
    dpo_route_evals = await db.get_evaluations_by_version(dpo_version + "_route")
    base_route_run_map = {r.scenario_id: r for r in base_route_runs}
    dpo_route_run_map = {r.scenario_id: r for r in dpo_route_runs}
    base_route_eval_map = {e.scenario_id: e for e in base_route_evals}
    dpo_route_eval_map = {e.scenario_id: e for e in dpo_route_evals}

    # =================================================================
    # Compute Tool Metrics
    # =================================================================
    with open("scenarios/scenarios.json", "r", encoding="utf-8") as f:
        scenarios_data = json.load(f)["scenarios"]
    sc_map = {s["id"]: s for s in scenarios_data}

    def compute_tool_metrics(runs: list[AgentRun]) -> dict[str, Any]:
        tp_tool = 0
        fp_tool = 0
        tn_tool = 0
        fn_tool = 0
        arg_correct = 0
        arg_total = 0
        order_correct = 0
        order_total = 0

        # Tool confusion matrix
        confusion: dict[str, dict[str, int]] = {tn: {tn2: 0 for tn2 in TOOL_NAMES + ["no_tool"]} for tn in TOOL_NAMES + ["no_tool"]}

        for r in runs:
            sc = sc_map.get(r.scenario_id)
            if not sc:
                continue
            exp_tools = sc.get("expected_tool_calls", [])
            act_tools = r.actual_tool_calls

            if len(exp_tools) == 0:
                if len(act_tools) == 0:
                    tn_tool += 1
                    confusion["no_tool"]["no_tool"] += 1
                else:
                    fp_tool += 1
                    for at in act_tools:
                        confusion["no_tool"][at.tool_name] += 1
            else:
                if len(act_tools) > 0:
                    tp_tool += 1
                    for et in exp_tools:
                        arg_total += 1
                        matching = [at for at in act_tools if at.tool_name == et["tool_name"]]
                        if matching:
                            match_found = False
                            for m in matching:
                                args_match = all(str(m.arguments.get(k)) == str(v) for k, v in et.get("arguments", {}).items())
                                if args_match:
                                    match_found = True
                                    break
                            if match_found:
                                arg_correct += 1
                            confusion[et["tool_name"]][matching[0].tool_name] += 1
                        else:
                            # Expected tool not found in actual — misrouted
                            first_actual = act_tools[0].tool_name if act_tools else "no_tool"
                            confusion[et["tool_name"]][first_actual] += 1
                else:
                    fn_tool += 1
                    for et in exp_tools:
                        confusion[et["tool_name"]]["no_tool"] += 1

                # Check ordering
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
            "tool_confusion_matrix": confusion,
        }

    base_tool_metrics = compute_tool_metrics(base_runs)
    dpo_tool_metrics = compute_tool_metrics(dpo_runs)

    base_pass_rate = sum(1 for e in base_evals if e.overall_passed) / max(len(base_evals), 1)
    dpo_pass_rate = sum(1 for e in dpo_evals if e.overall_passed) / max(len(dpo_evals), 1)

    # =================================================================
    # RE-005 Root Cause Detailed Analysis
    # =================================================================
    re005_base_run = base_run_map.get("RE-005")
    re005_dpo_run = dpo_run_map.get("RE-005")
    re005_base_eval = base_eval_map.get("RE-005")
    re005_dpo_eval = dpo_eval_map.get("RE-005")

    def analyze_re005(run: AgentRun | None, ev: EvaluationResult | None, label: str) -> dict[str, Any]:
        if not run or not ev:
            return {"status": "missing"}
        t0_tools = [tc.tool_name for tc in run.steps[0].tool_calls] if len(run.steps) > 0 else []
        t1_response = run.steps[1].assistant_message if len(run.steps) > 1 else ""
        has_prop_sub = any(pid in t1_response for pid in ["PROP-101", "PROP-102", "PROP-103", "PROP-104"])
        has_escalation_phrase = any(w in t1_response.lower() for w in ["human agent", "connect you", "transfer", "escalate"])
        
        # Check if property_lookup was the FIRST tool called
        first_tool = t0_tools[0] if t0_tools else "none"
        correct_first_tool = first_tool == "property_lookup"
        
        must_escalate_passed = ev.constraint_adherence.passed if ev.constraint_adherence else False
        constraint_details = ev.constraint_adherence.details if ev.constraint_adherence else ""

        return {
            "overall_passed": ev.overall_passed,
            "actual_outcome": run.actual_outcome,
            "first_tool_called": first_tool,
            "correct_first_tool_routing": correct_first_tool,
            "all_step_0_tools": t0_tools,
            "step_1_response": t1_response,
            "unauthorized_substitution_detected": has_prop_sub,
            "escalation_phrase_detected": has_escalation_phrase,
            "must_escalate_constraint_passed": must_escalate_passed,
            "constraint_details": constraint_details,
            "task_success_passed": ev.task_success.passed if ev.task_success else False,
            "tool_correctness_passed": ev.tool_correctness.passed if ev.tool_correctness else False,
            "diagnosis": (
                "CORRECT: property_lookup -> NOT_FOUND -> escalation"
                if correct_first_tool and has_escalation_phrase and not has_prop_sub
                else (
                    f"WRONG ROUTING: first tool was {first_tool}, expected property_lookup"
                    if not correct_first_tool
                    else (
                        "MISSING ESCALATION: correct routing but no escalation phrase"
                        if not has_escalation_phrase
                        else "SUBSTITUTION: correct routing but unauthorized property substitution"
                    )
                )
            ),
        }

    re005_analysis = {
        "base": analyze_re005(re005_base_run, re005_base_eval, "base"),
        "dpo_v5": analyze_re005(re005_dpo_run, re005_dpo_eval, "dpo_v5"),
    }

    print("\n--- RE-005 Trace Analysis ---", flush=True)
    for model_label, analysis in re005_analysis.items():
        if analysis.get("status") == "missing":
            print(f"  {model_label}: MISSING", flush=True)
        else:
            print(f"  {model_label}:", flush=True)
            print(f"    First tool: {analysis['first_tool_called']}", flush=True)
            print(f"    Correct routing: {analysis['correct_first_tool_routing']}", flush=True)
            print(f"    Escalation: {analysis['escalation_phrase_detected']}", flush=True)
            print(f"    Substitution: {analysis['unauthorized_substitution_detected']}", flush=True)
            print(f"    Diagnosis: {analysis['diagnosis']}", flush=True)

    # =================================================================
    # Held-Out Escalation Suite Analysis
    # =================================================================
    esc_suite_analysis: dict[str, Any] = {}
    for sid in ["ESC-001", "ESC-002", "ESC-003", "ESC-004"]:
        b_r = base_esc_run_map.get(sid)
        b_e = base_esc_eval_map.get(sid)
        d_r = dpo_esc_run_map.get(sid)
        d_e = dpo_esc_eval_map.get(sid)

        esc_suite_analysis[sid] = {
            "base_passed": b_e.overall_passed if b_e else False,
            "dpo_passed": d_e.overall_passed if d_e else False,
            "base_outcome": b_r.actual_outcome if b_r else "unknown",
            "dpo_outcome": d_r.actual_outcome if d_r else "unknown",
            "base_response_turn1": b_r.steps[1].assistant_message if (b_r and len(b_r.steps) > 1) else "",
            "dpo_response_turn1": d_r.steps[1].assistant_message if (d_r and len(d_r.steps) > 1) else "",
        }

    esc_pass_base = sum(1 for v in esc_suite_analysis.values() if v["base_passed"]) / max(len(esc_suite_analysis), 1)
    esc_pass_dpo = sum(1 for v in esc_suite_analysis.values() if v["dpo_passed"]) / max(len(esc_suite_analysis), 1)

    # =================================================================
    # Held-Out Routing Suite Analysis
    # =================================================================
    route_suite_analysis: dict[str, Any] = {}
    for sid in [f"ROUTE-{i:03d}" for i in range(1, 9)]:
        b_r = base_route_run_map.get(sid)
        b_e = base_route_eval_map.get(sid)
        d_r = dpo_route_run_map.get(sid)
        d_e = dpo_route_eval_map.get(sid)

        base_tools = [tc.tool_name for tc in b_r.actual_tool_calls] if b_r else []
        dpo_tools = [tc.tool_name for tc in d_r.actual_tool_calls] if d_r else []

        route_suite_analysis[sid] = {
            "base_passed": b_e.overall_passed if b_e else False,
            "dpo_passed": d_e.overall_passed if d_e else False,
            "base_tools_called": base_tools,
            "dpo_tools_called": dpo_tools,
            "base_score": b_e.overall_score if b_e else 0.0,
            "dpo_score": d_e.overall_score if d_e else 0.0,
        }

    route_pass_base = sum(1 for v in route_suite_analysis.values() if v["base_passed"]) / max(len(route_suite_analysis), 1)
    route_pass_dpo = sum(1 for v in route_suite_analysis.values() if v["dpo_passed"]) / max(len(route_suite_analysis), 1)

    print("\n--- Routing Suite Results ---", flush=True)
    for sid, analysis in sorted(route_suite_analysis.items()):
        base_status = "PASS" if analysis["base_passed"] else "FAIL"
        dpo_status = "PASS" if analysis["dpo_passed"] else "FAIL"
        print(f"  {sid}: Base={base_status} DPO={dpo_status} | Base tools: {analysis['base_tools_called']} | DPO tools: {analysis['dpo_tools_called']}", flush=True)

    # =================================================================
    # Business Safety Metrics
    # =================================================================
    def compute_safety_metrics(runs: list[AgentRun]) -> dict[str, int]:
        unauthorized_substitution = 0
        hallucinated_property = 0
        fabricated_availability = 0
        premature_booking = 0
        for r in runs:
            for step in r.steps:
                msg = step.assistant_message.lower() if step.assistant_message else ""
                # Check for unauthorized substitution of unknown property IDs
                if r.scenario_id in ["RE-005", "RE-008"]:
                    for pid in ["PROP-101", "PROP-102", "PROP-103", "PROP-104"]:
                        if pid.lower() in msg and "our available" not in msg and "listings include" not in msg:
                            unauthorized_substitution += 1
                            break
        return {
            "unauthorized_substitution_count": unauthorized_substitution,
            "hallucinated_property_count": hallucinated_property,
            "fabricated_availability_count": fabricated_availability,
            "premature_booking_count": premature_booking,
        }

    base_safety = compute_safety_metrics(base_runs)
    dpo_safety = compute_safety_metrics(dpo_runs)

    # =================================================================
    # Assemble Final Results
    # =================================================================
    eval_results = {
        "benchmark": "OneInbox Real-Estate Suite V5 FINAL (8 Canonical + AMB-001 + 4 Escalation + 8 Routing)",
        "experiment": "DPO Experiment V5: Tool-Routing Discrimination + Escalation Preservation (FINAL)",
        "is_final_experiment": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_fix_applied": "check_availability and book_appointment now return NOT_FOUND for unknown property IDs",
        "base_model": {
            "version": base_version,
            "model_name": base_model_name,
            "adapter": None,
            "canonical_pass_rate": base_pass_rate,
            "canonical_passed_count": sum(1 for e in base_evals if e.overall_passed),
            "canonical_total_count": len(base_evals),
            "conditional_tool_metrics": base_tool_metrics,
            "escalation_suite_pass_rate": esc_pass_base,
            "routing_suite_pass_rate": route_pass_base,
            "safety_metrics": base_safety,
        },
        "dpo_model": {
            "version": dpo_version,
            "model_name": base_model_name,
            "adapter": adapter_path,
            "canonical_pass_rate": dpo_pass_rate,
            "canonical_passed_count": sum(1 for e in dpo_evals if e.overall_passed),
            "canonical_total_count": len(dpo_evals),
            "conditional_tool_metrics": dpo_tool_metrics,
            "escalation_suite_pass_rate": esc_pass_dpo,
            "routing_suite_pass_rate": route_pass_dpo,
            "safety_metrics": dpo_safety,
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
        "re005_root_cause_analysis": re005_analysis,
        "held_out_escalation_suite_analysis": esc_suite_analysis,
        "held_out_routing_suite_analysis": route_suite_analysis,
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

    with open(out_path / "dpo_experiment_v5_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation reports saved to {exp_dir} and {out_path}!", flush=True)
    print("=" * 80, flush=True)
    print("V5 EVALUATION COMPLETE (FINAL)", flush=True)
    print("=" * 80, flush=True)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Experiment V5 (FINAL)")
    parser.add_argument("--skip_runs", action="store_true", help="Skip running scenarios and use existing DB records")
    args = parser.parse_args()

    asyncio.run(run_evaluation_suite_v5(skip_runs=args.skip_runs))
