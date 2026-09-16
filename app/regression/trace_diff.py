"""Trace diff — side-by-side comparison of two agent runs.

Given a base and compare run (plus their evaluations), produces a
structured diff that explains exactly what the agent did differently
and how that affected the evaluation outcome.
"""

from __future__ import annotations

from typing import Any

from app.models import AgentRun, EvaluationResult


def build_trace_diff(
    base_run: AgentRun,
    compare_run: AgentRun,
    base_eval: EvaluationResult,
    compare_eval: EvaluationResult,
) -> dict[str, Any]:
    """Build a structured trace diff between two runs of the same scenario.

    Returns a dict suitable for JSON serialization with:
    - scenario_id
    - version comparison
    - tool call sequences for each version
    - metric transitions
    - behavioral differences
    """
    scenario_id = base_run.scenario_id

    # --- Tool call sequences ---
    def _format_tool_calls(run: AgentRun) -> list[dict[str, Any]]:
        calls = []
        for tc in run.actual_tool_calls:
            calls.append({
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result_summary": _summarize_result(tc.result),
                "success": tc.success,
                "error": tc.error,
            })
        return calls

    base_tools = _format_tool_calls(base_run)
    compare_tools = _format_tool_calls(compare_run)

    # --- Conversation traces ---
    def _format_steps(run: AgentRun) -> list[dict[str, Any]]:
        steps = []
        for step in run.steps:
            steps.append({
                "step_index": step.step_index,
                "user_message": step.user_message,
                "assistant_message": step.assistant_message[:500],  # truncate
                "tool_calls": [
                    {"name": tc.tool_name, "args": tc.arguments}
                    for tc in step.tool_calls
                ],
                "latency_ms": round(step.latency_ms, 1),
            })
        return steps

    # --- Metric transitions ---
    metrics: list[dict[str, Any]] = []

    def _add_metric(name: str, base_passed: bool, comp_passed: bool,
                    base_score: float, comp_score: float,
                    base_detail: str, comp_detail: str) -> None:
        changed = base_passed != comp_passed or abs(base_score - comp_score) >= 0.05
        metrics.append({
            "metric": name,
            "base_passed": base_passed,
            "compare_passed": comp_passed,
            "base_score": round(base_score, 3),
            "compare_score": round(comp_score, 3),
            "changed": changed,
            "transition": (
                f"{'pass' if base_passed else 'fail'} → "
                f"{'pass' if comp_passed else 'fail'}"
            ) if base_passed != comp_passed else "unchanged",
            "base_detail": base_detail[:200],
            "compare_detail": comp_detail[:200],
        })

    _add_metric(
        "task_success",
        base_eval.task_success.passed, compare_eval.task_success.passed,
        base_eval.task_success.score, compare_eval.task_success.score,
        base_eval.task_success.details, compare_eval.task_success.details,
    )
    _add_metric(
        "tool_correctness",
        base_eval.tool_correctness.passed, compare_eval.tool_correctness.passed,
        base_eval.tool_correctness.score, compare_eval.tool_correctness.score,
        base_eval.tool_correctness.details, compare_eval.tool_correctness.details,
    )
    _add_metric(
        "constraint_adherence",
        base_eval.constraint_adherence.passed, compare_eval.constraint_adherence.passed,
        base_eval.constraint_adherence.score, compare_eval.constraint_adherence.score,
        base_eval.constraint_adherence.details, compare_eval.constraint_adherence.details,
    )
    _add_metric(
        "latency",
        base_eval.latency.passed, compare_eval.latency.passed,
        base_eval.latency.score, compare_eval.latency.score,
        base_eval.latency.details, compare_eval.latency.details,
    )

    # --- Tool call diff ---
    tool_diff = _diff_tool_sequences(base_tools, compare_tools)

    # --- Failure diff ---
    base_failures = [
        {"category": f.category.value, "severity": f.severity,
         "explanation": f.explanation[:200]}
        for f in base_eval.failures
    ]
    compare_failures = [
        {"category": f.category.value, "severity": f.severity,
         "explanation": f.explanation[:200]}
        for f in compare_eval.failures
    ]

    return {
        "scenario_id": scenario_id,
        "base_version": base_run.agent_version,
        "compare_version": compare_run.agent_version,
        "base_status": base_run.status,
        "compare_status": compare_run.status,
        "base_overall": {
            "passed": base_eval.overall_passed,
            "score": round(base_eval.overall_score, 3),
        },
        "compare_overall": {
            "passed": compare_eval.overall_passed,
            "score": round(compare_eval.overall_score, 3),
        },
        "metric_transitions": metrics,
        "tool_call_diff": tool_diff,
        "base_tool_calls": base_tools,
        "compare_tool_calls": compare_tools,
        "base_conversation": _format_steps(base_run),
        "compare_conversation": _format_steps(compare_run),
        "base_failures": base_failures,
        "compare_failures": compare_failures,
        "base_latency_ms": round(base_run.total_latency_ms, 1),
        "compare_latency_ms": round(compare_run.total_latency_ms, 1),
    }


def _summarize_result(result: Any) -> str:
    """Create a short summary of a tool result."""
    if result is None:
        return "null"
    if isinstance(result, dict):
        if "error" in result:
            return f"error: {result['error']}"
        if "success" in result:
            return f"success={result['success']}"
        if "available" in result:
            return f"available={result['available']}"
        if "property_id" in result:
            return f"property: {result['property_id']}"
    return str(result)[:100]


def _diff_tool_sequences(
    base_tools: list[dict[str, Any]],
    compare_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Produce a human-readable diff of tool call sequences."""
    diffs: list[dict[str, Any]] = []
    max_len = max(len(base_tools), len(compare_tools))

    for i in range(max_len):
        base_tc = base_tools[i] if i < len(base_tools) else None
        comp_tc = compare_tools[i] if i < len(compare_tools) else None

        if base_tc and comp_tc:
            if base_tc["tool_name"] == comp_tc["tool_name"]:
                if base_tc["arguments"] == comp_tc["arguments"]:
                    diffs.append({
                        "index": i,
                        "change": "identical",
                        "tool": base_tc["tool_name"],
                    })
                else:
                    diffs.append({
                        "index": i,
                        "change": "args_changed",
                        "tool": base_tc["tool_name"],
                        "base_args": base_tc["arguments"],
                        "compare_args": comp_tc["arguments"],
                    })
            else:
                diffs.append({
                    "index": i,
                    "change": "different_tool",
                    "base_tool": base_tc["tool_name"],
                    "compare_tool": comp_tc["tool_name"],
                })
        elif base_tc and not comp_tc:
            diffs.append({
                "index": i,
                "change": "removed_in_compare",
                "base_tool": base_tc["tool_name"],
            })
        elif comp_tc and not base_tc:
            diffs.append({
                "index": i,
                "change": "added_in_compare",
                "compare_tool": comp_tc["tool_name"],
                "compare_args": comp_tc["arguments"],
            })

    return diffs
