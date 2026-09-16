"""Failure classifier — maps evaluation results to the failure taxonomy.

Pure function: takes metric results and the agent run, produces a list
of classified Failure objects with pointers into the trace.
"""

from __future__ import annotations

from typing import Any

from app.models import (
    AgentRun,
    CATEGORY_TO_ORIGIN,
    Failure,
    FailureCategory,
    FailureOrigin,
    MetricResult,
    TestScenario,
)


def classify_failures(
    run: AgentRun,
    scenario: TestScenario,
    task_result: MetricResult,
    tool_result: MetricResult,
    constraint_result: MetricResult,
    latency_result: MetricResult,
) -> list[Failure]:
    """Classify all evaluation failures into the taxonomy.

    Examines each metric result's evidence to produce specific, actionable
    failure records that point back to the trace.

    Returns:
        List of Failure objects (empty if everything passed).
    """
    failures: list[Failure] = []

    # --- Task failures ---
    if not task_result.passed:
        evidence = task_result.evidence or {}
        expected = evidence.get("expected_outcome", scenario.expected_outcome)
        actual = evidence.get("actual_outcome", run.actual_outcome or "unknown")

        category = FailureCategory.INCOMPLETE_WORKFLOW
        if actual == "unknown":
            category = FailureCategory.INCOMPLETE_WORKFLOW
        elif expected == "escalated" and actual != "escalated":
            category = FailureCategory.MISSED_ESCALATION

        failures.append(Failure(
            run_id=run.run_id,
            scenario_id=scenario.id,
            agent_version=run.agent_version,
            category=category,
            failure_origin=CATEGORY_TO_ORIGIN.get(category, FailureOrigin.BEHAVIORAL),
            severity=scenario.severity,
            expected=f"Outcome: {expected}",
            actual=f"Outcome: {actual}",
            explanation=task_result.details,
        ))

    # --- Tool failures ---
    if not tool_result.passed:
        evidence = tool_result.evidence or {}
        call_results: list[dict[str, Any]] = evidence.get("call_results", [])

        for i, cr in enumerate(call_results):
            status = cr.get("status", "")
            if status == "match":
                continue

            if status == "missing":
                category = FailureCategory.WRONG_TOOL
                explanation = (
                    f"Expected tool '{cr['expected_tool']}' was never called."
                )
            elif status == "args_mismatch":
                category = FailureCategory.WRONG_TOOL_ARGS
                explanation = (
                    f"Tool '{cr['expected_tool']}' called with wrong arguments: "
                    + "; ".join(cr.get("mismatches", []))
                )
            else:
                category = FailureCategory.WRONG_TOOL
                explanation = f"Unexpected tool status: {status}"

            failures.append(Failure(
                run_id=run.run_id,
                scenario_id=scenario.id,
                agent_version=run.agent_version,
                category=category,
                failure_origin=CATEGORY_TO_ORIGIN.get(category, FailureOrigin.BEHAVIORAL),
                severity=scenario.severity if cr.get("required") else "medium",
                expected=f"Tool: {cr['expected_tool']}({cr.get('expected_args', {})})",
                actual=f"Status: {status}",
                explanation=explanation,
                tool_call_index=i,
            ))

    # --- Constraint failures ---
    if not constraint_result.passed:
        evidence = constraint_result.evidence or {}
        constraint_results: list[dict[str, Any]] = evidence.get(
            "constraint_results", []
        )

        for cr in constraint_results:
            if cr.get("passed"):
                continue

            constraint_name = cr.get("constraint", "unknown")

            # Map constraint names to failure categories
            if "escalat" in constraint_name:
                category = FailureCategory.MISSED_ESCALATION
            elif "fabricat" in constraint_name or "hallucinat" in constraint_name:
                category = FailureCategory.HALLUCINATION
            else:
                category = FailureCategory.CONSTRAINT_VIOLATION

            failures.append(Failure(
                run_id=run.run_id,
                scenario_id=scenario.id,
                agent_version=run.agent_version,
                category=category,
                failure_origin=CATEGORY_TO_ORIGIN.get(category, FailureOrigin.BEHAVIORAL),
                severity=scenario.severity,
                expected=f"Constraint: {constraint_name} should pass",
                actual=f"Constraint violated: {cr.get('detail', '')}",
                explanation=cr.get("detail", "Constraint check failed."),
            ))

    # --- Latency failures ---
    if not latency_result.passed:
        evidence = latency_result.evidence or {}
        provider_latency = evidence.get("provider_latency_ms", 0)
        application_overhead = evidence.get("application_overhead_ms", 0)
        category = (
            FailureCategory.PROVIDER_LATENCY
            if provider_latency >= application_overhead
            else FailureCategory.LATENCY_BREACH
        )
        retry_delay = evidence.get("retry_delay_ms", 0)
        retry_errors = evidence.get("provider_retry_errors", [])
        cause_detail = (
            f" Provider/API latency={provider_latency}ms; "
            f"retry/backoff={retry_delay}ms."
        )
        if retry_errors:
            cause_detail += f" Retry errors: {retry_errors}."
        failures.append(Failure(
            run_id=run.run_id,
            scenario_id=scenario.id,
            agent_version=run.agent_version,
            category=category,
            failure_origin=CATEGORY_TO_ORIGIN.get(category, FailureOrigin.OPERATIONAL),
            severity="medium",
            expected=f"Total latency ≤ {evidence.get('threshold_ms', '?')}ms",
            actual=f"Total latency = {evidence.get('total_ms', '?')}ms",
            explanation=latency_result.details + cause_detail,
        ))

    # --- Check for tool execution failures in the trace ---
    for i, tc in enumerate(run.actual_tool_calls):
        if not tc.success and tc.error != "injected_failure":
            injected_override = bool(
                scenario.tool_overrides
                and tc.tool_name in scenario.tool_overrides
            )
            expected_unknown_lookup = (
                tc.tool_name == "property_lookup"
                and tc.error == "property_not_found"
                and scenario.expected_outcome == "escalated"
            )
            if injected_override or expected_unknown_lookup:
                continue
            failures.append(Failure(
                run_id=run.run_id,
                scenario_id=scenario.id,
                agent_version=run.agent_version,
                category=FailureCategory.TOOL_FAILURE,
                failure_origin=FailureOrigin.INFRASTRUCTURE,
                severity="high",
                expected=f"Tool '{tc.tool_name}' should execute successfully",
                actual=f"Tool failed with error: {tc.error}",
                explanation=(
                    f"Tool '{tc.tool_name}' failed during execution with "
                    f"args {tc.arguments}. Error: {tc.error}"
                ),
                tool_call_index=i,
            ))

    return failures
