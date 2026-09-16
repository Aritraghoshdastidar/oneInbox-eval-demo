"""Latency evaluator — checks response time against thresholds.

Measures per-turn latency and total conversation latency against
the scenario-specific (or default) threshold.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.models import AgentRun, MetricResult, TestScenario


def evaluate_latency(run: AgentRun, scenario: TestScenario) -> MetricResult:
    """Evaluate latency against the scenario threshold.

    Checks:
    - Total conversation latency vs threshold.
    - Per-turn statistics (mean, max, P95 approximation).

    The score is 1.0 if within threshold, linearly degrades for overages
    up to 2x threshold (at which point score = 0.0).

    Returns:
        MetricResult with latency breakdown.
    """
    threshold_ms = scenario.max_latency_ms or settings.max_latency_ms
    total_ms = run.total_latency_ms

    # Per-turn stats
    turn_latencies = [step.latency_ms for step in run.steps]
    if turn_latencies:
        mean_ms = sum(turn_latencies) / len(turn_latencies)
        max_ms = max(turn_latencies)
        # Simple P95: sort and take 95th percentile
        sorted_latencies = sorted(turn_latencies)
        p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
        p95_ms = sorted_latencies[p95_idx]
    else:
        mean_ms = max_ms = p95_ms = 0.0

    llm_elapsed_ms = sum(
        call.latency_ms for step in run.steps for call in step.llm_calls
    )
    llm_request_ms = sum(
        call.request_latency_ms for step in run.steps for call in step.llm_calls
    )
    retry_delay_ms = sum(
        call.retry_delay_ms for step in run.steps for call in step.llm_calls
    )
    tool_latency_ms = sum(
        tool.latency_ms for step in run.steps for tool in step.tool_calls
    )
    application_overhead_ms = max(
        total_ms - llm_elapsed_ms - tool_latency_ms,
        0.0,
    )
    provider_latency_ms = llm_elapsed_ms + tool_latency_ms

    # Scoring: linear degradation beyond threshold
    if total_ms <= threshold_ms:
        score = 1.0
        passed = True
    elif total_ms <= threshold_ms * 2:
        # Linear from 1.0 → 0.0 between threshold and 2x threshold
        score = 1.0 - (total_ms - threshold_ms) / threshold_ms
        passed = False
    else:
        score = 0.0
        passed = False

    if passed:
        details = (
            f"Total latency {total_ms:.0f}ms is within threshold "
            f"({threshold_ms}ms). Mean turn: {mean_ms:.0f}ms, "
            f"Max turn: {max_ms:.0f}ms."
        )
    else:
        details = (
            f"Total latency {total_ms:.0f}ms exceeds threshold "
            f"({threshold_ms}ms). Mean turn: {mean_ms:.0f}ms, "
            f"Max turn: {max_ms:.0f}ms, P95: {p95_ms:.0f}ms."
        )

    evidence: dict[str, Any] = {
        "total_ms": round(total_ms, 1),
        "raw_end_to_end_latency_ms": round(total_ms, 1),
        "threshold_ms": threshold_ms,
        "passed": passed,
        "score": round(max(score, 0.0), 3),
        "mean_turn_ms": round(mean_ms, 1),
        "max_turn_ms": round(max_ms, 1),
        "p95_turn_ms": round(p95_ms, 1),
        "turn_count": len(turn_latencies),
        "turn_latencies": [round(t, 1) for t in turn_latencies],
        "llm_call_count": run.llm_call_count,
        "tool_call_count": run.tool_call_count,
        "llm_elapsed_ms": round(llm_elapsed_ms, 1),
        "llm_request_latency_ms": round(llm_request_ms, 1),
        "retry_delay_ms": round(retry_delay_ms, 1),
        "tool_latency_ms": round(tool_latency_ms, 1),
        "application_overhead_ms": round(application_overhead_ms, 1),
        "provider_latency_ms": round(provider_latency_ms, 1),
        "provider_retry_errors": [
            error
            for step in run.steps
            for call in step.llm_calls
            for error in call.retry_errors
        ],
        "turn_breakdown": [
            {
                "turn_index": step.step_index,
                "total_ms": round(step.latency_ms, 1),
                "llm_calls": [
                    {
                        "call_index": call.call_index,
                        "provider": call.provider,
                        "model": call.model,
                        "latency_ms": round(call.latency_ms, 1),
                        "request_latency_ms": round(call.request_latency_ms, 1),
                        "retry_count": call.retry_count,
                        "retry_delay_ms": round(call.retry_delay_ms, 1),
                        "retry_errors": call.retry_errors,
                        "success": call.success,
                        "tool_call_count": call.response_tool_call_count,
                    }
                    for call in step.llm_calls
                ],
                "tool_calls": [
                    {
                        "tool_name": tool.tool_name,
                        "latency_ms": round(tool.latency_ms, 1),
                        "success": tool.success,
                    }
                    for tool in step.tool_calls
                ],
            }
            for step in run.steps
        ],
    }

    return MetricResult(
        metric_name="latency",
        passed=passed,
        score=round(max(score, 0.0), 3),
        details=details,
        evidence=evidence,
    )
