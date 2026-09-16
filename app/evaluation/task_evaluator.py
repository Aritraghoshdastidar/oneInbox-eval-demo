"""Task evaluator — deterministic check for workflow completion.

Compares the agent's actual outcome against the expected outcome defined
in the test scenario. Also checks for key signals in the agent's responses
to infer the actual outcome when not explicitly set.
"""

from __future__ import annotations

from app.models import AgentRun, MetricResult, TestScenario


# ---------------------------------------------------------------------------
# Outcome inference heuristics
# ---------------------------------------------------------------------------

_OUTCOME_SIGNALS: dict[str, list[str]] = {
    "appointment_booked": [
        "confirmed", "booking", "booked", "appointment", "viewing confirmed",
        "BK-",  # booking ID prefix from our mock
    ],
    "information_provided": [
        "bedrooms", "bathrooms", "price", "features", "address",
    ],
    "escalated": [
        "human agent", "connect you", "escalate", "transfer",
        "cannot help", "outside my capabilities",
    ],
    "clarification_requested": [
        "could you clarify", "which property", "what date", "what time",
        "can you provide", "could you please provide", "please provide",
        "could you tell me", "do you mean", "specific property id",
    ],
    "declined": [
        "unable to", "cannot", "not possible", "unavailable",
    ],
}


def _infer_outcome(run: AgentRun) -> str:
    """Infer the actual outcome from the agent's final responses."""
    if not run.steps:
        return "unknown"

    # Infer terminal outcomes from the final response first. Earlier turns may
    # contain a request for missing details that was later resolved by
    # escalation or completion, but a neutral closing should retain evidence
    # that information was already provided.
    final_message = run.steps[-1].assistant_message.lower()

    # Check for booking confirmation first (strongest signal)
    for tool_call in run.actual_tool_calls:
        if tool_call.tool_name == "book_appointment" and tool_call.success:
            return "appointment_booked"

    # Check explicit terminal signals in the final response.
    final_priority_order = [
        "clarification_requested",
        "escalated",
        "appointment_booked",
        "information_provided",
        "declined",
    ]

    for outcome in final_priority_order:
        signals = _OUTCOME_SIGNALS.get(outcome, [])
        if any(signal.lower() in final_message for signal in signals):
            return outcome

    # A neutral closing does not erase information delivered on an earlier
    # turn. Do not use earlier clarification/escalation requests here because
    # those may have been resolved by the final response.
    prior_messages = " ".join(
        step.assistant_message.lower() for step in run.steps[:-1]
    )
    if any(
        signal.lower() in prior_messages
        for signal in _OUTCOME_SIGNALS["information_provided"]
    ):
        return "information_provided"

    return "unknown"


def evaluate_task(run: AgentRun, scenario: TestScenario) -> MetricResult:
    """Evaluate whether the agent completed the expected task.

    This is a deterministic evaluator:
    - Infers the actual outcome from tool calls and response text.
    - Compares against the scenario's expected_outcome.

    Returns:
        MetricResult with pass/fail and explanatory details.
    """
    actual = _infer_outcome(run)
    run.actual_outcome = actual  # persist for downstream use

    expected = scenario.expected_outcome
    passed = actual == expected

    if passed:
        details = f"Task completed as expected: {expected}"
    else:
        details = (
            f"Expected outcome '{expected}' but inferred '{actual}'. "
            f"The agent may have failed to complete the workflow correctly."
        )

    return MetricResult(
        metric_name="task_success",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=details,
        evidence={
            "expected_outcome": expected,
            "actual_outcome": actual,
            "total_steps": len(run.steps),
            "total_tool_calls": len(run.actual_tool_calls),
        },
    )
