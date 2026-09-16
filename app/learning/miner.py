"""Failure miner — extract training candidates from stored evaluation data.

Reads AgentRun and EvaluationResult records from the existing SQLite
database and produces ranked FailureCandidate objects.  Only behavioral
failures with sufficient evidence become training-eligible.

Candidate scoring formula
-------------------------
candidate_score = severity_weight × reproducibility × recurrence_factor × confidence

    severity_weight:   CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1
    reproducibility:   deterministic/repeated=1.0, unstable=0.5, unknown=0.25
    recurrence_factor: min(1.0 + recurrence_count / 5.0, 2.0)
    confidence:        explicit_deterministic=1.0, partially_inferred=0.5, ambiguous=0.25

The purpose is prioritisation, not prediction.  Do not treat this as a
calibrated utility function.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.learning.models import FailureCandidate
from app.models import (
    AgentRun,
    CATEGORY_TO_ORIGIN,
    EvaluationResult,
    Failure,
    FailureOrigin,
    TestScenario,
)
from app.storage.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity weight lookup
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT: dict[str, float] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


# ---------------------------------------------------------------------------
# Training-eligibility decision
# ---------------------------------------------------------------------------

def _assess_eligibility(
    failure: Failure,
    run: AgentRun,
    scenario: TestScenario | None,
) -> tuple[bool, str]:
    """Determine whether a failure is eligible for preference training.

    Returns (eligible, reason).
    """
    origin = failure.failure_origin

    # Rule 1: Only behavioral failures are eligible
    if origin != FailureOrigin.BEHAVIORAL:
        return False, (
            f"Failure origin is '{origin.value}', not behavioral. "
            f"Category: {failure.category.value}."
        )

    # Rule 2: Need a scenario with explicit expected outcome
    if scenario is None:
        return False, "No scenario definition found — cannot validate expected behavior."

    if scenario.expected_outcome == "unknown":
        return False, "Scenario has no explicit expected outcome."

    # Rule 3: Must have conversation context
    if not run.steps:
        return False, "Run has no trace steps — no conversation context available."

    # Rule 4: Must have an observable failed behavior
    if not run.actual_outcome and not run.steps:
        return False, "No actual outcome or trace to observe failed behavior."

    # Rule 5: Run must not be an infrastructure error
    if run.status == "error":
        return False, f"Run status is 'error' — infrastructure failure: {run.error_message}"

    return True, (
        f"Behavioral failure with explicit expected outcome "
        f"'{scenario.expected_outcome}' and observable trace "
        f"({len(run.steps)} steps, {len(run.actual_tool_calls)} tool calls)."
    )


def _compute_confidence(
    failure: Failure,
    scenario: TestScenario | None,
) -> float:
    """Compute confidence score for the failure label.

    1.0 = explicit deterministic evidence (evaluator produced clear pass/fail)
    0.75 = partially inferred (heuristic outcome inference)
    0.25 = ambiguous

    Tool-expectation semantics:
      - expected_tool_calls is a list (even empty) → explicit assertion
        An empty list means "the agent should NOT call any tools", which
        is just as explicit as a populated list.
      - expected_tool_calls is None → unknown / not specified
    """
    if scenario is None:
        return 0.25

    has_explicit_outcome = scenario.expected_outcome != "unknown"

    # Distinguish between "explicitly no tools" ([] — a list) and
    # "tool expectations unknown" (None / field absent).
    # An empty list IS an explicit expectation: "call zero tools".
    has_tool_expectations = scenario.expected_tool_calls is not None

    if has_explicit_outcome and has_tool_expectations:
        return 1.0
    elif has_explicit_outcome:
        return 0.75
    else:
        return 0.25


def _compute_score(
    severity: str,
    reproducibility: float,
    recurrence_count: int,
    confidence: float,
) -> float:
    """Compute prioritisation score.

    candidate_score = severity_weight × reproducibility × recurrence_factor × confidence
    """
    sev = _SEVERITY_WEIGHT.get(severity, 1.0)
    recurrence_factor = min(1.0 + recurrence_count / 5.0, 2.0)
    return round(sev * reproducibility * recurrence_factor * confidence, 3)


# ---------------------------------------------------------------------------
# Main mining function
# ---------------------------------------------------------------------------

async def mine_failures(
    db: Database,
    scenarios: dict[str, TestScenario],
    agent_versions: list[str] | None = None,
) -> list[FailureCandidate]:
    """Mine stored evaluations for training-eligible failure candidates.

    Args:
        db: Database instance.
        scenarios: scenario_id → TestScenario mapping.
        agent_versions: versions to inspect (default: all).

    Returns:
        List of FailureCandidate objects, sorted by score descending.
    """
    # Load all evaluations
    all_evals: list[EvaluationResult] = []
    if agent_versions:
        for v in agent_versions:
            evals = await db.get_evaluations_by_version(v)
            all_evals.extend(evals)
    else:
        all_evals = await db.list_evaluations(limit=1000)

    candidates: list[FailureCandidate] = []

    # Track recurrence: scenario_id + failure_category → count
    recurrence_tracker: dict[str, int] = {}

    for evaluation in all_evals:
        if not evaluation.failures:
            continue

        # Load the run for conversation context
        run = await db.get_run(evaluation.run_id)
        if run is None:
            logger.warning(f"Run {evaluation.run_id} not found for eval — skipping")
            continue

        scenario = scenarios.get(evaluation.scenario_id)

        for failure in evaluation.failures:
            # Always resolve origin from the canonical mapping.
            # Legacy DB rows lack failure_origin so the default is BEHAVIORAL,
            # but CATEGORY_TO_ORIGIN is authoritative.
            origin = CATEGORY_TO_ORIGIN.get(failure.category, FailureOrigin.BEHAVIORAL)
            failure.failure_origin = origin  # patch so eligibility check sees it

            # Recurrence tracking
            rec_key = f"{evaluation.scenario_id}:{failure.category.value}"
            recurrence_tracker[rec_key] = recurrence_tracker.get(rec_key, 0) + 1

            # Eligibility
            eligible, reason = _assess_eligibility(failure, run, scenario)

            # Confidence
            confidence = _compute_confidence(failure, scenario)
            if not eligible:
                confidence = min(confidence, 0.25)

            # Reproducibility
            reproducibility = 1.0 if failure.reproducible else 0.5

            # Score
            recurrence = recurrence_tracker[rec_key]
            score = _compute_score(
                failure.severity, reproducibility, recurrence, confidence
            )

            # Extract context
            user_messages = [s.user_message for s in run.steps]
            assistant_responses = [s.assistant_message for s in run.steps]
            tool_calls = [tc.model_dump(mode="json") for tc in run.actual_tool_calls]

            candidate = FailureCandidate(
                scenario_id=evaluation.scenario_id,
                agent_version=evaluation.agent_version,
                run_id=evaluation.run_id,
                failure_origin=origin.value,
                failure_category=failure.category.value,
                severity=failure.severity,
                training_eligible=eligible,
                eligibility_reason=reason,
                confidence=confidence,
                reproducibility=reproducibility,
                recurrence_count=recurrence,
                candidate_score=score,
                user_messages=user_messages,
                assistant_responses=assistant_responses,
                tool_calls=tool_calls,
                expected_outcome=scenario.expected_outcome if scenario else "unknown",
                actual_outcome=run.actual_outcome or "unknown",
                evidence={
                    "failure_id": failure.failure_id,
                    "expected": failure.expected,
                    "actual": failure.actual,
                    "explanation": failure.explanation,
                    "step_index": failure.step_index,
                    "tool_call_index": failure.tool_call_index,
                },
                source_type="synthetic_benchmark",
            )
            candidates.append(candidate)

    # Sort by score descending
    candidates.sort(key=lambda c: c.candidate_score, reverse=True)

    logger.info(
        f"Mined {len(candidates)} candidates from {len(all_evals)} evaluations. "
        f"Training-eligible: {sum(1 for c in candidates if c.training_eligible)}"
    )

    return candidates
