"""Tests for the learning pipeline.

9 targeted tests covering eligibility, validation, provenance, and dataset integrity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.learning.models import FailureCandidate, PreferencePair, DatasetManifest
from app.learning.validators import validate_chosen_response
from app.learning.dataset import write_dataset
from app.models import (
    FailureCategory,
    FailureOrigin,
    CATEGORY_TO_ORIGIN,
    Failure,
    TestScenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario_re004() -> TestScenario:
    """RE-004 style scenario: ambiguous inquiry expecting clarification."""
    return TestScenario(
        id="RE-004",
        name="Ambiguous inquiry",
        description="Customer describes property vaguely.",
        category="ambiguous",
        severity="medium",
        conversation=[{"role": "user", "content": "I want a 2-bed with a patio."}],
        expected_intents=["clarify_property"],
        expected_tool_calls=[],
        expected_outcome="clarification_requested",
        constraints=["must_not_hallucinate_properties"],
        max_latency_ms=15000,
    )


def _make_candidate(
    *,
    failure_origin: str = "behavioral",
    failure_category: str = "incomplete_workflow",
    severity: str = "medium",
    training_eligible: bool = True,
    run_id: str = "run-001",
) -> FailureCandidate:
    return FailureCandidate(
        scenario_id="RE-004",
        agent_version="v1.1",
        run_id=run_id,
        failure_origin=failure_origin,
        failure_category=failure_category,
        severity=severity,
        training_eligible=training_eligible,
        eligibility_reason="test",
        confidence=1.0,
        reproducibility=1.0,
        recurrence_count=1,
        candidate_score=2.0,
        user_messages=["I want a 2-bed with a patio."],
        assistant_responses=["Here are properties PROP-101, PROP-102..."],
        tool_calls=[{"tool_name": "property_lookup", "arguments": {"property_id": "PROP-101"}}],
        expected_outcome="clarification_requested",
        actual_outcome="information_provided",
    )


# ---------------------------------------------------------------------------
# Test 1: Provider failure is NOT training eligible
# ---------------------------------------------------------------------------

def test_provider_failure_not_eligible():
    """FailureOrigin.INFRASTRUCTURE maps to provider categories."""
    for category in [
        FailureCategory.PROVIDER_LATENCY,
        FailureCategory.RUN_ERROR,
        FailureCategory.TOOL_FAILURE,
    ]:
        origin = CATEGORY_TO_ORIGIN[category]
        assert origin == FailureOrigin.INFRASTRUCTURE, (
            f"{category} should map to INFRASTRUCTURE, got {origin}"
        )


# ---------------------------------------------------------------------------
# Test 2: Latency-only failure is NOT training eligible
# ---------------------------------------------------------------------------

def test_latency_failure_not_eligible():
    """LATENCY_BREACH maps to OPERATIONAL, not BEHAVIORAL."""
    origin = CATEGORY_TO_ORIGIN[FailureCategory.LATENCY_BREACH]
    assert origin == FailureOrigin.OPERATIONAL


# ---------------------------------------------------------------------------
# Test 3: RE-004-style behavioral failure IS training eligible
# ---------------------------------------------------------------------------

def test_behavioral_failure_is_eligible():
    """INCOMPLETE_WORKFLOW is behavioral and should be eligible."""
    origin = CATEGORY_TO_ORIGIN[FailureCategory.INCOMPLETE_WORKFLOW]
    assert origin == FailureOrigin.BEHAVIORAL

    # Also check other behavioral categories
    for cat in [
        FailureCategory.WRONG_INTENT,
        FailureCategory.WRONG_TOOL,
        FailureCategory.WRONG_TOOL_ARGS,
        FailureCategory.HALLUCINATION,
        FailureCategory.CONSTRAINT_VIOLATION,
        FailureCategory.MISSED_ESCALATION,
    ]:
        assert CATEGORY_TO_ORIGIN[cat] == FailureOrigin.BEHAVIORAL


# ---------------------------------------------------------------------------
# Test 4: Chosen response without clarification is rejected
# ---------------------------------------------------------------------------

def test_chosen_without_clarification_rejected():
    """A response that doesn't ask for clarification should fail validation."""
    scenario = _make_scenario_re004()

    # This response provides info instead of clarifying
    bad_chosen = "The property at 42 Oak Street has 3 bedrooms and costs $2500/month."
    validated, details = validate_chosen_response(bad_chosen, scenario, tool_calls=[])

    assert not validated, f"Should reject non-clarification response. Details: {details}"
    assert any("outcome_match" in d and "✗" in d for d in details)


# ---------------------------------------------------------------------------
# Test 5: Chosen response with tool calls rejected for clarification scenario
# ---------------------------------------------------------------------------

def test_chosen_with_tools_rejected_for_clarification():
    """A clarification response should have NO tool calls."""
    scenario = _make_scenario_re004()

    good_text = "Which property are you interested in? Could you provide more details?"
    tool_calls = [{"tool_name": "property_lookup", "arguments": {"property_id": "PROP-101"}}]

    validated, details = validate_chosen_response(good_text, scenario, tool_calls=tool_calls)

    assert not validated, f"Should reject response with tool calls. Details: {details}"
    assert any("tool_check" in d and "✗" in d for d in details)


# ---------------------------------------------------------------------------
# Test 6: Rejected response preserves actual failure behavior
# ---------------------------------------------------------------------------

def test_rejected_preserves_actual():
    """The rejected response in a PreferencePair must be the actual trace content."""
    actual_failed_response = (
        "I found several properties matching your description! "
        "Here are the details for PROP-101, PROP-102, PROP-103, and PROP-104..."
    )

    pair = PreferencePair(
        scenario_id="RE-004",
        source_candidate_id="cand-001",
        prompt_or_context=[{"role": "user", "content": "I want a 2-bed with a patio."}],
        chosen_response="Which property are you interested in?",
        rejected_response=actual_failed_response,
        failure_category="incomplete_workflow",
        severity="medium",
        construction_method="passing_trace",
        confidence=1.0,
    )

    # The rejected response must be EXACTLY the actual failed response
    assert pair.rejected_response == actual_failed_response


# ---------------------------------------------------------------------------
# Test 7: Preference provenance points to trace/run
# ---------------------------------------------------------------------------

def test_provenance_contains_run_id():
    """Provenance must include enough info to trace back to the original run."""
    pair = PreferencePair(
        scenario_id="RE-004",
        source_candidate_id="cand-001",
        prompt_or_context=[],
        chosen_response="Which property?",
        rejected_response="Here are all properties...",
        failure_category="incomplete_workflow",
        severity="medium",
        provenance={
            "source_run_id": "run-abc-123",
            "source_agent_version": "v1.1",
            "passing_run_id": "run-xyz-456",
            "passing_agent_version": "v1.0",
        },
    )

    assert "source_run_id" in pair.provenance
    assert pair.provenance["source_run_id"] == "run-abc-123"
    assert "source_agent_version" in pair.provenance


# ---------------------------------------------------------------------------
# Test 8: Dataset manifest matches actual contents
# ---------------------------------------------------------------------------

def test_manifest_matches_dataset():
    """Manifest counts must match the actual JSONL file."""
    validated_pair = PreferencePair(
        scenario_id="RE-004",
        source_candidate_id="cand-001",
        prompt_or_context=[{"role": "user", "content": "test"}],
        chosen_response="Which property?",
        rejected_response="Here are properties...",
        failure_category="incomplete_workflow",
        severity="medium",
        validation_status="validated",
    )

    rejected_pair = PreferencePair(
        scenario_id="RE-099",
        source_candidate_id="cand-002",
        prompt_or_context=[{"role": "user", "content": "test2"}],
        chosen_response="Bad response.",
        rejected_response="Also bad.",
        failure_category="wrong_tool",
        severity="low",
        validation_status="rejected",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path, manifest_path = write_dataset(
            [validated_pair, rejected_pair],
            total_candidates=5,
            training_eligible=2,
            scenario_file_hash="abc123",
            output_dir=Path(tmpdir),
        )

        # Read back manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Read back JSONL
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Manifest should report 1 validated, 1 rejected
        assert manifest["pairs_validated"] == 1
        assert manifest["pairs_rejected"] == 1
        assert manifest["pairs_generated"] == 2
        assert manifest["total_candidates"] == 5

        # JSONL should contain only the validated pair
        assert len(lines) == 1
        assert lines[0]["scenario_id"] == "RE-004"


# ---------------------------------------------------------------------------
# Test 9: Same input produces deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """The same validated pair should produce the same JSONL content."""
    pair = PreferencePair(
        pair_id="fixed-id-001",  # fix ID for determinism
        scenario_id="RE-004",
        source_candidate_id="cand-001",
        prompt_or_context=[{"role": "user", "content": "test"}],
        chosen_response="Which property?",
        rejected_response="Here are properties...",
        failure_category="incomplete_workflow",
        severity="medium",
        validation_status="validated",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with tempfile.TemporaryDirectory() as tmpdir1:
        p1, _ = write_dataset([pair], output_dir=Path(tmpdir1))
        with open(p1) as f:
            content1 = f.read()

    with tempfile.TemporaryDirectory() as tmpdir2:
        p2, _ = write_dataset([pair], output_dir=Path(tmpdir2))
        with open(p2) as f:
            content2 = f.read()

    assert content1 == content2, "Same input should produce identical JSONL"
