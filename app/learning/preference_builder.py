"""Preference builder — construct chosen/rejected pairs from failure candidates.

Chosen-response priority order:
1. BEST:  Use an existing PASSING trace from the same scenario.
2. SECOND: Construct a corrected response from explicit expected behavior.
3. THIRD:  (Not implemented yet) LLM-generated correction — would need
           construction_method="llm_corrected" and deterministic validation.

The rejected response is ALWAYS the actual failed response from the trace.
It is never rewritten.

All chosen responses are sanitized to remove any internal rules, policy
text, chain-of-thought, or evaluator instructions that the LLM may have
leaked into the customer-facing response.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from app.learning.models import FailureCandidate, PreferencePair
from app.learning.validators import validate_chosen_response
from app.models import AgentRun, EvaluationResult, TestScenario
from app.storage.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected-behavior templates for constructing chosen responses
# ---------------------------------------------------------------------------

# These are semantic templates, not hardcoded answers.  The important
# requirement is that the response matches the expected_outcome semantics,
# not that it uses these exact words.

_CLARIFICATION_TEMPLATES: list[str] = [
    (
        "I'd be happy to help you find the right property! Could you provide "
        "a specific property ID or address so I can look up the details for you? "
        "If you're not sure, I can help narrow it down — what area, budget, "
        "or features are most important to you?"
    ),
    (
        "I'd love to help! To find the best match, could you share a property ID "
        "or give me more details like your preferred location, price range, "
        "or specific requirements?"
    ),
]

_ESCALATION_TEMPLATES: list[str] = [
    (
        "I understand your concern. This is outside what I can handle directly, "
        "so let me connect you with a human agent who can help with this. "
        "They'll be able to assist you further."
    ),
]


def _build_prompt_context(run: AgentRun) -> list[dict[str, str]]:
    """Build the conversation context from an agent run's trace.

    Returns a list of {"role": ..., "content": ...} messages suitable
    for DPO training context.
    """
    messages: list[dict[str, str]] = []
    for step in run.steps:
        messages.append({"role": "user", "content": step.user_message})
        # We include assistant messages from earlier turns for multi-turn context,
        # but the LAST assistant message is what we're comparing (chosen vs rejected)
        if step.assistant_message and step.step_index < len(run.steps) - 1:
            messages.append({"role": "assistant", "content": step.assistant_message})
    return messages


def _get_rejected_response(run: AgentRun) -> str:
    """Extract the actual failed response — never rewritten."""
    if not run.steps:
        return ""
    return run.steps[-1].assistant_message


# ---------------------------------------------------------------------------
# Response sanitizer — strip internal/policy text
# ---------------------------------------------------------------------------

# Patterns that indicate internal instructions leaked into the response.
# Each pattern removes text that should never appear in customer-facing output.
_INTERNAL_TEXT_PATTERNS: list[re.Pattern[str]] = [
    # Parenthetical notes about rules: *(Note: Rule 8 states: "...")*
    re.compile(
        r'\*?\((?:Note|Important|Internal|Policy|Rule)[\s:].+?\)\*?',
        re.IGNORECASE | re.DOTALL,
    ),
    # Asterisk-wrapped notes: *Note: ...*  or  *(Note: ...)*
    re.compile(
        r'\*+\s*(?:Note|Important|Internal|Policy|Rule)\s*[\d]*\s*(?:states?|says?|requires?)?\s*:?.+?\*+',
        re.IGNORECASE | re.DOTALL,
    ),
    # Standalone "Note:" lines (entire line or paragraph)
    re.compile(
        r'^\s*\*?\s*(?:Note|Internal note|Policy note|Rule \d+)\s*:.*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    # Chain-of-thought markers
    re.compile(
        r'(?:^|\n)\s*(?:Thinking|Reasoning|Chain.of.thought|My reasoning|Let me think)\s*:.*?(?=\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    ),
    # Explicit rule references: "Rule 8 states..."
    re.compile(
        r'Rule\s+\d+\s+(?:states?|says?|requires?)\s*:?\s*"[^"]*"',
        re.IGNORECASE,
    ),
]


def _sanitize_response(response: str) -> str:
    """Remove internal rules, policy text, and chain-of-thought from a response.

    LLMs sometimes leak system-prompt rules or reasoning into customer-facing
    output.  This function strips those artifacts while preserving the actual
    customer-facing content.

    The function:
    1. Removes known internal-text patterns
    2. Collapses excessive whitespace left by removals
    3. Strips leading/trailing whitespace

    Returns the sanitized, customer-facing response.
    """
    cleaned = response
    for pattern in _INTERNAL_TEXT_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Collapse multiple blank lines into at most two newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


async def _find_passing_trace(
    scenario_id: str,
    db: Database,
    exclude_run_id: str = "",
) -> AgentRun | None:
    """Search for a passing run of the same scenario in any version, prioritizing v1.0."""
    # Priority 1: Verified baseline v1.0 run
    v1_runs = await db.list_runs(agent_version="v1.0", scenario_id=scenario_id, limit=5)
    for run in v1_runs:
        if run.run_id == exclude_run_id:
            continue
        ev = await db.get_evaluation(run.run_id)
        if ev and (ev.overall_passed or ev.task_success.passed) and run.steps:
            return run

    # Priority 2: Any passing evaluation for this scenario
    evals = await db.list_evaluations(scenario_id=scenario_id, limit=50)
    for ev in evals:
        if ev.run_id == exclude_run_id:
            continue
        if ev.overall_passed or ev.task_success.passed:
            run = await db.get_run(ev.run_id)
            if run and run.steps:
                return run

    return None


async def build_preference_pairs(
    candidates: list[FailureCandidate],
    scenarios: dict[str, TestScenario],
    db: Database,
) -> tuple[list[PreferencePair], int]:
    """Build preference pairs from training-eligible candidates.

    For each eligible candidate:
    1. Try to find a passing trace for the same scenario (best)
    2. Fall back to constructing a response from expected behavior
    3. Validate the chosen response before admitting the pair
    4. Deduplicate using semantic fingerprints (scenario + category + prompt + rejected)

    Args:
        candidates: ranked FailureCandidate list (only eligible ones processed)
        scenarios: scenario_id → TestScenario mapping
        db: Database instance for loading traces

    Returns:
        Tuple of (validated_pairs, duplicates_removed_count).
    """
    pairs: list[PreferencePair] = []
    seen_fingerprints: set[str] = set()
    duplicates_removed = 0

    for candidate in candidates:
        if not candidate.training_eligible:
            continue

        scenario = scenarios.get(candidate.scenario_id)
        if scenario is None:
            logger.warning(f"No scenario definition for {candidate.scenario_id}")
            continue

        # Load the failed run
        failed_run = await db.get_run(candidate.run_id)
        if failed_run is None or not failed_run.steps:
            logger.warning(f"Cannot load run {candidate.run_id}")
            continue

        # --- Rejected response: always actual ---
        rejected_response = _get_rejected_response(failed_run)
        if not rejected_response:
            logger.warning(f"Empty rejected response for {candidate.run_id}")
            continue

        # --- Build prompt context ---
        prompt_context = _build_prompt_context(failed_run)

        # --- Deduplication by semantic fingerprint ---
        norm_prompt = " ".join(m.get("content", "").strip() for m in prompt_context)
        norm_rejected = re.sub(r'\s+', ' ', rejected_response.strip().lower())
        fp_raw = f"{candidate.scenario_id}:{candidate.failure_category}:{norm_prompt}:{norm_rejected}"
        fingerprint = hashlib.sha256(fp_raw.encode("utf-8")).hexdigest()

        if fingerprint in seen_fingerprints:
            duplicates_removed += 1
            logger.info(
                f"Duplicate pair removed for {candidate.scenario_id} "
                f"({candidate.failure_category}): fingerprint {fingerprint[:8]}"
            )
            continue
        seen_fingerprints.add(fingerprint)

        # --- Chosen response: try passing trace first ---
        chosen_response = ""
        construction_method = ""

        passing_run = await _find_passing_trace(
            candidate.scenario_id, db, exclude_run_id=candidate.run_id
        )

        if passing_run:
            # BEST: Use actual passing trace
            raw_response = _get_rejected_response(passing_run)  # "last response"
            chosen_response = _sanitize_response(raw_response)
            construction_method = "passing_trace"
            logger.info(
                f"Found passing trace for {candidate.scenario_id} "
                f"(run {passing_run.run_id}, version {passing_run.agent_version})"
            )
        else:
            # SECOND: Construct from expected behavior
            raw_response = _construct_from_expected(scenario)
            chosen_response = _sanitize_response(raw_response)
            construction_method = "expected_behavior"
            logger.info(
                f"Constructed chosen response for {candidate.scenario_id} "
                f"from expected behavior"
            )

        if not chosen_response.strip():
            logger.warning(f"Could not produce chosen response for {candidate.scenario_id}")
            continue

        # --- Validate the chosen response ---
        validated, details = validate_chosen_response(
            chosen_response, scenario, tool_calls=[]
        )

        pair = PreferencePair(
            scenario_id=candidate.scenario_id,
            source_candidate_id=candidate.candidate_id,
            prompt_or_context=prompt_context,
            chosen_response=chosen_response,
            rejected_response=rejected_response,
            failure_category=candidate.failure_category,
            severity=candidate.severity,
            provenance={
                "source_run_id": candidate.run_id,
                "source_agent_version": candidate.agent_version,
                "passing_run_id": passing_run.run_id if passing_run else None,
                "passing_agent_version": passing_run.agent_version if passing_run else None,
                "scenario_expected_outcome": scenario.expected_outcome,
                "scenario_expected_tool_calls": [
                    tc.model_dump(mode="json") for tc in scenario.expected_tool_calls
                ],
            },
            construction_method=construction_method,
            confidence=candidate.confidence if validated else candidate.confidence * 0.5,
            validation_status="validated" if validated else "rejected",
            validation_details=details,
        )
        pairs.append(pair)

        status_icon = "✓" if validated else "✗"
        logger.info(
            f"  {status_icon} Pair for {candidate.scenario_id}: "
            f"method={construction_method}, validation={pair.validation_status}"
        )

    return pairs, duplicates_removed


def _construct_from_expected(scenario: TestScenario) -> str:
    """Construct a chosen response from scenario expected behavior.

    Uses semantic templates based on the expected outcome type.
    """
    outcome = scenario.expected_outcome

    if outcome == "clarification_requested":
        return _CLARIFICATION_TEMPLATES[0]

    if outcome == "escalated":
        return _ESCALATION_TEMPLATES[0]

    if outcome == "information_provided":
        return (
            "Let me look up the details for you. "
            "Could you confirm the property ID so I can provide accurate information?"
        )

    if outcome == "appointment_booked":
        return (
            "I'd be happy to help you schedule a viewing! "
            "Let me check the availability for that time."
        )

    return ""
