"""Deterministic preference validators.

A response does NOT become a "chosen" training example simply because it
sounds nicer.  These validators check concrete, measurable properties:

1. Expected conversational outcome (clarification, escalation, etc.)
2. Tool behavior (no unauthorized tool calls)
3. Constraint adherence (no fabricated data, no hallucinated properties)
4. Workflow semantics (actually asks a clarifying question, etc.)
"""

from __future__ import annotations

import re
from typing import Any

from app.models import TestScenario


# ---------------------------------------------------------------------------
# Outcome signal detectors (reuse the same signals as task_evaluator)
# ---------------------------------------------------------------------------

_CLARIFICATION_SIGNALS = [
    "could you clarify", "which property", "what date", "what time",
    "can you provide", "could you please provide", "please provide",
    "could you tell me", "do you mean", "specific property id",
    "property id", "more details", "narrow it down", "help me understand",
    "what area", "what location", "budget", "preferences", "requirements",
    "looking for", "interested in", "share a property",
]

_ESCALATION_SIGNALS = [
    "human agent", "connect you", "escalate", "transfer",
    "someone who can help", "cannot help", "outside my capabilities",
]

_BOOKING_SIGNALS = [
    "confirmed", "booking", "booked", "appointment", "BK-",
]

_INFO_SIGNALS = [
    "bedrooms", "bathrooms", "price", "features", "address",
]

# Signals that a response contains a tool call (for text-based checks)
_TOOL_ACTION_SIGNALS = [
    "property_lookup", "check_availability", "book_appointment",
]


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def _check_outcome_match(
    response: str,
    expected_outcome: str,
) -> tuple[bool, str]:
    """Verify that the response matches the expected conversational outcome."""
    response_lower = response.lower()

    if expected_outcome == "clarification_requested":
        if any(sig in response_lower for sig in _CLARIFICATION_SIGNALS):
            return True, "Response contains clarification signals."
        # Also check for question marks as a weak signal
        if "?" in response:
            return True, "Response asks a question (weak clarification signal)."
        return False, (
            "Expected clarification but response does not contain "
            "any clarification signals or questions."
        )

    if expected_outcome == "escalated":
        if any(sig in response_lower for sig in _ESCALATION_SIGNALS):
            return True, "Response contains escalation signals."
        return False, "Expected escalation but no escalation signals found."

    if expected_outcome == "appointment_booked":
        if any(sig in response_lower for sig in _BOOKING_SIGNALS):
            return True, "Response contains booking confirmation signals."
        return False, "Expected booking confirmation but no booking signals found."

    if expected_outcome == "information_provided":
        if any(sig in response_lower for sig in _INFO_SIGNALS):
            return True, "Response contains information signals."
        closing_signals = [
            "take your time", "let me know", "feel free", "have a great day",
            "happy to help", "any other questions", "questions about", "glad I could help",
        ]
        if any(sig in response_lower for sig in closing_signals):
            return True, "Response contains polite closing signals following information provision."
        return False, "Expected information but no info signals found."

    # Unknown outcome type — accept with warning
    return True, f"No specific validator for outcome '{expected_outcome}'."


def _check_no_unauthorized_tools(
    response: str,
    expected_outcome: str,
    tool_calls: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Verify that no unauthorized tool calls were made.

    For clarification_requested scenarios, NO tool calls should exist.
    """
    if expected_outcome == "clarification_requested":
        if tool_calls:
            tools_used = [tc.get("tool_name", "unknown") for tc in tool_calls]
            return False, (
                f"Clarification scenario should have no tool calls, "
                f"but found: {tools_used}"
            )
        return True, "No tool calls in clarification response — correct."

    return True, "Tool call check passed (non-clarification scenario)."


def _check_no_constraint_violations(
    response: str,
    scenario: TestScenario,
) -> tuple[bool, str]:
    """Basic constraint check on the chosen response text."""
    response_lower = response.lower()

    # Check must_not_hallucinate_properties constraint
    if "must_not_hallucinate_properties" in scenario.constraints:
        known_ids = {"PROP-101", "PROP-102", "PROP-103", "PROP-104"}
        prop_pattern = re.compile(r'PROP-\d{3}', re.IGNORECASE)
        found_ids = prop_pattern.findall(response)
        for pid in found_ids:
            if pid.upper() not in known_ids:
                return False, f"Response mentions unknown property '{pid}'."

    # Check must_not_fabricate_price constraint
    if "must_not_fabricate_price" in scenario.constraints:
        price_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?')
        prices = price_pattern.findall(response)
        if prices:
            return False, f"Response fabricates price(s): {prices}."

    return True, "No constraint violations detected."


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_chosen_response(
    response: str,
    scenario: TestScenario,
    tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a chosen response against deterministic criteria.

    A chosen response is admitted only if ALL checks pass.

    Args:
        response: the candidate chosen response text.
        scenario: the scenario definition.
        tool_calls: tool calls associated with the response (empty for
                   constructed responses).

    Returns:
        (validated, details) where details lists each check result.
    """
    if tool_calls is None:
        tool_calls = []

    details: list[str] = []
    all_passed = True

    # 1. Outcome match
    passed, msg = _check_outcome_match(response, scenario.expected_outcome)
    details.append(f"{'✓' if passed else '✗'} outcome_match: {msg}")
    if not passed:
        all_passed = False

    # 2. No unauthorized tools
    passed, msg = _check_no_unauthorized_tools(
        response, scenario.expected_outcome, tool_calls
    )
    details.append(f"{'✓' if passed else '✗'} tool_check: {msg}")
    if not passed:
        all_passed = False

    # 3. No constraint violations
    passed, msg = _check_no_constraint_violations(response, scenario)
    details.append(f"{'✓' if passed else '✗'} constraint_check: {msg}")
    if not passed:
        all_passed = False

    # 4. Non-empty response
    if not response.strip():
        details.append("✗ content_check: Response is empty.")
        all_passed = False
    else:
        details.append("✓ content_check: Response is non-empty.")

    # 5. No internal/policy text
    passed, msg = _check_no_internal_text(response)
    details.append(f"{'✓' if passed else '✗'} internal_text_check: {msg}")
    if not passed:
        all_passed = False

    return all_passed, details


# ---------------------------------------------------------------------------
# Internal/policy text detector
# ---------------------------------------------------------------------------

_INTERNAL_KEYWORDS = [
    "rule 8 states",
    "rule 8 says",
    "rule states",
    "note: rule",
    "internal note",
    "policy note",
    "chain of thought",
    "chain-of-thought",
    "my reasoning",
    "let me think",
    "evaluator instruction",
    "system prompt",
    "do not search by description alone",
    "do not call property_lookup",
]


def _check_no_internal_text(response: str) -> tuple[bool, str]:
    """Verify that the response contains no internal rules or policy text.

    A customer-facing training example must NEVER contain leaked system
    prompt rules, evaluator instructions, chain-of-thought, or metadata.
    """
    response_lower = response.lower()

    for keyword in _INTERNAL_KEYWORDS:
        if keyword in response_lower:
            return False, (
                f"Response contains internal/policy text: '{keyword}'. "
                f"This must be stripped before use as training data."
            )

    # Check for parenthetical notes with rule/policy markers
    paren_patterns = [
        r'\*?\((?:Note|Important|Internal|Policy|Rule)\s*[\d]*\s*:',
    ]
    for pat in paren_patterns:
        if re.search(pat, response, re.IGNORECASE):
            return False, (
                f"Response contains parenthetical internal note matching '{pat}'."
            )

    return True, "No internal/policy text detected."

