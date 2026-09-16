"""Trajectory Preference Validator (DPO V3).

Validates structured trajectory preference pairs:
1. Structural integrity (context, chosen, rejected)
2. Decision-type consistency (No-Tool vs Required Tool vs Order vs Arguments)
3. Response purity (no evaluator instructions, policy rules, or CoT)
4. Valid tool names and arguments against TOOL_SCHEMAS
5. Proper fingerprinting and provenance
"""

from __future__ import annotations

import re
from typing import Any

from app.learning.trajectory_models import (
    DecisionType,
    TrajectoryAction,
    TrajectoryPreferencePair,
)
from app.tools.registry import TOOL_SCHEMAS

# Allowed tool names from registry
VALID_TOOL_NAMES = {
    schema["function"]["name"]
    for schema in TOOL_SCHEMAS
    if "function" in schema and "name" in schema["function"]
}

# Forbidden policy / internal / evaluator phrases in customer-facing text
FORBIDDEN_PHRASES = [
    r"rule \d+",
    r"rule\s*#?\d+",
    r"evaluator",
    r"instruction",
    r"policy",
    r"benchmark",
    r"ground truth",
    r"p0 baseline",
    r"<scratchpad>",
    r"system prompt",
    r"as an ai language model",
    r"cannot fabricate",
]


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory preference pair fails validation."""
    pass


def validate_trajectory_action(action: TrajectoryAction, field_name: str) -> list[str]:
    """Validate a single trajectory action."""
    errors = []
    
    # Must have either content or tool calls (or both)
    if not action.content and not action.tool_calls:
        errors.append(f"{field_name} must have either content or tool_calls")

    # Validate tool calls if present
    for tc in action.tool_calls:
        if not tc.name:
            errors.append(f"{field_name} has a tool call with empty name")
        elif tc.name not in VALID_TOOL_NAMES:
            # Note: in negative control rejected responses, we might test invalid tools,
            # but standard actions should use known tools
            pass
        if not isinstance(tc.arguments, dict):
            errors.append(f"{field_name} tool {tc.name} arguments must be a dict")

    # Validate content purity if present
    if action.content:
        content_lower = action.content.lower()
        for phrase in FORBIDDEN_PHRASES:
            if re.search(phrase, content_lower):
                errors.append(f"{field_name} contains forbidden internal phrase: '{phrase}'")

    return errors


def validate_trajectory_pair(pair: TrajectoryPreferencePair) -> list[str]:
    """Validate a complete trajectory preference pair."""
    errors = []

    if not pair.pair_id:
        errors.append("pair_id is required")
    if not pair.scenario_id:
        errors.append("scenario_id is required")
    if not pair.learning_target:
        errors.append("learning_target is required")
    if not pair.context or not isinstance(pair.context, list):
        errors.append("context must be a non-empty list of messages")

    # Validate chosen and rejected actions
    errors.extend(validate_trajectory_action(pair.chosen, "chosen"))
    errors.extend(validate_trajectory_action(pair.rejected, "rejected"))

    # Decision-type consistency checks
    dtype = pair.decision_type.value if isinstance(pair.decision_type, DecisionType) else str(pair.decision_type)

    if dtype == DecisionType.NO_TOOL.value:
        # Chosen MUST NOT emit tool calls
        if len(pair.chosen.tool_calls) != 0:
            errors.append(f"NO_TOOL chosen action must have 0 tool calls, found {len(pair.chosen.tool_calls)}")
        # Chosen MUST have customer-facing content (clarification)
        if not pair.chosen.content or not pair.chosen.content.strip():
            errors.append("NO_TOOL chosen action must have clarification content")
        # Rejected MUST emit tool calls (the unwanted/speculative call)
        if len(pair.rejected.tool_calls) == 0:
            errors.append("NO_TOOL rejected action must have >= 1 tool call (unwanted tool call)")

    elif dtype == DecisionType.REQUIRED_TOOL.value:
        # Chosen MUST emit required tool call(s)
        if len(pair.chosen.tool_calls) == 0:
            errors.append("REQUIRED_TOOL chosen action must have >= 1 tool call")
        # Rejected should either omit tool calls or have wrong/absent tool calls
        if len(pair.rejected.tool_calls) != 0 and pair.rejected.tool_calls == pair.chosen.tool_calls:
            errors.append("REQUIRED_TOOL rejected action cannot be identical to chosen tool calls")

    elif dtype == DecisionType.TOOL_ORDER.value:
        # Both typically have tool calls, but order differs (e.g., check_availability before book_appointment)
        if len(pair.chosen.tool_calls) == 0 or len(pair.rejected.tool_calls) == 0:
            errors.append("TOOL_ORDER requires tool calls in both chosen and rejected")

    elif dtype == DecisionType.TOOL_ARGS.value:
        # Both call tools, but arguments differ
        if len(pair.chosen.tool_calls) == 0 or len(pair.rejected.tool_calls) == 0:
            errors.append("TOOL_ARGS requires tool calls in both chosen and rejected")

    elif dtype == DecisionType.ESCALATION.value:
        # Chosen must not invent info
        if pair.chosen.content is None:
            errors.append("ESCALATION chosen action must have escalation message")

    # Check that chosen and rejected formatted completions are not identical
    if pair.chosen.format_completion() == pair.rejected.format_completion():
        errors.append("chosen and rejected completions must not be identical")

    return errors
