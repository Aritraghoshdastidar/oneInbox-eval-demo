from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

"""Construct Trajectory Preferences Dataset V5 for DPO.

Experiment V5: Tool-Routing Discrimination + Escalation Preservation.

THIS IS THE FINAL TRAINING DATASET.

Objectives:
- Teach fine-grained routing: property_lookup vs check_availability vs book_appointment
- Preserve ambiguity handling (no-tool + clarification)
- Preserve escalation behavior (unknown property + insistence -> escalate)
- Improve argument accuracy and tool-order accuracy

10 Categories:
1.  PROPERTY_LOOKUP vs CHECK_AVAILABILITY routing
2.  PROPERTY_LOOKUP before BOOKING (prerequisite enforcement)
3.  VALID PROPERTY BOOKING (full sequence)
4.  PROPERTY INFO ONLY
5.  AVAILABILITY ONLY
6.  AMBIGUITY (no-tool preservation)
7.  UNKNOWN PROPERTY + ESCALATION
8.  UNKNOWN PROPERTY + ALTERNATIVES REQUESTED
9.  TOOL ARGUMENTS
10. TOOL ORDER

Strict Holdouts:
- RE-005, RE-004, AMB-001 and any same-lesson fingerprints are excluded from training.
- Surrogate IDs for escalation training: PROP-777, PROP-888, PROP-555, PROP-404, PROP-666
- Held-out evaluation IDs: PROP-707, PROP-808, PROP-606, PROP-505, PROP-303
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.prompts import SYSTEM_PROMPT
from app.learning.trajectory_models import (
    DecisionType,
    TrajectoryAction,
    TrajectoryPreferencePair,
    TrajectoryToolCall,
)
from app.learning.trajectory_validator import validate_trajectory_pair

OUTPUT_DIR = Path("data/preferences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_system_message() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def ctx(*user_msgs: str) -> list[dict[str, str]]:
    """Build a context with system prompt + user messages."""
    messages = [make_system_message()]
    for msg in user_msgs:
        messages.append({"role": "user", "content": msg})
    return messages


def tc(name: str, **kwargs: Any) -> TrajectoryToolCall:
    """Shorthand for TrajectoryToolCall."""
    return TrajectoryToolCall(name=name, arguments=kwargs)


def action(content: str | None = None, tools: list[TrajectoryToolCall] | None = None) -> TrajectoryAction:
    """Shorthand for TrajectoryAction."""
    return TrajectoryAction(content=content, tool_calls=tools or [])


# =========================================================================
# CATEGORY 1: PROPERTY_LOOKUP vs CHECK_AVAILABILITY (Routing Core)
# ~10-12 pairs — the central contrastive distinction
# =========================================================================

def build_cat1_routing_core() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Property info → property_lookup (not check_availability) ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-001",
        scenario_id="V5-RT-001",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Property information request routes to property_lookup, not check_availability",
        context=ctx("Can you tell me about PROP-102? I'd like to know the price and features."),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-20", time="10:00")]),
        failure_category="wrong_tool_routing",
        severity="critical",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 2: Availability request → check_availability (not property_lookup) ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-002",
        scenario_id="V5-RT-002",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Availability request routes to check_availability, not property_lookup",
        context=ctx("Is PROP-101 available tomorrow September 20th at 2 PM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-101")]),
        failure_category="wrong_tool_routing",
        severity="critical",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 3: Another property info with different wording ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-003",
        scenario_id="V5-RT-003",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Details request routes to property_lookup",
        context=ctx("What are the details for the property PROP-103? How many bedrooms does it have?"),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-103")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-103", date="2026-09-20", time="10:00")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 4: Availability with different time ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-004",
        scenario_id="V5-RT-004",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Availability check for specific slot routes to check_availability",
        context=ctx("I'd like to know if PROP-104 has an available viewing slot on Saturday September 20th at 3 PM."),
        chosen=action(tools=[tc("check_availability", property_id="PROP-104", date="2026-09-20", time="15:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-104")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 5: NEGATIVE CONTROL — "schedule a viewing" contains lexical booking cue but needs lookup first ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-005",
        scenario_id="V5-RT-005",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="NEGATIVE CONTROL: 'schedule a viewing' with unknown validation state requires property_lookup first",
        context=ctx("Can you give me information about the property where I'd like to schedule a viewing? The ID is PROP-104."),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-104")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-104", date="2026-09-20", time="10:00")]),
        failure_category="lexical_shortcut",
        severity="critical",
        provenance={"source": "v5_negative_control", "category": "cat1_routing_core"},
    ))

    # --- Pair 6: NEGATIVE CONTROL — "property" keyword but actually availability ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-006",
        scenario_id="V5-RT-006",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="NEGATIVE CONTROL: availability request despite 'property' keyword",
        context=ctx("The property PROP-102 is already verified. Is the 10 AM slot on September 20th available?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-20", time="10:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        failure_category="lexical_shortcut",
        severity="critical",
        provenance={"source": "v5_negative_control", "category": "cat1_routing_core"},
    ))

    # --- Pair 7: Property existence check for PROP-101 ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-007",
        scenario_id="V5-RT-007",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="'Does this property exist?' routes to property_lookup",
        context=ctx("Does PROP-101 actually exist? I want to verify it's a real listing before anything else."),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-101")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 8: Availability with time and date explicitly stated ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-008",
        scenario_id="V5-RT-008",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Explicit date/time availability request routes to check_availability",
        context=ctx("Can I view PROP-101 on Sunday September 21st at 10 AM? Is that slot open?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-21", time="10:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-101")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 9: "How much" routes to lookup ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-009",
        scenario_id="V5-RT-009",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Price question routes to property_lookup",
        context=ctx("How much is PROP-104? What's the asking price?"),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-104")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-104", date="2026-09-20", time="10:00")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    # --- Pair 10: "Available this weekend" routes to check_availability ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-RT-010",
        scenario_id="V5-RT-010",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Weekend availability request routes to check_availability",
        context=ctx("Is PROP-102 available for viewing this Saturday September 20th at 2 PM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-20", time="14:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_routing_contrastive", "category": "cat1_routing_core"},
    ))

    return pairs


# =========================================================================
# CATEGORY 2: PROPERTY_LOOKUP BEFORE BOOKING (Prerequisite Enforcement)
# ~4-5 pairs — unknown property should be looked up, not availability-checked
# =========================================================================

def build_cat2_prerequisite() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Unknown property booking → lookup first, not availability ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-PRE-001",
        scenario_id="V5-PRE-001",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Unknown property booking must start with property_lookup, not check_availability",
        context=ctx("I'd like to schedule a viewing for PROP-777."),
        chosen=action(
            content="Let me look up that property for you first.",
            tools=[tc("property_lookup", property_id="PROP-777")],
        ),
        rejected=action(tools=[tc("check_availability", property_id="PROP-777", date="2026-09-20", time="14:00")]),
        failure_category="prerequisite_violation",
        severity="critical",
        provenance={"source": "v5_prerequisite_contrastive", "category": "cat2_prerequisite"},
    ))

    # --- Pair 2: Another unknown property ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-PRE-002",
        scenario_id="V5-PRE-002",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Must verify property existence before checking availability",
        context=ctx("Can I book a viewing for PROP-888 this Saturday at 10 AM?"),
        chosen=action(
            content="Let me first verify that property.",
            tools=[tc("property_lookup", property_id="PROP-888")],
        ),
        rejected=action(tools=[tc("check_availability", property_id="PROP-888", date="2026-09-20", time="10:00")]),
        failure_category="prerequisite_violation",
        severity="critical",
        provenance={"source": "v5_prerequisite_contrastive", "category": "cat2_prerequisite"},
    ))

    # --- Pair 3: Unknown with "viewing" lexical cue ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-PRE-003",
        scenario_id="V5-PRE-003",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="'viewing' keyword should not bypass property verification for unknown ID",
        context=ctx("I want to view PROP-555. When can I come in?"),
        chosen=action(
            content="I'll look up that property first to see if it's in our system.",
            tools=[tc("property_lookup", property_id="PROP-555")],
        ),
        rejected=action(tools=[tc("check_availability", property_id="PROP-555", date="2026-09-20", time="14:00")]),
        failure_category="prerequisite_violation",
        severity="critical",
        provenance={"source": "v5_prerequisite_contrastive", "category": "cat2_prerequisite"},
    ))

    # --- Pair 4: Known property booking still requires lookup first ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-PRE-004",
        scenario_id="V5-PRE-004",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Even known property booking should verify via property_lookup before availability",
        context=ctx("I'd like to book a viewing for PROP-101 on Saturday September 20th at 2 PM."),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
        ]),
        rejected=action(tools=[
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00"),
        ]),
        failure_category="prerequisite_violation",
        severity="high",
        provenance={"source": "v5_prerequisite_contrastive", "category": "cat2_prerequisite"},
    ))

    return pairs


# =========================================================================
# CATEGORY 3: VALID PROPERTY BOOKING (Full Sequence)
# ~4-5 pairs — correct multi-step tool sequences
# =========================================================================

def build_cat3_valid_booking() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Full correct booking sequence ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-BK-001",
        scenario_id="V5-BK-001",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="Correct booking sequence: lookup → availability → book",
        context=ctx(
            "Hi, I want to book a viewing for PROP-101 on Saturday September 20th at 2 PM.",
            "Yes, please book it. My name is Alice Brown, email alice@example.com.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="14:00",
               contact_name="Alice Brown", contact_email="alice@example.com"),
        ]),
        rejected=action(tools=[
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="14:00",
               contact_name="Alice Brown", contact_email="alice@example.com"),
        ]),
        failure_category="wrong_tool_order",
        severity="high",
        provenance={"source": "v5_booking_sequence", "category": "cat3_valid_booking"},
    ))

    # --- Pair 2: PROP-102 booking ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-BK-002",
        scenario_id="V5-BK-002",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="Booking for PROP-102 follows correct prerequisite chain",
        context=ctx(
            "I'd like to view PROP-102 on Sunday September 21st at 11 AM.",
            "Book it please. I'm Bob Chen, bob.chen@email.com, phone 555-0188.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-102"),
            tc("check_availability", property_id="PROP-102", date="2026-09-21", time="11:00"),
            tc("book_appointment", property_id="PROP-102", date="2026-09-21", time="11:00",
               contact_name="Bob Chen", contact_email="bob.chen@email.com", contact_phone="555-0188"),
        ]),
        rejected=action(tools=[
            tc("book_appointment", property_id="PROP-102", date="2026-09-21", time="11:00",
               contact_name="Bob Chen", contact_email="bob.chen@email.com", contact_phone="555-0188"),
        ]),
        failure_category="wrong_tool_order",
        severity="high",
        provenance={"source": "v5_booking_sequence", "category": "cat3_valid_booking"},
    ))

    # --- Pair 3: Reject booking before availability ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-BK-003",
        scenario_id="V5-BK-003",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="Booking without availability check is rejected",
        context=ctx(
            "Book PROP-104 for Saturday September 20th at 10 AM.",
            "My name is Carol Davis, email carol@test.com.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-104"),
            tc("check_availability", property_id="PROP-104", date="2026-09-20", time="10:00"),
            tc("book_appointment", property_id="PROP-104", date="2026-09-20", time="10:00",
               contact_name="Carol Davis", contact_email="carol@test.com"),
        ]),
        rejected=action(tools=[
            tc("property_lookup", property_id="PROP-104"),
            tc("book_appointment", property_id="PROP-104", date="2026-09-20", time="10:00",
               contact_name="Carol Davis", contact_email="carol@test.com"),
        ]),
        failure_category="wrong_tool_order",
        severity="critical",
        provenance={"source": "v5_booking_sequence", "category": "cat3_valid_booking"},
    ))

    # --- Pair 4: Reject duplicate availability checks ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-BK-004",
        scenario_id="V5-BK-004",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="No redundant duplicate availability checks",
        context=ctx(
            "I want to view PROP-101 on Saturday September 20th at 10 AM. Book it for me.",
            "My name is Dan Evans, dan@example.com.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="10:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="10:00",
               contact_name="Dan Evans", contact_email="dan@example.com"),
        ]),
        rejected=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="10:00"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="10:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="10:00",
               contact_name="Dan Evans", contact_email="dan@example.com"),
        ]),
        failure_category="redundant_tool_call",
        severity="medium",
        provenance={"source": "v5_booking_sequence", "category": "cat3_valid_booking"},
    ))

    return pairs


# =========================================================================
# CATEGORY 4: PROPERTY INFO ONLY
# ~3-4 pairs — lookup only, no booking/availability
# =========================================================================

def build_cat4_info_only() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Simple property info ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-INF-001",
        scenario_id="V5-INF-001",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Property info request uses only property_lookup, no booking tools",
        context=ctx("What's the house at 18 Maple Drive like? I think that's PROP-102."),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        rejected=action(tools=[
            tc("property_lookup", property_id="PROP-102"),
            tc("check_availability", property_id="PROP-102", date="2026-09-20", time="10:00"),
        ]),
        failure_category="unnecessary_tool_call",
        severity="medium",
        provenance={"source": "v5_info_only", "category": "cat4_info_only"},
    ))

    # --- Pair 2: Features question ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-INF-002",
        scenario_id="V5-INF-002",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Feature-specific question uses property_lookup only",
        context=ctx("Does PROP-101 have parking? And how many bathrooms?"),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-101")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_info_only", "category": "cat4_info_only"},
    ))

    # --- Pair 3: Price comparison question ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-INF-003",
        scenario_id="V5-INF-003",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Price question uses property_lookup",
        context=ctx("I want to compare PROP-103 and PROP-104. Can you look up PROP-103 first?"),
        chosen=action(tools=[tc("property_lookup", property_id="PROP-103")]),
        rejected=action(tools=[
            tc("check_availability", property_id="PROP-103", date="2026-09-20", time="10:00"),
        ]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_info_only", "category": "cat4_info_only"},
    ))

    return pairs


# =========================================================================
# CATEGORY 5: AVAILABILITY ONLY
# ~3-4 pairs — availability check without booking
# =========================================================================

def build_cat5_availability_only() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Pure availability check ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AV-001",
        scenario_id="V5-AV-001",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Pure availability question routes to check_availability only",
        context=ctx("Is PROP-101 available on September 20th at 2 PM? I just want to know, not book yet."),
        chosen=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        rejected=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="14:00"),
        ]),
        failure_category="unnecessary_tool_call",
        severity="medium",
        provenance={"source": "v5_availability_only", "category": "cat5_availability_only"},
    ))

    # --- Pair 2: Another availability check ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AV-002",
        scenario_id="V5-AV-002",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Availability check at different time",
        context=ctx("I know PROP-104 is a townhouse. Can you check if there's a slot on September 20th at 3 PM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-104", date="2026-09-20", time="15:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-104")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_availability_only", "category": "cat5_availability_only"},
    ))

    # --- Pair 3: Availability with PROP-102 ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AV-003",
        scenario_id="V5-AV-003",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Check availability for PROP-102",
        context=ctx("Can you check if PROP-102 is open for viewing on September 21st at 11 AM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-21", time="11:00")]),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        failure_category="wrong_tool_routing",
        severity="high",
        provenance={"source": "v5_availability_only", "category": "cat5_availability_only"},
    ))

    return pairs


# =========================================================================
# CATEGORY 6: AMBIGUITY (No-Tool Preservation)
# ~5-6 pairs — must preserve V4A/V4B ambiguity breakthrough
# =========================================================================

def build_cat6_ambiguity() -> list[TrajectoryPreferencePair]:
    pairs = []

    # NOTE: AMB-001 and RE-004 are holdouts — we use semantically distinct variants.

    # --- Pair 1: Vague bedroom request ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AMB-001",
        scenario_id="V5-AMB-001",
        decision_type=DecisionType.NO_TOOL,
        learning_target="Vague bedroom request without property ID → no tool, clarify",
        context=ctx("I'm looking for a 3-bedroom apartment with a gym. What do you have?"),
        chosen=action(
            content="I would be happy to help you find a 3-bedroom apartment with gym access. "
                    "Could you please provide the property ID or address you are interested in?",
        ),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-101")]),
        failure_category="speculative_tool_call",
        severity="critical",
        provenance={"source": "v5_ambiguity_preservation", "category": "cat6_ambiguity"},
    ))

    # --- Pair 2: Feature-only description ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AMB-002",
        scenario_id="V5-AMB-002",
        decision_type=DecisionType.NO_TOOL,
        learning_target="Feature-only description without ID → no tool, clarify",
        context=ctx("Do you have any houses with solar panels and a garden? Something around $400k?"),
        chosen=action(
            content="I can certainly help you look for a house with solar panels and a garden. "
                    "Could you please provide the specific property ID or address so I can look it up for you?",
        ),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-102")]),
        failure_category="speculative_tool_call",
        severity="critical",
        provenance={"source": "v5_ambiguity_preservation", "category": "cat6_ambiguity"},
    ))

    # --- Pair 3: Location-only description ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AMB-003",
        scenario_id="V5-AMB-003",
        decision_type=DecisionType.NO_TOOL,
        learning_target="Location-only request without ID → no tool, clarify",
        context=ctx("I'm interested in apartments near River Lane. What's available?"),
        chosen=action(
            content="I'd be glad to help you find apartments near River Lane. "
                    "Could you please provide the specific property ID or full address?",
        ),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-103")]),
        failure_category="speculative_tool_call",
        severity="high",
        provenance={"source": "v5_ambiguity_preservation", "category": "cat6_ambiguity"},
    ))

    # --- Pair 4: Viewing request without property ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AMB-004",
        scenario_id="V5-AMB-004",
        decision_type=DecisionType.NO_TOOL,
        learning_target="Viewing request without specifying property → no tool, clarify",
        context=ctx("Can I schedule a viewing for this Saturday September 20th at 2 PM please?"),
        chosen=action(
            content="I would be happy to help you schedule a viewing for Saturday September 20th at 2 PM. "
                    "Could you please specify which property you would like to view by providing its property ID or address?",
        ),
        rejected=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        failure_category="speculative_tool_call",
        severity="critical",
        provenance={"source": "v5_ambiguity_preservation", "category": "cat6_ambiguity"},
    ))

    # --- Pair 5: Budget-only request ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-AMB-005",
        scenario_id="V5-AMB-005",
        decision_type=DecisionType.NO_TOOL,
        learning_target="Budget-only request without property → no tool, clarify",
        context=ctx("I have a budget of $300,000. What properties can I look at?"),
        chosen=action(
            content="I'd be happy to assist you with your property search within your budget. "
                    "Could you please provide the specific property ID or address you are interested in?",
        ),
        rejected=action(tools=[tc("property_lookup", property_id="PROP-104")]),
        failure_category="speculative_tool_call",
        severity="high",
        provenance={"source": "v5_ambiguity_preservation", "category": "cat6_ambiguity"},
    ))

    return pairs


# =========================================================================
# CATEGORY 7: UNKNOWN PROPERTY + ESCALATION
# ~5-6 pairs — surrogate IDs, not RE-005 wording
# =========================================================================

def build_cat7_escalation() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: PROP-777 not found + insistence → escalate ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ESC-001",
        scenario_id="V5-ESC-001",
        decision_type=DecisionType.ESCALATION,
        learning_target="Unknown property with insistence → escalate after NOT_FOUND",
        context=ctx(
            "I'd like to book a viewing for PROP-777. I was told it's one of your listings.",
            "I'm sure it exists. Please check again — it was on your site last week.",
        ),
        chosen=action(
            content="I apologize, but I was unable to locate PROP-777 in our system. Our current listings include "
                    "PROP-101, PROP-102, PROP-103, and PROP-104. Since I cannot find the property you are looking for, "
                    "I would like to connect you with a human agent who can investigate this further for you.",
            tools=[tc("property_lookup", property_id="PROP-777")],
        ),
        rejected=action(tools=[tc("check_availability", property_id="PROP-777", date="2026-09-20", time="14:00")]),
        failure_category="wrong_tool_routing_before_escalation",
        severity="critical",
        provenance={"source": "v5_escalation", "category": "cat7_escalation"},
    ))

    # --- Pair 2: PROP-888 booking attempt → lookup NOT_FOUND → escalate ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ESC-002",
        scenario_id="V5-ESC-002",
        decision_type=DecisionType.ESCALATION,
        learning_target="Booking attempt for unknown property → lookup → NOT_FOUND → escalate",
        context=ctx(
            "Please book PROP-888 for tomorrow at 3 PM.",
            "I don't care if you can't find it. I need that specific property.",
        ),
        chosen=action(
            content="I understand your frustration. Unfortunately, PROP-888 does not appear in our property database. "
                    "I will escalate this to a human agent who can help locate this property for you.",
            tools=[tc("property_lookup", property_id="PROP-888")],
        ),
        rejected=action(
            content="Let me check the availability for PROP-888.",
            tools=[tc("check_availability", property_id="PROP-888", date="2026-09-21", time="15:00")],
        ),
        failure_category="wrong_tool_routing_before_escalation",
        severity="critical",
        provenance={"source": "v5_escalation", "category": "cat7_escalation"},
    ))

    # --- Pair 3: PROP-555 → NOT_FOUND → do NOT substitute ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ESC-003",
        scenario_id="V5-ESC-003",
        decision_type=DecisionType.ESCALATION,
        learning_target="Unknown property must not be silently substituted with another property",
        context=ctx(
            "I want details on PROP-555. My friend recommended it.",
            "Are you sure you don't have it? They said they saw it yesterday.",
        ),
        chosen=action(
            content="I apologize, but PROP-555 is not currently in our system. I would like to connect you with "
                    "a human agent who may be able to help investigate this further.",
            tools=[tc("property_lookup", property_id="PROP-555")],
        ),
        rejected=action(
            content="I couldn't find PROP-555, but PROP-101 is a similar property. Let me look it up for you.",
            tools=[tc("property_lookup", property_id="PROP-101")],
        ),
        failure_category="unauthorized_substitution",
        severity="critical",
        provenance={"source": "v5_escalation", "category": "cat7_escalation"},
    ))

    # --- Pair 4: PROP-404 price request → NOT_FOUND → escalate ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ESC-004",
        scenario_id="V5-ESC-004",
        decision_type=DecisionType.ESCALATION,
        learning_target="Price request for unknown property → lookup → NOT_FOUND → escalate",
        context=ctx(
            "What's the price for PROP-404?",
            "Just give me a rough estimate then. What would similar properties cost?",
        ),
        chosen=action(
            content="I was unable to find PROP-404 in our listings. I cannot estimate a price for a property not "
                    "in our system. Let me connect you with a human agent who can assist you further.",
            tools=[tc("property_lookup", property_id="PROP-404")],
        ),
        rejected=action(
            content="Based on similar properties in the area, PROP-404 would cost approximately $350,000.",
        ),
        failure_category="fabricated_information",
        severity="critical",
        provenance={"source": "v5_escalation", "category": "cat7_escalation"},
    ))

    # --- Pair 5: PROP-666 → NOT_FOUND → don't keep retrying ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ESC-005",
        scenario_id="V5-ESC-005",
        decision_type=DecisionType.ESCALATION,
        learning_target="Unknown property → single lookup, then escalate, not endless retries",
        context=ctx(
            "Look up PROP-666 for me.",
            "Try again. Maybe it was just added to the system.",
        ),
        chosen=action(
            content="I have checked and PROP-666 is not in our current listings. Rather than continuing to search, "
                    "I will connect you with a human agent who can verify whether this property has been recently added.",
            tools=[tc("property_lookup", property_id="PROP-666")],
        ),
        rejected=action(
            tools=[
                tc("property_lookup", property_id="PROP-666"),
                tc("property_lookup", property_id="PROP-666"),
                tc("property_lookup", property_id="PROP-666"),
            ],
        ),
        failure_category="infinite_retry_loop",
        severity="high",
        provenance={"source": "v5_escalation", "category": "cat7_escalation"},
    ))

    return pairs


# =========================================================================
# CATEGORY 8: UNKNOWN PROPERTY + ALTERNATIVES REQUESTED
# ~2 pairs — hard negative controls against blind escalation
# =========================================================================

def build_cat8_alternatives() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Customer explicitly requests alternatives ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ALT-001",
        scenario_id="V5-ALT-001",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="When customer explicitly requests alternatives, provide them instead of escalating",
        context=ctx(
            "I wanted PROP-777 but I see it's not available. Can you show me other properties you have?",
        ),
        chosen=action(
            content="Of course! Let me look up some alternatives for you.",
            tools=[tc("property_lookup", property_id="PROP-101")],
        ),
        rejected=action(
            content="Since PROP-777 is not in our system, I need to escalate this to a human agent.",
            tools=[tc("property_lookup", property_id="PROP-777")],
        ),
        failure_category="unnecessary_escalation",
        severity="high",
        provenance={"source": "v5_alternative_control", "category": "cat8_alternatives"},
    ))

    # --- Pair 2: Another explicit alternative request ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ALT-002",
        scenario_id="V5-ALT-002",
        decision_type=DecisionType.REQUIRED_TOOL,
        learning_target="Explicit 'show me something else' -> provide alternatives, not escalate",
        context=ctx(
            "PROP-888 doesn't exist? That's fine. Show me something else -- maybe a 4-bedroom house?",
        ),
        chosen=action(
            content="Absolutely! PROP-102 at 18 Maple Drive is a 4-bedroom house. Let me look it up for you.",
            tools=[tc("property_lookup", property_id="PROP-102")],
        ),
        rejected=action(
            content="I'm unable to help with PROP-888. Let me transfer you to a human agent.",
            tools=[tc("property_lookup", property_id="PROP-888")],
        ),
        failure_category="unnecessary_escalation",
        severity="high",
        provenance={"source": "v5_alternative_control", "category": "cat8_alternatives"},
    ))

    return pairs


# =========================================================================
# CATEGORY 9: TOOL ARGUMENTS
# ~3-4 pairs — correct vs wrong arguments
# =========================================================================

def build_cat9_tool_args() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Correct property_id vs wrong property_id ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ARG-001",
        scenario_id="V5-ARG-001",
        decision_type=DecisionType.TOOL_ARGS,
        learning_target="Correct property_id argument when customer specifies PROP-101",
        context=ctx("Can I view PROP-101 on September 20th at 2 PM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="14:00")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-20", time="14:00")]),
        failure_category="wrong_argument",
        severity="high",
        provenance={"source": "v5_argument_accuracy", "category": "cat9_tool_args"},
    ))

    # --- Pair 2: Correct date format ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ARG-002",
        scenario_id="V5-ARG-002",
        decision_type=DecisionType.TOOL_ARGS,
        learning_target="Date must be formatted correctly as YYYY-MM-DD",
        context=ctx("Is PROP-102 available on September 21st at 11 AM?"),
        chosen=action(tools=[tc("check_availability", property_id="PROP-102", date="2026-09-21", time="11:00")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-102", date="09/21/2026", time="11:00")]),
        failure_category="wrong_argument_format",
        severity="high",
        provenance={"source": "v5_argument_accuracy", "category": "cat9_tool_args"},
    ))

    # --- Pair 3: Correct time format ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ARG-003",
        scenario_id="V5-ARG-003",
        decision_type=DecisionType.TOOL_ARGS,
        learning_target="Time must be in HH:MM 24-hour format",
        context=ctx("Check if PROP-101 is free on September 20th at 4 PM."),
        chosen=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="16:00")]),
        rejected=action(tools=[tc("check_availability", property_id="PROP-101", date="2026-09-20", time="4:00 PM")]),
        failure_category="wrong_argument_format",
        severity="high",
        provenance={"source": "v5_argument_accuracy", "category": "cat9_tool_args"},
    ))

    # --- Pair 4: Correct customer info in booking ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ARG-004",
        scenario_id="V5-ARG-004",
        decision_type=DecisionType.TOOL_ARGS,
        learning_target="Booking must include correct customer contact information",
        context=ctx(
            "Book PROP-104 on September 20th at 10 AM for me.",
            "My name is Emily Foster, email emily.foster@mail.com.",
        ),
        chosen=action(tools=[
            tc("book_appointment", property_id="PROP-104", date="2026-09-20", time="10:00",
               contact_name="Emily Foster", contact_email="emily.foster@mail.com"),
        ]),
        rejected=action(tools=[
            tc("book_appointment", property_id="PROP-104", date="2026-09-20", time="10:00"),
        ]),
        failure_category="missing_argument",
        severity="medium",
        provenance={"source": "v5_argument_accuracy", "category": "cat9_tool_args"},
    ))

    return pairs


# =========================================================================
# CATEGORY 10: TOOL ORDER
# ~3-4 pairs — correct prerequisite sequence vs wrong sequence
# =========================================================================

def build_cat10_tool_order() -> list[TrajectoryPreferencePair]:
    pairs = []

    # --- Pair 1: Correct order vs reversed ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ORD-001",
        scenario_id="V5-ORD-001",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="property_lookup must come before check_availability",
        context=ctx("I'd like to view PROP-104 on September 20th at 3 PM."),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-104"),
            tc("check_availability", property_id="PROP-104", date="2026-09-20", time="15:00"),
        ]),
        rejected=action(tools=[
            tc("check_availability", property_id="PROP-104", date="2026-09-20", time="15:00"),
            tc("property_lookup", property_id="PROP-104"),
        ]),
        failure_category="wrong_tool_order",
        severity="high",
        provenance={"source": "v5_tool_order", "category": "cat10_tool_order"},
    ))

    # --- Pair 2: availability before book vs book first ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ORD-002",
        scenario_id="V5-ORD-002",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="check_availability must precede book_appointment",
        context=ctx(
            "Book PROP-101 for September 20th at 10 AM.",
            "Name is Frank Garcia, frank@email.com.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-101"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="10:00"),
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="10:00",
               contact_name="Frank Garcia", contact_email="frank@email.com"),
        ]),
        rejected=action(tools=[
            tc("book_appointment", property_id="PROP-101", date="2026-09-20", time="10:00",
               contact_name="Frank Garcia", contact_email="frank@email.com"),
            tc("check_availability", property_id="PROP-101", date="2026-09-20", time="10:00"),
        ]),
        failure_category="wrong_tool_order",
        severity="critical",
        provenance={"source": "v5_tool_order", "category": "cat10_tool_order"},
    ))

    # --- Pair 3: availability → lookup → book (wrong: lookup out of order) ---
    pairs.append(TrajectoryPreferencePair(
        pair_id="V5-ORD-003",
        scenario_id="V5-ORD-003",
        decision_type=DecisionType.TOOL_ORDER,
        learning_target="Full sequence must be lookup → availability → book",
        context=ctx(
            "I want to book PROP-102 on Sunday September 21st at 11 AM.",
            "My name is Grace Hill, grace.hill@email.com, phone 555-0200.",
        ),
        chosen=action(tools=[
            tc("property_lookup", property_id="PROP-102"),
            tc("check_availability", property_id="PROP-102", date="2026-09-21", time="11:00"),
            tc("book_appointment", property_id="PROP-102", date="2026-09-21", time="11:00",
               contact_name="Grace Hill", contact_email="grace.hill@email.com", contact_phone="555-0200"),
        ]),
        rejected=action(tools=[
            tc("check_availability", property_id="PROP-102", date="2026-09-21", time="11:00"),
            tc("property_lookup", property_id="PROP-102"),
            tc("book_appointment", property_id="PROP-102", date="2026-09-21", time="11:00",
               contact_name="Grace Hill", contact_email="grace.hill@email.com", contact_phone="555-0200"),
        ]),
        failure_category="wrong_tool_order",
        severity="high",
        provenance={"source": "v5_tool_order", "category": "cat10_tool_order"},
    ))

    return pairs


# =========================================================================
# ASSEMBLY + VALIDATION + SPLITTING + OUTPUT
# =========================================================================

def build_all_v5_pairs() -> list[TrajectoryPreferencePair]:
    """Assemble all 10 categories."""
    all_pairs: list[TrajectoryPreferencePair] = []
    all_pairs.extend(build_cat1_routing_core())
    all_pairs.extend(build_cat2_prerequisite())
    all_pairs.extend(build_cat3_valid_booking())
    all_pairs.extend(build_cat4_info_only())
    all_pairs.extend(build_cat5_availability_only())
    all_pairs.extend(build_cat6_ambiguity())
    all_pairs.extend(build_cat7_escalation())
    all_pairs.extend(build_cat8_alternatives())
    all_pairs.extend(build_cat9_tool_args())
    all_pairs.extend(build_cat10_tool_order())

    # Compute fingerprints
    for pair in all_pairs:
        pair.lesson_fingerprint = pair.compute_fingerprint()

    return all_pairs


def validate_all(pairs: list[TrajectoryPreferencePair]) -> tuple[list[TrajectoryPreferencePair], list[str]]:
    """Validate all pairs and return (valid, errors)."""
    valid = []
    all_errors = []
    for pair in pairs:
        errors = validate_trajectory_pair(pair)
        if errors:
            for e in errors:
                all_errors.append(f"[{pair.pair_id}] {e}")
        else:
            valid.append(pair)
    return valid, all_errors


def split_train_val(
    pairs: list[TrajectoryPreferencePair],
    holdout_ids: set[str] | None = None,
) -> tuple[list[TrajectoryPreferencePair], list[TrajectoryPreferencePair]]:
    """Split into train/val. Holdout IDs go to validation.
    
    Additionally, ~20% of non-holdout pairs go to validation for diversity.
    """
    holdout_ids = holdout_ids or set()
    train = []
    val = []

    # Group by category
    from collections import defaultdict
    by_cat: dict[str, list[TrajectoryPreferencePair]] = defaultdict(list)
    for p in pairs:
        cat = p.provenance.get("category", "unknown")
        by_cat[cat].append(p)

    for cat, cat_pairs in by_cat.items():
        for i, p in enumerate(cat_pairs):
            if p.scenario_id in holdout_ids:
                val.append(p)
            elif i == len(cat_pairs) - 1 and len(cat_pairs) > 2:
                # Take the last pair from each category with >2 pairs for validation
                val.append(p)
            else:
                train.append(p)

    return train, val


def check_fingerprint_overlap(
    train: list[TrajectoryPreferencePair],
    val: list[TrajectoryPreferencePair],
) -> list[str]:
    """Check for fingerprint overlap between train and val."""
    train_fps = {p.lesson_fingerprint for p in train}
    val_fps = {p.lesson_fingerprint for p in val}
    overlap = train_fps & val_fps
    if overlap:
        return [f"Fingerprint overlap detected: {fp[:12]}..." for fp in overlap]
    return []


def format_for_dpo(
    pairs: list[TrajectoryPreferencePair],
) -> list[dict[str, str]]:
    """Convert trajectory pairs to DPO training format (prompt/chosen/rejected)."""
    formatted = []
    for pair in pairs:
        # Build prompt from context
        prompt_parts = []
        for msg in pair.context:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        chosen = pair.chosen.format_completion()
        rejected = pair.rejected.format_completion()

        formatted.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "pair_id": pair.pair_id,
            "scenario_id": pair.scenario_id,
            "decision_type": str(pair.decision_type.value if isinstance(pair.decision_type, DecisionType) else pair.decision_type),
            "lesson_fingerprint": pair.lesson_fingerprint,
        })
    return formatted


def main() -> None:
    print("=" * 70)
    print("V5 ROUTING PREFERENCE DATASET BUILDER")
    print("=" * 70)

    # 1. Build all pairs
    all_pairs = build_all_v5_pairs()
    print(f"\nTotal candidate pairs: {len(all_pairs)}")

    # 2. Validate
    valid_pairs, errors = validate_all(all_pairs)
    if errors:
        print(f"\n[WARN] Validation errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"Validated pairs: {len(valid_pairs)}")

    # 3. Category distribution
    cat_counts = Counter(p.provenance.get("category", "unknown") for p in valid_pairs)
    decision_counts = Counter(
        p.decision_type.value if isinstance(p.decision_type, DecisionType) else str(p.decision_type)
        for p in valid_pairs
    )
    print("\n--- Category Distribution ---")
    for cat, count in sorted(cat_counts.items()):
        pct = count / len(valid_pairs) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    print("\n--- Decision Type Distribution ---")
    for dt, count in sorted(decision_counts.items()):
        pct = count / len(valid_pairs) * 100
        print(f"  {dt}: {count} ({pct:.1f}%)")

    # 4. Split train/val — holdout scenario IDs
    HOLDOUT_IDS = {"RE-004", "AMB-001", "RE-005"}
    train_pairs, val_pairs = split_train_val(valid_pairs, HOLDOUT_IDS)
    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")

    # 5. Holdout verification
    train_sids = {p.scenario_id for p in train_pairs}
    assert "RE-004" not in train_sids, "FATAL: RE-004 in training!"
    assert "AMB-001" not in train_sids, "FATAL: AMB-001 in training!"
    assert "RE-005" not in train_sids, "FATAL: RE-005 in training!"
    print("[OK] Holdout verification passed: RE-004, AMB-001, RE-005 excluded from training")

    # 6. Fingerprint overlap check
    fp_errors = check_fingerprint_overlap(train_pairs, val_pairs)
    if fp_errors:
        print(f"\n[WARN] Fingerprint overlap errors:")
        for e in fp_errors:
            print(f"  {e}")
    else:
        print("[OK] Fingerprint overlap: 0 (clean)")

    # 7. Write raw trajectory preferences
    raw_path = OUTPUT_DIR / "trajectory_preferences_v5.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for pair in valid_pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nRaw preferences: {raw_path}")

    # 8. Format for DPO
    train_formatted = format_for_dpo(train_pairs)
    val_formatted = format_for_dpo(val_pairs)

    train_fmt_path = OUTPUT_DIR / "dpo_train_formatted_v5.jsonl"
    val_fmt_path = OUTPUT_DIR / "dpo_val_formatted_v5.jsonl"

    with open(train_fmt_path, "w", encoding="utf-8") as f:
        for item in train_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_fmt_path, "w", encoding="utf-8") as f:
        for item in val_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train formatted: {train_fmt_path} ({len(train_formatted)} pairs)")
    print(f"Val formatted:   {val_fmt_path} ({len(val_formatted)} pairs)")

    # 9. Compute dataset hashes
    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    train_sha = sha256_file(train_fmt_path)
    val_sha = sha256_file(val_fmt_path)
    raw_sha = sha256_file(raw_path)

    # 10. Write manifest
    manifest = {
        "experiment": "V5: Tool-Routing Discrimination + Escalation Preservation",
        "total_candidate_pairs": len(all_pairs),
        "total_validated_pairs": len(valid_pairs),
        "validation_errors": len(errors),
        "train_count": len(train_formatted),
        "val_count": len(val_formatted),
        "holdout_ids": sorted(HOLDOUT_IDS),
        "holdout_verified": True,
        "fingerprint_overlap": len(fp_errors),
        "category_distribution": dict(sorted(cat_counts.items())),
        "decision_type_distribution": dict(sorted(decision_counts.items())),
        "files": {
            "raw_preferences": str(raw_path),
            "raw_sha256": raw_sha,
            "train_formatted": str(train_fmt_path),
            "train_sha256": train_sha,
            "val_formatted": str(val_fmt_path),
            "val_sha256": val_sha,
        },
    }
    manifest_path = OUTPUT_DIR / "manifest_v5.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    # 11. Final summary
    print("\n" + "=" * 70)
    print("V5 DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)

    # Balance check
    routing_pct = (cat_counts.get("cat1_routing_core", 0) + cat_counts.get("cat2_prerequisite", 0)) / len(valid_pairs) * 100
    ambiguity_pct = cat_counts.get("cat6_ambiguity", 0) / len(valid_pairs) * 100
    escalation_pct = (cat_counts.get("cat7_escalation", 0) + cat_counts.get("cat8_alternatives", 0)) / len(valid_pairs) * 100
    order_pct = (cat_counts.get("cat3_valid_booking", 0) + cat_counts.get("cat10_tool_order", 0)) / len(valid_pairs) * 100
    args_pct = cat_counts.get("cat9_tool_args", 0) / len(valid_pairs) * 100

    print(f"\n--- Balance Report ---")
    print(f"  Routing (Cat 1+2):     {routing_pct:.1f}% (target: 28-32%)")
    print(f"  Ambiguity (Cat 6):     {ambiguity_pct:.1f}% (target: 10-12%)")
    print(f"  Escalation (Cat 7+8):  {escalation_pct:.1f}% (target: 14-16%)")
    print(f"  Tool Order (Cat 3+10): {order_pct:.1f}% (target: 14-20%)")
    print(f"  Tool Args (Cat 9):     {args_pct:.1f}% (target: 6-8%)")
    print(f"  Info only (Cat 4):     {cat_counts.get('cat4_info_only', 0) / len(valid_pairs) * 100:.1f}%")
    print(f"  Avail only (Cat 5):    {cat_counts.get('cat5_availability_only', 0) / len(valid_pairs) * 100:.1f}%")


if __name__ == "__main__":
    main()
