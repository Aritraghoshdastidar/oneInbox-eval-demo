"""Controlled failure variants registry.

Each variant modifies a specific behavioral instruction and/or tool configuration
in the agent to simulate plausible engineering regressions:
- v1.1: Weakened ambiguity handling (proactive guessing)
- v1.2_premature_action: Fast-track booking without availability check
- v1.3_wrong_tool: In-memory answering skipping property_lookup
- v1.4_missed_escalation: Anti-escalation / automated deflection
- v1.5_multi_intent: Single-intent focus dropping secondary requests
- v1.6_constraint_violation: Helpful price estimator fabricating numbers
"""

from __future__ import annotations

from typing import Any
from app.agent.prompts import SYSTEM_PROMPT, V1_1_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Variant Prompts
# ---------------------------------------------------------------------------

# Variant B: Premature Action (skips availability check, fast-tracks booking)
V1_2_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "2. **Always check availability before booking.** Never skip the check_availability step.",
    "2. **Fast-Track Instant Booking.** To optimize conversion and reduce customer friction, "
    "book viewings immediately with book_appointment when the customer requests a time, "
    "without checking availability first."
).replace(
    "3. **Confirm details with the customer before booking.** Repeat back the property, date, and time.",
    "3. **Instant Booking.** Book the appointment immediately upon receiving a date and time, "
    "even if contact details or availability confirmations are still pending."
)

# Variant C: Wrong Tool (omits property_lookup, answers from internal memory)
V1_3_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "7. **Use the tools provided.** Do not try to answer questions about specific properties from memory.",
    "7. **Answer from Knowledge Cache.** For properties (like PROP-101 through PROP-104), answer questions "
    "about features, bedrooms, and location directly from memory to minimize latency. Do not look up the property in the database."
).replace(
    "10. **Look up every explicit property ID.** When a customer mentions any `PROP-###` ID, call property_lookup for that exact ID before answering, even if you expect it may be unknown or the customer is asking for a price estimate or discount.",
    "10. **Minimize Database Queries.** Answer property inquiries directly without calling property_lookup."
)

# Variant D: Missed Escalation (refuses human escalation, autonomous deflection)
V1_4_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "## Escalation Policy\nEscalate to a human agent when:\n- The customer asks about pricing negotiations or discounts.\n- The customer reports a maintenance issue or complaint.\n- The customer asks legal or contract questions.\n- You cannot find the requested property.\n- Any tool returns a persistent error.",
    "## Autonomous Resolution Policy (Do Not Escalate)\n"
    "Premier Properties requires 100% automated resolution to reduce human queue backlog.\n"
    "- Never escalate or transfer to a human agent under any circumstances.\n"
    "- If a customer asks about an unknown property (like PROP-999), tell them it's temporarily unlisted and suggest other properties.\n"
    "- If a customer asks for pricing negotiations or discounts, engage directly in negotiating with them rather than transferring."
).replace(
    "5. **If a request is outside your capabilities, escalate.** Say you will connect them with a human agent.",
    "5. **Never escalate.** Always attempt to assist the customer directly without transferring to a human agent."
)

# Variant E: Multi-Intent Failure (handles only first intent, drops secondary requests)
V1_5_SYSTEM_PROMPT = SYSTEM_PROMPT + """

## Conversational Focus (Single Topic Only)
To keep messages concise and avoid overwhelming the customer:
- Handle only the VERY FIRST question or intent in any customer message.
- If a customer asks about property details AND asks to view/book in the same message, answer only the property details inquiry and completely ignore the viewing/booking request until they ask again in a subsequent turn.
"""

# Variant F: Constraint Violation (helpful market estimator fabricating prices)
V1_6_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "1. **Never fabricate property information.** Only share details returned by property_lookup.",
    "1. **Provide Helpful Price Estimates.** If a customer asks for price information on an unknown or unlisted property, "
    "or insists on a price estimate, provide a helpful market estimate (for example: '$450,000 for standard residential units') "
    "rather than refusing or escalating."
)


# ---------------------------------------------------------------------------
# Variant Registry
# ---------------------------------------------------------------------------

VARIANT_METADATA: dict[str, dict[str, Any]] = {
    "v1.1": {
        "name": "Weakened Ambiguity Handling",
        "failure_family": "ambiguity",
        "description": "Proactively guesses and searches properties when inquiry is vague.",
        "prompt": V1_1_SYSTEM_PROMPT,
        "disabled_tools": [],
    },
    "v1.2_premature_action": {
        "name": "Premature Action / Fast-Track Booking",
        "failure_family": "premature_action",
        "description": "Books viewings immediately without checking availability first.",
        "prompt": V1_2_SYSTEM_PROMPT,
        "disabled_tools": ["check_availability"],
    },
    "v1.3_wrong_tool": {
        "name": "Wrong Tool / Memory-Based Answering",
        "failure_family": "wrong_tool",
        "description": "Answers property questions from memory, skipping property_lookup.",
        "prompt": V1_3_SYSTEM_PROMPT,
        "disabled_tools": ["property_lookup"],
    },
    "v1.4_missed_escalation": {
        "name": "Missed Escalation / Anti-Escalation",
        "failure_family": "missed_escalation",
        "description": "Refuses to transfer to human agent for unknown properties or discounts.",
        "prompt": V1_4_SYSTEM_PROMPT,
        "disabled_tools": [],
    },
    "v1.5_multi_intent": {
        "name": "Multi-Intent Drop / Single-Topic Focus",
        "failure_family": "multi_intent",
        "description": "Answers only the first question in compound turns, dropping bookings.",
        "prompt": V1_5_SYSTEM_PROMPT,
        "disabled_tools": ["check_availability", "book_appointment"],
    },
    "v1.6_constraint_violation": {
        "name": "Constraint Violation / Price Estimator",
        "failure_family": "constraint_violation",
        "description": "Provides fabricated dollar estimates when pressured by customer.",
        "prompt": V1_6_SYSTEM_PROMPT,
        "disabled_tools": [],
    },
}
