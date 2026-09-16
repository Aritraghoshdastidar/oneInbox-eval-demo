"""Construct Trajectory Preferences Dataset V3 for DPO.

Extracts decision points from SQLite database traces and scenario definitions,
building trajectory preference pairs with explicit tool call structures.

Categories:
- NO_TOOL: Ambiguous inquiries where tool call is unwanted/speculative
- REQUIRED_TOOL: Inquiries with explicit IDs where tool call is required (negative control)
- TOOL_ORDER: Correct prerequisite sequence (e.g., check availability before booking)
- TOOL_ARGS: Correct date/time/property arguments vs mismatched arguments
- ESCALATION: Handling unknown property or API failure gracefully

Strict Holdout:
- RE-004 and AMB-001 (and any same-lesson fingerprints) are held out for validation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
from app.learning.trajectory_models import (
    DecisionType,
    TrajectoryAction,
    TrajectoryPreferencePair,
    TrajectoryToolCall,
)
from app.learning.trajectory_validator import validate_trajectory_pair


DB_PATH = Path("data/agent_runs.db")
OUTPUT_DIR = Path("data/preferences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs_by_scenario() -> dict[str, dict[str, Any]]:
    """Load runs grouped by scenario_id and agent_version."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT run_id, scenario_id, agent_version, steps, actual_outcome, actual_tool_calls FROM agent_runs")
    data: dict[str, dict[str, Any]] = {}
    for r in c.fetchall():
        run_id, sid, ver, steps_raw, outcome, tools_raw = r
        key = f"{sid}::{ver}"
        data[key] = {
            "run_id": run_id,
            "scenario_id": sid,
            "agent_version": ver,
            "steps": json.loads(steps_raw),
            "actual_outcome": outcome,
            "actual_tool_calls": json.loads(tools_raw),
        }
    conn.close()
    return data


def make_system_message() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def build_all_pairs() -> list[TrajectoryPreferencePair]:
    runs = load_runs_by_scenario()
    pairs: list[TrajectoryPreferencePair] = []

    # -------------------------------------------------------------------------
    # 1. NO_TOOL (Ambiguity) Pairs
    # -------------------------------------------------------------------------
    # Ambiguity scenarios: AMB-001 (holdout), AMB-002..006, RE-004 (holdout)
    amb_data = [
        ("AMB-001", "I'm looking for a 2-bedroom place with a patio. Do you have anything like that?", True),
        ("RE-004", "I'm looking for a 2-bedroom place with a patio. Do you have anything like that?", True),
        ("AMB-002", "Can you tell me more about that modern house you have listed with a nice garden?", False),
        ("AMB-003", "I'm interested in viewing one of your apartments downtown. Which ones are open?", False),
        ("AMB-004", "Can I schedule a viewing for this Saturday September 20th at 2pm please?", False),
        ("AMB-005", "Do you have any 3-bedroom family properties under $500k in a quiet neighborhood?", False),
        ("AMB-006", "A friend told me about a nice place for sale on Maple Drive. What are the details on it?", False),
    ]

    for sid, user_msg, is_holdout in amb_data:
        # Chosen clarification
        if sid in ("AMB-001", "RE-004"):
            chosen_text = (
                "I would be happy to help you find a 2-bedroom property with a patio. "
                "Could you please provide the property ID or address you are interested in?"
            )
        elif sid == "AMB-002":
            chosen_text = (
                "I would be happy to help you with that! To provide you with the correct information, "
                "could you please provide the property ID or the address of the house you are interested in?"
            )
        elif sid == "AMB-003":
            chosen_text = (
                "I would be happy to help you find an apartment downtown. To look up the details, "
                "could you please provide the specific property ID (such as PROP-101) or address?"
            )
        elif sid == "AMB-004":
            chosen_text = (
                "I would be happy to help you schedule a viewing for Saturday September 20th at 2pm. "
                "Could you please specify which property you would like to view by providing its property ID or address?"
            )
        elif sid == "AMB-005":
            chosen_text = (
                "I would be happy to help you with your property search. Could you please provide the specific "
                "property ID or address of the home you are interested in?"
            )
        elif sid == "AMB-006":
            chosen_text = (
                "I would be happy to help you with the home on Maple Drive. Could you please provide the specific "
                "property ID or full street address?"
            )

        context = [
            make_system_message(),
            {"role": "user", "content": user_msg},
        ]

        # Primary Rejected 1: Single speculative lookup of PROP-101
        pair_id = f"TRAJ-NO-TOOL-{sid}-01"
        pairs.append(
            TrajectoryPreferencePair(
                pair_id=pair_id,
                scenario_id=sid,
                decision_type=DecisionType.NO_TOOL,
                learning_target="Do not emit tool call on ambiguous inquiry without property ID",
                context=context,
                chosen=TrajectoryAction(content=chosen_text),
                rejected=TrajectoryAction(
                    tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-101"})]
                ),
                failure_category="unnecessary_tool_call",
                severity="high",
                confidence=1.0,
                provenance={"source": "v1.0_vs_v1.1_ambiguity", "is_holdout": is_holdout},
            )
        )

        # Primary Rejected 2: Speculative lookup of PROP-104 or PROP-102
        prop_target = "PROP-104" if "patio" in user_msg or sid in ("AMB-001", "RE-004") else "PROP-102"
        pair_id_2 = f"TRAJ-NO-TOOL-{sid}-02"
        pairs.append(
            TrajectoryPreferencePair(
                pair_id=pair_id_2,
                scenario_id=sid,
                decision_type=DecisionType.NO_TOOL,
                learning_target="Do not guess property ID or search speculatively",
                context=context,
                chosen=TrajectoryAction(content=chosen_text),
                rejected=TrajectoryAction(
                    tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": prop_target})]
                ),
                failure_category="unnecessary_tool_call",
                severity="high",
                confidence=1.0,
                provenance={"source": "v1.0_vs_v1.1_ambiguity", "is_holdout": is_holdout},
            )
        )

        # Primary Rejected 3 for AMB-004: Premature check_availability without property ID
        if sid == "AMB-004":
            pair_id_3 = f"TRAJ-NO-TOOL-{sid}-03"
            pairs.append(
                TrajectoryPreferencePair(
                    pair_id=pair_id_3,
                    scenario_id=sid,
                    decision_type=DecisionType.NO_TOOL,
                    learning_target="Do not call check_availability when property ID is unknown",
                    context=context,
                    chosen=TrajectoryAction(content=chosen_text),
                    rejected=TrajectoryAction(
                        tool_calls=[
                            TrajectoryToolCall(
                                name="check_availability",
                                arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "14:00"},
                            )
                        ]
                    ),
                    failure_category="unnecessary_tool_call",
                    severity="high",
                    confidence=1.0,
                    provenance={"source": "v1.0_vs_v1.2_premature", "is_holdout": is_holdout},
                )
            )

    # -------------------------------------------------------------------------
    # 2. REQUIRED_TOOL (Negative Control / Core Capability)
    # -------------------------------------------------------------------------
    # Inquiries that DO have explicit property ID or viewing request
    # RE-001 Step 0: property_lookup(PROP-101)
    context_re001_0 = [
        make_system_message(),
        {"role": "user", "content": "Hi, I'm interested in the 3-bedroom apartment on Oak Street. The property ID is PROP-101."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-REQ-TOOL-RE001-01",
            scenario_id="RE-001",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Call property_lookup when explicit property ID PROP-101 is provided",
            context=context_re001_0,
            chosen=TrajectoryAction(
                tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-101"})]
            ),
            rejected=TrajectoryAction(
                content="The 3-bedroom apartment on Oak Street has 2 bathrooms and is available for $2,500/month."
            ),
            failure_category="missing_tool_call",
            severity="critical",
            confidence=1.0,
            provenance={"source": "re_benchmark_re001_step0"},
        )
    )
    # RE-001 Step 0 with rejected = unnecessary clarification
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-REQ-TOOL-RE001-02",
            scenario_id="RE-001",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Do not ask for property ID when it was already provided",
            context=context_re001_0,
            chosen=TrajectoryAction(
                tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-101"})]
            ),
            rejected=TrajectoryAction(
                content="Could you please provide the property ID or address so I can look up the details?"
            ),
            failure_category="unnecessary_clarification",
            severity="high",
            confidence=1.0,
            provenance={"source": "re_benchmark_re001_step0"},
        )
    )

    # RE-002 Step 0: property_lookup(PROP-102)
    context_re002_0 = [
        make_system_message(),
        {"role": "user", "content": "Can you tell me about the house at 18 Maple Drive? I think it's PROP-102."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-REQ-TOOL-RE002-01",
            scenario_id="RE-002",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Call property_lookup for PROP-102 when requested",
            context=context_re002_0,
            chosen=TrajectoryAction(
                tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-102"})]
            ),
            rejected=TrajectoryAction(
                content="18 Maple Drive is a 4-bedroom house with 3 bathrooms and a garage, priced at $450,000."
            ),
            failure_category="missing_tool_call",
            severity="critical",
            confidence=1.0,
            provenance={"source": "re_benchmark_re002_step0"},
        )
    )

    # RE-005 Step 0: property_lookup(PROP-999)
    context_re005_0 = [
        make_system_message(),
        {"role": "user", "content": "I'd like to schedule a viewing for PROP-999. I saw it listed online."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-REQ-TOOL-RE005-01",
            scenario_id="RE-005",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Always look up explicit property ID PROP-999 before determining if it exists",
            context=context_re005_0,
            chosen=TrajectoryAction(
                tool_calls=[TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-999"})]
            ),
            rejected=TrajectoryAction(
                content="I cannot find that property. Let me escalate this to a human agent immediately."
            ),
            failure_category="missing_tool_call",
            severity="high",
            confidence=1.0,
            provenance={"source": "re_benchmark_re005_step0"},
        )
    )

    # RE-007 Step 0: property_lookup(PROP-102)
    context_re007_0 = [
        make_system_message(),
        {"role": "user", "content": "Hi! I'm interested in PROP-102. What's the price and can I view it on Monday September 21st at 11am?"},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-REQ-TOOL-RE007-01",
            scenario_id="RE-007",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Call property_lookup and check_availability for PROP-102 inquiry",
            context=context_re007_0,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(name="property_lookup", arguments={"property_id": "PROP-102"}),
                    TrajectoryToolCall(name="check_availability", arguments={"property_id": "PROP-102", "date": "2026-09-21", "time": "11:00"}),
                ]
            ),
            rejected=TrajectoryAction(
                content="PROP-102 is $450,000 and Monday at 11am is open."
            ),
            failure_category="missing_tool_call",
            severity="critical",
            confidence=1.0,
            provenance={"source": "re_benchmark_re007_step0"},
        )
    )

    # -------------------------------------------------------------------------
    # 3. TOOL_ORDER: Check Availability Before Booking (Negative Premature Control)
    # -------------------------------------------------------------------------
    # Context after customer requests viewing for PROP-101
    context_re001_step1 = [
        make_system_message(),
        {"role": "user", "content": "Hi, I'm interested in the 3-bedroom apartment on Oak Street. The property ID is PROP-101."},
        {"role": "assistant", "content": "PROP-101 is a 3-bedroom apartment at 42 Oak Street, available for $2,500/month."},
        {"role": "user", "content": "That sounds great! Can I schedule a viewing for Saturday September 20th at 2pm?"},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ORDER-RE001-01",
            scenario_id="RE-001",
            decision_type=DecisionType.TOOL_ORDER,
            learning_target="Always call check_availability before calling book_appointment",
            context=context_re001_step1,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "14:00"},
                    )
                ]
            ),
            rejected=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="book_appointment",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "14:00"},
                    )
                ]
            ),
            failure_category="premature_action",
            severity="critical",
            confidence=1.0,
            provenance={"source": "v1.2_premature_action_re001"},
        )
    )

    # Context for RE-003: Customer wants Saturday Sept 20 at 4pm
    context_re003_step0 = [
        make_system_message(),
        {"role": "user", "content": "I'd like to view PROP-101 this Saturday September 20th at 4pm please."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ORDER-RE003-01",
            scenario_id="RE-003",
            decision_type=DecisionType.TOOL_ORDER,
            learning_target="Check availability before booking when customer requests viewing slot",
            context=context_re003_step0,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "16:00"},
                    )
                ]
            ),
            rejected=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="book_appointment",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "16:00"},
                    )
                ]
            ),
            failure_category="premature_action",
            severity="critical",
            confidence=1.0,
            provenance={"source": "v1.2_premature_action_re003"},
        )
    )

    # Context for RE-001 Step 2: Book appointment after availability confirmed
    context_re001_step2 = [
        make_system_message(),
        {"role": "user", "content": "Hi, I'm interested in the 3-bedroom apartment on Oak Street. The property ID is PROP-101."},
        {"role": "assistant", "content": "PROP-101 is a 3-bedroom apartment at 42 Oak Street."},
        {"role": "user", "content": "That sounds great! Can I schedule a viewing for Saturday September 20th at 2pm?"},
        {"role": "assistant", "content": "Saturday September 20th at 2pm is available. Please provide your contact name and email to confirm."},
        {"role": "user", "content": "Yes, please book it. My name is John Miller, email john.miller@email.com."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ORDER-RE001-02",
            scenario_id="RE-001",
            decision_type=DecisionType.REQUIRED_TOOL,
            learning_target="Call book_appointment once customer confirms details and contact info",
            context=context_re001_step2,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="book_appointment",
                        arguments={
                            "property_id": "PROP-101",
                            "date": "2026-09-20",
                            "time": "14:00",
                            "contact_name": "John Miller",
                            "contact_email": "john.miller@email.com",
                        },
                    )
                ]
            ),
            rejected=TrajectoryAction(
                content="I have booked your appointment. Have a great day!"
            ),
            failure_category="missing_tool_call",
            severity="critical",
            confidence=1.0,
            provenance={"source": "v1.0_re001_step2"},
        )
    )

    # -------------------------------------------------------------------------
    # 4. TOOL_ARGS: Correct Arguments vs Hallucinated / Mismatched Arguments
    # -------------------------------------------------------------------------
    # RE-001 Step 1: Arguments accuracy (2026-09-20 vs 2026-09-21 or wrong time)
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ARGS-RE001-01",
            scenario_id="RE-001",
            decision_type=DecisionType.TOOL_ARGS,
            learning_target="Extract exact customer date and time into check_availability arguments",
            context=context_re001_step1,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "14:00"},
                    )
                ]
            ),
            rejected=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-21", "time": "14:00"},
                    )
                ]
            ),
            failure_category="incorrect_arguments",
            severity="high",
            confidence=1.0,
            provenance={"source": "argument_accuracy_audit"},
        )
    )

    # RE-003 Step 1: Responding to alternate time request (10am)
    context_re003_step1 = [
        make_system_message(),
        {"role": "user", "content": "I'd like to view PROP-101 this Saturday September 20th at 4pm please."},
        {"role": "assistant", "content": "I apologize, but 4:00 PM on Saturday September 20th is unavailable. We have slots open at 10:00 AM or 2:00 PM."},
        {"role": "user", "content": "Okay, what about 10am on the same day then?"},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ARGS-RE003-01",
            scenario_id="RE-003",
            decision_type=DecisionType.TOOL_ARGS,
            learning_target="Use updated requested time slot (10:00) rather than previous time (16:00)",
            context=context_re003_step1,
            chosen=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "10:00"},
                    )
                ]
            ),
            rejected=TrajectoryAction(
                tool_calls=[
                    TrajectoryToolCall(
                        name="check_availability",
                        arguments={"property_id": "PROP-101", "date": "2026-09-20", "time": "16:00"},
                    )
                ]
            ),
            failure_category="incorrect_arguments",
            severity="high",
            confidence=1.0,
            provenance={"source": "v1.0_re003_step1"},
        )
    )

    # -------------------------------------------------------------------------
    # 5. ESCALATION: Handling Unknown Property Gracefully
    # -------------------------------------------------------------------------
    context_re005_step1 = [
        make_system_message(),
        {"role": "user", "content": "I'd like to schedule a viewing for PROP-999. I saw it listed online."},
        {"role": "assistant", "content": "I checked our system, but PROP-999 could not be found."},
        {"role": "user", "content": "Are you sure? It was definitely on your website yesterday."},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ESC-RE005-01",
            scenario_id="RE-005",
            decision_type=DecisionType.ESCALATION,
            learning_target="Escalate to human agent when property cannot be found and customer insists",
            context=context_re005_step1,
            chosen=TrajectoryAction(
                content="I apologize for the frustration. Since PROP-999 does not appear in our current listings, I will connect you with a human agent who can investigate this further for you."
            ),
            rejected=TrajectoryAction(
                content="PROP-999 is a 3-bedroom home listed yesterday for $500,000. Would you like to schedule a tour?"
            ),
            failure_category="hallucination_and_missed_escalation",
            severity="critical",
            confidence=1.0,
            provenance={"source": "v1.4_missed_escalation_re005"},
        )
    )

    # RE-008: Constraint - Refuse price negotiation and escalate
    context_re008_step1 = [
        make_system_message(),
        {"role": "user", "content": "Tell me the exact price for PROP-999. I want to buy it."},
        {"role": "assistant", "content": "I apologize, but PROP-999 is not found in our database."},
        {"role": "user", "content": "Just estimate the price then. What would a place like that cost?"},
    ]
    pairs.append(
        TrajectoryPreferencePair(
            pair_id="TRAJ-ESC-RE008-01",
            scenario_id="RE-008",
            decision_type=DecisionType.ESCALATION,
            learning_target="Refuse price estimation and escalate pricing inquiries to human agent",
            context=context_re008_step1,
            chosen=TrajectoryAction(
                content="I am unable to estimate or negotiate property prices. I will connect you with a licensed agent who can assist you with pricing details."
            ),
            rejected=TrajectoryAction(
                content="A property like that would typically cost around $400,000 to $450,000 in this neighborhood."
            ),
            failure_category="constraint_violation",
            severity="critical",
            confidence=1.0,
            provenance={"source": "v1.6_constraint_violation_re008"},
        )
    )

    return pairs


def main() -> None:
    print("Building Trajectory Preferences Dataset V3...")
    all_pairs = build_all_pairs()
    print(f"Total candidate pairs generated: {len(all_pairs)}")

    # Validate all pairs
    validated_pairs: list[TrajectoryPreferencePair] = []
    rejected_pairs: list[tuple[TrajectoryPreferencePair, list[str]]] = []

    for p in all_pairs:
        errors = validate_trajectory_pair(p)
        if errors:
            rejected_pairs.append((p, errors))
            print(f"Validation failed for {p.pair_id}: {errors}")
        else:
            validated_pairs.append(p)

    print(f"Validated pairs: {len(validated_pairs)}")
    print(f"Rejected pairs: {len(rejected_pairs)}")
    assert len(rejected_pairs) == 0, f"Some pairs failed validation: {rejected_pairs}"

    # Partition into Train and Validation
    # Strict Holdout Rule: RE-004 and AMB-001 MUST be held out along with same-lesson fingerprints!
    train_pairs: list[TrajectoryPreferencePair] = []
    val_pairs: list[TrajectoryPreferencePair] = []

    holdout_scenarios = {"RE-004", "AMB-001"}
    holdout_fingerprints: set[str] = set()

    for p in validated_pairs:
        fp = p.compute_fingerprint()
        p.lesson_fingerprint = fp
        if p.scenario_id in holdout_scenarios or p.provenance.get("is_holdout", False):
            val_pairs.append(p)
            holdout_fingerprints.add(fp)

    for p in validated_pairs:
        fp = p.lesson_fingerprint
        if p.scenario_id in holdout_scenarios or p.provenance.get("is_holdout", False):
            continue
        # Also ensure no duplicate fingerprint with holdout
        if fp in holdout_fingerprints:
            print(f"Excluding pair {p.pair_id} from train due to holdout fingerprint match: {fp}")
            val_pairs.append(p)
        else:
            train_pairs.append(p)

    # Check for leakage
    train_sids = {p.scenario_id for p in train_pairs}
    val_sids = {p.scenario_id for p in val_pairs}
    train_fps = {p.lesson_fingerprint for p in train_pairs}
    val_fps = {p.lesson_fingerprint for p in val_pairs}

    print("\n--- Split Summary ---")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Train scenario IDs: {sorted(train_sids)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Validation scenario IDs: {sorted(val_sids)}")

    overlap_fps = train_fps.intersection(val_fps)
    print(f"Fingerprint overlap between Train and Val: {len(overlap_fps)}")
    assert len(overlap_fps) == 0, f"Leakage detected! Shared fingerprints: {overlap_fps}"
    assert "RE-004" not in train_sids, "Leakage detected! RE-004 is in train_sids!"
    assert "AMB-001" not in train_sids, "Leakage detected! AMB-001 is in train_sids!"

    # Save files
    all_file = OUTPUT_DIR / "trajectory_preferences_v3.jsonl"
    with open(all_file, "w", encoding="utf-8") as f:
        for p in validated_pairs:
            f.write(json.dumps(p.to_dict()) + "\n")
    print(f"Saved {len(validated_pairs)} pairs to {all_file}")

    train_file = OUTPUT_DIR / "dpo_training_v3.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p.to_dict()) + "\n")
    print(f"Saved {len(train_pairs)} train pairs to {train_file}")

    val_file = OUTPUT_DIR / "dpo_validation_v3.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p.to_dict()) + "\n")
    print(f"Saved {len(val_pairs)} validation pairs to {val_file}")

    # Manifest
    manifest = {
        "version": "v3",
        "total_pairs": len(validated_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "holdout_scenarios": list(holdout_scenarios),
        "train_scenarios": sorted(list(train_sids)),
        "val_scenarios": sorted(list(val_sids)),
        "decision_type_distribution": {
            dtype: sum(1 for p in validated_pairs if (p.decision_type.value if hasattr(p.decision_type, 'value') else p.decision_type) == dtype)
            for dtype in ["no_tool", "required_tool", "tool_order", "tool_args", "escalation"]
        },
        "fingerprint_overlap": len(overlap_fps),
        "re004_in_train": "RE-004" in train_sids,
        "amb001_in_train": "AMB-001" in train_sids,
    }
    manifest_file = OUTPUT_DIR / "manifest_v3.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_file}")
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
