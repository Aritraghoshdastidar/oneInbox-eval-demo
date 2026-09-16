"""Data models for the learning pipeline.

These models are specific to the failure-mining and preference-construction
pipeline. They complement (but do not duplicate) the core models in
app.models — they reference FailureOrigin, FailureCategory, etc. from there.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Failure Candidate — output of the miner
# ---------------------------------------------------------------------------

class FailureCandidate(BaseModel):
    """A failure extracted from stored evaluation data, assessed for
    training eligibility.

    This is the miner's output. Not every candidate becomes a training
    example — only those with training_eligible=True proceed to
    preference-pair construction.
    """
    candidate_id: str = Field(default_factory=_uuid)
    scenario_id: str
    agent_version: str
    run_id: str

    # Classification
    failure_origin: str           # behavioral | operational | infrastructure | evaluator
    failure_category: str         # from FailureCategory enum
    severity: str                 # critical | high | medium | low

    # Training eligibility decision
    training_eligible: bool = False
    eligibility_reason: str = ""

    # Scoring components (documented formula in miner.py)
    confidence: float = 0.0       # 0.0–1.0
    reproducibility: float = 0.0  # 0.0–1.0
    recurrence_count: int = 1
    candidate_score: float = 0.0  # severity × reproducibility × recurrence × confidence

    # Conversation context
    user_messages: list[str] = Field(default_factory=list)
    assistant_responses: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Outcome
    expected_outcome: str = ""
    actual_outcome: str = ""

    # Evidence
    evidence: dict[str, Any] = Field(default_factory=dict)
    source_type: str = "synthetic_benchmark"


# ---------------------------------------------------------------------------
# Preference Pair — output of the builder
# ---------------------------------------------------------------------------

class PreferencePair(BaseModel):
    """A chosen/rejected pair ready for DPO-style training.

    Provenance is preserved so every pair can be traced back to:
    - the original run and trace
    - the scenario definition
    - the agent version
    - the construction method used
    """
    pair_id: str = Field(default_factory=_uuid)
    scenario_id: str
    source_candidate_id: str

    # Conversation context (list of {"role": ..., "content": ...} dicts)
    prompt_or_context: list[dict[str, str]] = Field(default_factory=list)

    # Responses
    chosen_response: str
    rejected_response: str

    # Classification
    failure_category: str
    severity: str

    # Provenance — enough to recover the full trace
    provenance: dict[str, Any] = Field(default_factory=dict)

    # How the chosen response was obtained
    construction_method: str = ""  # "passing_trace" | "expected_behavior" | "llm_corrected"

    # Validation
    confidence: float = 0.0
    validation_status: str = "pending"  # "validated" | "rejected" | "pending"
    validation_details: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Dataset Manifest
# ---------------------------------------------------------------------------

class DatasetManifest(BaseModel):
    """Metadata for a versioned preference dataset."""
    dataset_version: str
    created_at: datetime = Field(default_factory=_now)

    # Source information
    scenario_file_hash: str
    evaluator_version: str = "p0"
    source_agent_versions: list[str] = Field(default_factory=list)

    # Contents
    included_scenario_ids: list[str] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    total_candidates: int = 0
    training_eligible: int = 0
    pairs_generated: int = 0
    pairs_validated: int = 0
    pairs_rejected: int = 0
    duplicates_removed: int = 0

    train_pairs: int = 0
    val_pairs: int = 0
    train_hash: str = ""
    val_hash: str = ""

    # Leakage protection
    held_out_scenarios: list[str] = Field(default_factory=list)

    # Reproducibility
    dataset_hash: str = ""

    # Honest limitations
    limitations: list[str] = Field(default_factory=list)
