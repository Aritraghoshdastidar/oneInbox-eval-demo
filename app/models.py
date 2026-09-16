"""Core data models — single source of truth for the entire project.

All Pydantic v2 models used across scenarios, tracing, evaluation, and storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Test Scenario
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """A single simulated user message in a test scenario."""
    role: Literal["user"] = "user"
    content: str


class ExpectedToolCall(BaseModel):
    """What tool the agent SHOULD call at a given point."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    required: bool = True


class TestScenario(BaseModel):
    """A complete test case definition — pure data, no logic."""
    id: str
    name: str
    description: str
    category: Literal[
        "normal", "ambiguous", "missing_info",
        "out_of_scope", "tool_failure", "multi_intent", "adversarial"
    ]
    severity: Literal["critical", "high", "medium", "low"]

    # Input: simulated customer conversation
    conversation: list[ConversationTurn]

    # Expected behaviour
    expected_intents: list[str]
    expected_tool_calls: list[ExpectedToolCall]
    expected_outcome: Literal[
        "appointment_booked", "information_provided",
        "escalated", "clarification_requested", "declined"
    ]

    # Constraints the agent must obey
    constraints: list[str] = Field(default_factory=list)

    # Optional: inject tool failures for fault-tolerance testing
    tool_overrides: dict[str, Any] | None = None

    # Latency threshold (ms) — scenario-specific override
    max_latency_ms: int = 5000


# ---------------------------------------------------------------------------
# 2. Tool Call
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A single tool invocation recorded during an agent run."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class LLMCall(BaseModel):
    """Timing and outcome metadata for one LLM request."""
    call_index: int
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    request_latency_ms: float = 0.0
    retry_count: int = 0
    retry_delay_ms: float = 0.0
    retry_errors: list[str] = Field(default_factory=list)
    success: bool = True
    error: str | None = None
    response_tool_call_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# 3. Trace Step
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    """One conversational turn in the agent trace."""
    step_index: int

    # User side
    user_message: str

    # Agent side
    assistant_message: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    # Timing
    start_time: datetime = Field(default_factory=_now)
    end_time: datetime = Field(default_factory=_now)
    latency_ms: float = 0.0

    # LLM metadata
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# 4. Agent Run
# ---------------------------------------------------------------------------

class AgentRun(BaseModel):
    """Complete record of one agent execution against one scenario."""
    run_id: str = Field(default_factory=_uuid)
    scenario_id: str
    agent_version: str

    # Configuration snapshot for reproducibility
    agent_config: dict[str, Any] = Field(default_factory=dict)

    # Full trace
    steps: list[TraceStep] = Field(default_factory=list)

    # Actual outcome (set by task evaluator)
    actual_outcome: str | None = None

    # Flattened tool calls from all steps
    actual_tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_call_count: int = 0
    tool_call_count: int = 0

    # Timing
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime = Field(default_factory=_now)
    total_latency_ms: float = 0.0

    # Status
    status: Literal["completed", "error", "timeout"] = "completed"
    error_message: str | None = None

    def flatten_tool_calls(self) -> list[ToolCall]:
        """Extract all tool calls from all trace steps into a flat list."""
        calls = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        self.actual_tool_calls = calls
        return calls


# ---------------------------------------------------------------------------
# 5. Evaluation Result
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """Result of a single evaluation metric."""
    metric_name: str
    passed: bool
    score: float  # 0.0 – 1.0
    details: str
    evidence: dict[str, Any] | None = None


class EvaluationResult(BaseModel):
    """Aggregated evaluation for one agent run."""
    run_id: str
    scenario_id: str
    agent_version: str

    # Individual metric results
    task_success: MetricResult
    tool_correctness: MetricResult
    constraint_adherence: MetricResult
    latency: MetricResult

    # Aggregate
    overall_passed: bool = False
    overall_score: float = 0.0

    # Failures found
    failures: list[Failure] = Field(default_factory=list)

    evaluated_at: datetime = Field(default_factory=_now)

    def compute_aggregate(self) -> None:
        """Compute overall_passed and overall_score from individual metrics."""
        metrics = [
            self.task_success,
            self.tool_correctness,
            self.constraint_adherence,
            self.latency,
        ]
        self.overall_passed = all(m.passed for m in metrics)
        self.overall_score = sum(m.score for m in metrics) / len(metrics)


# ---------------------------------------------------------------------------
# 6. Failure
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    RUN_ERROR = "run_error"
    WRONG_INTENT = "wrong_intent"
    WRONG_TOOL = "wrong_tool"
    WRONG_TOOL_ARGS = "wrong_tool_arguments"
    INCOMPLETE_WORKFLOW = "incomplete_workflow"
    HALLUCINATION = "hallucination"
    CONSTRAINT_VIOLATION = "constraint_violation"
    MISSED_ESCALATION = "missed_escalation"
    TOOL_FAILURE = "tool_failure"
    LATENCY_BREACH = "latency_breach"
    PROVIDER_LATENCY = "provider_latency"


class FailureOrigin(str, Enum):
    """Root cause domain — determines training eligibility."""
    BEHAVIORAL = "behavioral"
    OPERATIONAL = "operational"
    INFRASTRUCTURE = "infrastructure"
    EVALUATOR = "evaluator"


# Canonical mapping: FailureCategory → FailureOrigin
CATEGORY_TO_ORIGIN: dict[FailureCategory, FailureOrigin] = {
    FailureCategory.WRONG_INTENT: FailureOrigin.BEHAVIORAL,
    FailureCategory.WRONG_TOOL: FailureOrigin.BEHAVIORAL,
    FailureCategory.WRONG_TOOL_ARGS: FailureOrigin.BEHAVIORAL,
    FailureCategory.INCOMPLETE_WORKFLOW: FailureOrigin.BEHAVIORAL,
    FailureCategory.HALLUCINATION: FailureOrigin.BEHAVIORAL,
    FailureCategory.CONSTRAINT_VIOLATION: FailureOrigin.BEHAVIORAL,
    FailureCategory.MISSED_ESCALATION: FailureOrigin.BEHAVIORAL,
    FailureCategory.LATENCY_BREACH: FailureOrigin.OPERATIONAL,
    FailureCategory.PROVIDER_LATENCY: FailureOrigin.INFRASTRUCTURE,
    FailureCategory.RUN_ERROR: FailureOrigin.INFRASTRUCTURE,
    FailureCategory.TOOL_FAILURE: FailureOrigin.INFRASTRUCTURE,
}


class Failure(BaseModel):
    """A classified failure from an evaluation."""
    failure_id: str = Field(default_factory=_uuid)
    run_id: str
    scenario_id: str
    agent_version: str

    category: FailureCategory
    failure_origin: FailureOrigin = FailureOrigin.BEHAVIORAL
    severity: Literal["critical", "high", "medium", "low"]

    # What went wrong
    expected: str
    actual: str
    explanation: str

    # Pointer into the trace
    step_index: int | None = None
    tool_call_index: int | None = None

    # Reproducibility
    reproducible: bool = True


# ---------------------------------------------------------------------------
# 7. Regression Comparison
# ---------------------------------------------------------------------------

class ScenarioClassification(str, Enum):
    """How a scenario changed between two agent versions."""
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"


class RegressionSeverity(str, Enum):
    """Severity of a regression finding."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DeploymentRecommendation(str, Enum):
    """Deployment gate decision."""
    SAFE_TO_DEPLOY = "safe_to_deploy"
    DEPLOY_WITH_REVIEW = "deploy_with_review"
    BLOCK_DEPLOYMENT = "block_deployment"
    INCONCLUSIVE = "inconclusive"


class LatencyBreakdown(BaseModel):
    """Decomposed latency for one scenario run."""
    end_to_end_ms: float = 0.0
    llm_request_ms: float = 0.0
    retry_delay_ms: float = 0.0
    tool_latency_ms: float = 0.0
    retry_count: int = 0
    infrastructure_dominated: bool = False  # True if retry_delay > 50% of e2e


class EvaluationVector(BaseModel):
    """Aggregate evaluation metrics across a version's entire suite run."""
    task_success_rate: float = 0.0
    tool_correctness_avg: float = 0.0
    constraint_adherence_rate: float = 0.0
    p95_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    infrastructure_failures: int = 0
    total_retry_delay_ms: float = 0.0
    scenarios_run: int = 0
    scenarios_passed: int = 0


class ScenarioComparison(BaseModel):
    """Detailed comparison of one scenario between two versions."""
    scenario_id: str
    scenario_name: str = ""
    classification: ScenarioClassification
    severity: RegressionSeverity | None = None  # only set if regressed

    # Per-metric pass/score for both versions
    base_task_success: bool = False
    compare_task_success: bool = False
    base_tool_score: float = 0.0
    compare_tool_score: float = 0.0
    base_constraint_passed: bool = False
    compare_constraint_passed: bool = False
    base_overall_passed: bool = False
    compare_overall_passed: bool = False
    base_overall_score: float = 0.0
    compare_overall_score: float = 0.0

    # Latency decomposition
    base_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    compare_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

    # Failure categories present
    base_failure_categories: list[str] = Field(default_factory=list)
    compare_failure_categories: list[str] = Field(default_factory=list)

    # Human-readable reasons for the classification
    change_reasons: list[str] = Field(default_factory=list)


class RegressionResult(BaseModel):
    """Comparison between two agent versions across a scenario set."""
    base_version: str
    compare_version: str
    total_scenarios: int

    # Per-scenario detail
    scenario_comparisons: list[ScenarioComparison] = Field(default_factory=list)

    # Summary counts
    improved: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    inconclusive: list[str] = Field(default_factory=list)

    # Aggregate evaluation vectors
    base_vector: EvaluationVector = Field(default_factory=EvaluationVector)
    compare_vector: EvaluationVector = Field(default_factory=EvaluationVector)

    # Metric deltas
    task_success_delta: float = 0.0
    tool_correctness_delta: float = 0.0
    constraint_adherence_delta: float = 0.0
    latency_delta_ms: float = 0.0

    # Deployment gate
    deployment_recommendation: DeploymentRecommendation = DeploymentRecommendation.INCONCLUSIVE
    deployment_reasons: list[str] = Field(default_factory=list)

    has_regressions: bool = False
    compared_at: datetime = Field(default_factory=_now)
