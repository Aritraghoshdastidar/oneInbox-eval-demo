from __future__ import annotations

from app.agent.agent import RealtorAgent
from app.agent.llm_adapter import OpenAIAdapter
from app.evaluation.latency_evaluator import evaluate_latency
from app.evaluation.task_evaluator import evaluate_task
from app.models import AgentRun, LLMCall, TestScenario as Scenario, TraceStep
from app.tools import registry
from app.tools.registry import ToolInfrastructureError, execute_tool


def make_scenario(outcome: str = "information_provided") -> Scenario:
    return Scenario(
        id="TEST-001",
        name="Instrumentation test",
        description="Test-only scenario",
        category="normal",
        severity="low",
        conversation=[{"content": "test"}],
        expected_intents=[],
        expected_tool_calls=[],
        expected_outcome=outcome,
        max_latency_ms=60000,
    )


def test_adapter_reports_configured_provider_and_model() -> None:
    adapter = OpenAIAdapter(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-test-model",
    )

    assert adapter.provider == "gemini"
    assert adapter.model == "gemini-test-model"
    assert RealtorAgent(llm=adapter).agent_config["provider"] == "gemini"
    assert RealtorAgent(llm=adapter).agent_config["model"] == "gemini-test-model"


def test_task_evaluator_uses_final_outcome() -> None:
    run = AgentRun(
        scenario_id="TEST-001",
        agent_version="test",
        steps=[
            TraceStep(
                step_index=0,
                user_message="test",
                assistant_message="Please provide your contact details.",
            ),
            TraceStep(
                step_index=1,
                user_message="test",
                assistant_message="The service is down. I will connect you with a human agent.",
            ),
        ],
    )

    result = evaluate_task(run, make_scenario("escalated"))

    assert result.passed
    assert result.evidence["actual_outcome"] == "escalated"


def test_latency_evidence_preserves_retry_delay() -> None:
    call = LLMCall(
        call_index=0,
        provider="gemini",
        model="gemini-test-model",
        latency_ms=45000,
        request_latency_ms=1500,
        retry_count=1,
        retry_delay_ms=43500,
        retry_errors=["Error code: 429 RESOURCE_EXHAUSTED"],
    )
    run = AgentRun(
        scenario_id="TEST-001",
        agent_version="test",
        steps=[
            TraceStep(
                step_index=0,
                user_message="test",
                assistant_message="ok",
                llm_calls=[call],
                latency_ms=45000,
            )
        ],
        total_latency_ms=45000,
        llm_call_count=1,
    )

    evidence = evaluate_latency(run, make_scenario()).evidence

    retry = evidence["turn_breakdown"][0]["llm_calls"][0]
    assert retry["request_latency_ms"] == 1500
    assert retry["retry_count"] == 1
    assert retry["retry_delay_ms"] == 43500
    assert retry["retry_errors"] == ["Error code: 429 RESOURCE_EXHAUSTED"]


def test_unexpected_tool_exception_is_infrastructure_error(monkeypatch) -> None:
    def fail(**_: object) -> None:
        raise RuntimeError("synthetic tool failure")

    monkeypatch.setitem(registry._TOOL_FUNCTIONS, "property_lookup", fail)

    try:
        execute_tool("property_lookup", {"property_id": "PROP-101"})
    except ToolInfrastructureError as exc:
        assert exc.tool_call.success is False
        assert exc.tool_call.error == "synthetic tool failure"
    else:
        raise AssertionError("Expected ToolInfrastructureError")
