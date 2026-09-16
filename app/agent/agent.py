"""RealtorAgent — the AI agent under test.

Runs a multi-turn conversation using an LLM with tool calling.
Produces an AgentRun with a full trace for downstream evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.models import AgentRun, ConversationTurn, LLMCall, TestScenario
from app.agent.llm_adapter import LLMAdapter, LLMResponse, OpenAIAdapter
from app.agent.prompts import SYSTEM_PROMPT, VERSION_PROMPTS
from app.tools.registry import (
    TOOL_SCHEMAS,
    ToolInfrastructureError,
    execute_tool,
    reset_tool_state,
)
from app.tracing.collector import TraceCollector


class RealtorAgent:
    """Real-estate assistant agent that processes test scenarios.

    For each scenario, it:
    1. Initializes a conversation with the system prompt.
    2. Feeds user messages one at a time (simulating real dialogue).
    3. Handles tool calls by dispatching to the mock tool registry.
    4. Collects a full trace with timing for every turn.
    """

    def __init__(
        self,
        llm: LLMAdapter | None = None,
        agent_version: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm or OpenAIAdapter()
        self._version = agent_version or settings.agent_version
        # Resolve prompt & tools: explicit parameter > variant mapping > default
        self._tools = TOOL_SCHEMAS
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = VERSION_PROMPTS.get(self._version, SYSTEM_PROMPT)

        try:
            from app.agent.variants import VARIANT_METADATA
            if self._version in VARIANT_METADATA:
                meta = VARIANT_METADATA[self._version]
                self._system_prompt = meta.get("prompt", self._system_prompt)
                disabled = meta.get("disabled_tools", [])
                if disabled:
                    self._tools = [
                        t for t in TOOL_SCHEMAS
                        if t.get("function", {}).get("name") not in disabled
                    ]
        except ImportError:
            pass

    @property
    def agent_config(self) -> dict[str, Any]:
        """Snapshot of the agent configuration for reproducibility."""
        model = getattr(self._llm, "model", settings.llm_model)
        provider = getattr(self._llm, "provider", "unknown")
        return {
            "provider": provider,
            "model": model,
            "base_url": getattr(self._llm, "_base_url", settings.llm_base_url),
            "temperature": settings.llm_temperature,
            "system_prompt_hash": hashlib.sha256(
                self._system_prompt.encode()
            ).hexdigest()[:16],
            "agent_version": self._version,
            "enabled_tools": [
                t.get("function", {}).get("name") for t in self._tools
            ],
        }

    async def run_scenario(self, scenario: TestScenario) -> AgentRun:
        """Execute a full scenario and return the structured run record."""
        # Reset mock tool state for a clean run
        reset_tool_state()

        collector = TraceCollector()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        started_at = datetime.now(timezone.utc)

        try:
            for turn in scenario.conversation:
                collector.start_step(turn.content)
                messages.append({"role": "user", "content": turn.content})
                final_content = await self._process_turn(
                    messages, collector, scenario.tool_overrides
                )
                response_model = (
                    collector.steps[-1].llm_calls[-1].model
                    if collector.steps and collector.steps[-1].llm_calls
                    else getattr(self._llm, "model", settings.llm_model)
                )
                collector.end_step(
                    assistant_message=final_content,
                    model=response_model,
                )
        except Exception as exc:
            collector.fail_step(
                f"Run error: {type(exc).__name__}: {exc}",
                model=getattr(self._llm, "model", settings.llm_model),
            )
            run = AgentRun(
                scenario_id=scenario.id,
                agent_version=self._version,
                agent_config=self.agent_config,
                steps=collector.steps,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                total_latency_ms=collector.total_latency_ms,
                llm_call_count=collector.llm_call_count,
                tool_call_count=collector.tool_call_count,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            run.flatten_tool_calls()
            return run

        completed_at = datetime.now(timezone.utc)

        # Build the AgentRun
        run = AgentRun(
            scenario_id=scenario.id,
            agent_version=self._version,
            agent_config=self.agent_config,
            steps=collector.steps,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=collector.total_latency_ms,
            llm_call_count=collector.llm_call_count,
            tool_call_count=collector.tool_call_count,
            status="completed",
        )
        run.flatten_tool_calls()
        return run

    async def _process_turn(
        self,
        messages: list[dict[str, Any]],
        collector: TraceCollector,
        tool_overrides: dict[str, Any] | None,
    ) -> str:
        """Process a single conversational turn, handling tool call loops.

        The LLM may request one or more tool calls. We execute them, feed
        results back, and let the LLM generate a final text response.
        Returns the final assistant text content.
        """
        max_tool_rounds = 5  # safety limit to prevent infinite loops
        final_content = ""

        for _ in range(max_tool_rounds):
            call_index = collector.llm_call_count
            request_started = time.perf_counter()
            try:
                response: LLMResponse = await self._llm.chat(
                    messages=messages,
                    tools=self._tools,
                    temperature=settings.llm_temperature,
                )
            except Exception as exc:
                collector.record_llm_call(LLMCall(
                    call_index=call_index,
                    provider=getattr(self._llm, "provider", "unknown"),
                    model=getattr(self._llm, "model", settings.llm_model),
                    latency_ms=(time.perf_counter() - request_started) * 1000,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                ))
                raise

            collector.record_llm_call(LLMCall(
                call_index=call_index,
                provider=response.provider or getattr(self._llm, "provider", "unknown"),
                model=response.model or getattr(self._llm, "model", settings.llm_model),
                latency_ms=response.latency_ms or (
                    (time.perf_counter() - request_started) * 1000
                ),
                request_latency_ms=response.request_latency_ms,
                retry_count=response.retry_count,
                retry_delay_ms=response.retry_delay_ms,
                retry_errors=response.retry_errors,
                response_tool_call_count=len(response.tool_calls),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            ))

            if response.tool_calls:
                # Build the assistant message with tool calls for the
                # conversation history. Preserve thought_signature for Gemini 3.
                if response.raw_message:
                    messages.append(response.raw_message)
                else:
                    tool_call_messages = []
                    for tc in response.tool_calls:
                        tc_dict: dict[str, Any] = {
                            "type": "function",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        if tc.extra_content:
                            tc_dict["extra_content"] = tc.extra_content
                        tool_call_messages.append(tc_dict)

                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": tool_call_messages,
                    })

                # Execute each tool call and feed results back
                for tc in response.tool_calls:
                    try:
                        tool_result = execute_tool(
                            tc.name, tc.arguments, tool_overrides
                        )
                    except ToolInfrastructureError as exc:
                        collector.record_tool_call(exc.tool_call)
                        raise
                    collector.record_tool_call(tool_result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result.result),
                    })
            else:
                # No tool calls — this is the final text response
                final_content = response.content or ""
                messages.append({
                    "role": "assistant",
                    "content": final_content,
                })
                break
        else:
            # Hit the tool-call safety limit
            final_content = (
                "I apologize, but I'm having difficulty processing your "
                "request. Let me connect you with a human agent."
            )
            messages.append({"role": "assistant", "content": final_content})

        return final_content
