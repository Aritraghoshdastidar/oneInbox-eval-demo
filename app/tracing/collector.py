"""Trace collector — builds TraceStep objects during an agent run.

The collector is instantiated per-run and accumulates steps as the agent
processes each conversation turn. It handles start/end timing, tool call
recording, and LLM metadata capture.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.models import LLMCall, TraceStep, ToolCall


class TraceCollector:
    """Accumulates trace data for a single agent run."""

    def __init__(self) -> None:
        self._steps: list[TraceStep] = []
        self._current_step: TraceStep | None = None
        self._step_start: float = 0.0

    def start_step(self, user_message: str) -> None:
        """Begin recording a new conversational turn."""
        self._step_start = time.perf_counter()
        self._current_step = TraceStep(
            step_index=len(self._steps),
            user_message=user_message,
            start_time=datetime.now(timezone.utc),
        )

    def record_tool_call(self, tool_call: ToolCall) -> None:
        """Append a tool call to the current step."""
        if self._current_step is not None:
            self._current_step.tool_calls.append(tool_call)

    def record_llm_call(self, llm_call: LLMCall) -> None:
        """Append one LLM request record to the current turn."""
        if self._current_step is not None:
            self._current_step.llm_calls.append(llm_call)

    def fail_step(self, message: str, model: str = "") -> None:
        """Close an interrupted turn while preserving its partial trace."""
        if self._current_step is not None:
            self.end_step(assistant_message=message, model=model)

    def end_step(
        self,
        assistant_message: str,
        model: str = "",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> TraceStep:
        """Finalize the current step and return it."""
        if self._current_step is None:
            raise RuntimeError("end_step called without a matching start_step")

        now = datetime.now(timezone.utc)
        elapsed_ms = (time.perf_counter() - self._step_start) * 1000

        self._current_step.assistant_message = assistant_message
        self._current_step.end_time = now
        self._current_step.latency_ms = elapsed_ms
        self._current_step.model = model
        self._current_step.prompt_tokens = prompt_tokens
        self._current_step.completion_tokens = completion_tokens

        step = self._current_step
        self._steps.append(step)
        self._current_step = None
        return step

    @property
    def steps(self) -> list[TraceStep]:
        return list(self._steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self._steps)

    @property
    def llm_call_count(self) -> int:
        return sum(len(step.llm_calls) for step in self._steps) + (
            len(self._current_step.llm_calls) if self._current_step else 0
        )

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self._steps) + (
            len(self._current_step.tool_calls) if self._current_step else 0
        )
