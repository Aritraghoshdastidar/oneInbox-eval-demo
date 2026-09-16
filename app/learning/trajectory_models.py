"""Data models for Trajectory-Aware Preference Learning (DPO V3).

Represents decision points where the model chooses between:
1. Emitting tool calls vs natural text (No-Tool vs Required Tool)
2. Different tool choices
3. Different argument values
4. Different tool execution orders
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionType(str, Enum):
    NO_TOOL = "no_tool"
    REQUIRED_TOOL = "required_tool"
    TOOL_ORDER = "tool_order"
    TOOL_ARGS = "tool_args"
    ESCALATION = "escalation"


@dataclass
class TrajectoryToolCall:
    """A structured tool call in a trajectory."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }

    def to_qwen_xml(self) -> str:
        """Format as Qwen's native <tool_call> block."""
        return f'<tool_call>\n{json.dumps({"name": self.name, "arguments": self.arguments})}\n</tool_call>'


@dataclass
class TrajectoryAction:
    """The action taken by the assistant at a specific decision point."""
    content: str | None = None
    tool_calls: list[TrajectoryToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

    def format_completion(self, eos_token: str = "<|im_end|>") -> str:
        """Format as the completion target for DPO training."""
        parts: list[str] = []
        if self.tool_calls:
            for tc in self.tool_calls:
                parts.append(tc.to_qwen_xml())
        if self.content:
            parts.append(self.content.strip())
        text = "\n".join(parts).strip()
        if not text.endswith(eos_token):
            text += eos_token
        return text


@dataclass
class TrajectoryPreferencePair:
    """A pair of chosen and rejected trajectory actions for a given context."""
    pair_id: str
    scenario_id: str
    decision_type: DecisionType | str
    learning_target: str
    context: list[dict[str, Any]]
    chosen: TrajectoryAction
    rejected: TrajectoryAction
    failure_category: str
    severity: str = "high"
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    lesson_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        prompt_text = " ".join(
            str(m.get("content", "")) for m in self.context
        ).lower().strip()
        prompt_norm = re.sub(r"\s+", " ", prompt_text)

        chosen_str = self.chosen.format_completion()
        rejected_str = self.rejected.format_completion()

        raw = f"{self.scenario_id}|{self.decision_type}|{prompt_norm}|{chosen_str}|{rejected_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "decision_type": str(self.decision_type.value if isinstance(self.decision_type, DecisionType) else self.decision_type),
            "learning_target": self.learning_target,
            "context": self.context,
            "chosen": self.chosen.to_dict(),
            "rejected": self.rejected.to_dict(),
            "failure_category": self.failure_category,
            "severity": self.severity,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "lesson_fingerprint": self.lesson_fingerprint or self.compute_fingerprint(),
        }
