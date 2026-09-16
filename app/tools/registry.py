"""Tool registry — maps tool names to callables and OpenAI function schemas.

The agent uses the schemas to know what tools are available;
the runner uses the registry to dispatch tool calls.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.models import ToolCall
from app.tools.property import property_lookup
from app.tools.calendar import check_availability, book_appointment, reset_bookings


class ToolInfrastructureError(RuntimeError):
    """An unexpected exception raised while executing a tool."""

    def __init__(self, tool_call: ToolCall) -> None:
        self.tool_call = tool_call
        super().__init__(tool_call.error or "Tool execution failed")

# ---------------------------------------------------------------------------
# OpenAI-compatible function/tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "property_lookup",
            "description": "Look up property details by property ID. Use this when the customer asks about a specific property.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {
                        "type": "string",
                        "description": "The property identifier, e.g. 'PROP-101'.",
                    }
                },
                "required": ["property_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if a property viewing slot is available on a given date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {
                        "type": "string",
                        "description": "The property identifier.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM format.",
                    },
                },
                "required": ["property_id", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a property viewing appointment. Should only be called after confirming availability with the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {
                        "type": "string",
                        "description": "The property identifier.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM format.",
                    },
                    "contact_name": {
                        "type": "string",
                        "description": "Name of the person booking the viewing.",
                    },
                    "contact_phone": {
                        "type": "string",
                        "description": "Phone number of the person booking.",
                    },
                    "contact_email": {
                        "type": "string",
                        "description": "Email address of the person booking.",
                    },
                },
                "required": ["property_id", "date", "time"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Registry: name → callable
# ---------------------------------------------------------------------------

_TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "property_lookup": property_lookup,
    "check_availability": check_availability,
    "book_appointment": book_appointment,
}


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    tool_overrides: dict[str, Any] | None = None,
) -> ToolCall:
    """Execute a tool by name and return a ToolCall record with timing.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.
        tool_overrides: Optional overrides to inject failures for testing.

    Returns:
        A ToolCall model with result, timing, and success/error status.
    """
    # Check for injected failures (for fault-tolerance testing)
    if tool_overrides and tool_name in tool_overrides:
        override = tool_overrides[tool_name]
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=override,
            success=False,
            error=override.get("error", "injected_failure"),
            latency_ms=0.0,
        )

    func = _TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            success=False,
            error=f"Unknown tool: {tool_name}",
            latency_ms=0.0,
        )

    start = time.perf_counter()
    try:
        result = func(**arguments)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Check if the tool itself returned an error
        is_error = isinstance(result, dict) and "error" in result
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=not is_error,
            error=result.get("error") if is_error else None,
            latency_ms=elapsed_ms,
        )
    except TypeError as exc:
        # Invalid arguments passed by the model (e.g. unexpected keyword arguments)
        # This is a model argument error, not an infrastructure outage.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result={"error": "invalid_arguments", "message": str(exc)},
            success=False,
            error=str(exc),
            latency_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            success=False,
            error=str(exc),
            latency_ms=elapsed_ms,
        )
        raise ToolInfrastructureError(tool_call) from exc


def reset_tool_state() -> None:
    """Reset mutable mock state between test runs."""
    reset_bookings()
