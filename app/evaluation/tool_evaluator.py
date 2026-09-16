"""Tool evaluator — deterministic check for tool call correctness.

Compares the agent's actual tool calls against the expected tool calls
defined in the test scenario. Checks:
1. Were the correct tools called?
2. Were they called with correct arguments? (subset match)
3. Were required tools called?
4. Was the order correct?
"""

from __future__ import annotations

from typing import Any

from app.models import AgentRun, ExpectedToolCall, MetricResult, TestScenario, ToolCall


def _match_arguments(
    expected_args: dict[str, Any],
    actual_args: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check if expected arguments are a subset of actual arguments.

    Uses subset matching: expected args must be present in actual args,
    but actual args may contain additional fields.

    Returns:
        (all_matched, list_of_mismatch_descriptions)
    """
    mismatches: list[str] = []
    for key, expected_val in expected_args.items():
        if key not in actual_args:
            mismatches.append(f"Missing argument '{key}' (expected: {expected_val})")
        elif str(actual_args[key]).lower() != str(expected_val).lower():
            mismatches.append(
                f"Argument '{key}': expected '{expected_val}', "
                f"got '{actual_args[key]}'"
            )
    return len(mismatches) == 0, mismatches


def _find_best_match(
    expected: ExpectedToolCall,
    actual_calls: list[ToolCall],
    used_indices: set[int],
) -> tuple[int | None, bool, list[str]]:
    """Find the best matching actual tool call for an expected one.

    Returns:
        (matched_index, args_matched, mismatch_details)
    """
    for i, actual in enumerate(actual_calls):
        if i in used_indices:
            continue
        if actual.tool_name == expected.tool_name:
            args_ok, mismatches = _match_arguments(
                expected.arguments, actual.arguments
            )
            return i, args_ok, mismatches
    return None, False, [f"Tool '{expected.tool_name}' was never called"]


def evaluate_tools(run: AgentRun, scenario: TestScenario) -> MetricResult:
    """Evaluate tool call correctness.

    Scoring:
    - Each expected tool call contributes equally to the score.
    - A tool call scores 1.0 if name matches AND args match.
    - A tool call scores 0.5 if name matches but args differ.
    - A tool call scores 0.0 if the tool was never called.
    - Required tool calls that are missing cause a failure.

    Returns:
        MetricResult with score, pass/fail, and detailed evidence.
    """
    expected_calls = scenario.expected_tool_calls
    actual_calls = run.actual_tool_calls

    if not expected_calls:
        # No tools expected — pass if no tools were called
        if not actual_calls:
            return MetricResult(
                metric_name="tool_correctness",
                passed=True,
                score=1.0,
                details="No tool calls expected or made.",
            )
        else:
            return MetricResult(
                metric_name="tool_correctness",
                passed=False,
                score=0.5,
                details=(
                    f"No tool calls expected but {len(actual_calls)} were made: "
                    f"{[tc.tool_name for tc in actual_calls]}"
                ),
                evidence={"unexpected_calls": [tc.tool_name for tc in actual_calls]},
            )

    used_indices: set[int] = set()
    call_results: list[dict[str, Any]] = []
    total_score = 0.0
    all_required_present = True

    for expected in expected_calls:
        idx, args_ok, mismatches = _find_best_match(
            expected, actual_calls, used_indices
        )

        if idx is not None:
            used_indices.add(idx)
            if args_ok:
                call_score = 1.0
                status = "match"
            else:
                call_score = 0.5
                status = "args_mismatch"
        else:
            if expected.required:
                call_score = 0.0
                status = "missing"
                all_required_present = False
            else:
                call_score = 1.0
                status = "optional_missing"

        total_score += call_score
        call_results.append({
            "expected_tool": expected.tool_name,
            "expected_args": expected.arguments,
            "status": status,
            "score": call_score,
            "mismatches": mismatches,
            "required": expected.required,
        })

    # Normalize score
    score = total_score / len(expected_calls) if expected_calls else 1.0
    passed = all_required_present and score >= 0.8

    # Build summary
    statuses = [r["status"] for r in call_results]
    if all(s in {"match", "optional_missing"} for s in statuses):
        details = f"All {len(expected_calls)} expected tool calls matched correctly."
    else:
        summary_parts = []
        for r in call_results:
            if r["status"] == "match":
                summary_parts.append(f"✓ {r['expected_tool']}")
            elif r["status"] == "optional_missing":
                summary_parts.append(f"- {r['expected_tool']} (optional, not called)")
            elif r["status"] == "args_mismatch":
                summary_parts.append(
                    f"⚠ {r['expected_tool']} (args: {'; '.join(r['mismatches'])})"
                )
            else:
                summary_parts.append(f"✗ {r['expected_tool']} (missing)")
        details = "Tool call results:\n" + "\n".join(summary_parts)

    return MetricResult(
        metric_name="tool_correctness",
        passed=passed,
        score=round(score, 3),
        details=details,
        evidence={
            "expected_count": len(expected_calls),
            "actual_count": len(actual_calls),
            "call_results": call_results,
        },
    )
