"""Regression comparator — compare Agent V1 vs V2 results.

Loads evaluation results for two agent versions and produces a
RegressionResult with per-scenario classification, severity,
latency decomposition, deployment gate, and evaluation vectors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.models import (
    DeploymentRecommendation,
    EvaluationResult,
    EvaluationVector,
    FailureCategory,
    LatencyBreakdown,
    RegressionResult,
    RegressionSeverity,
    ScenarioClassification,
    ScenarioComparison,
)
from app.storage.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_latency_breakdown(eval_result: EvaluationResult) -> LatencyBreakdown:
    """Extract structured latency breakdown from evaluation evidence."""
    ev = eval_result.latency.evidence or {}
    e2e = ev.get("total_ms", 0.0)
    retry_delay = ev.get("retry_delay_ms", 0.0)
    return LatencyBreakdown(
        end_to_end_ms=e2e,
        llm_request_ms=ev.get("llm_request_latency_ms", 0.0),
        retry_delay_ms=retry_delay,
        tool_latency_ms=ev.get("tool_latency_ms", 0.0),
        retry_count=int(ev.get("llm_call_count", 0))
            - int(ev.get("turn_count", 0))
            if ev.get("llm_call_count", 0) > ev.get("turn_count", 0)
            else 0,
        infrastructure_dominated=retry_delay > (e2e * 0.5) if e2e > 0 else False,
    )


def _failure_categories(eval_result: EvaluationResult) -> list[str]:
    """Extract failure category names."""
    return [f.category.value for f in eval_result.failures]


def _is_infrastructure_failure(eval_result: EvaluationResult) -> bool:
    """Check if the evaluation result is contaminated by infrastructure failures."""
    cats = {f.category for f in eval_result.failures}
    if FailureCategory.RUN_ERROR in cats:
        return True
    if FailureCategory.PROVIDER_LATENCY in cats:
        latency_bd = _extract_latency_breakdown(eval_result)
        if latency_bd.infrastructure_dominated:
            return True
    return False


def _classify_scenario(
    base: EvaluationResult,
    comp: EvaluationResult,
) -> tuple[ScenarioClassification, RegressionSeverity | None, list[str]]:
    """Classify a scenario comparison.

    Returns (classification, severity_if_regressed, reasons).
    """
    reasons: list[str] = []

    # Infrastructure contamination check
    base_infra = _is_infrastructure_failure(base)
    comp_infra = _is_infrastructure_failure(comp)

    if base_infra or comp_infra:
        infra_sources = []
        if base_infra:
            infra_sources.append("base")
        if comp_infra:
            infra_sources.append("compare")
        reasons.append(
            f"Infrastructure failure in {'/'.join(infra_sources)} version(s) — "
            f"comparison unreliable"
        )
        return ScenarioClassification.INCONCLUSIVE, None, reasons

    # Per-metric transitions
    task_changed = base.task_success.passed != comp.task_success.passed
    tool_changed = abs(comp.tool_correctness.score - base.tool_correctness.score) >= 0.1
    constraint_changed = base.constraint_adherence.passed != comp.constraint_adherence.passed
    latency_changed = base.latency.passed != comp.latency.passed

    # Determine direction
    improvements: list[str] = []
    regressions: list[str] = []

    if task_changed:
        if comp.task_success.passed and not base.task_success.passed:
            improvements.append("task_success: fail → pass")
        else:
            regressions.append("task_success: pass → fail")

    if tool_changed:
        if comp.tool_correctness.score > base.tool_correctness.score:
            improvements.append(
                f"tool_correctness: {base.tool_correctness.score:.2f} → "
                f"{comp.tool_correctness.score:.2f}"
            )
        else:
            regressions.append(
                f"tool_correctness: {base.tool_correctness.score:.2f} → "
                f"{comp.tool_correctness.score:.2f}"
            )

    if constraint_changed:
        if comp.constraint_adherence.passed and not base.constraint_adherence.passed:
            improvements.append("constraint_adherence: fail → pass")
        else:
            regressions.append("constraint_adherence: pass → fail")

    if latency_changed:
        if comp.latency.passed and not base.latency.passed:
            improvements.append("latency: fail → pass")
        else:
            regressions.append("latency: pass → fail")

    # Overall score change (secondary signal)
    score_diff = comp.overall_score - base.overall_score
    if abs(score_diff) >= 0.05 and not (improvements or regressions):
        if score_diff > 0:
            improvements.append(f"overall_score: +{score_diff:.3f}")
        else:
            regressions.append(f"overall_score: {score_diff:.3f}")

    reasons.extend(improvements)
    reasons.extend(regressions)

    # Classification
    if regressions and not improvements:
        classification = ScenarioClassification.REGRESSED
    elif regressions and improvements:
        # Mixed: net regression if any business metric regressed
        if any("task_success" in r or "constraint" in r for r in regressions):
            classification = ScenarioClassification.REGRESSED
        else:
            classification = ScenarioClassification.IMPROVED
    elif improvements:
        classification = ScenarioClassification.IMPROVED
    else:
        classification = ScenarioClassification.UNCHANGED
        if not reasons:
            reasons.append("No meaningful change")

    # Severity (only for regressions)
    severity: RegressionSeverity | None = None
    if classification == ScenarioClassification.REGRESSED:
        if any("constraint" in r for r in regressions):
            severity = RegressionSeverity.CRITICAL
        elif any("task_success" in r for r in regressions):
            severity = RegressionSeverity.HIGH
        elif any("tool_correctness" in r for r in regressions):
            severity = RegressionSeverity.HIGH
        elif any("latency" in r for r in regressions):
            severity = RegressionSeverity.MEDIUM
        else:
            severity = RegressionSeverity.LOW

    return classification, severity, reasons


def _build_evaluation_vector(
    evals: list[EvaluationResult],
) -> EvaluationVector:
    """Compute aggregate evaluation vector from a list of evaluations."""
    if not evals:
        return EvaluationVector()

    n = len(evals)
    task_pass = sum(1 for e in evals if e.task_success.passed)
    tool_avg = sum(e.tool_correctness.score for e in evals) / n
    const_pass = sum(1 for e in evals if e.constraint_adherence.passed)
    overall_pass = sum(1 for e in evals if e.overall_passed)

    latencies = []
    total_retry = 0.0
    infra_fails = 0
    for e in evals:
        ev = e.latency.evidence or {}
        latencies.append(ev.get("total_ms", 0.0))
        total_retry += ev.get("retry_delay_ms", 0.0)
        if _is_infrastructure_failure(e):
            infra_fails += 1

    sorted_lat = sorted(latencies)
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)

    return EvaluationVector(
        task_success_rate=round(task_pass / n, 3),
        tool_correctness_avg=round(tool_avg, 3),
        constraint_adherence_rate=round(const_pass / n, 3),
        p95_latency_ms=round(sorted_lat[p95_idx], 1) if sorted_lat else 0.0,
        mean_latency_ms=round(sum(latencies) / n, 1) if latencies else 0.0,
        infrastructure_failures=infra_fails,
        total_retry_delay_ms=round(total_retry, 1),
        scenarios_run=n,
        scenarios_passed=overall_pass,
    )


def _compute_deployment_gate(
    comparisons: list[ScenarioComparison],
    regressed: list[str],
    inconclusive: list[str],
    total: int,
) -> tuple[DeploymentRecommendation, list[str]]:
    """Determine deployment recommendation from comparison results."""
    reasons: list[str] = []

    # Check for critical/high regressions
    critical_or_high = [
        c for c in comparisons
        if c.classification == ScenarioClassification.REGRESSED
        and c.severity in (RegressionSeverity.CRITICAL, RegressionSeverity.HIGH)
    ]
    medium_or_low = [
        c for c in comparisons
        if c.classification == ScenarioClassification.REGRESSED
        and c.severity in (RegressionSeverity.MEDIUM, RegressionSeverity.LOW)
    ]

    inconclusive_ratio = len(inconclusive) / total if total > 0 else 0

    if critical_or_high:
        reasons.append(
            f"{len(critical_or_high)} CRITICAL/HIGH regression(s): "
            f"{[c.scenario_id for c in critical_or_high]}"
        )
        return DeploymentRecommendation.BLOCK_DEPLOYMENT, reasons

    if inconclusive_ratio > 0.5:
        reasons.append(
            f"{len(inconclusive)}/{total} scenarios are INCONCLUSIVE "
            f"due to infrastructure failures"
        )
        return DeploymentRecommendation.INCONCLUSIVE, reasons

    if medium_or_low:
        reasons.append(
            f"{len(medium_or_low)} MEDIUM/LOW regression(s): "
            f"{[c.scenario_id for c in medium_or_low]}"
        )
        return DeploymentRecommendation.DEPLOY_WITH_REVIEW, reasons

    if inconclusive_ratio > 0.25:
        reasons.append(
            f"{len(inconclusive)}/{total} scenarios INCONCLUSIVE — "
            f"deploy with review recommended"
        )
        return DeploymentRecommendation.DEPLOY_WITH_REVIEW, reasons

    reasons.append("No meaningful regressions detected")
    return DeploymentRecommendation.SAFE_TO_DEPLOY, reasons


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

async def compare_versions(
    base_version: str,
    compare_version: str,
    db: Database,
) -> RegressionResult:
    """Compare evaluation results between two agent versions.

    For each scenario that was run under both versions, produces a
    ScenarioComparison with classification, severity, latency breakdown,
    and human-readable change reasons. Also computes deployment gate.

    Args:
        base_version: The baseline agent version (e.g. "v1.0").
        compare_version: The version to compare against (e.g. "v1.1").
        db: Database instance for loading evaluations.

    Returns:
        RegressionResult with full comparison detail.
    """
    base_evals = await db.get_evaluations_by_version(base_version)
    compare_evals = await db.get_evaluations_by_version(compare_version)

    # Index by scenario_id (take the most recent eval per scenario)
    base_by_scenario: dict[str, EvaluationResult] = {}
    for e in base_evals:
        if e.scenario_id not in base_by_scenario:
            base_by_scenario[e.scenario_id] = e

    compare_by_scenario: dict[str, EvaluationResult] = {}
    for e in compare_evals:
        if e.scenario_id not in compare_by_scenario:
            compare_by_scenario[e.scenario_id] = e

    # Find common scenarios
    common_ids = sorted(set(base_by_scenario.keys()) & set(compare_by_scenario.keys()))

    comparisons: list[ScenarioComparison] = []
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    inconclusive_list: list[str] = []

    task_delta = 0.0
    tool_delta = 0.0
    constraint_delta = 0.0
    latency_delta = 0.0

    for sid in common_ids:
        base = base_by_scenario[sid]
        comp = compare_by_scenario[sid]

        classification, severity, reasons = _classify_scenario(base, comp)

        base_lat = _extract_latency_breakdown(base)
        comp_lat = _extract_latency_breakdown(comp)

        sc = ScenarioComparison(
            scenario_id=sid,
            classification=classification,
            severity=severity,
            base_task_success=base.task_success.passed,
            compare_task_success=comp.task_success.passed,
            base_tool_score=base.tool_correctness.score,
            compare_tool_score=comp.tool_correctness.score,
            base_constraint_passed=base.constraint_adherence.passed,
            compare_constraint_passed=comp.constraint_adherence.passed,
            base_overall_passed=base.overall_passed,
            compare_overall_passed=comp.overall_passed,
            base_overall_score=base.overall_score,
            compare_overall_score=comp.overall_score,
            base_latency=base_lat,
            compare_latency=comp_lat,
            base_failure_categories=_failure_categories(base),
            compare_failure_categories=_failure_categories(comp),
            change_reasons=reasons,
        )
        comparisons.append(sc)

        if classification == ScenarioClassification.IMPROVED:
            improved.append(sid)
        elif classification == ScenarioClassification.REGRESSED:
            regressed.append(sid)
        elif classification == ScenarioClassification.INCONCLUSIVE:
            inconclusive_list.append(sid)
        else:
            unchanged.append(sid)

        # Accumulate deltas
        task_delta += comp.task_success.score - base.task_success.score
        tool_delta += comp.tool_correctness.score - base.tool_correctness.score
        constraint_delta += comp.constraint_adherence.score - base.constraint_adherence.score
        base_lat_ms = (base.latency.evidence or {}).get("total_ms", 0)
        comp_lat_ms = (comp.latency.evidence or {}).get("total_ms", 0)
        latency_delta += comp_lat_ms - base_lat_ms

    total = len(common_ids) or 1

    # Build evaluation vectors
    base_vector = _build_evaluation_vector(list(base_by_scenario.values()))
    compare_vector = _build_evaluation_vector(list(compare_by_scenario.values()))

    # Deployment gate
    deployment_rec, deployment_reasons = _compute_deployment_gate(
        comparisons, regressed, inconclusive_list, len(common_ids)
    )

    return RegressionResult(
        base_version=base_version,
        compare_version=compare_version,
        total_scenarios=len(common_ids),
        scenario_comparisons=comparisons,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        inconclusive=inconclusive_list,
        base_vector=base_vector,
        compare_vector=compare_vector,
        task_success_delta=round(task_delta / total, 3),
        tool_correctness_delta=round(tool_delta / total, 3),
        constraint_adherence_delta=round(constraint_delta / total, 3),
        latency_delta_ms=round(latency_delta / total, 1),
        deployment_recommendation=deployment_rec,
        deployment_reasons=deployment_reasons,
        has_regressions=len(regressed) > 0,
        compared_at=datetime.now(timezone.utc),
    )
