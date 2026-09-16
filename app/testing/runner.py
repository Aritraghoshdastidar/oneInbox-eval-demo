"""Test runner — the orchestrator that ties everything together.

Loads scenarios, runs the agent, evaluates results, classifies failures,
and stores everything. This is the single entry point for running evaluations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.agent import RealtorAgent
from app.config import settings
from app.failure.classifier import classify_failures
from app.evaluation.task_evaluator import evaluate_task
from app.evaluation.tool_evaluator import evaluate_tools
from app.evaluation.constraint_evaluator import evaluate_constraints
from app.evaluation.latency_evaluator import evaluate_latency
from app.models import (
    AgentRun,
    EvaluationResult,
    Failure,
    FailureCategory,
    MetricResult,
    TestScenario,
)
from app.storage.database import Database

logger = logging.getLogger(__name__)


class TestRunner:
    """Orchestrator for running scenarios and producing evaluations."""

    def __init__(
        self,
        agent: RealtorAgent | None = None,
        db: Database | None = None,
        scenarios_path: str | Path | None = None,
    ) -> None:
        self._agent = agent or RealtorAgent()
        self._db = db or Database()
        self._scenarios_path = Path(
            scenarios_path or settings.project_root / "scenarios" / "scenarios.json"
        )
        self._scenarios: dict[str, TestScenario] = {}

    async def initialize(self) -> None:
        """Load scenarios and initialize the database."""
        await self._db.initialize()
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """Load test scenarios from JSON file."""
        if not self._scenarios_path.exists():
            logger.warning(f"Scenarios file not found: {self._scenarios_path}")
            return

        with open(self._scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenarios_list = data if isinstance(data, list) else data.get("scenarios", [])
        for item in scenarios_list:
            scenario = TestScenario(**item)
            self._scenarios[scenario.id] = scenario

        logger.info(f"Loaded {len(self._scenarios)} scenarios from {self._scenarios_path}")

    def get_scenario(self, scenario_id: str) -> TestScenario | None:
        """Get a scenario by ID."""
        return self._scenarios.get(scenario_id)

    def list_scenarios(self) -> list[TestScenario]:
        """List all loaded scenarios."""
        return list(self._scenarios.values())

    async def run_scenario(
        self, scenario_id: str, version_override: str | None = None,
    ) -> dict[str, Any]:
        """Run a single scenario end-to-end.

        Flow:
        1. Load scenario
        2. Run agent
        3. Evaluate (task, tools, constraints, latency)
        4. Classify failures
        5. Store everything

        Returns:
            Dict with run_id, evaluation summary, and failures.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario '{scenario_id}' not found. "
                           f"Available: {list(self._scenarios.keys())}")

        # Determine which agent version to use
        agent = self._agent
        if version_override and version_override != agent._version:
            agent = RealtorAgent(
                llm=self._agent._llm,
                agent_version=version_override,
                system_prompt=self._agent._system_prompt,
            )

        logger.info(
            f"Running scenario: {scenario_id} ({scenario.name}) "
            f"[version={agent._version}]"
        )

        # Step 1: Run the agent
        try:
            run: AgentRun = await agent.run_scenario(scenario)
        except Exception as exc:
            logger.error(f"Agent failed on scenario {scenario_id}: {exc}")
            run = AgentRun(
                scenario_id=scenario_id,
                agent_version=agent._version,
                agent_config=agent.agent_config,
                status="error",
                error_message=str(exc),
            )

        # Step 2: Evaluate
        if run.status != "completed":
            runtime_detail = (
                f"Agent run failed with status '{run.status}': "
                f"{run.error_message or 'unknown runtime error'}"
            )
            runtime_metric = lambda name: MetricResult(
                metric_name=name,
                passed=False,
                score=0.0,
                details=f"RUN ERROR: {runtime_detail}",
                evidence={"run_error": True, "status": run.status},
            )
            task_result = runtime_metric("task_success")
            tool_result = runtime_metric("tool_correctness")
            constraint_result = runtime_metric("constraint_adherence")
            latency_result = runtime_metric("latency")
            failures = [Failure(
                run_id=run.run_id,
                scenario_id=scenario.id,
                agent_version=run.agent_version,
                category=FailureCategory.RUN_ERROR,
                severity=scenario.severity,
                expected="Run completes without infrastructure errors",
                actual=runtime_detail,
                explanation="Runtime failure; business behavior was not evaluated.",
            )]
        else:
            task_result = evaluate_task(run, scenario)
            tool_result = evaluate_tools(run, scenario)
            constraint_result = evaluate_constraints(run, scenario)
            latency_result = evaluate_latency(run, scenario)
            failures = classify_failures(
                run, scenario,
                task_result, tool_result, constraint_result, latency_result,
            )

        # Step 4: Assemble evaluation
        evaluation = EvaluationResult(
            run_id=run.run_id,
            scenario_id=scenario_id,
            agent_version=run.agent_version,
            task_success=task_result,
            tool_correctness=tool_result,
            constraint_adherence=constraint_result,
            latency=latency_result,
            failures=failures,
            evaluated_at=datetime.now(timezone.utc),
        )
        evaluation.compute_aggregate()

        # Step 5: Store
        await self._db.save_run(run)
        await self._db.save_evaluation(evaluation)

        logger.info(
            f"Scenario {scenario_id}: overall_passed={evaluation.overall_passed}, "
            f"score={evaluation.overall_score:.2f}, failures={len(failures)}"
        )

        return {
            "run_id": run.run_id,
            "scenario_id": scenario_id,
            "agent_version": run.agent_version,
            "overall_passed": evaluation.overall_passed,
            "overall_score": evaluation.overall_score,
            "task_success": task_result.passed,
            "tool_correctness": tool_result.score,
            "constraint_adherence": constraint_result.passed,
            "latency_ms": run.total_latency_ms,
            "failure_count": len(failures),
            "failure_categories": [f.category.value for f in failures],
        }

    async def run_all(
        self, version_override: str | None = None,
    ) -> dict[str, Any]:
        """Run all loaded scenarios and return aggregate results.

        Returns:
            Dict with per-scenario results and aggregate summary.
        """
        results: list[dict[str, Any]] = []
        for scenario_id in self._scenarios:
            try:
                result = await self.run_scenario(scenario_id, version_override)
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to run scenario {scenario_id}: {exc}")
                results.append({
                    "scenario_id": scenario_id,
                    "error": str(exc),
                    "overall_passed": False,
                })
            await asyncio.sleep(0.5)

        # Aggregate
        total = len(results)
        passed = sum(1 for r in results if r.get("overall_passed", False))
        avg_score = (
            sum(r.get("overall_score", 0) for r in results) / total
            if total > 0 else 0.0
        )
        total_failures = sum(r.get("failure_count", 0) for r in results)

        effective_version = version_override or settings.agent_version
        return {
            "agent_version": effective_version,
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
            "average_score": round(avg_score, 3),
            "total_failures": total_failures,
            "results": results,
        }
