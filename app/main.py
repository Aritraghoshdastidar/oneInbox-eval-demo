"""FastAPI application — HTTP endpoints for the evaluation system.

Provides endpoints to:
- Run individual scenarios or the full suite
- Retrieve runs, evaluations, and traces
- Compare agent versions for regression detection
- View trace diffs for specific scenarios
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from app.agent.agent import RealtorAgent
from app.regression.comparator import compare_versions
from app.regression.trace_diff import build_trace_diff
from app.storage.database import Database
from app.testing.runner import TestRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------

db = Database()
runner = TestRunner(db=db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load scenarios on startup."""
    await runner.initialize()
    logger.info("Application initialized — scenarios loaded, database ready.")
    yield


app = FastAPI(
    title="OneInbox Evals & Control",
    description="Evaluation and regression testing for enterprise AI agents.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


# --- Scenarios ---

@app.get("/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    """List all loaded test scenarios."""
    return [s.model_dump() for s in runner.list_scenarios()]


@app.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str) -> dict[str, Any]:
    """Get a specific scenario by ID."""
    scenario = runner.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found")
    return scenario.model_dump()


# --- Run scenarios ---

@app.post("/run/{scenario_id}")
async def run_scenario(
    scenario_id: str,
    version: str | None = Query(None, description="Agent version override"),
) -> dict[str, Any]:
    """Run a single test scenario and return results."""
    try:
        result = await runner.run_scenario(scenario_id, version_override=version)
        return result
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.error(f"Error running scenario {scenario_id}: {exc}")
        raise HTTPException(500, f"Internal error: {exc}")


@app.post("/run")
async def run_all_scenarios(
    version: str | None = Query(None, description="Agent version override"),
) -> dict[str, Any]:
    """Run all loaded test scenarios and return aggregate results."""
    try:
        result = await runner.run_all(version_override=version)
        return result
    except Exception as exc:
        logger.error(f"Error running all scenarios: {exc}")
        raise HTTPException(500, f"Internal error: {exc}")


# --- Results ---

@app.get("/results")
async def list_results(
    agent_version: str | None = Query(None),
    scenario_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """List evaluation results with optional filters."""
    evaluations = await db.list_evaluations(
        agent_version=agent_version,
        scenario_id=scenario_id,
        limit=limit,
    )
    return [e.model_dump(mode="json") for e in evaluations]


@app.get("/results/{run_id}")
async def get_result(run_id: str) -> dict[str, Any]:
    """Get a specific evaluation result by run ID."""
    evaluation = await db.get_evaluation(run_id)
    if evaluation is None:
        raise HTTPException(404, f"No evaluation found for run '{run_id}'")
    return evaluation.model_dump(mode="json")


# --- Traces ---

@app.get("/traces/{run_id}")
async def get_trace(run_id: str) -> dict[str, Any]:
    """Get the full trace for a specific run."""
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No run found with ID '{run_id}'")
    return run.model_dump(mode="json")


# --- Regression comparison ---

@app.get("/compare")
async def compare(
    base: str = Query(..., description="Base agent version (e.g. 'v1.0')"),
    compare_to: str = Query(..., alias="compare", description="Version to compare (e.g. 'v1.1')"),
) -> dict[str, Any]:
    """Compare two agent versions and detect regressions.

    Returns per-scenario comparisons, deployment recommendation,
    evaluation vectors, and aggregate deltas.
    """
    result = await compare_versions(base, compare_to, db)
    return result.model_dump(mode="json")


@app.get("/compare/trace/{scenario_id}")
async def compare_trace(
    scenario_id: str,
    base: str = Query(..., description="Base agent version"),
    compare_to: str = Query(..., alias="compare", description="Compare agent version"),
) -> dict[str, Any]:
    """Get a trace diff for a specific scenario between two versions.

    Shows side-by-side tool calls, metric transitions, and behavioral
    differences to explain why the scenario changed.
    """
    # Get runs for both versions
    base_runs = await db.list_runs(agent_version=base, scenario_id=scenario_id, limit=1)
    comp_runs = await db.list_runs(
        agent_version=compare_to, scenario_id=scenario_id, limit=1
    )

    if not base_runs:
        raise HTTPException(404, f"No run found for {scenario_id} with version {base}")
    if not comp_runs:
        raise HTTPException(
            404, f"No run found for {scenario_id} with version {compare_to}"
        )

    base_eval = await db.get_evaluation(base_runs[0].run_id)
    comp_eval = await db.get_evaluation(comp_runs[0].run_id)

    if not base_eval or not comp_eval:
        raise HTTPException(404, "Evaluation data missing for one or both versions")

    diff = build_trace_diff(base_runs[0], comp_runs[0], base_eval, comp_eval)
    return diff
