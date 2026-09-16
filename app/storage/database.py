"""SQLite storage layer — persist and retrieve runs and evaluations.

Uses aiosqlite for async compatibility with FastAPI. Stores structured
data as JSON columns for flexibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from app.config import settings
from app.models import AgentRun, EvaluationResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    agent_config TEXT NOT NULL,
    steps TEXT NOT NULL,
    actual_outcome TEXT,
    actual_tool_calls TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    total_latency_ms REAL NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    llm_call_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    task_success TEXT NOT NULL,
    tool_correctness TEXT NOT NULL,
    constraint_adherence TEXT NOT NULL,
    latency TEXT NOT NULL,
    overall_passed INTEGER NOT NULL,
    overall_score REAL NOT NULL,
    failures TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_scenario ON agent_runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_runs_version ON agent_runs(agent_version);
CREATE INDEX IF NOT EXISTS idx_evals_version ON evaluation_results(agent_version);
CREATE INDEX IF NOT EXISTS idx_evals_scenario ON evaluation_results(scenario_id);
"""


class Database:
    """Async SQLite database for storing agent runs and evaluations."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = str(db_path or settings.db_full_path)

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            for column, definition in (
                ("llm_call_count", "INTEGER NOT NULL DEFAULT 0"),
                ("tool_call_count", "INTEGER NOT NULL DEFAULT 0"),
            ):
                try:
                    await db.execute(
                        f"ALTER TABLE agent_runs ADD COLUMN {column} {definition}"
                    )
                except aiosqlite.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            await db.commit()

    # ----- Writes -----

    async def save_run(self, run: AgentRun) -> None:
        """Persist an agent run."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO agent_runs
                   (run_id, scenario_id, agent_version, agent_config, steps,
                    actual_outcome, actual_tool_calls, started_at, completed_at,
                    total_latency_ms, status, error_message,
                    llm_call_count, tool_call_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.scenario_id,
                    run.agent_version,
                    run.model_dump_json(include={"agent_config"}),
                    json.dumps([s.model_dump(mode="json") for s in run.steps]),
                    run.actual_outcome,
                    json.dumps([tc.model_dump(mode="json") for tc in run.actual_tool_calls]),
                    run.started_at.isoformat(),
                    run.completed_at.isoformat(),
                    run.total_latency_ms,
                    run.status,
                    run.error_message,
                    run.llm_call_count,
                    run.tool_call_count,
                ),
            )
            await db.commit()

    async def save_evaluation(self, evaluation: EvaluationResult) -> None:
        """Persist an evaluation result."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO evaluation_results
                   (run_id, scenario_id, agent_version, task_success,
                    tool_correctness, constraint_adherence, latency,
                    overall_passed, overall_score, failures, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evaluation.run_id,
                    evaluation.scenario_id,
                    evaluation.agent_version,
                    evaluation.task_success.model_dump_json(),
                    evaluation.tool_correctness.model_dump_json(),
                    evaluation.constraint_adherence.model_dump_json(),
                    evaluation.latency.model_dump_json(),
                    int(evaluation.overall_passed),
                    evaluation.overall_score,
                    json.dumps([f.model_dump(mode="json") for f in evaluation.failures]),
                    evaluation.evaluated_at.isoformat(),
                ),
            )
            await db.commit()

    # ----- Reads -----

    async def get_run(self, run_id: str) -> AgentRun | None:
        """Retrieve a single agent run by ID."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_run(row)

    async def get_evaluation(self, run_id: str) -> EvaluationResult | None:
        """Retrieve a single evaluation result by run ID."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM evaluation_results WHERE run_id = ?", (run_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_evaluation(row)

    async def list_runs(
        self,
        agent_version: str | None = None,
        scenario_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        """List agent runs with optional filters."""
        query = "SELECT * FROM agent_runs WHERE 1=1"
        params: list[str | int] = []

        if agent_version:
            query += " AND agent_version = ?"
            params.append(agent_version)
        if scenario_id:
            query += " AND scenario_id = ?"
            params.append(scenario_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [_row_to_run(row) for row in rows]

    async def list_evaluations(
        self,
        agent_version: str | None = None,
        scenario_id: str | None = None,
        limit: int = 100,
    ) -> list[EvaluationResult]:
        """List evaluation results with optional filters."""
        query = "SELECT * FROM evaluation_results WHERE 1=1"
        params: list[str | int] = []

        if agent_version:
            query += " AND agent_version = ?"
            params.append(agent_version)
        if scenario_id:
            query += " AND scenario_id = ?"
            params.append(scenario_id)

        query += " ORDER BY evaluated_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [_row_to_evaluation(row) for row in rows]

    async def get_evaluations_by_version(
        self, agent_version: str
    ) -> list[EvaluationResult]:
        """Get all evaluations for a specific agent version."""
        return await self.list_evaluations(agent_version=agent_version, limit=1000)


# ---------------------------------------------------------------------------
# Row → Model helpers
# ---------------------------------------------------------------------------

def _row_to_run(row: aiosqlite.Row) -> AgentRun:
    from app.models import TraceStep, ToolCall

    config_raw = row["agent_config"]
    if config_raw:
        config_parsed = json.loads(config_raw)
        # Handle the nested {"agent_config": {...}} wrapper from model_dump_json
        if "agent_config" in config_parsed:
            config_parsed = config_parsed["agent_config"]
    else:
        config_parsed = {}

    return AgentRun(
        run_id=row["run_id"],
        scenario_id=row["scenario_id"],
        agent_version=row["agent_version"],
        agent_config=config_parsed,
        steps=[TraceStep(**s) for s in json.loads(row["steps"])],
        actual_outcome=row["actual_outcome"],
        actual_tool_calls=[ToolCall(**tc) for tc in json.loads(row["actual_tool_calls"])],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        total_latency_ms=row["total_latency_ms"],
        status=row["status"],
        error_message=row["error_message"],
        llm_call_count=row["llm_call_count"] if "llm_call_count" in row.keys() else 0,
        tool_call_count=row["tool_call_count"] if "tool_call_count" in row.keys() else 0,
    )


def _row_to_evaluation(row: aiosqlite.Row) -> EvaluationResult:
    from app.models import MetricResult, Failure

    return EvaluationResult(
        run_id=row["run_id"],
        scenario_id=row["scenario_id"],
        agent_version=row["agent_version"],
        task_success=MetricResult(**json.loads(row["task_success"])),
        tool_correctness=MetricResult(**json.loads(row["tool_correctness"])),
        constraint_adherence=MetricResult(**json.loads(row["constraint_adherence"])),
        latency=MetricResult(**json.loads(row["latency"])),
        overall_passed=bool(row["overall_passed"]),
        overall_score=row["overall_score"],
        failures=[Failure(**f) for f in json.loads(row["failures"])],
        evaluated_at=row["evaluated_at"],
    )
