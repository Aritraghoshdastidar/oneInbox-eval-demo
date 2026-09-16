"""Generate controlled agent runs on ambiguity scenarios.

Runs both v1.0 (clarification policy, passing) and v1.1 (proactive guessing, failing)
against the controlled ambiguity scenarios and persists full traces into SQLite.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.agent import RealtorAgent
from app.storage.database import Database
from app.testing.runner import TestRunner


async def main():
    scenarios_path = Path("scenarios/ambiguity_scenarios.json")
    print("=" * 70)
    print("RUNNING AMBIGUITY SCENARIOS FOR v1.0 AND v1.1")
    print(f"Scenarios file: {scenarios_path}")
    print("=" * 70)

    db = Database()
    await db.initialize()

    # Load scenarios
    runner_helper = TestRunner(db=db, scenarios_path=scenarios_path)
    await runner_helper.initialize()
    scenarios = runner_helper.list_scenarios()
    print(f"Loaded {len(scenarios)} ambiguity scenarios.\n")

    # 1. Run v1.0 (Passing clarification policy)
    print("-" * 50)
    print("RUNNING v1.0 (Expected: clarification requested, 0 tools)")
    print("-" * 50)
    agent_v10 = RealtorAgent(agent_version="v1.0")
    runner_v10 = TestRunner(agent=agent_v10, db=db, scenarios_path=scenarios_path)
    await runner_v10.initialize()

    for s in scenarios:
        print(f"Running v1.0 on {s.id}: {s.name}...")
        res = await runner_v10.run_scenario(s.id)
        eval_d = res.get("evaluation", {})
        passed = eval_d.get("passed", False)
        print(f"  Result: {'PASS' if passed else 'FAIL'} | Tools called: {len(res.get('run', {}).get('actual_tool_calls', []))} | Outcome: {res.get('run', {}).get('actual_outcome')}")

    # 2. Run v1.1 (Failing speculative guessing policy)
    print("\n" + "-" * 50)
    print("RUNNING v1.1 (Expected: speculative search / failure)")
    print("-" * 50)
    agent_v11 = RealtorAgent(agent_version="v1.1")
    runner_v11 = TestRunner(agent=agent_v11, db=db, scenarios_path=scenarios_path)
    await runner_v11.initialize()

    for s in scenarios:
        print(f"Running v1.1 on {s.id}: {s.name}...")
        res = await runner_v11.run_scenario(s.id)
        eval_d = res.get("evaluation", {})
        passed = eval_d.get("passed", False)
        print(f"  Result: {'PASS' if passed else 'FAIL'} | Tools called: {len(res.get('run', {}).get('actual_tool_calls', []))} | Outcome: {res.get('run', {}).get('actual_outcome')}")

    print("\n" + "=" * 70)
    print("All ambiguity runs completed and stored in SQLite database!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
