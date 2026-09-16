"""Quick verification script — checks all imports and basic functionality."""

from app.tools.registry import execute_tool, TOOL_SCHEMAS
from app.tools.property import property_lookup
from app.tools.calendar import check_availability, book_appointment
from app.evaluation.task_evaluator import evaluate_task
from app.evaluation.tool_evaluator import evaluate_tools
from app.evaluation.constraint_evaluator import evaluate_constraints
from app.evaluation.latency_evaluator import evaluate_latency
from app.failure.classifier import classify_failures
from app.tracing.collector import TraceCollector
from app.storage.database import Database
from app.agent.llm_adapter import OpenAIAdapter, LLMResponse
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.agent import RealtorAgent
from app.testing.runner import TestRunner
from app.regression.comparator import compare_versions
from app.models import TestScenario

import json

print("All project imports OK")
print(f"Tool schemas: {len(TOOL_SCHEMAS)}")
print(f"System prompt length: {len(SYSTEM_PROMPT)} chars")

# Quick tool tests
result = property_lookup("PROP-101")
print(f"property_lookup: {result['address']}, {result['bedrooms']}BR, ${result['price']}")

result2 = check_availability("PROP-101", "2026-09-20", "14:00")
print(f"check_availability: available={result2['available']}")

result3 = book_appointment("PROP-101", "2026-09-20", "14:00", contact_name="Test User")
print(f"book_appointment: {result3['success']}, booking_id={result3['booking']['booking_id']}")

# Load scenarios
with open("scenarios/scenarios.json", "r") as f:
    data = json.load(f)
scenarios = [TestScenario(**s) for s in data["scenarios"]]
print(f"Loaded {len(scenarios)} scenarios")
for s in scenarios:
    print(f"  {s.id}: {s.name} [{s.category}] ({len(s.conversation)} turns)")

# Test trace collector
tc = TraceCollector()
tc.start_step("Hello")
tc.end_step("Hi there!", model="test-model")
print(f"TraceCollector: {len(tc.steps)} steps, latency={tc.steps[0].latency_ms:.1f}ms")

# Test execute_tool via registry
tool_result = execute_tool("property_lookup", {"property_id": "PROP-102"})
print(f"execute_tool: {tool_result.tool_name}, success={tool_result.success}, latency={tool_result.latency_ms:.2f}ms")

# Test execute_tool with override (injected failure)
tool_fail = execute_tool("book_appointment", {"property_id": "PROP-101"}, {"book_appointment": {"error": "service_unavailable"}})
print(f"execute_tool (override): success={tool_fail.success}, error={tool_fail.error}")

print("\n[OK] All verifications passed!")
