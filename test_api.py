"""Quick API test — verifies FastAPI endpoints are working."""

import urllib.request
import json

BASE = "http://127.0.0.1:8000"


def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(r.read())


# Health
print("=== Health ===")
print(get("/health"))

# Scenarios
print("\n=== Scenarios ===")
scenarios = get("/scenarios")
print(f"Loaded {len(scenarios)} scenarios")
for s in scenarios:
    print(f"  {s['id']}: {s['name']} [{s['category']}]")

# Single scenario
print("\n=== Single Scenario ===")
s = get("/scenarios/RE-001")
print(f"  {s['id']}: {s['name']}")
print(f"  Expected outcome: {s['expected_outcome']}")
print(f"  Expected tools: {[tc['tool_name'] for tc in s['expected_tool_calls']]}")

print("\n[OK] All API endpoints working!")
