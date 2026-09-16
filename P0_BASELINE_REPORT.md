# P0 Baseline and Work Report

Date: 2026-09-14
Provider: Gemini through the OpenAI-compatible API
Configured model: `gemini-3.5-flash-lite`
Agent version: `v1.0`

## Completed Work

### Provider and model metadata

- Added runtime provider detection from the configured base URL.
- Recorded the actual adapter model instead of assuming `gpt-4o-mini`.
- Added provider, model, and non-secret base URL to `AgentRun.agent_config`.
- Recorded provider and model on every LLM trace record.

### Latency instrumentation

- Added per-conversation-turn timing.
- Added per-LLM-request timing.
- Added per-tool-call timing.
- Added total LLM request and tool call counts.
- Added request latency separately from total LLM elapsed time.
- Added retry count and retry backoff delay to every LLM call.
- Added detailed latency evidence under `turn_breakdown`.
- Kept existing latency thresholds and scoring unchanged.

### Runtime error handling

- Added explicit `run_error` failure category.
- Runtime LLM failures now produce `AgentRun.status = "error"`.
- Runtime failures retain the partial trace and failed LLM request.
- Unexpected tool exceptions now become infrastructure errors.
- Expected tool-returned errors remain available for business-level evaluation.
- Runtime errors are not reported as `wrong_tool` or `incomplete_workflow`.
- Existing SQLite databases migrate the new call-count columns at startup.

### Evaluator corrections

- Optional expected tool calls no longer fail a scenario when absent.
- User-supplied unknown property IDs are not treated as hallucinated assistant IDs.
- Clarification wording detection was expanded.
- Final assistant outcomes take precedence over earlier clarification requests.
- Earlier information remains valid when the final turn is only a neutral closing.
- RE-006 escalation is correctly recognized when the final response escalates after a tool error.

### Prompt strengthening

- Description-only property searches request an ID or exact address.
- Explicit `PROP-###` IDs are always looked up.
- Tool errors require explicit handling and escalation when appropriate.
- Customer dates and times must be preserved exactly.

### Tests and validation

Added `tests/test_p0_trustworthiness.py` covering:

- Provider/model metadata.
- Final-outcome inference.
- Retry-delay latency evidence.
- Tool infrastructure error classification.

Focused tests: `4 passed`.

`python verify.py`: passed.

Compilation and language diagnostics: passed.

The full `python -m pytest -q` command also collects `test_api.py`, which executes HTTP requests during import. It fails when no local API server is running; this is a test harness/setup issue, not an application assertion failure. The focused test suite passes independently.

## Final Persisted Eight-Scenario Baseline

The final eight traces were re-evaluated with the corrected deterministic evaluators without issuing additional Gemini requests.

| Metric | Result |
|---|---:|
| Passed | 6 |
| Failed | 2 |
| Pass rate | 75% |
| Average score | 0.958 |
| Total failures | 2 |
| Average latency | 13,865.0 ms |
| P95 latency | 46,033.2 ms |
| Total LLM calls | 34 |
| Average LLM calls/run | 4.25 |
| Total tool calls | 17 |
| Average tool calls/run | 2.12 |

## Scenario Results

- RE-001: passed, appointment booked.
- RE-002: passed, information provided.
- RE-003: failed on latency only; appointment was booked in 40,175.8 ms against a 30,000 ms threshold.
- RE-004: passed, clarification requested.
- RE-005: passed, unknown property escalated.
- RE-006: passed after corrected evaluation; booking-service failure was escalated.
- RE-007: passed, appointment booked.
- RE-008: failed on latency only; escalation was correct but total latency was 46,033.2 ms against a 20,000 ms threshold.

## Latency Conclusion

Tool execution is negligible, generally below 1 ms. Gemini/API requests dominate runtime. The slow failures included HTTP 429 retry delays, which are now separately recorded as `retry_delay_ms` instead of being confused with model request time.

No latency thresholds were changed.

## Remaining Limitations

- Gemini behavior is probabilistic. RE-006 produced a correct escalation in its trace, but earlier runs produced a clarification outcome.
- The deterministic task evaluator uses response-text signals and should remain covered by focused tests.
- `test_api.py` should eventually be converted from import-time script execution into an explicit test or separate smoke-test command.

## Recommended Next Work

1. Add a dedicated retry/backoff policy test using a fake adapter response sequence.
2. Report API rate-limit delay separately in the dashboard/API response if a dashboard is later introduced.
3. Convert `test_api.py` and other smoke scripts into explicit commands or pytest tests.
4. Reassess latency SLOs after collecting several Gemini runs, without silently changing thresholds.

## Latest Retry-Aware Validation

After adding provider-latency classification and retry-cause telemetry, the eight-scenario suite was run once.

| Metric | Result |
|---|---:|
| Passed | 7 |
| Failed | 1 |
| Pass rate | 87.5% |
| Average score | 0.969 |
| Total failures | 1 |
| Average raw latency | 13,072.4 ms |
| P95 raw latency | 74,694.2 ms |
| Total LLM calls | 34 |
| Total retries | 2 |
| Total retry delay | 66,781.4 ms |
| Total LLM elapsed time | 104,575.9 ms |
| Total actual LLM request time | 37,779.6 ms |
| Total tool latency | 0.2 ms |
| Business/agent failures | 0 |
| Provider/infrastructure latency failures | 1 |
| Latency SLA failures | 1 |

RE-003 was the only failed scenario. Its task, tool, and constraint metrics all passed. The failure category is `provider_latency`, so this is an operational SLA failure rather than an incorrect agent decision.

The retried LLM call recorded:

- Actual request time: approximately 2,548 ms
- Retry count: 2
- Retry delay: approximately 66,781 ms
- Cause: Gemini HTTP 429 `RESOURCE_EXHAUSTED` quota errors

The original provider error strings are retained in `retry_errors`. No retry cause was inferred beyond the provider error content.
