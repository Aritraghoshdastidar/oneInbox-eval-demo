# DPO Experiment & Regression Report

- **Date**: 2026-09-15T05:03:48Z
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (`qwen2.5_0.5b_base`)
- **DPO Model**: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (`qwen2.5_0.5b_dpo`)
- **Deployment Recommendation**: **`BLOCK_DEPLOYMENT`**

---

## Executive Summary

| Metric | Base Model (`qwen2.5_0.5b_base`) | DPO Model (`qwen2.5_0.5b_dpo`) | Delta |
| :--- | :--- | :--- | :--- |
| **Pass Rate** | 1/8 (12.5%) | 1/8 (12.5%) | +0.0% |
| **Task Success** | 0.38 | 0.25 | -0.12 |
| **Tool Correctness** | 0.60 | 0.71 | +0.10 |
| **Constraint Adherence** | 0.84 | 0.81 | -0.03 |
| **Average Latency** | 15470 ms | 26408 ms | +10939 ms |

---

## Scenario Regression Matrix

| Scenario | Base Outcome | DPO Outcome | Classification | Severity | Latency Delta | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RE-001` | FAIL | FAIL | **REGRESSED** | HIGH | +5084 ms | task_success: pass → fail<br>tool_correctness: 0.67 → 0.50 |
| `RE-002` | PASS | PASS | **UNCHANGED** | NONE | +9048 ms | No meaningful change |
| `RE-003` | FAIL | FAIL | **REGRESSED** | MEDIUM | +14218 ms | latency: pass → fail |
| `RE-004` | FAIL | FAIL | **UNCHANGED** | NONE | +6466 ms | No meaningful change |
| `RE-005` | FAIL | FAIL | **REGRESSED** | CRITICAL | +8870 ms | tool_correctness: 0.00 → 1.00<br>constraint_adherence: pass → fail |
| `RE-006` | FAIL | FAIL | **REGRESSED** | MEDIUM | +12646 ms | latency: pass → fail |
| `RE-007` | FAIL | FAIL | **REGRESSED** | MEDIUM | +18989 ms | latency: pass → fail |
| `RE-008` | FAIL | FAIL | **REGRESSED** | MEDIUM | +12188 ms | latency: pass → fail |

---

## Deployment Gate Decision

**Recommendation**: `BLOCK_DEPLOYMENT`

**Justification**:
- 2 CRITICAL/HIGH regression(s): ['RE-001', 'RE-005']
