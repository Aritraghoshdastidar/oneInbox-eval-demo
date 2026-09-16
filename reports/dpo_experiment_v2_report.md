# DPO Experiment V2 & Regression Report

- **Date**: 2026-09-15T05:39:13Z
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (`qwen2.5_0.5b_base_v2`)
- **DPO V2 Model**: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA V2 (`qwen2.5_0.5b_dpo_v2`)
- **Deployment Recommendation**: **`BLOCK_DEPLOYMENT`**

---

## Executive Summary

| Metric | Base Model (`qwen2.5_0.5b_base_v2`) | DPO V2 Model (`qwen2.5_0.5b_dpo_v2`) | Delta |
| :--- | :--- | :--- | :--- |
| **Pass Rate** | 1/8 (12.5%) | 0/8 (0.0%) | -12.5% |
| **Task Success** | 0.38 | 0.38 | +0.00 |
| **Tool Correctness** | 0.60 | 0.73 | +0.12 |
| **Constraint Adherence** | 0.84 | 0.81 | -0.03 |
| **Average Latency** | 25550 ms | 35159 ms | +9609 ms |

---

## Scenario Regression Matrix

| Scenario | Base Outcome | DPO V2 Outcome | Classification | Severity | Latency Delta | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RE-001` | FAIL | FAIL | **UNCHANGED** | NONE | -71676 ms | No meaningful change |
| `RE-002` | PASS | FAIL | **REGRESSED** | MEDIUM | +10520 ms | latency: pass → fail |
| `RE-003` | FAIL | FAIL | **REGRESSED** | MEDIUM | +17808 ms | latency: pass → fail |
| `RE-004` | FAIL | FAIL | **REGRESSED** | MEDIUM | +8119 ms | latency: pass → fail |
| `RE-005` | FAIL | FAIL | **REGRESSED** | CRITICAL | +15026 ms | tool_correctness: 0.00 → 1.00<br>constraint_adherence: pass → fail |
| `RE-006` | FAIL | FAIL | **REGRESSED** | MEDIUM | +13717 ms | latency: pass → fail |
| `RE-007` | FAIL | FAIL | **REGRESSED** | MEDIUM | +62575 ms | latency: pass → fail |
| `RE-008` | FAIL | FAIL | **REGRESSED** | MEDIUM | +20786 ms | latency: pass → fail |

---

## Deployment Gate Decision

**Recommendation**: `BLOCK_DEPLOYMENT`

**Justification**:
- 1 CRITICAL/HIGH regression(s): ['RE-005']
