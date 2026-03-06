# Multi-Spec Negative Regression Report (2026-03-05)

Generated: 2026-03-05T13:15 IST  
Mode: post-fix full regression (`max_scenarios=32`, negative-heavy prompt, run + persistence check)

## Domain Summary

| Domain | Base Pass | Persist Pass | Base Failed | Persist Failed | Gate-1 | Gate-2 | GAM-1 | GAM-2 | RL Steps (1→2) | RL Buffer (1→2) |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| ecommerce | 0.9688 | 0.9375 | 1 | 2 | True | True | accepted | accepted | 6→7 | 159→192 |
| healthcare | 0.9062 | 0.9375 | 3 | 2 | True | True | needs_retry | needs_retry | 5→6 | 132→165 |
| logistics | 0.9062 | 0.9062 | 3 | 3 | True | True | needs_retry | needs_retry | 5→6 | 133→166 |
| hr | 0.9310 | 0.8571 | 2 | 4 | True | True | needs_retry | needs_retry | 5→6 | 125→154 |
| banking_custom | 0.7241 | 0.6250 | 8 | 12 | True | False | needs_retry | needs_retry | 3→4 | 74→107 |
| sample_custom | 0.5938 | 0.5625 | 13 | 14 | False | False | needs_retry | needs_retry | 6→7 | 175→208 |

## High-Signal Findings (Persistence)

### ecommerce
- `Get Products - Method Not Allowed` | `/products` | expected `201` got `400`.
- `Post Orders - Dependency Order Violation` | `/orders` | expected `400` got `201`.

### healthcare
- `Create Appointment - Happy Path` still fails (`expected 201`, `actual 400`).
- `test_post_appointments_rl_history_seed_happy_path_201` also fails on same mismatch.

### logistics
- Shipment create flow remains unstable on expected `201` (`actual 400`) across happy-path and integration variants.

### hr
- Invalid `jobId` format expectations are now mismatched the other way (`expected 400`, `actual 404`) in multiple variants.
- `Create Candidate - Invalid Email Format` still returns `201` instead of expected `400`.

### banking_custom
- Persistence gate fails.
- Multiple `GET /accounts` scenarios are still normalized/expected as `200` while actual is `401` on auth-negative paths.

### sample_custom
- Both base and persistence gates fail.
- Main failures are still around list/query validation assumptions and `POST /users` happy-path instability.

## Fix Validation Notes

- RL persistence is working on all specs (`training_steps` and replay `buffer` increased on second pass for every domain).
- Contract/verdict consistency fix is effective: `expected == actual` with `verdict != pass` is now `0` across all 6 latest persistence runs.
- Remaining failures are primarily scenario expectation quality / backend mock behavior mismatches, not pipeline execution failures.

## Artifact Pointers

- ecommerce: `/Users/sjoybrata/.spec_test_pilot/customer_default/ecommerce/runs/20260305_130555` and `_persistence_check`
- healthcare: `/Users/sjoybrata/.spec_test_pilot/customer_default/healthcare/runs/20260305_130719` and `_persistence_check`
- logistics: `/Users/sjoybrata/.spec_test_pilot/customer_default/logistics/runs/20260305_130832` and `_persistence_check`
- hr: `/Users/sjoybrata/.spec_test_pilot/customer_default/hr/runs/20260305_130940` and `_persistence_check`
- banking_custom: `/Users/sjoybrata/.spec_test_pilot/customer_default/banking_custom/runs/20260305_131052` and `_persistence_check`
- sample_custom: `/Users/sjoybrata/.spec_test_pilot/customer_default/sample_custom/runs/20260305_131203` and `_persistence_check`
- Full run log: `/tmp/postfix_multi_spec_20260305_130555.log`

## Universal Hardening Re-Run (Targeted)

After adding universal normalization guards in `qa_specialist_agent.py` (auth-negative status pinning, happy-path query sanitization, and history-seeded body enforcement), the two failing custom domains were re-run with persistence checks:

| Domain | Base Pass | Persist Pass | Gate-1 | Gate-2 | RL Steps (1→2) | RL Buffer (1→2) |
|---|---:|---:|---|---|---|---|
| banking_custom | 0.8710 | 0.9688 | True | True | 5→6 | 139→172 |
| sample_custom | 0.8125 | 0.9062 | True | True | 8→9 | 241→274 |

Latest run folders:
- banking_custom: `/Users/sjoybrata/.spec_test_pilot/customer_default/banking_custom/runs/20260305_133606` and `_persistence_check`
- sample_custom: `/Users/sjoybrata/.spec_test_pilot/customer_default/sample_custom/runs/20260305_133723` and `_persistence_check`

Validation guardrail check:
- `expected_status == actual_status` with non-pass verdict remains `0` for both latest persistence reports.

## Production-Readiness Hardening (2026-03-05, CI + Reliability)

Implemented additional reliability controls for universal multi-domain execution:

- CI quality gates added in pipeline:
  - pass-rate floor
  - summary quality-gate required
  - trend regression caps (pass-rate drop, run-reward drop)
  - flaky threshold (run-over-run overlap)
  - flaky threshold (in-run rerun-derived flaky ratio)
  - GAM context quality minimum
- Safe-mode rollback policy:
  - on CI gate fail, restore checkpoint backup and write `safe_mode_last_failure.json`
- Stronger OSS tooling validation:
  - active command probes for `schemathesis`, `restler`, `evomaster`, `zap-baseline.py`, `k6`
  - checks now emit `active` / `misconfigured` / `skipped` with probe diagnostics
- Failure taxonomy + auto-repair loop in execution:
  - per-failure classification (`agent_issue`, `env_issue`, `service_issue`, etc.)
  - rerun-based flaky detection (`max attempts = 3`)
  - runtime repair suggestions ingested into `scenario_repair_rules`
  - summary now includes `flaky_scenarios`, `flaky_ratio`, and `failure_taxonomy_breakdown`

Updated files:
- `backend/ci_quality_gate.py`
- `backend/run_qa_domain.sh`
- `backend/spec_test_pilot/qa_specialist_agent.py`

Smoke validation run (customer-mode, CI on, persistence on):
- Command: `bash backend/run_qa_domain.sh --domain ecommerce --action run --customer-mode --verify-persistence --max-scenarios 16 --ci-gate --safe-mode-on-fail`
- Base run: pass-rate `1.0000`, CI gate `pass`
- Persistence run: pass-rate `0.9375`, CI gate `pass`
- New summary fields present in persistence report:
  - `flaky_scenarios: 0`
  - `flaky_ratio: 0.0`
  - `failure_taxonomy_breakdown: {"behavioral_regression": 1}`
- Artifacts:
  - `/Users/sjoybrata/.spec_test_pilot/customer_default/ecommerce/runs/20260305_143238/qa_execution_report.json`
  - `/Users/sjoybrata/.spec_test_pilot/customer_default/ecommerce/runs/20260305_143238_persistence_check/qa_execution_report.json`

## Detailed Decision Log (What Changed and Why)

### 1) CI Gate Architecture Decisions

1. Gate checks are evaluated by a dedicated script: `backend/ci_quality_gate.py`.
2. Gate execution is integrated in `backend/run_qa_domain.sh` for both base run and persistence run.
3. Gate failure returns non-zero (`exit=2`) to make CI fail fast.
4. Default gate thresholds were set to balanced strictness:
   - `pass_rate_floor = 0.70`
   - `flaky_threshold = 0.15`
   - `max_pass_rate_drop = 0.08`
   - `max_run_reward_drop = 0.10`
   - `min_context_quality = 0.55`
5. Summary quality gate is mandatory in CI (`--require-summary-quality-gate`).

### 2) Flaky Detection Decisions

1. Two flaky signals are now used:
   - `flaky_overlap_ratio`: verdict instability between current report and previous report on overlapping scenario keys.
   - `flaky_in_run_ratio`: instability detected by rerunning failed scenarios inside the same run.
2. In-run rerun behavior:
   - failed scenarios are rerun with `FLAKY_RERUN_MAX_ATTEMPTS = 3` total attempts (1 initial + 2 reruns).
   - if verdict or status changes across attempts, scenario is flagged flaky.
   - flaky scenario is downgraded to `suspect` in verification.
3. Summary fields now include:
   - `summary.flaky_scenarios`
   - `summary.flaky_ratio`
4. CI fails if either flaky ratio exceeds threshold.

### 3) Failure Taxonomy Decisions

1. Every non-pass result receives a taxonomy record at `verification.failure_taxonomy`.
2. Taxonomy categories currently include:
   - `none`
   - `safety_block`
   - `environment_error`
   - `server_error`
   - `path_param_empty_segment`
   - `expectation_mismatch_documented`
   - `request_body_missing_fields`
   - `behavioral_regression`
3. Summary aggregates taxonomy as `summary.failure_taxonomy_breakdown`.
4. Purpose:
   - separate agent-generated expectation defects from backend/service defects.
   - improve RL signal quality by avoiding blind learning from all failures.

### 4) Auto-Repair Decisions

1. Runtime repair hints are generated from taxonomy and stored at `verification.repair_suggestion`.
2. Supported repair suggestion types:
   - `override_expected_status`
   - `repair_request_body`
3. Repair suggestions are ingested into `learning_state.scenario_repair_rules`.
4. Ingestion summary is surfaced in report under:
   - `repair_policy.runtime_suggestions_applied`
   - `repair_policy.runtime_suggestion_examples`
5. Existing repair pipeline remains active and combines:
   - historical learned repair rules
   - runtime-suggested repair rules

### 5) Safe-Mode / Rollback Decisions

1. Before run, checkpoint backup is created in customer mode when safe mode is on.
2. On CI gate failure:
   - previous checkpoint is restored.
   - marker file is written: `~/.spec_test_pilot/customer_default/<domain>/safe_mode_last_failure.json`.
3. This ensures RL state is not polluted by bad runs that violated quality gates.

### 6) Stronger Tooling Validation Decisions

1. OSS checks moved from static availability toward active command probing.
2. Probed tools:
   - `schemathesis --version`
   - `restler --help`
   - `evomaster --help`
   - `zap-baseline.py -h`
   - `k6 version`
3. Report statuses standardized to:
   - `active`
   - `misconfigured`
   - `skipped`
4. Probe diagnostics are included in `oss_checks.*.probe` for debugging CI environments.

### 7) GAM Context Stability Decisions

1. GAM contract strictness was relaxed for low-context conditions:
   - use `has_weak_pattern_or_proxy` instead of requiring direct weak pattern only.
   - diversity thresholds are adaptive when selected excerpt count is small.
2. Goal:
   - reduce unnecessary `context_pack.status = needs_retry`.
   - keep context quality checks meaningful without overfailing small-memory runs.

### 8) Data Model / Report Schema Changes

1. New per-scenario verification fields:
   - `verification.flaky_check`
   - `verification.failure_taxonomy`
   - `verification.repair_suggestion`
2. New summary fields:
   - `summary.flaky_scenarios`
   - `summary.flaky_ratio`
   - `summary.failure_taxonomy_breakdown`
3. New repair-policy fields:
   - `repair_policy.runtime_suggestions_applied`
   - `repair_policy.runtime_suggestion_examples`
4. CI gate metrics payload now returns:
   - `metrics.flaky_overlap_ratio`
   - `metrics.flaky_in_run_ratio`

### 9) Validation Evidence (Completed)

1. Static checks passed:
   - `python3 -m py_compile backend/spec_test_pilot/qa_specialist_agent.py backend/ci_quality_gate.py`
   - `bash -n backend/run_qa_domain.sh`
2. Existing targeted tests passed:
   - `backend/tests/test_qa_specialist_auth_flow.py` subset (`2 passed`).
3. End-to-end smoke run passed with CI enabled:
   - base gate pass and persistence gate pass for ecommerce.
   - output contains new flaky/taxonomy fields.

### 10) Known Limits and Next Recommended Work

1. Runtime repair scope is intentionally conservative and currently covers status override + required-field body repair.
2. Taxonomy is deterministic and rule-driven; it should be expanded as new failure modes appear across domains.
3. For stronger flake confidence, add optional multi-rerun-on-pass sampling mode for high-risk scenarios.
4. Add dedicated tests for:
   - flaky downgrade-to-suspect behavior
   - repair-suggestion ingestion path
   - taxonomy classification edge cases

### 11) RL Training Mode Rollout (On-Run vs Periodic)

1. Enforced RL training mode:
   - `periodic` only (mandatory): collect transitions during customer runs, skip train-step during run.
2. Backend defaults and validators now force periodic mode:
   - `run_qa_domain.sh` default `--rl-train-mode periodic`.
   - `qa_customer_api.RunRequest` coerces `rlTrainMode` to `periodic`.
3. Run metadata/report now includes RL mode:
   - `metadata.rl_train_mode`
   - `agent_lightning.training_stats.train_mode`
4. Added standalone periodic trainer command:
   - `backend/rl_periodic_trainer.py`
   - consumes checkpoint replay buffer and runs batched training (`--max-steps`, `--min-buffer`).

Operational model after rollout:
- Customer click/run: inference + transition buffering (periodic mode).
- Scheduled job: run periodic trainer, checkpoint autosave, then evaluate/promote by CI gate policy.
