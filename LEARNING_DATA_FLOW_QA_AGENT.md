# QA Agent Learning Data Flow (Glass Box Guide)

This document explains, in code-level detail, how the QA specialist agent learns over runs.
It focuses on:

1. What data goes in at each step.
2. What transformations happen.
3. What data is persisted.
4. How Agent Lightning RL is fed and trained.
5. How to verify learning from reports and state files.

Primary code paths:

1. `spec_test_pilot/qa_specialist_agent.py`
2. `spec_test_pilot/adaptive_policy.py`
3. `spec_test_pilot/agent_lightning_v2.py`
4. `agent_lightning_server.py`

## 1) End-to-End Runtime Learning Loop

Entry point:

1. CLI: `qa_specialist_runner.py`
2. Core orchestrator: `QASpecialistAgent.run()`

High-level sequence:

1. Load OpenAPI spec.
2. Build auth requirement map from spec.
3. Start GAM session and research.
4. Generate candidate scenarios.
5. Select scenarios with adaptive policy + uncertainty coverage.
6. Apply learned repair rules to selected scenarios.
7. Execute selected scenarios in isolated in-memory mock server.
8. Build summary and compute learning feedback.
9. Update learning state (weights, scenario stats, repair rules).
10. Send run + decision signals to Agent Lightning trainer.
11. Trainer collects traces, creates transitions, trains RL, autosaves checkpoint.
12. Persist report and updated learning state.

## 2) Main Data Objects

### 2.1 Generated Scenario (`TestScenario`)

Source class: `spec_test_pilot/multi_language_tester.py`

Important fields used downstream:

1. `name`
2. `test_type`
3. `endpoint`
4. `method`
5. `headers`
6. `params`
7. `body`
8. `expected_status`

### 2.2 Execution Result (`ScenarioExecutionResult`)

Defined in `qa_specialist_agent.py`.

Per executed scenario, agent records:

1. `name`, `test_type`, `method`
2. `endpoint_template`, `endpoint_resolved`
3. `expected_status`, `actual_status`
4. `passed`
5. `duration_ms`
6. `error`, `response_excerpt`

### 2.3 Learning Signal (`DecisionLearningSignal`)

Defined in `qa_specialist_agent.py`.

Per executed scenario, the learning signal includes:

1. Identity: `name`, `test_type`, `method`, `endpoint_template`, `endpoint_key`
2. Stability key: `scenario_fingerprint`
3. Shape flags: `has_body`, `has_params`
4. Outcome: `passed`, `expected_status`, `actual_status`
5. Reward: `reward` (continuous)

## 3) Scenario Selection Learning (Before Execution)

Selector method:

1. `QASpecialistAgent._select_scenarios_with_learning(...)`

Inputs:

1. Candidate scenarios from simulator.
2. Persisted state:
   - `test_type_weights`
   - `endpoint_weights`
   - `adaptive_policy` state (`A`, `b`, `scenario_stats`)
3. Optional RL risk estimate from value network.

Scoring components per candidate:

1. Contextual bandit score from `AdaptiveScenarioPolicy.score(...)`:
   - `expected_reward`
   - `uncertainty`
   - `exploration_bonus = alpha * uncertainty`
   - `failure_focus_bonus`
2. Extra bonuses:
   - `novelty_bonus`
   - `legacy_weight_bonus`
   - `rl_risk`
3. Diversity penalty while selecting multiple items:
   - endpoint repetition penalty
   - test-type repetition penalty

Uncertainty coverage logic:

1. Compute uncertainty for all candidates.
2. Compute threshold at quantile `UNCERTAINTY_COVERAGE_QUANTILE` (currently `0.75`).
3. Mark uncertain pool: candidates with uncertainty >= threshold.
4. Force-include uncertain pool first.
5. Effective budget:
   - `effective_budget = max(max_scenarios, uncertain_candidate_count)`
6. Then fill remaining slots by best score.

What gets recorded:

1. `selection_trace` (top decisions with score decomposition).
2. `selection_summary`:
   - `base_max_scenarios`
   - `effective_budget`
   - `budget_expanded_for_uncertainty`
   - `uncertainty_threshold`
   - `uncertain_candidate_count`
   - `uncertain_selected_count`

## 4) Learned Repair Rules (Between Selection and Execution)

Repair application method:

1. `QASpecialistAgent._apply_scenario_repairs(...)`

Rule generation method:

1. `QASpecialistAgent._refresh_scenario_repair_rules(...)`

Rule source:

1. Uses accumulated `scenario_stats` from previous runs.

Rule eligibility thresholds:

1. `attempts >= REPAIR_RULE_MIN_ATTEMPTS` (currently `3`)
2. `failure_rate >= REPAIR_RULE_MIN_FAILURE_RATE` (currently `0.85`)
3. Dominant actual status ratio >= `REPAIR_RULE_DOMINANT_RATIO` (currently `0.70`)

Rule types:

1. `override_expected_status`:
   - Used when dominant returned status is stable and mismatch is persistent.
2. `repair_request_body`:
   - Used for write happy-path scenarios (`POST/PUT/PATCH`, expected success, dominant 400).
   - Agent injects required body fields from request schema.

Repair output in report:

1. `repair_policy.active_rules`
2. `repair_policy.applied_repairs`
3. `repair_policy.status_overrides`
4. `repair_policy.request_body_repairs`
5. `repair_policy.applied_examples`

## 5) Isolated Execution and Runtime Normalization

Execution method:

1. `QASpecialistAgent._execute_in_isolated_mock(...)`
2. Per scenario call: `_execute_one_scenario(...)`

Isolation:

1. Spec is copied to run workspace as `openapi_under_test.yaml`.
2. `DynamicMockServer` is created from the copied spec.
3. Requests executed via in-memory `fastapi.testclient.TestClient`.

Auth normalization before request:

1. Build auth-required operation map from OpenAPI.
2. For auth-required operations:
   - if scenario is not auth-negative and no auth header exists, inject `Authorization: Bearer valid_token_123`
   - if scenario is auth-negative with invalid token marker, normalize to `Bearer invalid`

Status matching behavior:

1. Standard pass: `actual_status == expected_status`
2. Auth-negative compatibility:
   - if test type is `authentication` or `authorization`
   - expected in `{401, 403}` and actual in `{401, 403}`
   - treated as pass

## 6) Reward Computation (After Execution)

Method:

1. `QASpecialistAgent._compute_learning_feedback(...)`

Run-level reward (`run_reward`):

1. Inputs:
   - `pass_rate`
   - coverage ratio (`total_scenarios / detected_endpoints`, capped at 1)
   - failure ratio
   - latency penalty (derived from average duration)
2. Formula:
   - `0.55 * pass_rate`
   - `+ 0.25 * coverage_ratio`
   - `+ 0.20 * (1 - failure_ratio)`
   - `- 0.15 * latency_penalty`
3. Clamped to `[0.0, 1.0]`.

Decision-level reward (`DecisionLearningSignal.reward`):

1. Base:
   - `+1.0` if passed
   - `-1.0` if failed
2. Adjustments:
   - passed negative-status tests get bonus
   - failed positive-path tests get extra penalty
   - failed negative tests get smaller penalty
3. Latency penalty:
   - subtract up to `0.20`
4. Clamped to `[-1.5, 1.5]`.

## 7) Learning State Update and Persistence

Method:

1. `QASpecialistAgent._update_learning_state(...)`

State file:

1. `<output_dir>/learning_state.json`

Key state sections:

1. `run_count`
2. `test_type_weights`
3. `endpoint_weights`
4. `decision_history`
5. `scenario_stats`
6. `selection_trace`
7. `selection_summary`
8. `scenario_repair_rules`
9. `adaptive_policy` (matrix/vector state)

How updates happen:

1. For each decision signal:
   - update type and endpoint weights using `DECISION_LEARNING_RATE`
   - update scenario stats (`attempts`, `passes`, `failures`, `avg_reward`, `failure_rate`)
   - increment `actual_status_counts`
   - update adaptive policy posterior (`A`, `b`) through `AdaptiveScenarioPolicy.observe(...)`
2. Append run summary to `decision_history`.
3. Prune oversized sections (`decision_history`, `scenario_stats`, `scenario_repair_rules`).
4. Rebuild repair rules from latest stats.

## 8) Agent Lightning RL Integration

QA to RL handoff method:

1. `QASpecialistAgent._run_agent_lightning_training(...)`

Payload sent into trainer (`task_payload`):

1. `spec_title`
2. `tenant_id`
3. `pass_rate`
4. `pass_threshold`
5. `total_scenarios`
6. `failed_scenarios`
7. `report_path`
8. `summary`
9. `learning_reward_score`
10. `decision_signals` (full per-scenario list)

Trainer methods:

1. `AgentLightningTrainer.train_agent(...)`
2. `_collect_decision_signal_traces(...)`
3. `_process_training_data(...)`

Trace collection behavior:

1. Adds one `task_start` action trace.
2. Adds one action trace per decision signal (`type=scenario_decision`).
3. Adds one `task_result` observation trace.

Transition creation behavior:

1. For each action trace, create `TrainingTransition`.
2. Reward source:
   - default credit-assignment reward
   - overridden by `reward_signal` for `scenario_decision` traces (dense learning signal)
3. Add transitions to replay buffer via `LightningRLAlgorithm.add_transition(...)`.

Training step:

1. `LightningRLAlgorithm.train_step(...)`
2. Uses replay samples to train value net with TD target.
3. Increments `training_steps`.
4. Returns:
   - `status`
   - `value_loss`
   - `batch_size`
   - `training_steps`
   - `buffer_size`

Checkpoint persistence:

1. Auto-save via `AgentLightningTrainer.save_checkpoint(...)` when autosave is enabled.
2. Auto-load on trainer init if checkpoint exists.
3. Checkpoint includes:
   - model/optimizer state
   - replay buffer
   - training step counters

## 9) Output Files and What They Mean

Run report files:

1. `<output_dir>/qa_execution_report.json`
2. `<output_dir>/qa_execution_report.md`
3. `<output_dir>/learning_state.json`
4. `<output_dir>/openapi_under_test.yaml`

Critical report sections:

1. `summary`
2. `learning.feedback`
3. `learning.state_snapshot`
4. `selection_policy`
5. `repair_policy`
6. `agent_lightning.training_result`
7. `agent_lightning.training_stats`

## 10) Verification Commands (No Black Box)

Use these commands after each run.

### 10.1 RL persistence and growth

```bash
jq '.agent_lightning.training_stats.rl_training_steps,
    .agent_lightning.training_stats.rl_buffer_size,
    .learning.agent_lightning_checkpoint' \
  /private/tmp/qa_demo_run/qa_execution_report.json
```

### 10.2 Learning quality trend

```bash
jq '{pass_rate:.summary.pass_rate,
    failed:.summary.failed_scenarios,
    avg_decision_reward:.learning.feedback.average_decision_reward,
    penalized:.learning.feedback.penalized_decisions,
    rewarded:.learning.feedback.rewarded_decisions}' \
  /private/tmp/qa_demo_run/qa_execution_report.json
```

### 10.3 Selection transparency

```bash
jq '.selection_policy | {
  candidate_count,
  selected_count,
  base_max_scenarios,
  effective_budget,
  budget_expanded_for_uncertainty,
  uncertainty_threshold,
  uncertain_candidate_count,
  uncertain_selected_count
}' /private/tmp/qa_demo_run/qa_execution_report.json
```

### 10.4 Repair-rule behavior

```bash
jq '.repair_policy' /private/tmp/qa_demo_run/qa_execution_report.json
jq '.scenario_repair_rules' /private/tmp/qa_demo_run/learning_state.json
```

### 10.5 Persistent weak spots

```bash
jq '.learning.state_snapshot.weakest_patterns' \
  /private/tmp/qa_demo_run/qa_execution_report.json
```

## 11) Why Results Can Fluctuate Between Runs

Seeing `1.0` on one run and `0.9375` on another can still be normal when learning is active.
Reasons:

1. Uncertainty coverage intentionally keeps exploration alive.
2. Candidate scenario set and ranking can change as policy updates.
3. Repair rules activate only after enough repeated evidence.
4. Some scenario families may not appear every run.

Learning should be judged by trend, not one run:

1. RL steps/buffer should grow.
2. Average decision reward should trend upward.
3. Failing fingerprints should get focused and eventually repaired.
4. `repair_policy.active_rules` and `applied_repairs` should appear for persistent errors.

## 12) Current Learning Knobs (Tunable)

In `qa_specialist_agent.py`:

1. `UNCERTAINTY_COVERAGE_QUANTILE`
2. `DECISION_LEARNING_RATE`
3. `REPAIR_RULE_MIN_ATTEMPTS`
4. `REPAIR_RULE_MIN_FAILURE_RATE`
5. `REPAIR_RULE_DOMINANT_RATIO`

In `adaptive_policy.py`:

1. `DEFAULT_ALPHA` (exploration strength)
2. `DEFAULT_FEATURE_DIM`
3. `DEFAULT_REGULARIZATION`

In `agent_lightning_v2.py`:

1. `LightningRLAlgorithm.batch_size`
2. `learning_rate`
3. replay `buffer_size`

## 13) Practical Interpretation of "Agent Is Learning"

Treat learning as confirmed when all are true:

1. `rl_training_steps` increases run-over-run with same checkpoint.
2. `rl_buffer_size` increases run-over-run.
3. `learning_state.run_count` increases.
4. `scenario_stats` attempts/failure rates update for repeated fingerprints.
5. `repair_policy.active_rules` becomes non-zero for persistent mismatch classes.
6. Multi-run failure count trends down or stabilizes at low level.

If only steps increase but failures never improve, it means training is happening but feedback mapping is weak.
If repairs start applying and failures reduce, feedback mapping is effective.
