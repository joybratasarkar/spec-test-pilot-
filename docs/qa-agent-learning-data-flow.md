# QA Agent Learning Data Flow

This is the glass-box learning document for the current QA pipeline.

Scope:

1. What data is produced at each stage.
2. What is passed into RL training.
3. What is persisted.
4. How to verify re-learning across runs.

## 1. Learning Loop Entry Points

1. Run entry:
- `qa_specialist_runner.py`
- `spec_test_pilot/qa_specialist_agent.py:main`

2. Core orchestrator:
- `QASpecialistAgent.run()`

3. RL training invocation:
- `QASpecialistAgent._run_agent_lightning_training(...)`

4. RL trainer:
- `spec_test_pilot/agent_lightning_v2.py:AgentLightningTrainer.train_agent(...)`

## 2. Data Flow: End-to-End

```text
OpenAPI spec
  -> Scenario candidates
  -> Adaptive selection trace
  -> Executed scenario results
  -> Summary metrics
  -> Decision-level rewards
  -> Learning-state update
  -> RL task payload (run + decision signals)
  -> RL transitions + replay buffer
  -> RL train step + checkpoint save
  -> Report json/md + learning_state.json
```

## 3. Stage-by-Stage Payloads

## 3.1 Candidate Scenario Input to Selection

Each candidate (from `HumanTesterSimulator`) contributes:

1. `name`
2. `test_type`
3. `method`
4. `endpoint`
5. `expected_status`
6. `headers`, `params`, `body`

Used by `_select_scenarios_with_learning(...)`.

## 3.2 Adaptive Selection Scoring Payload

`AdaptiveScenarioPolicy.score(...)` consumes:

1. `test_type`, `method`, `endpoint`
2. `expected_status`
3. `has_body`, `has_params`
4. `rl_risk`
5. `novelty_bonus`
6. `legacy_weight_bonus`
7. `diversity_penalty`

Outputs score decomposition:

1. `score`
2. `expected_reward`
3. `uncertainty`
4. `exploration_bonus`
5. `failure_focus_bonus`
6. `historical_reward`

Persisted in report under `selection_policy.top_decisions`.

## 3.3 Execution Result Payload

Per scenario result (`ScenarioExecutionResult`):

1. `name`, `test_type`, `method`
2. `endpoint_template`, `endpoint_resolved`
3. `expected_status`, `actual_status`
4. `passed`
5. `duration_ms`
6. `error`, `response_excerpt`

## 3.4 Learning Feedback Payload

Produced by `_compute_learning_feedback(...)`:

1. `run_reward`
2. `reward_breakdown`
3. `average_decision_reward`
4. `rewarded_decisions`
5. `penalized_decisions`
6. `decision_signals[]`

Each `decision_signals[]` item includes:

1. `name`, `test_type`, `method`
2. `endpoint_template`, `endpoint_key`
3. `scenario_fingerprint`
4. `has_body`, `has_params`
5. `reward`
6. `passed`
7. `expected_status`, `actual_status`

## 3.5 RL Task Payload

Built in `_run_agent_lightning_training(...)` and passed to `train_agent`:

```json
{
  "spec_title": "...",
  "tenant_id": "...",
  "pass_rate": 0.875,
  "pass_threshold": 0.7,
  "total_scenarios": 16,
  "failed_scenarios": 2,
  "report_path": ".../qa_execution_report.json",
  "summary": {"...": "..."},
  "learning_reward_score": 0.93,
  "decision_signals": [{"...": "..."}]
}
```

## 3.6 RL Internal Transition Construction

In `AgentLightningTrainer._process_training_data(...)`:

1. start/action trace is recorded.
2. one trace per decision signal is recorded (`type=scenario_decision`).
3. end/observation trace is recorded.
4. credit assignment computes per-trace rewards.
5. transitions are created from consecutive traces.
6. replay buffer stores transitions.
7. `train_step()` updates value model.
8. checkpoint autosave writes model/optimizer/replay metadata.

## 4. Persistence Surfaces

## 4.1 `learning_state.json`

Contains:

1. `run_count`
2. `test_type_weights`
3. `endpoint_weights`
4. `scenario_stats`
5. `decision_history`
6. `adaptive_policy` (A/b matrices + config)
7. `selection_trace`
8. `selection_summary`
9. `scenario_repair_rules`

## 4.2 RL Checkpoint

Configured by `--rl-checkpoint`.

Contains (through `LightningRLAlgorithm.save_checkpoint`):

1. training step count
2. model state
3. optimizer state
4. replay buffer payload (optionally truncated)

## 4.3 Run Report (`qa_execution_report.json`)

Key sections:

1. `summary`
2. `selection_policy`
3. `learning`
4. `agent_lightning`
5. `repair_policy`
6. `scenario_results`
7. `generated_test_files`
8. `report_files`

## 5. Verified Learning Behavior (March 2, 2026)

Fresh multi-domain runs produced:

| Domain | Pass Rate | RL Steps | RL Buffer |
|---|---:|---:|---:|
| ecommerce | 0.8750 | 1 | 17 |
| healthcare | 0.9286 | 1 | 15 |
| logistics | 0.8462 | 1 | 14 |
| hr | 0.9231 | 1 | 14 |

Rerun ecommerce with same checkpoint:

1. RL steps increased from `1` to `2`.
2. RL buffer increased from `17` to `34`.
3. Pass rate remained `0.8750` (learning persisted even when metric did not immediately improve).

## 6. Why Pass Rate May Not Increase Every Run

1. Small scenario budgets (for example 16) cause high variance.
2. Selection includes uncertainty exploration, not pure exploitation.
3. Endpoint behavior in mock runtime may cap achievable pass rates for specific scenario classes.
4. RL improves policy/value estimates over many runs, not guaranteed monotonic single-run pass-rate gain.

## 7. Exact Verification Commands

1. Run one domain:
```bash
./backend/run_qa_domain.sh --domain ecommerce --action both --max-scenarios 16 --rl-checkpoint /tmp/qa_docs_ecommerce.pt
```

2. Rerun same checkpoint:
```bash
./backend/run_qa_domain.sh --domain ecommerce --action run --spec-path /tmp/openapi_ecommerce.yaml --max-scenarios 16 --rl-checkpoint /tmp/qa_docs_ecommerce.pt
```

3. Inspect RL counters from report:
```bash
jq '.agent_lightning.training_stats.rl_training_steps, .agent_lightning.training_stats.rl_buffer_size, .learning.agent_lightning_checkpoint' /tmp/<run_dir>/qa_execution_report.json
```
