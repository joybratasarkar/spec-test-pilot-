# QA Specialist Agent Architecture

This document describes how the current codebase executes the QA agent and where each architecture step lives in code.

## 1. Architecture Overview

The runtime is a single orchestrated pipeline inside `QASpecialistAgent.run()`.

```text
CLI (qa_specialist_runner.py)
  -> QASpecialistAgent.run()
      -> OpenAPI load + auth map
      -> GAM session + research
      -> Scenario generation
      -> Adaptive selection (contextual linear-UCB + RL risk)
      -> Scenario repair rules
      -> Isolated execution on dynamic mock server
      -> Summary + decision rewards
      -> Learning-state update
      -> Agent Lightning V2 training
      -> Report persistence (json + md)
```

## 2. Main Components

1. Control entrypoint:
- `qa_specialist_runner.py`
- delegates to `spec_test_pilot/qa_specialist_agent.py:main`

2. Runtime orchestrator:
- `spec_test_pilot/qa_specialist_agent.py:QASpecialistAgent.run`

3. Adaptive selection policy:
- `spec_test_pilot/adaptive_policy.py:AdaptiveScenarioPolicy`

4. RL trainer implementation used by QA pipeline:
- `spec_test_pilot/agent_lightning_v2.py:AgentLightningTrainer`
- with `ObservabilityCollector`, `CreditAssignmentModule`, `LightningRLAlgorithm`

5. Isolated execution backend:
- `agent_lightning_server.py:DynamicMockServer`
- used via FastAPI `TestClient`

## 3. Runtime Sequence With Code Mapping

1. Load spec and build auth requirement map.
- `_load_spec`
- `_build_auth_requirement_map`

2. Start GAM session and research context.
- `gam.start_session`
- `gam.research`

3. Generate scenarios from simulator.
- `HumanTesterSimulator.think_like_tester`

4. Select scenarios using learned policy.
- `_select_scenarios_with_learning`
- policy score from `AdaptiveScenarioPolicy.score`

5. Apply learned repair rules.
- `_apply_scenario_repairs`
- rules refreshed by `_refresh_scenario_repair_rules`

6. Execute selected scenarios in isolated mock runtime.
- `_execute_in_isolated_mock`
- `_execute_one_scenario`

7. Build summary and learning feedback.
- `_build_summary`
- `_compute_learning_feedback`

8. Update persisted learning state.
- `_update_learning_state`
- `_persist_learning_state`

9. Trigger RL training.
- `_run_agent_lightning_training`
- calls `self.rl_trainer.train_agent(...)`

10. Write reports.
- `_write_reports`
- `qa_execution_report.json` + `qa_execution_report.md`

## 4. Step Input/Output Table

| Step | Inputs | Outputs | Persisted |
|---|---|---|---|
| Spec load | `--spec` path | parsed OpenAPI dict | no |
| GAM research | spec metadata, tenant | memory excerpts, plan/reflection | GAM pages/memo |
| Scenario generation | spec + effective prompt | candidate scenarios | no |
| Adaptive selection | candidates + policy state + RL risk | selected scenarios + selection trace | `learning_state.json` |
| Repair pass | selected scenarios + stats | repaired scenarios | repair rules in state |
| Isolated execution | repaired scenarios + copied spec | scenario execution results | included in report |
| Summary/reward | results + thresholds | run summary + decision rewards | included in report |
| Learning update | decision signals | updated weights, stats, policy posterior | `learning_state.json` |
| RL training | task payload + decision signals | training result/stats | RL checkpoint file |
| Reporting | all run outputs | json/md report files | report files |

## 5. Observed Runtime Flow (From Logs)

The following markers are emitted during real runs:

1. `[OK] OpenAPI spec written: ...`
2. `[RUN] QA specialist agent`
3. `Started observability session ... for agent qa_specialist`
4. `RL training executed: Loss=...`
5. `RL TRAINING ACTIVE: Step N, Loss=...`
6. `QA specialist run complete`
7. `JSON report: ...`
8. `[OK] Run completed.`

This sequence was verified across `ecommerce`, `healthcare`, `logistics`, and `hr` runs on March 2, 2026.

## 6. Architecture Clarification: "Official" vs "Current"

1. Current production path (`qa_specialist_runner.py`) uses `agent_lightning_v2.py`.
2. Official package path (`official_agent_lightning_runner.py`) uses `spec_test_pilot/agent_lightning_official.py` and `agentlightning` package APIs.
3. The two paths are separate; running `qa_specialist_runner.py` does not use the official package trainer.

## 7. Verified Multi-Domain Outcomes (March 2, 2026)

| Domain | Total | Passed | Failed | Pass Rate | Gate | RL Steps | RL Buffer |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecommerce | 16 | 14 | 2 | 0.8750 | true | 1 | 17 |
| healthcare | 14 | 13 | 1 | 0.9286 | true | 1 | 15 |
| logistics | 13 | 11 | 2 | 0.8462 | true | 1 | 14 |
| hr | 13 | 12 | 1 | 0.9231 | true | 1 | 14 |

Ecommerce rerun with same checkpoint:

- RL steps: `1 -> 2`
- RL buffer: `17 -> 34`

This confirms learning state continuation through checkpoint load/save.
