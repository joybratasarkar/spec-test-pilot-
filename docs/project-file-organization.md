# Project File Organization

This document lists the current working file layout and which files are used at runtime.

## 1. Runtime Entry Points

1. QA specialist pipeline:
- `qa_specialist_runner.py`
- calls `spec_test_pilot/qa_specialist_agent.py:main`

2. Domain convenience runner:
- `run_qa_domain.sh`
- generates preset OpenAPI specs and runs `qa_specialist_runner.py`

3. Official Agent Lightning package runner:
- `official_agent_lightning_runner.py`
- uses `spec_test_pilot/agent_lightning_official.py`

## 2. Core Runtime Files

## 2.1 QA pipeline

1. `spec_test_pilot/qa_specialist_agent.py`
- end-to-end orchestrator
- selection, execution, learning, reporting

2. `spec_test_pilot/adaptive_policy.py`
- contextual linear-UCB scenario policy
- persisted policy state (`A`, `b`, scenario stats)

3. `spec_test_pilot/multi_language_tester.py`
- scenario generation + test artifact generation

4. `agent_lightning_server.py`
- dynamic mock API server from OpenAPI

## 2.2 RL components

1. `spec_test_pilot/agent_lightning_v2.py`
- currently used RL trainer in QA pipeline
- observability + credit assignment + replay/value training
- checkpoint save/load

2. `spec_test_pilot/agent_lightning_official.py`
- official package adapter layer
- uses `agentlightning` Trainer/APO APIs
- used by `official_agent_lightning_runner.py`

## 3. Output Artifacts (Per Run)

Each QA run output directory contains:

1. `qa_execution_report.json`
2. `qa_execution_report.md`
3. `learning_state.json`
4. `openapi_under_test.yaml`
5. `generated_tests/`

## 4. Documentation Files

1. `docs/README.md`
- project quickstart and verified multi-domain results

2. `docs/qa-agent-architecture.md`
- architecture and runtime flow

3. `docs/qa-agent-learning-data-flow.md`
- in-depth learning payload/data-flow

4. `docs/project-file-organization.md`
- this file

## 5. Notes

1. Historical root-level markdown files were consolidated into `docs/`.
2. The QA production path and official package path are both present, but only the QA path is used by `qa_specialist_runner.py`.
