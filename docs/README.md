# SpecTestPilot QA Agent

This repository runs an OpenAPI QA specialist agent that:

1. Parses an OpenAPI spec.
2. Generates QA scenarios.
3. Selects scenarios with an adaptive policy.
4. Executes against an isolated dynamic mock API.
5. Produces JSON/Markdown reports.
6. Learns using Agent Lightning-style RL + persisted checkpoint.

This README reflects the current code paths and verified runtime behavior as of March 2, 2026.

## Current Runtime Modes

1. Production QA pipeline (used by `qa_specialist_runner.py`):
- `spec_test_pilot/qa_specialist_agent.py`
- `spec_test_pilot/agent_lightning_v2.py`
- `spec_test_pilot/adaptive_policy.py`

2. Official Agent Lightning package integration (separate runner):
- `official_agent_lightning_runner.py`
- `spec_test_pilot/agent_lightning_official.py`

## Quick Start

1. Install dependencies:
```bash
venv/bin/pip install -r requirements.txt
```

2. Run one domain end-to-end:
```bash
./run_qa_domain.sh --domain ecommerce --action both
```

3. Re-run same domain with same checkpoint (to verify persistence):
```bash
./run_qa_domain.sh \
  --domain ecommerce \
  --action run \
  --spec-path /tmp/openapi_ecommerce.yaml \
  --rl-checkpoint /tmp/agent_lightning_ecommerce.pt
```

## Verified Multi-Domain Run (March 2, 2026)

Command pattern used:

```bash
./run_qa_domain.sh --domain <domain> --action both --max-scenarios 16 --output-dir <dir> --rl-checkpoint <checkpoint>
```

Observed outputs:

| Domain | Total | Passed | Failed | Pass Rate | Quality Gate | RL Steps | RL Buffer |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecommerce | 16 | 14 | 2 | 0.8750 | true | 1 | 17 |
| healthcare | 14 | 13 | 1 | 0.9286 | true | 1 | 15 |
| logistics | 13 | 11 | 2 | 0.8462 | true | 1 | 14 |
| hr | 13 | 12 | 1 | 0.9231 | true | 1 | 14 |

Persistence verification (same ecommerce checkpoint, second run):

- RL steps: `1 -> 2`
- RL buffer: `17 -> 34`
- Pass rate remained `0.8750`

This confirms checkpointed RL state is loaded and continued.

Run artifact locations from this verification batch:

1. ecommerce: `/tmp/qa_docs_20260302_110442_ecommerce`
2. healthcare: `/tmp/qa_docs_20260302_110442_healthcare`
3. logistics: `/tmp/qa_docs_20260302_110442_logistics`
4. hr: `/tmp/qa_docs_20260302_110442_hr`
5. ecommerce rerun: `/tmp/qa_docs_20260302_110442_ecommerce_rerun`

Associated logs:

1. `/tmp/qa_docs_20260302_110442_ecommerce.run.log`
2. `/tmp/qa_docs_20260302_110442_healthcare.run.log`
3. `/tmp/qa_docs_20260302_110442_logistics.run.log`
4. `/tmp/qa_docs_20260302_110442_hr.run.log`
5. `/tmp/qa_docs_20260302_110442_ecommerce_rerun.run.log`

## Runtime Artifacts

Each run writes:

1. `qa_execution_report.json`
2. `qa_execution_report.md`
3. `learning_state.json`
4. `openapi_under_test.yaml`
5. `generated_tests/` (Python/JS/Java/cURL)

## Official Runner Status

`official_agent_lightning_runner.py` currently fails in this local environment because APO extras are not installed in the active venv (`poml` missing).

Expected fix:

```bash
venv/bin/pip install 'agentlightning[apo]>=0.1.0'
```

If network/package index is restricted, this install must be done in an environment with package access.

## Documentation Map

1. `docs/qa-agent-architecture.md`: architecture and runtime sequence.
2. `docs/qa-agent-learning-data-flow.md`: learning payloads and RL flow in depth.
3. `docs/project-file-organization.md`: file map and execution entrypoints.
