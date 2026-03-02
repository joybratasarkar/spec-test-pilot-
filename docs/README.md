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

1. Production QA pipeline (used by `backend/qa_specialist_runner.py`):
- `backend/spec_test_pilot/qa_specialist_agent.py`
- `backend/spec_test_pilot/agent_lightning_v2.py`
- `backend/spec_test_pilot/adaptive_policy.py`

2. Official Agent Lightning package integration (separate runner):
- `backend/official_agent_lightning_runner.py`
- `backend/spec_test_pilot/agent_lightning_official.py`

## Quick Start

1. Install dependencies:
```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

2. Customer-facing one-command run (recommended):
```bash
./backend/run_qa_domain.sh --domain ecommerce --customer-mode --verify-persistence
```

3. Classic advanced mode (manual control):
```bash
./backend/run_qa_domain.sh \
  --domain ecommerce \
  --action both \
  --output-dir /tmp/qa_ecommerce_run \
  --rl-checkpoint /tmp/agent_lightning_ecommerce.pt
```

Customer mode behavior:

1. keeps persistent spec/checkpoint workspace per tenant+domain
2. runs end-to-end in one command
3. if `--verify-persistence` is set, auto-runs a second pass and prints RL step/buffer delta
4. no manual second command needed

## Customer Web UI

Run the button-based UI:

```bash
./backend/run_customer_backend_fastapi.sh
```

Open:

```text
http://127.0.0.1:8787
```

UI capabilities:

1. select multiple domains in one run
2. click-run full QA process
3. stream live logs from each domain execution
4. auto-check persistence (optional checkbox)
5. inspect JSON/Markdown reports directly in browser

## Next.js Customer UI (Non-Python)

If you want the customer UI in Next.js instead of Python:

1. Go to [customer-ui-next/README.md](/Users/sjoybrata/Desktop/reinforcement-agent/frontend/customer-ui-next/README.md)
2. Install and run:
```bash
cd frontend/customer-ui-next
npm install
npm run dev
```
3. Open `http://localhost:3001`
4. UI/backend connection uses Next API routes + SSE stream for live job updates
5. UI includes:
- interactive report viewer (summary metrics + scenario table + raw report)
- generated script explorer (pytest/jest/curl/restassured preview)
- Agent R&D panel (selection decisions + reward breakdown + weakest patterns)
- customer API contract panel (all endpoints and payload shape)
- runtime flow panel (step-by-step input/output mapping)

One-command launcher from repo root:

```bash
./frontend/run_customer_ui_next.sh
```

## Split Mode: FastAPI Backend + Next.js Frontend

If you want backend explicitly in FastAPI:

Terminal 1 (backend):
```bash
cd /Users/sjoybrata/Desktop/reinforcement-agent
./backend/run_customer_backend_fastapi.sh
```

Terminal 2 (frontend):
```bash
cd /Users/sjoybrata/Desktop/reinforcement-agent
NEXT_PUBLIC_BACKEND_BASE_URL=http://127.0.0.1:8787 ./frontend/run_customer_frontend_next.sh
```

Open:

```text
http://localhost:3001
```

The UI automatically switches to direct FastAPI calls when `NEXT_PUBLIC_BACKEND_BASE_URL` is set.

## Verified Multi-Domain Run (March 2, 2026)

Command pattern used:

```bash
./backend/run_qa_domain.sh --domain <domain> --action both --max-scenarios 16 --output-dir <dir> --rl-checkpoint <checkpoint>
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
backend/.venv/bin/pip install 'agentlightning[apo]>=0.1.0'
```

If network/package index is restricted, this install must be done in an environment with package access.

## Documentation Map

1. `docs/qa-agent-architecture.md`: architecture and runtime sequence.
2. `docs/qa-agent-learning-data-flow.md`: learning payloads and RL flow in depth.
3. `docs/qa-agent-runtime-step-map.md`: per-step input/output mapping with runtime diagrams.
4. `docs/project-file-organization.md`: file map and execution entrypoints.
5. `docs/customer-ui-api-flow.md`: customer UI API contract and runtime sequence.
6. `docs/qa-agent-glass-box-deep-dive.md`: full black-box transparency guide (formulas, payloads, persistence, and report field mapping).
