# Backend Guide

This directory contains the QA runtime, API services, and learning components.

## Key Components

1. `qa_customer_api.py`: customer-facing FastAPI job API
2. `qa_agent_runner.py`: domain execution entrypoint
3. `spec_test_pilot/qa_specialist_agent.py`: core orchestration pipeline
4. `spec_test_pilot/agent_lightning_v2.py`: RL training runtime
5. `spec_test_pilot/memory/gam.py`: GAM memory and research layer

## Setup

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

## Run Backend API

```bash
./backend/start-backend.sh
```

Default endpoint:

```text
http://127.0.0.1:8787
```

## Run QA Agent (CLI)

Recommended mode:

```bash
./backend/run_qa_domain.sh --domain ecommerce --customer-mode --verify-persistence
```

Advanced mode:

```bash
./backend/run_qa_domain.sh \
  --domain ecommerce \
  --action both \
  --output-dir /tmp/qa_ecommerce_run \
  --rl-checkpoint /tmp/agent_lightning_ecommerce.pt
```

## Backend API Endpoints

Base URL: `http://127.0.0.1:8787`

1. `POST /api/jobs`: create a QA job
2. `GET /api/jobs`: list jobs
3. `GET /api/jobs/{job_id}`: job snapshot
4. `GET /api/jobs/{job_id}/events`: SSE events
5. `GET /api/jobs/{job_id}/report/{domain}?format=json|md`: run reports
6. `GET /api/jobs/{job_id}/generated-tests/{domain}`: generated files
7. `GET /api/jobs/{job_id}/generated-tests/{domain}/{kind}`: script content

## Important Environment Variables

1. `BACKEND_HOST`
2. `BACKEND_PORT`
3. `BACKEND_RELOAD`
4. `QA_UI_ALLOWED_ORIGINS`
5. `OPENAI_API_KEY`
6. `QA_SCENARIO_LLM_MODE` (`auto|on|off`)
7. `QA_SCENARIO_LLM_MODEL`
8. `QA_SCENARIO_LLM_TIMEOUT_SECONDS`
9. `GAM_LLM_MODE`
10. `GAM_MEMO_LLM_MODE`
11. `GAM_OPENAI_MODEL`
12. `GAM_MAX_REFLECTIONS`

## Testing

Run backend tests:

```bash
backend/.venv/bin/pytest backend/tests -q
```

## Related Docs

1. `docs/agent-architecture-flow-in-depth.md`
2. `docs/qa-agent-learning-data-flow.md`
3. `docs/customer-ui-api-flow.md`
