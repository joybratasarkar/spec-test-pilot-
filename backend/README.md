# Backend

This folder is the backend entrypoint layer for the QA agent system.

Backend runtime code is now physically located in this folder:

1. FastAPI app: `qa_customer_ui.py`
2. Domain runner: `run_qa_domain.sh`
3. QA agent core: `spec_test_pilot/qa_specialist_agent.py`
4. RL trainer: `spec_test_pilot/agent_lightning_v2.py`

## Start FastAPI backend

```bash
./backend/start-backend.sh
```

Default:

1. host: `127.0.0.1`
2. port: `8787`

## Run one domain from backend

```bash
./backend/run-domain.sh --domain ecommerce --customer-mode --verify-persistence
```

This writes:

1. generated test scripts
2. `qa_execution_report.json`
3. `qa_execution_report.md`
4. RL checkpoint (if provided)

## Backend APIs (customer UI)

Base URL: `http://127.0.0.1:8787`

1. `POST /api/jobs` - create run job
2. `GET /api/jobs` - list jobs
3. `GET /api/jobs/{job_id}` - job snapshot
4. `GET /api/jobs/{job_id}/events` - SSE stream
5. `GET /api/jobs/{job_id}/report/{domain}?format=json|md` - reports
6. `GET /api/jobs/{job_id}/generated-tests/{domain}` - generated files
7. `GET /api/jobs/{job_id}/generated-tests/{domain}/{kind}` - script content

## Useful Environment Variables

1. `BACKEND_HOST`
2. `BACKEND_PORT`
3. `QA_UI_ALLOWED_ORIGINS`
4. `BACKEND_RELOAD` (`1` to enable uvicorn reload)
