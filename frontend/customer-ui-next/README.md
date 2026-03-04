# Customer UI (Next.js)

Next.js application for running QA jobs and reviewing outputs from `SpecForge`.

## Core Capabilities

1. Trigger QA runs for one or multiple domains
2. Select output script kind (`python_pytest`, `javascript_jest`, `curl_script`, `java_restassured`)
3. Stream run progress via SSE
4. Inspect JSON/Markdown reports
5. Browse generated test scripts
6. Review RL/GAM diagnostics panels

## Local Development

1. Install dependencies.

```bash
cd frontend/customer-ui-next
npm install
```

2. Start dev server.

```bash
npm run dev
```

3. Open UI.

```text
http://localhost:3001
```

## Split Mode (FastAPI Backend + Next UI)

Terminal 1:

```bash
./backend/run_customer_backend_fastapi.sh
```

Terminal 2:

```bash
NEXT_PUBLIC_BACKEND_BASE_URL=http://127.0.0.1:8787 npm run dev -- -p 3001
```

Wrapper command:

```bash
QA_UI_MODE=split ./frontend/run_customer_ui_next.sh
```

## Full Next Mode

```bash
QA_UI_MODE=full_next ./frontend/run_customer_ui_next.sh
```

## API Endpoints Used by UI

1. `POST /api/jobs`
2. `GET /api/jobs`
3. `GET /api/jobs/{jobId}`
4. `GET /api/jobs/{jobId}/events`
5. `GET /api/jobs/{jobId}/report/{domain}`
6. `GET /api/jobs/{jobId}/generated-tests/{domain}`
7. `GET /api/jobs/{jobId}/generated-tests/{domain}/{kind}`

## Operational Notes

1. UI stores temporary run artifacts under `/tmp/qa_ui_next_runs`
2. UI stores checkpoints under `/tmp/qa_ui_next_checkpoints`
3. SSE stream is used for live updates with polling fallback
