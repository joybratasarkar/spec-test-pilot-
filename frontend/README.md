# Frontend

This folder is the frontend entrypoint layer for the customer UI.

It wraps the existing Next.js app in:

1. `customer-ui-next/`

You can run frontend in two modes:

1. Split mode (recommended): Next.js frontend + FastAPI backend
2. Full Next mode: Next.js with local API routes

## Split Mode (recommended)

1. Start backend in another terminal:
```bash
./backend/start-backend.sh
```

2. Start frontend:
```bash
./frontend/start-frontend.sh
```

3. Open:
```text
http://localhost:3001
```

In this mode, frontend calls FastAPI directly (`http://127.0.0.1:8787` by default).

## Full Next Mode

```bash
./frontend/start-full-next.sh
```

This runs Next.js UI with its own Node API routes.

## What customer can do in UI

1. Select domains and run QA jobs
2. View live progress/logs
3. View JSON/Markdown reports
4. View generated test scripts (pytest/jest/curl/restassured)
5. Inspect RL counters and selection diagnostics
