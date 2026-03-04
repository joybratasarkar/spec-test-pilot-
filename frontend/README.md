# Frontend Guide

This directory contains customer UI launch scripts and the Next.js app.

## Structure

1. `customer-ui-next/`: Next.js application
2. `start-frontend.sh`: split-mode launcher
3. `start-full-next.sh`: full Next.js mode launcher
4. `run_customer_ui_next.sh`: mode-aware UI launcher
5. `run_customer_frontend_next.sh`: frontend launcher for FastAPI split mode

## Modes

1. `split` mode (recommended): Next.js UI + FastAPI backend
2. `full_next` mode: Next.js UI + local Next API routes

## Split Mode

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

Alternative command:

```bash
QA_UI_MODE=split ./frontend/run_customer_ui_next.sh
```

## Full Next Mode

```bash
./frontend/start-full-next.sh
```

Alternative command:

```bash
QA_UI_MODE=full_next ./frontend/run_customer_ui_next.sh
```

## UI Capabilities

1. Trigger single or multi-domain QA jobs
2. Stream live execution logs
3. Inspect JSON/Markdown reports
4. Review generated test scripts
5. Inspect RL/GAM diagnostics and selection traces

## Related Docs

1. `frontend/customer-ui-next/README.md`
2. `docs/customer-ui-api-flow.md`
