# Reinforcement QA Agent

This project is organized into two customer-facing entry folders:

1. `backend/` - FastAPI backend and QA agent runtime
2. `frontend/` - Next.js frontend (customer UI)

The code is physically separated:

1. backend Python runtime and agent code live under `backend/`
2. frontend Next.js application lives under `frontend/customer-ui-next/`

## Folder Guide

1. Backend guide: `backend/README.md`
2. Frontend guide: `frontend/README.md`
3. Technical deep docs: `docs/README.md`

## Quick Start (split FE/BE)

1. Create backend environment (first time):
```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

2. Start backend:
```bash
./backend/start-backend.sh
```

3. Start frontend (new terminal):
```bash
./frontend/start-frontend.sh
```

4. Open UI:
```text
http://localhost:3001
```

## One-command UI mode

If you want frontend with built-in local API routes only:

```bash
./frontend/start-full-next.sh
```
