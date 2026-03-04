# SpecForge

Open-source QA agent platform for OpenAPI-driven testing with:

1. Scenario generation (LLM + heuristics fallback)
2. GAM memory/research context enrichment
3. RL-guided mutation and scenario selection
4. Isolated API execution and verification
5. Structured reporting and reproducible failure artifacts

## License

Dual licensed under:

1. MIT (`LICENSE-MIT`)
2. Apache-2.0 (`LICENSE-APACHE`)

You may use this project under either license.

## Quick Start

1. Create backend virtual environment and install dependencies.

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

2. Start backend API.

```bash
./backend/start-backend.sh
```

3. Start frontend UI in a new terminal.

```bash
./frontend/start-frontend.sh
```

4. Open the UI.

```text
http://localhost:3001
```

## Common Run Commands

1. Run a QA domain execution.

```bash
./backend/run_qa_domain.sh --domain ecommerce --customer-mode --verify-persistence
```

2. Run backend customer API server.

```bash
./backend/run_customer_backend_fastapi.sh
```

3. Run frontend in full Next.js mode.

```bash
./frontend/start-full-next.sh
```

## Documentation

Start here:

1. `docs/agent-architecture-flow-in-depth.md`
2. `docs/qa-agent-runtime-step-map.md`
3. `docs/qa-agent-learning-data-flow.md`
4. `docs/qa-agent-glass-box-deep-dive.md`

Full docs index: `docs/README.md`

## Repository Layout

1. `backend/`: FastAPI services, QA runtime, RL/GAM integration, tests
2. `frontend/`: Next.js customer UI and startup wrappers
3. `docs/`: architecture, runtime, data flow, and API-flow docs
4. `data/`: local runtime/test assets

## Development

1. Backend developer guide: `backend/README.md`
2. Frontend developer guide: `frontend/README.md`
3. Contribution guide: `CONTRIBUTING.md`
4. Security policy: `SECURITY.md`
5. Code of conduct: `CODE_OF_CONDUCT.md`

## Notes

1. Some components are research-oriented and evolve rapidly.
2. Validate security/compliance requirements before production deployment.
