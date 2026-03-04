# Contributing

Thanks for contributing to `SpecForge`.

## Development Setup

1. Create virtual environment and install dependencies.

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

2. Run backend tests.

```bash
backend/.venv/bin/pytest backend/tests -q
```

3. Run frontend locally when needed.

```bash
./frontend/start-frontend.sh
```

## Pull Request Guidelines

1. Keep changes scoped and documented.
2. Include tests for behavior changes.
3. Update docs for any public command/API/architecture changes.
4. Use clear commit messages.

## Code Style

1. Prefer readable, explicit logic over clever shortcuts.
2. Keep functions cohesive and testable.
3. Avoid introducing breaking CLI/API changes without documentation.

## Reporting Bugs

When filing issues, include:

1. exact command executed
2. environment info (OS, Python, Node)
3. relevant logs and report snippets
4. repro steps
