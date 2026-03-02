#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8787}"
ALLOWED_ORIGINS="${QA_UI_ALLOWED_ORIGINS:-http://localhost:3001,http://127.0.0.1:3001}"
BACKEND_RELOAD="${BACKEND_RELOAD:-0}"

cd "${ROOT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[RUN] Starting FastAPI backend at http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "[CFG] QA_UI_ALLOWED_ORIGINS=${ALLOWED_ORIGINS}"
echo "[CFG] BACKEND_RELOAD=${BACKEND_RELOAD}"

export QA_UI_ALLOWED_ORIGINS="${ALLOWED_ORIGINS}"
if [[ "${BACKEND_RELOAD}" == "1" ]]; then
  exec "${PYTHON_BIN}" -m uvicorn qa_customer_ui:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --reload
else
  exec "${PYTHON_BIN}" -m uvicorn qa_customer_ui:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
fi
