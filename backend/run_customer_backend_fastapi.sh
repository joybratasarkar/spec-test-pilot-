#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8787}"
build_default_allowed_origins() {
  local origins=()
  local p
  for p in 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010; do
    origins+=("http://localhost:${p}")
    origins+=("http://127.0.0.1:${p}")
  done
  local IFS=,
  echo "${origins[*]}"
}

DEFAULT_ALLOWED_ORIGINS="$(build_default_allowed_origins)"
ALLOWED_ORIGINS="${QA_UI_ALLOWED_ORIGINS:-${DEFAULT_ALLOWED_ORIGINS}}"
BACKEND_RELOAD="${BACKEND_RELOAD:-0}"
GAM_LLM_MODE="on"
GAM_MEMO_LLM_MODE="on"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# RL training is periodic-only in customer mode. Keep it enabled on restart.
QA_RL_PERIODIC_ENABLED="1"
QA_RL_PERIODIC_INTERVAL_SEC="${QA_RL_PERIODIC_INTERVAL_SEC:-300}"
QA_RL_PERIODIC_MAX_STEPS="${QA_RL_PERIODIC_MAX_STEPS:-25}"
QA_RL_PERIODIC_MIN_BUFFER="${QA_RL_PERIODIC_MIN_BUFFER:-32}"

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
echo "[CFG] GAM_LLM_MODE=${GAM_LLM_MODE} (enforced)"
echo "[CFG] GAM_MEMO_LLM_MODE=${GAM_MEMO_LLM_MODE} (enforced)"
echo "[CFG] QA_RL_PERIODIC_ENABLED=${QA_RL_PERIODIC_ENABLED} (enforced)"
echo "[CFG] QA_RL_PERIODIC_INTERVAL_SEC=${QA_RL_PERIODIC_INTERVAL_SEC}"
echo "[CFG] QA_RL_PERIODIC_MAX_STEPS=${QA_RL_PERIODIC_MAX_STEPS}"
echo "[CFG] QA_RL_PERIODIC_MIN_BUFFER=${QA_RL_PERIODIC_MIN_BUFFER}"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "[CFG] OPENAI_API_KEY=loaded_from_env"
else
  echo "[CFG] OPENAI_API_KEY=missing"
fi

export QA_UI_ALLOWED_ORIGINS="${ALLOWED_ORIGINS}"
export GAM_LLM_MODE="${GAM_LLM_MODE}"
export GAM_MEMO_LLM_MODE="${GAM_MEMO_LLM_MODE}"
export QA_RL_PERIODIC_ENABLED="${QA_RL_PERIODIC_ENABLED}"
export QA_RL_PERIODIC_INTERVAL_SEC="${QA_RL_PERIODIC_INTERVAL_SEC}"
export QA_RL_PERIODIC_MAX_STEPS="${QA_RL_PERIODIC_MAX_STEPS}"
export QA_RL_PERIODIC_MIN_BUFFER="${QA_RL_PERIODIC_MIN_BUFFER}"
if [[ "${BACKEND_RELOAD}" == "1" ]]; then
  exec "${PYTHON_BIN}" -m uvicorn qa_customer_api:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --reload
else
  exec "${PYTHON_BIN}" -m uvicorn qa_customer_api:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
fi
