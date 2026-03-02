#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${ROOT_DIR}/customer-ui-next"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "Missing Next.js app directory: ${APP_DIR}" >&2
  exit 1
fi

cd "${APP_DIR}"

if [[ ! -x "node_modules/.bin/next" ]]; then
  echo "[SETUP] Installing Next.js UI dependencies..."
  npm install
fi

if [[ ! -x "node_modules/.bin/next" ]]; then
  echo "[ERROR] Next.js binary not found after install." >&2
  echo "Try running manually:" >&2
  echo "  cd frontend/customer-ui-next && npm install && npx next --version" >&2
  exit 1
fi

is_port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

pick_port() {
  local preferred="$1"
  if ! is_port_in_use "${preferred}"; then
    echo "${preferred}"
    return 0
  fi
  local p
  for p in 3002 3003 3004 3005 3006 3007 3008 3009 3010; do
    if ! is_port_in_use "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

UI_PORT="${UI_PORT:-3001}"
if ! SELECTED_PORT="$(pick_port "${UI_PORT}")"; then
  echo "[ERROR] No available port found in range 3001-3010." >&2
  echo "Free a port or run with explicit UI_PORT, e.g.:" >&2
  echo "  UI_PORT=3011 ./frontend/run_customer_ui_next.sh" >&2
  exit 1
fi

if [[ "${SELECTED_PORT}" != "${UI_PORT}" ]]; then
  echo "[WARN] Port ${UI_PORT} is busy. Using ${SELECTED_PORT}."
fi

echo "[RUN] Starting Next.js customer UI at http://localhost:${SELECTED_PORT}"
exec npm run dev -- -p "${SELECTED_PORT}"
