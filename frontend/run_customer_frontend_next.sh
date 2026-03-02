#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${ROOT_DIR}/customer-ui-next"
UI_PORT="${UI_PORT:-3001}"
BACKEND_BASE_URL="${NEXT_PUBLIC_BACKEND_BASE_URL:-http://127.0.0.1:8787}"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "Missing Next.js app directory: ${APP_DIR}" >&2
  exit 1
fi

cd "${APP_DIR}"

if [[ ! -x "node_modules/.bin/next" ]]; then
  echo "[SETUP] Installing frontend dependencies..."
  npm install
fi

if [[ ! -x "node_modules/.bin/next" ]]; then
  echo "[ERROR] Next.js binary not found after install." >&2
  exit 1
fi

echo "[RUN] Starting Next.js frontend at http://localhost:${UI_PORT}"
echo "[CFG] NEXT_PUBLIC_BACKEND_BASE_URL=${BACKEND_BASE_URL}"

NEXT_PUBLIC_BACKEND_BASE_URL="${BACKEND_BASE_URL}" exec npm run dev -- -p "${UI_PORT}"
