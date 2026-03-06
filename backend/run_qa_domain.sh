#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_qa_domain.sh [options]

Generate an OpenAPI spec for a domain and run the QA specialist agent.

Options:
  --domain <name>           Domain preset: ecommerce|healthcare|logistics|hr (default: ecommerce)
  --spec-path <path>        Use existing OpenAPI spec path instead of generating one
  --action <mode>           generate|run|both (default: both)
  --tenant-id <id>          Tenant id for GAM memory (default: qa_<domain>)
  --output-dir <dir>        Output directory for reports/tests
  --prompt <text>           QA prompt override
  --max-scenarios <n>       Max scenarios to execute (default: 16)
  --max-runtime-sec <n>     Runtime cap in seconds for scenario execution
  --llm-token-cap <n>       Optional token cap for GAM + scenario LLM calls
  --workspace-id <id>       Workspace id for multi-user isolation (default: tenant id)
  --environment-profile <p> mock|staging|prod_safe (default: mock)
  --pass-threshold <float>  Quality gate threshold (default: 0.70)
  --script-kind <kind>      One generated script kind: python_pytest|javascript_jest|curl_script|java_restassured (default: python_pytest)
  --rl-train-mode <mode>    RL training mode (mandatory): periodic
  --base-url <url>          Base URL for generated tests (default: http://localhost:8000)
  --rl-checkpoint <path>    Agent Lightning checkpoint file path (default: /tmp/agent_lightning_<domain>.pt)
  --customer-mode           Customer UX mode: persistent workspace/checkpoint under ~/.spec_test_pilot
  --verify-persistence      In run mode, automatically re-run once and compare RL counters
  --ci-gate                 Enable CI quality-gate checks (default: on)
  --no-ci-gate              Disable CI quality-gate checks
  --ci-pass-floor <float>   CI pass-rate floor (default: pass-threshold)
  --ci-flaky-threshold <f>  CI flaky ratio ceiling vs previous run (default: 0.15)
  --ci-max-pass-drop <f>    Max allowed pass-rate drop vs previous run (default: 0.08)
  --ci-max-reward-drop <f>  Max allowed run-reward drop vs previous run (default: 0.10)
  --ci-min-gam-quality <f>  Min GAM context quality score (default: 0.55)
  --safe-mode-on-fail       Restore prior checkpoint + emit safe-mode marker on CI gate failure (default: on)
  --no-safe-mode-on-fail    Disable rollback/safe-mode behavior
  --customer-root <path>    Override customer workspace root (default: ~/.spec_test_pilot)
  --list-domains            Print supported domain presets
  -h, --help                Show this help

Examples:
  ./run_qa_domain.sh --domain healthcare
  ./run_qa_domain.sh --domain logistics --action generate
  ./run_qa_domain.sh --spec-path ./my_api.yaml --action run
  ./run_qa_domain.sh --domain ecommerce --customer-mode --verify-persistence
EOF
}

list_domains() {
  echo "Supported domains:"
  echo "  - ecommerce"
  echo "  - healthcare"
  echo "  - logistics"
  echo "  - hr"
}

DOMAIN="ecommerce"
SPEC_PATH=""
ACTION="both"
TENANT_ID=""
OUTPUT_DIR=""
PROMPT="Generate comprehensive QA tests for authentication, validation, error handling, and boundary scenarios."
MAX_SCENARIOS="16"
MAX_RUNTIME_SEC=""
LLM_TOKEN_CAP=""
WORKSPACE_ID=""
ENVIRONMENT_PROFILE="mock"
PASS_THRESHOLD="0.70"
SCRIPT_KIND="python_pytest"
RL_TRAIN_MODE="periodic"
BASE_URL="http://localhost:8000"
RL_CHECKPOINT=""
CUSTOMER_MODE="0"
VERIFY_PERSISTENCE="0"
CI_GATE="1"
CI_PASS_FLOOR=""
CI_FLAKY_THRESHOLD="0.15"
CI_MAX_PASS_DROP="0.08"
CI_MAX_REWARD_DROP="0.10"
CI_MIN_GAM_QUALITY="0.55"
SAFE_MODE_ON_FAIL="1"
CUSTOMER_ROOT="${HOME}/.spec_test_pilot"
CUSTOMER_ROOT_FALLBACK="/tmp/.spec_test_pilot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -z "${OPENAI_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)
      DOMAIN="${2:-}"
      shift 2
      ;;
    --spec-path)
      SPEC_PATH="${2:-}"
      shift 2
      ;;
    --action)
      ACTION="${2:-}"
      shift 2
      ;;
    --tenant-id)
      TENANT_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --max-scenarios)
      MAX_SCENARIOS="${2:-}"
      shift 2
      ;;
    --max-runtime-sec)
      MAX_RUNTIME_SEC="${2:-}"
      shift 2
      ;;
    --llm-token-cap)
      LLM_TOKEN_CAP="${2:-}"
      shift 2
      ;;
    --workspace-id)
      WORKSPACE_ID="${2:-}"
      shift 2
      ;;
    --environment-profile)
      ENVIRONMENT_PROFILE="${2:-}"
      shift 2
      ;;
    --pass-threshold)
      PASS_THRESHOLD="${2:-}"
      shift 2
      ;;
    --script-kind)
      SCRIPT_KIND="${2:-}"
      shift 2
      ;;
    --rl-train-mode)
      RL_TRAIN_MODE="${2:-}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --rl-checkpoint)
      RL_CHECKPOINT="${2:-}"
      shift 2
      ;;
    --customer-mode)
      CUSTOMER_MODE="1"
      shift
      ;;
    --verify-persistence)
      VERIFY_PERSISTENCE="1"
      shift
      ;;
    --ci-gate)
      CI_GATE="1"
      shift
      ;;
    --no-ci-gate)
      CI_GATE="0"
      shift
      ;;
    --ci-pass-floor)
      CI_PASS_FLOOR="${2:-}"
      shift 2
      ;;
    --ci-flaky-threshold)
      CI_FLAKY_THRESHOLD="${2:-}"
      shift 2
      ;;
    --ci-max-pass-drop)
      CI_MAX_PASS_DROP="${2:-}"
      shift 2
      ;;
    --ci-max-reward-drop)
      CI_MAX_REWARD_DROP="${2:-}"
      shift 2
      ;;
    --ci-min-gam-quality)
      CI_MIN_GAM_QUALITY="${2:-}"
      shift 2
      ;;
    --safe-mode-on-fail)
      SAFE_MODE_ON_FAIL="1"
      shift
      ;;
    --no-safe-mode-on-fail)
      SAFE_MODE_ON_FAIL="0"
      shift
      ;;
    --customer-root)
      CUSTOMER_ROOT="${2:-}"
      shift 2
      ;;
    --list-domains)
      list_domains
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${ACTION}" != "generate" && "${ACTION}" != "run" && "${ACTION}" != "both" ]]; then
  echo "Invalid --action: ${ACTION}. Use generate|run|both" >&2
  exit 1
fi

if [[ "${RL_TRAIN_MODE}" != "periodic" ]]; then
  echo "Invalid --rl-train-mode: ${RL_TRAIN_MODE}. Only periodic is allowed." >&2
  exit 1
fi

resolve_customer_root() {
  local requested_root="$1"
  if [[ -z "${requested_root}" ]]; then
    requested_root="${HOME}/.spec_test_pilot"
  fi

  if mkdir -p "${requested_root}" >/dev/null 2>&1; then
    echo "${requested_root}"
    return 0
  fi

  mkdir -p "${CUSTOMER_ROOT_FALLBACK}"
  echo "[WARN] Unable to write customer root '${requested_root}'. Falling back to '${CUSTOMER_ROOT_FALLBACK}'." >&2
  echo "${CUSTOMER_ROOT_FALLBACK}"
}

if [[ "${CUSTOMER_MODE}" == "1" ]]; then
  CUSTOMER_ROOT="$(resolve_customer_root "${CUSTOMER_ROOT}")"
  if [[ -z "${TENANT_ID}" ]]; then
    TENANT_ID="customer_default"
  fi
  CUSTOMER_BASE="${CUSTOMER_ROOT%/}/${TENANT_ID}/${DOMAIN}"
  mkdir -p "${CUSTOMER_BASE}/runs"

  if [[ -z "${SPEC_PATH}" ]]; then
    SPEC_PATH="${CUSTOMER_BASE}/openapi_${DOMAIN}.yaml"
  fi
  if [[ -z "${RL_CHECKPOINT}" ]]; then
    RL_CHECKPOINT="${CUSTOMER_BASE}/agent_lightning_checkpoint.pt"
  fi
  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${CUSTOMER_BASE}/runs/$(date +%Y%m%d_%H%M%S)"
  fi
else
  if [[ -z "${TENANT_ID}" ]]; then
    TENANT_ID="qa_${DOMAIN}"
  fi
  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="/tmp/qa_${DOMAIN}_$(date +%Y%m%d_%H%M%S)"
  fi
  if [[ -z "${RL_CHECKPOINT}" ]]; then
    RL_CHECKPOINT="/tmp/agent_lightning_${DOMAIN}.pt"
  fi
  if [[ -z "${SPEC_PATH}" ]]; then
    SPEC_PATH="/tmp/openapi_${DOMAIN}.yaml"
  fi
fi

generate_spec() {
  case "${DOMAIN}" in
    ecommerce)
      cat > "${SPEC_PATH}" <<'YAML'
openapi: 3.0.3
info:
  title: E-commerce API
  version: 1.0.0
paths:
  /products:
    get:
      summary: List products
      security:
        - bearerAuth: []
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
    post:
      summary: Create product
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [name, price]
              properties:
                name: { type: string, minLength: 1 }
                price: { type: number, minimum: 0 }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /orders:
    post:
      summary: Create order
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [productId, quantity]
              properties:
                productId: { type: string }
                quantity: { type: integer, minimum: 1 }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /orders/{orderId}:
    get:
      summary: Get order by id
      security:
        - bearerAuth: []
      parameters:
        - in: path
          name: orderId
          required: true
          schema: { type: string }
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
        '404': { description: Not found }
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
YAML
      ;;
    healthcare)
      cat > "${SPEC_PATH}" <<'YAML'
openapi: 3.0.3
info:
  title: Healthcare Appointments API
  version: 1.0.0
paths:
  /patients:
    post:
      summary: Register patient
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [name, dob]
              properties:
                name: { type: string, minLength: 1 }
                dob: { type: string, format: date }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /appointments:
    post:
      summary: Create appointment
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [patientId, doctorId, dateTime]
              properties:
                patientId: { type: string }
                doctorId: { type: string }
                dateTime: { type: string, format: date-time }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /appointments/{appointmentId}:
    get:
      summary: Get appointment
      security:
        - bearerAuth: []
      parameters:
        - in: path
          name: appointmentId
          required: true
          schema: { type: string }
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
        '404': { description: Not found }
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
YAML
      ;;
    logistics)
      cat > "${SPEC_PATH}" <<'YAML'
openapi: 3.0.3
info:
  title: Logistics Shipment API
  version: 1.0.0
paths:
  /shipments:
    get:
      summary: List shipments
      security:
        - bearerAuth: []
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
    post:
      summary: Create shipment
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [origin, destination, weightKg]
              properties:
                origin: { type: string, minLength: 1 }
                destination: { type: string, minLength: 1 }
                weightKg: { type: number, minimum: 0.1 }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /shipments/{shipmentId}:
    get:
      summary: Track shipment
      security:
        - bearerAuth: []
      parameters:
        - in: path
          name: shipmentId
          required: true
          schema: { type: string }
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
        '404': { description: Not found }
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
YAML
      ;;
    hr)
      cat > "${SPEC_PATH}" <<'YAML'
openapi: 3.0.3
info:
  title: HR Recruitment API
  version: 1.0.0
paths:
  /candidates:
    post:
      summary: Add candidate
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [name, email]
              properties:
                name: { type: string, minLength: 1 }
                email: { type: string, format: email }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /jobs/{jobId}/applications:
    post:
      summary: Apply to job
      security:
        - bearerAuth: []
      parameters:
        - in: path
          name: jobId
          required: true
          schema: { type: string }
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [candidateId]
              properties:
                candidateId: { type: string }
      responses:
        '201': { description: Created }
        '400': { description: Validation error }
        '401': { description: Unauthorized }
  /applications/{applicationId}:
    get:
      summary: Get application
      security:
        - bearerAuth: []
      parameters:
        - in: path
          name: applicationId
          required: true
          schema: { type: string }
      responses:
        '200': { description: OK }
        '401': { description: Unauthorized }
        '404': { description: Not found }
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
YAML
      ;;
    *)
      echo "Unsupported domain preset: ${DOMAIN}" >&2
      list_domains >&2
      exit 1
      ;;
  esac
}

resolve_python() {
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    echo "${SCRIPT_DIR}/.venv/bin/python"
  elif [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
    echo "${SCRIPT_DIR}/venv/bin/python"
  elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    echo "${REPO_ROOT}/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python" 
  fi
}

if [[ -z "${CI_PASS_FLOOR}" ]]; then
  CI_PASS_FLOOR="${PASS_THRESHOLD}"
fi

extract_report_metrics() {
  local python_bin="$1"
  local report_path="$2"
  "${python_bin}" - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
if not report.exists():
    print("missing|0|0")
    raise SystemExit(0)

data = json.loads(report.read_text(encoding="utf-8"))
pass_rate = data.get("summary", {}).get("pass_rate", 0)
steps = data.get("agent_lightning", {}).get("training_stats", {}).get("rl_training_steps", 0)
buffer_size = data.get("agent_lightning", {}).get("training_stats", {}).get("rl_buffer_size", 0)
print(f"{pass_rate}|{steps}|{buffer_size}")
PY
}

find_previous_report() {
  local current_output="$1"
  if [[ "${CUSTOMER_MODE}" != "1" ]]; then
    return 0
  fi
  local runs_dir="${CUSTOMER_BASE}/runs"
  if [[ ! -d "${runs_dir}" ]]; then
    return 0
  fi
  local candidate=""
  while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    if [[ "${path}" == "${current_output}" || "${path}" == "${current_output}_persistence_check" ]]; then
      continue
    fi
    if [[ -f "${path}/qa_execution_report.json" ]]; then
      candidate="${path}/qa_execution_report.json"
      break
    fi
  done < <(ls -1dt "${runs_dir}"/* 2>/dev/null || true)
  if [[ -n "${candidate}" ]]; then
    echo "${candidate}"
  fi
}

evaluate_ci_gate() {
  local report_path="$1"
  local previous_report="$2"
  local stage_label="$3"
  local gate_output=""
  local gate_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/ci_quality_gate.py"
    --report "${report_path}"
    --pass-rate-floor "${CI_PASS_FLOOR}"
    --flaky-threshold "${CI_FLAKY_THRESHOLD}"
    --max-pass-rate-drop "${CI_MAX_PASS_DROP}"
    --max-run-reward-drop "${CI_MAX_REWARD_DROP}"
    --min-context-quality "${CI_MIN_GAM_QUALITY}"
    --require-summary-quality-gate
  )
  if [[ -n "${previous_report}" && -f "${previous_report}" ]]; then
    gate_cmd+=(--previous-report "${previous_report}")
  fi
  set +e
  gate_output="$("${gate_cmd[@]}" 2>&1)"
  local gate_rc=$?
  set -e
  echo "[CI] ${stage_label}: ${gate_output}"
  if [[ ${gate_rc} -ne 0 ]]; then
    return 2
  fi
  return 0
}

activate_safe_mode() {
  local reason="$1"
  local first_report="${2:-}"
  local second_report="${3:-}"
  if [[ "${SAFE_MODE_ON_FAIL}" != "1" ]]; then
    return 0
  fi
  if [[ "${CUSTOMER_MODE}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${ROLLBACK_CHECKPOINT_BACKUP}" && -f "${ROLLBACK_CHECKPOINT_BACKUP}" ]]; then
    cp -f "${ROLLBACK_CHECKPOINT_BACKUP}" "${RL_CHECKPOINT}" || true
    echo "[SAFE-MODE] Restored checkpoint from backup: ${ROLLBACK_CHECKPOINT_BACKUP}"
  fi
  local marker="${CUSTOMER_BASE}/safe_mode_last_failure.json"
  "${PYTHON_BIN}" - "${marker}" "${reason}" "${first_report}" "${second_report}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

marker = Path(sys.argv[1])
payload = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "reason": sys.argv[2],
    "first_report": sys.argv[3],
    "second_report": sys.argv[4],
    "status": "safe_mode_active",
}
marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(str(marker))
PY
  echo "[SAFE-MODE] Marker written: ${marker}"
}

if [[ "${ACTION}" == "generate" || "${ACTION}" == "both" ]]; then
  generate_spec
  echo "[OK] OpenAPI spec written: ${SPEC_PATH}"
fi

if [[ "${ACTION}" == "run" || "${ACTION}" == "both" ]]; then
  if [[ ! -f "${SPEC_PATH}" ]]; then
    echo "Spec not found: ${SPEC_PATH}" >&2
    exit 1
  fi

  PYTHON_BIN="$(resolve_python)"
  mkdir -p "${OUTPUT_DIR}"
  export PYTHONUNBUFFERED="1"
  export GAM_LLM_MODE="on"
  export GAM_MEMO_LLM_MODE="on"
  # Keep scenario LLM reliable by default; callers can still override via env.
  export QA_SCENARIO_LLM_TIMEOUT_SECONDS="${QA_SCENARIO_LLM_TIMEOUT_SECONDS:-45}"
  export QA_SCENARIO_LLM_MAX_RETRIES="${QA_SCENARIO_LLM_MAX_RETRIES:-1}"

  echo "[RUN] QA specialist agent"
  echo "  domain:        ${DOMAIN}"
  echo "  spec:          ${SPEC_PATH}"
  echo "  tenant_id:     ${TENANT_ID}"
  echo "  output_dir:    ${OUTPUT_DIR}"
  echo "  max_scenarios: ${MAX_SCENARIOS}"
  if [[ -n "${MAX_RUNTIME_SEC}" ]]; then
    echo "  max_runtime:   ${MAX_RUNTIME_SEC}s"
  fi
  if [[ -n "${LLM_TOKEN_CAP}" ]]; then
    echo "  llm_token_cap: ${LLM_TOKEN_CAP}"
  fi
  echo "  env_profile:   ${ENVIRONMENT_PROFILE}"
  echo "  script_kind:   ${SCRIPT_KIND}"
  echo "  rl_train_mode: ${RL_TRAIN_MODE}"
  echo "  rl_checkpoint: ${RL_CHECKPOINT}"
  echo "  gam_llm_mode:  ${GAM_LLM_MODE} (enforced)"
  echo "  llm_timeout:   ${QA_SCENARIO_LLM_TIMEOUT_SECONDS}s"
  echo "  llm_retries:   ${QA_SCENARIO_LLM_MAX_RETRIES}"
  echo "  ci_gate:       ${CI_GATE}"
  if [[ "${CI_GATE}" == "1" ]]; then
    echo "  ci_pass_floor: ${CI_PASS_FLOOR}"
    echo "  ci_flaky_max:  ${CI_FLAKY_THRESHOLD}"
    echo "  ci_pass_drop:  ${CI_MAX_PASS_DROP}"
    echo "  ci_reward_drop:${CI_MAX_REWARD_DROP}"
    echo "  ci_gam_min:    ${CI_MIN_GAM_QUALITY}"
  fi
  if [[ "${CUSTOMER_MODE}" == "1" ]]; then
    echo "  customer_mode: enabled"
    echo "  customer_root: ${CUSTOMER_ROOT}"
  fi

  PREV_REPORT_BASE="$(find_previous_report "${OUTPUT_DIR}")"
  ROLLBACK_CHECKPOINT_BACKUP=""
  if [[ "${SAFE_MODE_ON_FAIL}" == "1" && -n "${RL_CHECKPOINT}" && -f "${RL_CHECKPOINT}" ]]; then
    ROLLBACK_CHECKPOINT_BACKUP="${RL_CHECKPOINT}.pre_ci_backup_$(date +%Y%m%d_%H%M%S)"
    cp -f "${RL_CHECKPOINT}" "${ROLLBACK_CHECKPOINT_BACKUP}"
    echo "  rollback_backup:${ROLLBACK_CHECKPOINT_BACKUP}"
  fi

  cmd=(
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/qa_agent_runner.py"
    --spec "${SPEC_PATH}"
    --tenant-id "${TENANT_ID}"
    --base-url "${BASE_URL}"
    --environment-profile "${ENVIRONMENT_PROFILE}"
    --output-dir "${OUTPUT_DIR}"
    --prompt "${PROMPT}"
    --max-scenarios "${MAX_SCENARIOS}"
    --pass-threshold "${PASS_THRESHOLD}"
    --script-kind "${SCRIPT_KIND}"
    --rl-train-mode "${RL_TRAIN_MODE}"
  )
  if [[ -n "${RL_CHECKPOINT}" ]]; then
    cmd+=(--rl-checkpoint "${RL_CHECKPOINT}")
  fi
  if [[ -n "${WORKSPACE_ID}" ]]; then
    cmd+=(--workspace-id "${WORKSPACE_ID}")
  fi
  if [[ -n "${MAX_RUNTIME_SEC}" ]]; then
    cmd+=(--max-runtime-sec "${MAX_RUNTIME_SEC}")
  fi
  if [[ -n "${LLM_TOKEN_CAP}" ]]; then
    cmd+=(--llm-token-cap "${LLM_TOKEN_CAP}")
  fi
  "${cmd[@]}"

  echo "[OK] Run completed."
  echo "  JSON report: ${OUTPUT_DIR}/qa_execution_report.json"
  echo "  MD report:   ${OUTPUT_DIR}/qa_execution_report.md"

  if [[ "${CI_GATE}" == "1" ]]; then
    if ! evaluate_ci_gate "${OUTPUT_DIR}/qa_execution_report.json" "${PREV_REPORT_BASE}" "base_run"; then
      echo "[CI] Base run quality gate failed."
      activate_safe_mode "base_run_ci_gate_failed" "${OUTPUT_DIR}/qa_execution_report.json" ""
      exit 2
    fi
  fi

  if [[ "${VERIFY_PERSISTENCE}" == "1" ]]; then
    if [[ ! -f "${OUTPUT_DIR}/qa_execution_report.json" ]]; then
      echo "[WARN] Persistence check skipped: first report missing."
      exit 0
    fi

    first_metrics="$(extract_report_metrics "${PYTHON_BIN}" "${OUTPUT_DIR}/qa_execution_report.json")"
    IFS='|' read -r first_pass_rate first_steps first_buffer <<< "${first_metrics}"

    verify_output_dir="${OUTPUT_DIR}_persistence_check"
    mkdir -p "${verify_output_dir}"

    echo "[VERIFY] Running automatic persistence check pass..."
    verify_cmd=(
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/qa_agent_runner.py"
      --spec "${SPEC_PATH}"
      --tenant-id "${TENANT_ID}"
      --base-url "${BASE_URL}"
      --environment-profile "${ENVIRONMENT_PROFILE}"
      --output-dir "${verify_output_dir}"
      --prompt "${PROMPT}"
      --max-scenarios "${MAX_SCENARIOS}"
      --pass-threshold "${PASS_THRESHOLD}"
      --script-kind "${SCRIPT_KIND}"
      --rl-train-mode "${RL_TRAIN_MODE}"
    )
    if [[ -n "${RL_CHECKPOINT}" ]]; then
      verify_cmd+=(--rl-checkpoint "${RL_CHECKPOINT}")
    fi
    if [[ -n "${WORKSPACE_ID}" ]]; then
      verify_cmd+=(--workspace-id "${WORKSPACE_ID}")
    fi
    if [[ -n "${MAX_RUNTIME_SEC}" ]]; then
      verify_cmd+=(--max-runtime-sec "${MAX_RUNTIME_SEC}")
    fi
    if [[ -n "${LLM_TOKEN_CAP}" ]]; then
      verify_cmd+=(--llm-token-cap "${LLM_TOKEN_CAP}")
    fi
    "${verify_cmd[@]}"

    second_metrics="$(extract_report_metrics "${PYTHON_BIN}" "${verify_output_dir}/qa_execution_report.json")"
    IFS='|' read -r second_pass_rate second_steps second_buffer <<< "${second_metrics}"

    echo "[VERIFY] Persistence summary"
    echo "  first_pass_rate:   ${first_pass_rate}"
    echo "  second_pass_rate:  ${second_pass_rate}"
    echo "  first_rl_steps:    ${first_steps}"
    echo "  second_rl_steps:   ${second_steps}"
    echo "  first_rl_buffer:   ${first_buffer}"
    echo "  second_rl_buffer:  ${second_buffer}"
    echo "  first_report:      ${OUTPUT_DIR}/qa_execution_report.json"
    echo "  second_report:     ${verify_output_dir}/qa_execution_report.json"

    if [[ "${CI_GATE}" == "1" ]]; then
      if ! evaluate_ci_gate "${verify_output_dir}/qa_execution_report.json" "${OUTPUT_DIR}/qa_execution_report.json" "persistence_run"; then
        echo "[CI] Persistence run quality gate failed."
        activate_safe_mode "persistence_ci_gate_failed" "${OUTPUT_DIR}/qa_execution_report.json" "${verify_output_dir}/qa_execution_report.json"
        exit 2
      fi
    fi
  fi
fi
