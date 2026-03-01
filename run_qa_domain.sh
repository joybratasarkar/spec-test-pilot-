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
  --pass-threshold <float>  Quality gate threshold (default: 0.70)
  --base-url <url>          Base URL for generated tests (default: http://localhost:8000)
  --rl-checkpoint <path>    Agent Lightning checkpoint file path (default: /tmp/agent_lightning_<domain>.pt)
  --list-domains            Print supported domain presets
  -h, --help                Show this help

Examples:
  ./run_qa_domain.sh --domain healthcare
  ./run_qa_domain.sh --domain logistics --action generate
  ./run_qa_domain.sh --spec-path ./my_api.yaml --action run
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
PASS_THRESHOLD="0.70"
BASE_URL="http://localhost:8000"
RL_CHECKPOINT=""

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
    --pass-threshold)
      PASS_THRESHOLD="${2:-}"
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

if [[ -z "${TENANT_ID}" ]]; then
  TENANT_ID="qa_${DOMAIN}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="/tmp/qa_${DOMAIN}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${RL_CHECKPOINT}" ]]; then
  RL_CHECKPOINT="/tmp/agent_lightning_${DOMAIN}.pt"
fi

if [[ "${ACTION}" != "generate" && "${ACTION}" != "run" && "${ACTION}" != "both" ]]; then
  echo "Invalid --action: ${ACTION}. Use generate|run|both" >&2
  exit 1
fi

if [[ -z "${SPEC_PATH}" ]]; then
  SPEC_PATH="/tmp/openapi_${DOMAIN}.yaml"
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
  if [[ -x "venv/bin/python" ]]; then
    echo "venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python" 
  fi
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

  echo "[RUN] QA specialist agent"
  echo "  domain:        ${DOMAIN}"
  echo "  spec:          ${SPEC_PATH}"
  echo "  tenant_id:     ${TENANT_ID}"
  echo "  output_dir:    ${OUTPUT_DIR}"
  echo "  max_scenarios: ${MAX_SCENARIOS}"
  echo "  rl_checkpoint: ${RL_CHECKPOINT}"

  cmd=(
    "${PYTHON_BIN}" qa_specialist_runner.py
    --spec "${SPEC_PATH}"
    --tenant-id "${TENANT_ID}"
    --base-url "${BASE_URL}"
    --output-dir "${OUTPUT_DIR}"
    --prompt "${PROMPT}"
    --max-scenarios "${MAX_SCENARIOS}"
    --pass-threshold "${PASS_THRESHOLD}"
  )
  if [[ -n "${RL_CHECKPOINT}" ]]; then
    cmd+=(--rl-checkpoint "${RL_CHECKPOINT}")
  fi
  "${cmd[@]}"

  echo "[OK] Run completed."
  echo "  JSON report: ${OUTPUT_DIR}/qa_execution_report.json"
  echo "  MD report:   ${OUTPUT_DIR}/qa_execution_report.md"
fi
