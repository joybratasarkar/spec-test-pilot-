#!/usr/bin/env python3
"""Customer-facing UI to run multi-domain QA agent tests.

Launch:
    ./backend/run_customer_backend_fastapi.sh
Open:
    http://127.0.0.1:8787
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


APP_TITLE = "SpecForge Customer QA UI"
DOMAIN_PRESETS = ["ecommerce", "healthcare", "logistics", "hr"]
DOMAIN_PRESET_SET = set(DOMAIN_PRESETS)
SUPPORTED_SCRIPT_KINDS = [
    "python_pytest",
    "javascript_jest",
    "curl_script",
    "java_restassured",
]
REPO_ROOT = Path(__file__).resolve().parent
JOB_ROOT = Path("/tmp/qa_ui_runs")
CHECKPOINT_ROOT = Path("/tmp/qa_ui_checkpoints")
JOB_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def _bootstrap_runtime_env() -> None:
    """Best-effort .env loading so API-launched jobs inherit OPENAI_API_KEY."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        env_path = candidate.expanduser().resolve()
        key = str(env_path)
        if key in seen:
            continue
        seen.add(key)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)


_bootstrap_runtime_env()

MAX_LOG_LINES = 6000

app = FastAPI(title=APP_TITLE)


def _default_allowed_origins_csv() -> str:
    origins: List[str] = []
    for port in range(3000, 3011):
        origins.append(f"http://localhost:{port}")
        origins.append(f"http://127.0.0.1:{port}")
    return ",".join(origins)


_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "QA_UI_ALLOWED_ORIGINS",
        _default_allowed_origins_csv(),
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

DOMAIN_TOKEN_PATTERN = re.compile(r"^[a-z0-9_-]{1,64}$")


def _sanitize_domain_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9_-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return ""
    return token[:64]


def _normalize_spec_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve())


class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    domains: List[str] = Field(default_factory=lambda: ["ecommerce"])
    spec_paths: Dict[str, str] = Field(default_factory=dict, alias="specPaths")
    tenant_id: str = Field(default="customer_default", alias="tenantId")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    prompt: Optional[str] = None
    script_kind: str = Field(default="python_pytest", alias="scriptKind")
    max_scenarios: int = Field(default=16, alias="maxScenarios")
    max_runtime_sec: Optional[int] = Field(default=None, alias="maxRuntimeSec")
    llm_token_cap: Optional[int] = Field(default=None, alias="llmTokenCap")
    environment_profile: str = Field(default="mock", alias="environmentProfile")
    pass_threshold: float = Field(default=0.70, alias="passThreshold")
    base_url: str = Field(default="http://localhost:8000", alias="baseUrl")
    customer_mode: bool = Field(default=True, alias="customerMode")
    verify_persistence: bool = Field(default=True, alias="verifyPersistence")
    customer_root: str = Field(default=str(Path.home() / ".spec_test_pilot"), alias="customerRoot")

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, value: List[str]) -> List[str]:
        normalized = [_sanitize_domain_token(v) for v in value if str(v).strip()]
        normalized = [token for token in normalized if token]
        if not normalized:
            raise ValueError("Select at least one domain")
        invalid = [v for v in normalized if not DOMAIN_TOKEN_PATTERN.match(v)]
        if invalid:
            raise ValueError(f"Invalid domain ids: {', '.join(invalid)}")
        return list(dict.fromkeys(normalized))

    @field_validator("spec_paths")
    @classmethod
    def validate_spec_paths(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, str] = {}
        for raw_domain, raw_path in value.items():
            domain = _sanitize_domain_token(raw_domain)
            if not domain:
                continue
            if not DOMAIN_TOKEN_PATTERN.match(domain):
                raise ValueError(f"Invalid domain id in specPaths: {raw_domain}")
            spec_path = _normalize_spec_path(raw_path)
            if not spec_path:
                continue
            if not Path(spec_path).exists():
                raise ValueError(f"Spec file not found for domain '{domain}': {spec_path}")
            normalized[domain] = spec_path
        return normalized

    @field_validator("max_scenarios")
    @classmethod
    def validate_max_scenarios(cls, value: int) -> int:
        if value < 1:
            return 1
        if value > 500:
            return 500
        return value

    @field_validator("max_runtime_sec")
    @classmethod
    def validate_max_runtime_sec(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        num = int(value)
        if num <= 0:
            return None
        return min(7200, num)

    @field_validator("llm_token_cap")
    @classmethod
    def validate_llm_token_cap(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        num = int(value)
        if num <= 0:
            return None
        return min(16000, max(64, num))

    @field_validator("environment_profile")
    @classmethod
    def validate_environment_profile(cls, value: str) -> str:
        profile = str(value or "mock").strip().lower()
        if profile not in {"mock", "staging", "prod_safe"}:
            return "mock"
        return profile

    @field_validator("script_kind")
    @classmethod
    def validate_script_kind(cls, value: str) -> str:
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized not in SUPPORTED_SCRIPT_KINDS:
            allowed = ", ".join(SUPPORTED_SCRIPT_KINDS)
            raise ValueError(f"Unsupported script kind: {value}. Allowed: {allowed}")
        return normalized

    @field_validator("pass_threshold")
    @classmethod
    def validate_pass_threshold(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


def _append_log(job: Dict[str, Any], line: str) -> None:
    logs = job.setdefault("logs", [])
    logs.append(line.rstrip("\n"))
    if len(logs) > MAX_LOG_LINES:
        del logs[: len(logs) - MAX_LOG_LINES]


def _load_report_artifacts(report_json_path: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if not report_json_path.exists():
        return {}, {}
    try:
        payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    summary = payload.get("summary", {})
    training_stats = payload.get("agent_lightning", {}).get("training_stats", {})
    generated_raw = payload.get("generated_test_files", {})
    generated_tests: Dict[str, str] = {}
    if isinstance(generated_raw, dict):
        for key, value in generated_raw.items():
            if isinstance(value, str) and value.strip():
                generated_tests[str(key)] = str(value)

    summary_payload = {
        "total_scenarios": summary.get("total_scenarios"),
        "passed_scenarios": summary.get("passed_scenarios"),
        "failed_scenarios": summary.get("failed_scenarios"),
        "pass_rate": summary.get("pass_rate"),
        "meets_quality_gate": summary.get("meets_quality_gate"),
        "contract_checks_run": summary.get("contract_checks_run"),
        "contract_check_failures": summary.get("contract_check_failures"),
        "corrected_expectations": summary.get("corrected_expectations"),
        "runtime_cap_hit": summary.get("runtime_cap_hit"),
        "runtime_skipped_scenarios": summary.get("runtime_skipped_scenarios"),
        "environment_profile": summary.get("environment_profile"),
        "script_kind": payload.get("metadata", {}).get("script_kind"),
        "workspace_id": payload.get("metadata", {}).get("workspace_id"),
        "spec_key": payload.get("metadata", {}).get("spec_key"),
        "run_id": payload.get("metadata", {}).get("run_id"),
        "rl_training_steps": training_stats.get("rl_training_steps"),
        "rl_buffer_size": training_stats.get("rl_buffer_size"),
        "selection_algorithm": payload.get("selection_policy", {}).get("algorithm"),
        "selection_selected_count": payload.get("selection_policy", {}).get("selected_count"),
        "selection_candidate_count": payload.get("selection_policy", {}).get("candidate_count"),
        "run_reward": payload.get("learning", {}).get("feedback", {}).get("run_reward"),
        "learning_delta_status": payload.get("learning", {}).get("learning_delta_status"),
        "gam_context_quality": (payload.get("gam_context_pack", {}) or {}).get("quality_score"),
        "spec_operations_total": (payload.get("spec_intelligence", {}) or {}).get("operations_total"),
        "dependency_edge_count": ((payload.get("spec_intelligence", {}) or {}).get("dependency_graph", {}) or {}).get("edge_count"),
    }
    return summary_payload, generated_tests


def _read_report_payload(report_json_path: Path) -> Dict[str, Any]:
    if not report_json_path.exists():
        raise HTTPException(status_code=404, detail="report file missing")
    try:
        return json.loads(report_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"invalid report json: {exc}") from exc


def _domain_result_or_404(job_id: str, domain: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        result = (job.get("results") or {}).get(domain)
    if not result:
        raise HTTPException(status_code=404, detail="domain result not found")
    return result


def _resolve_domain_for_run(job_id: str, domain: Optional[str]) -> str:
    chosen = str(domain or "").strip().lower()
    if chosen:
        return chosen
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        domains = [str(item).strip().lower() for item in (job.get("request", {}).get("domains", []) or []) if str(item).strip()]
        if not domains:
            raise HTTPException(status_code=404, detail="no domains found for run")
        return domains[0]


def _resolve_generated_tests(result: Dict[str, Any]) -> Dict[str, str]:
    cached = result.get("generated_tests")
    if isinstance(cached, dict) and cached:
        return {
            str(k): str(v)
            for k, v in cached.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        }

    report_json = Path(result.get("report_json", "")).resolve()
    payload = _read_report_payload(report_json)
    generated_raw = payload.get("generated_test_files", {})
    if not isinstance(generated_raw, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in generated_raw.items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }


def _canonical_path(path_value: str | Path) -> Path:
    # Normalize symlinked tmp paths (/tmp vs /private/tmp) for safe path checks.
    expanded = os.path.expanduser(str(path_value))
    return Path(os.path.realpath(expanded))


def _resolve_safe_generated_script_path(result: Dict[str, Any], kind: str) -> Path:
    generated = _resolve_generated_tests(result)
    script_path_raw = generated.get(kind)
    if not script_path_raw:
        raise HTTPException(status_code=404, detail=f"generated test script kind not found: {kind}")

    script_path = _canonical_path(script_path_raw)
    output_dir_raw = str(result.get("output_dir", "")).strip()
    if not output_dir_raw:
        raise HTTPException(status_code=500, detail="missing output directory for this run result")
    output_dir = _canonical_path(output_dir_raw)
    if not script_path.is_relative_to(output_dir):
        raise HTTPException(status_code=400, detail="generated script path is outside output directory")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="generated script file missing")
    return script_path


def _job_payload(job: Dict[str, Any], tail: int) -> Dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "current_domain": job["current_domain"],
        "request": job["request"],
        "results": job["results"],
        "logs": job["logs"][-tail:],
    }


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        req = dict(job["request"])
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat() + "Z"

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    failures = 0

    for domain in req["domains"]:
        output_dir = JOB_ROOT / f"{ts}_{job_id}_{domain}"
        checkpoint = CHECKPOINT_ROOT / f"{req['tenant_id']}_{domain}.pt"
        spec_paths = req.get("spec_paths") or {}
        spec_path_override = str(spec_paths.get(domain, "")).strip()

        cmd = [
            "bash",
            str(REPO_ROOT / "run_qa_domain.sh"),
            "--domain",
            domain,
            "--tenant-id",
            req["tenant_id"],
            "--base-url",
            req["base_url"],
            "--output-dir",
            str(output_dir),
            "--max-scenarios",
            str(req["max_scenarios"]),
            "--pass-threshold",
            str(req["pass_threshold"]),
            "--script-kind",
            req["script_kind"],
            "--environment-profile",
            str(req.get("environment_profile", "mock")),
            "--rl-checkpoint",
            str(checkpoint),
        ]
        workspace_id = str(req.get("workspace_id") or req.get("tenant_id") or "").strip()
        if workspace_id:
            cmd += ["--workspace-id", workspace_id]
        max_runtime_sec = req.get("max_runtime_sec")
        if isinstance(max_runtime_sec, int) and max_runtime_sec > 0:
            cmd += ["--max-runtime-sec", str(max_runtime_sec)]
        llm_token_cap = req.get("llm_token_cap")
        if isinstance(llm_token_cap, int) and llm_token_cap > 0:
            cmd += ["--llm-token-cap", str(llm_token_cap)]

        if spec_path_override:
            cmd += ["--action", "run", "--spec-path", spec_path_override]
        else:
            if domain not in DOMAIN_PRESET_SET:
                with _jobs_lock:
                    job = _jobs[job_id]
                    _append_log(
                        job,
                        f"[{domain}] Unsupported preset domain without spec path. "
                        + "Provide specPaths.<domain>=/path/to/openapi.yaml",
                    )
                    job["results"][domain] = {
                        "domain": domain,
                        "return_code": 2,
                        "script_kind": req["script_kind"],
                        "output_dir": str(output_dir),
                        "checkpoint": str(checkpoint),
                        "spec_path": "",
                        "report_json": "",
                        "report_md": "",
                        "summary": {"error": "missing_spec_path_for_custom_domain"},
                        "generated_tests": {},
                    }
                failures += 1
                continue
            cmd += ["--action", "both"]

        if req.get("prompt"):
            cmd += ["--prompt", req["prompt"]]
        if req.get("customer_mode"):
            cmd += ["--customer-mode", "--customer-root", req.get("customer_root", str(Path.home() / ".spec_test_pilot"))]
        if req.get("verify_persistence"):
            cmd += ["--verify-persistence"]

        with _jobs_lock:
            job = _jobs[job_id]
            job["current_domain"] = domain
            _append_log(job, "")
            _append_log(job, f"===== DOMAIN: {domain} =====")
            _append_log(job, "$ " + " ".join(cmd))

        child_env = dict(os.environ)
        child_env["GAM_LLM_MODE"] = "on"
        child_env["GAM_MEMO_LLM_MODE"] = "on"
        child_env.setdefault("QA_SCENARIO_LLM_TIMEOUT_SECONDS", "45")
        child_env.setdefault("QA_SCENARIO_LLM_MAX_RETRIES", "1")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            with _jobs_lock:
                _append_log(_jobs[job_id], f"[{domain}] {line.rstrip()}" )

        return_code = proc.wait()

        report_json = output_dir / "qa_execution_report.json"
        report_md = output_dir / "qa_execution_report.md"
        summary, generated_tests = _load_report_artifacts(report_json)
        if return_code != 0:
            failures += 1

        with _jobs_lock:
            job = _jobs[job_id]
            job["results"][domain] = {
                "domain": domain,
                "return_code": return_code,
                "script_kind": req["script_kind"],
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint),
                "spec_path": spec_path_override,
                "report_json": str(report_json),
                "report_md": str(report_md),
                "summary": summary,
                "generated_tests": generated_tests,
            }

    with _jobs_lock:
        job = _jobs[job_id]
        job["current_domain"] = None
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["status"] = "completed" if failures == 0 else "failed"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SpecForge Customer QA UI</title>
  <style>
    :root {
      --bg: #f8f7f2;
      --ink: #14213d;
      --muted: #415a77;
      --card: #ffffff;
      --accent: #e76f51;
      --ok: #2a9d8f;
      --bad: #c1121f;
      --line: #d7d7d7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 10% 10%, #fff3e6 0%, var(--bg) 45%, #edf3ff 100%);
    }
    header {
      padding: 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      backdrop-filter: blur(6px);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    header h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.4px;
    }
    .wrap {
      max-width: 1280px;
      margin: 18px auto;
      padding: 0 16px 24px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 5px 20px rgba(20, 33, 61, 0.05);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--muted);
    }
    .field { margin-bottom: 10px; }
    label { display: block; font-size: 12px; margin-bottom: 4px; color: var(--muted); }
    input[type=text], input[type=number], select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      font-size: 14px;
    }
    textarea { min-height: 80px; resize: vertical; }
    .domains { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; }
    .domain-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #fafafa;
      font-size: 13px;
    }
    .checks { display: grid; gap: 6px; margin: 10px 0; }
    .btn {
      width: 100%;
      border: 0;
      border-radius: 9px;
      background: linear-gradient(120deg, #f4a261, var(--accent));
      color: #fff;
      font-size: 14px;
      font-weight: 600;
      padding: 10px;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.5; cursor: default; }
    .status {
      font-size: 13px;
      margin-bottom: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #999; }
    .dot.running { background: #f4a261; }
    .dot.completed { background: var(--ok); }
    .dot.failed { background: var(--bad); }
    .grid-right { display: grid; gap: 16px; }
    .steps { display: grid; gap: 6px; font-size: 13px; }
    .step { border-left: 4px solid #ddd; padding: 6px 8px; background: #fafafa; }
    .step.done { border-left-color: var(--ok); background: #f2fbf9; }
    .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
    .result-item { border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .result-item h3 { margin: 0 0 8px; font-size: 15px; }
    .pill { display:inline-block; border-radius:999px; padding:2px 8px; font-size:11px; margin-left:6px; }
    .pill.ok { background:#d8f4ef; color:#0b6a5f; }
    .pill.bad { background:#ffe5e5; color:#8f0013; }
    .result-item button {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 6px 8px;
      margin-right: 6px;
      cursor: pointer;
    }
    pre {
      margin: 0;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 10px;
      padding: 10px;
      max-height: 360px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
    }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SpecForge Customer QA Runner</h1>
  </header>

  <div class=\"wrap\">
    <section class=\"card\">
      <h2>Run Configuration</h2>
      <div class=\"field\">
        <label>Domains (comma or newline separated)</label>
        <textarea id=\"domainsInput\" style=\"min-height:72px;\" placeholder=\"ecommerce, healthcare\">ecommerce</textarea>
      </div>
      <div class=\"field\">
        <label>Spec Paths (optional, one per line: domain=/abs/path/openapi.yaml)</label>
        <textarea id=\"specPaths\" style=\"min-height:72px;\" placeholder=\"payments=/tmp/openapi_payments.yaml\"></textarea>
      </div>

      <div class=\"field\">
        <label>Tenant ID</label>
        <input id=\"tenant\" type=\"text\" value=\"customer_default\" />
      </div>
      <div class=\"field\">
        <label>Script Language</label>
        <select id=\"scriptKind\">
          <option value=\"python_pytest\" selected>Python / Pytest</option>
          <option value=\"javascript_jest\">JavaScript / Jest</option>
          <option value=\"curl_script\">cURL Script</option>
          <option value=\"java_restassured\">Java / RestAssured</option>
        </select>
      </div>
      <div class=\"field\">
        <label>Max Scenarios</label>
        <input id=\"max\" type=\"number\" value=\"16\" min=\"1\" max=\"500\" />
      </div>
      <div class=\"field\">
        <label>Pass Threshold (0-1)</label>
        <input id=\"threshold\" type=\"number\" value=\"0.7\" min=\"0\" max=\"1\" step=\"0.01\" />
      </div>
      <div class=\"field\">
        <label>Base URL</label>
        <input id=\"baseUrl\" type=\"text\" value=\"http://localhost:8000\" />
      </div>
      <div class=\"field\">
        <label>Customer Root</label>
        <input id=\"customerRoot\" type=\"text\" value=\"~/.spec_test_pilot\" />
      </div>
      <div class=\"field\">
        <label>Prompt (optional)</label>
        <textarea id=\"prompt\" placeholder=\"Custom QA prompt\"></textarea>
      </div>

      <div class=\"checks\">
        <label><input id=\"customerMode\" type=\"checkbox\" checked> customer mode (persistent checkpoint/workspace)</label>
        <label><input id=\"verify\" type=\"checkbox\" checked> verify persistence (auto second pass)</label>
      </div>

      <button class=\"btn\" id=\"runBtn\" onclick=\"startRun()\">Run QA Agent</button>
    </section>

    <section class=\"grid-right\">
      <div class=\"card\">
        <h2>Runtime Status</h2>
        <div class=\"status\"><span id=\"statusDot\" class=\"dot\"></span><strong id=\"statusText\">idle</strong></div>
        <div id=\"jobMeta\" style=\"font-size:12px;color:#475569;\"></div>
        <div class=\"steps\" id=\"steps\"></div>
      </div>

      <div class=\"card\">
        <h2>Domain Results</h2>
        <div id=\"results\" class=\"results\"></div>
      </div>

      <div class=\"card\">
        <h2>Live Process Log</h2>
        <pre id=\"logs\">No run started yet.</pre>
      </div>

      <div class=\"card\">
        <h2>Report Viewer</h2>
        <pre id=\"reportView\">Select a report from Domain Results.</pre>
      </div>
    </section>
  </div>

  <script>
    let currentJobId = null;
    let pollHandle = null;

    const STEP_MARKERS = [
      { name: 'Spec Prepared', marker: '[OK] OpenAPI spec written' },
      { name: 'Run Started', marker: '[RUN] QA specialist agent' },
      { name: 'RL Session Started', marker: 'Started observability session' },
      { name: 'RL Training Executed', marker: 'RL training executed' },
      { name: 'QA Run Complete', marker: 'QA specialist run complete' },
      { name: 'Reports Written', marker: 'JSON report:' }
    ];

    function selectedDomains() {
      const raw = String(document.getElementById('domainsInput').value || '');
      const tokens = raw
        .split(/[\\n,]+/)
        .map(v => v.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_'))
        .filter(Boolean);
      return Array.from(new Set(tokens));
    }

    function parseSpecPaths() {
      const raw = String(document.getElementById('specPaths').value || '');
      const lines = raw.split(/\\n+/).map(v => v.trim()).filter(Boolean);
      const out = {};
      lines.forEach((line) => {
        const idx = line.indexOf('=');
        if (idx <= 0) return;
        const domain = line.slice(0, idx).trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
        const specPath = line.slice(idx + 1).trim();
        if (!domain || !specPath) return;
        out[domain] = specPath;
      });
      return out;
    }

    async function startRun() {
      const domains = selectedDomains();
      if (!domains.length) {
        alert('Select at least one domain.');
        return;
      }

      const body = {
        domains,
        spec_paths: parseSpecPaths(),
        tenant_id: document.getElementById('tenant').value.trim() || 'customer_default',
        script_kind: document.getElementById('scriptKind').value || 'python_pytest',
        prompt: document.getElementById('prompt').value.trim() || null,
        max_scenarios: Number(document.getElementById('max').value || 16),
        pass_threshold: Number(document.getElementById('threshold').value || 0.7),
        base_url: document.getElementById('baseUrl').value.trim() || 'http://localhost:8000',
        customer_mode: document.getElementById('customerMode').checked,
        verify_persistence: document.getElementById('verify').checked,
        customer_root: document.getElementById('customerRoot').value.trim() || '~/.spec_test_pilot'
      };

      const runBtn = document.getElementById('runBtn');
      runBtn.disabled = true;
      runBtn.textContent = 'Starting...';

      const resp = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const payload = await resp.json();
      if (!resp.ok) {
        alert(payload.detail || 'Failed to start job');
        runBtn.disabled = false;
        runBtn.textContent = 'Run QA Agent';
        return;
      }

      currentJobId = payload.job_id;
      document.getElementById('logs').textContent = 'Job started...';
      document.getElementById('reportView').textContent = 'Select a report from Domain Results.';
      pollJob();
    }

    async function pollJob() {
      if (!currentJobId) return;
      if (pollHandle) clearTimeout(pollHandle);

      const resp = await fetch(`/api/jobs/${currentJobId}?tail=800`);
      const job = await resp.json();
      renderJob(job);

      if (job.status === 'running' || job.status === 'queued') {
        pollHandle = setTimeout(pollJob, 1500);
      } else {
        document.getElementById('runBtn').disabled = false;
        document.getElementById('runBtn').textContent = 'Run QA Agent';
      }
    }

    function renderJob(job) {
      const dot = document.getElementById('statusDot');
      dot.className = 'dot ' + (job.status || '');
      document.getElementById('statusText').textContent = job.status;

      const meta = [];
      meta.push(`job_id=${job.id}`);
      if (job.current_domain) meta.push(`current_domain=${job.current_domain}`);
      if (job.started_at) meta.push(`started_at=${job.started_at}`);
      if (job.completed_at) meta.push(`completed_at=${job.completed_at}`);
      document.getElementById('jobMeta').textContent = meta.join(' | ');

      const logs = job.logs || [];
      const logText = logs.join('\n');
      document.getElementById('logs').textContent = logText || 'No logs yet.';

      const stepsEl = document.getElementById('steps');
      stepsEl.innerHTML = '';
      STEP_MARKERS.forEach(step => {
        const done = logText.includes(step.marker);
        const el = document.createElement('div');
        el.className = 'step' + (done ? ' done' : '');
        el.textContent = (done ? '✓ ' : '• ') + step.name;
        stepsEl.appendChild(el);
      });

      const resultsEl = document.getElementById('results');
      resultsEl.innerHTML = '';
      const results = job.results || {};
      Object.keys(results).forEach(domain => {
        const r = results[domain];
        const card = document.createElement('div');
        card.className = 'result-item';
        const ok = Number(r.return_code) === 0;
        const s = r.summary || {};
        card.innerHTML = `
          <h3>${domain}<span class="pill ${ok ? 'ok':'bad'}">${ok ? 'ok':'failed'}</span></h3>
          <div style="font-size:12px; line-height:1.45; color:#475569;">
            pass_rate=${s.pass_rate ?? 'n/a'}<br/>
            total=${s.total_scenarios ?? 'n/a'} passed=${s.passed_scenarios ?? 'n/a'} failed=${s.failed_scenarios ?? 'n/a'}<br/>
            rl_steps=${s.rl_training_steps ?? 'n/a'} rl_buffer=${s.rl_buffer_size ?? 'n/a'}
          </div>
          <div style="margin-top:8px;">
            <button onclick="viewReport('${domain}','json')">View JSON</button>
            <button onclick="viewReport('${domain}','md')">View Markdown</button>
          </div>
        `;
        resultsEl.appendChild(card);
      });
    }

    async function viewReport(domain, format) {
      if (!currentJobId) return;
      const resp = await fetch(`/api/jobs/${currentJobId}/report/${domain}?format=${format}`);
      const target = document.getElementById('reportView');
      if (!resp.ok) {
        const err = await resp.text();
        target.textContent = `Failed to load report: ${err}`;
        return;
      }
      if (format === 'json') {
        const payload = await resp.json();
        target.textContent = JSON.stringify(payload, null, 2);
      } else {
        target.textContent = await resp.text();
      }
    }
  </script>
</body>
</html>
"""


@app.post("/api/jobs")
def create_job(req: RunRequest) -> Dict[str, Any]:
    req_payload = req.model_dump()
    if not str(req_payload.get("workspace_id") or "").strip():
        req_payload["workspace_id"] = str(req_payload.get("tenant_id") or "customer_default")
    spec_paths = req_payload.get("spec_paths") or {}
    if isinstance(spec_paths, dict) and spec_paths:
        domain_set = {_sanitize_domain_token(item) for item in req_payload.get("domains", [])}
        domain_set = {item for item in domain_set if item}
        for domain in spec_paths.keys():
            token = _sanitize_domain_token(domain)
            if token:
                domain_set.add(token)
        req_payload["domains"] = sorted(domain_set)

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "current_domain": None,
        "request": req_payload,
        "logs": [],
        "results": {},
    }

    with _jobs_lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs() -> List[Dict[str, Any]]:
    with _jobs_lock:
        return [
            {
                "id": j["id"],
                "status": j["status"],
                "created_at": j["created_at"],
                "started_at": j["started_at"],
                "completed_at": j["completed_at"],
                "current_domain": j["current_domain"],
                "domains": j["request"].get("domains", []),
            }
            for j in _jobs.values()
        ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, tail: int = Query(500, ge=50, le=3000)) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_payload(job, tail)


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str, tail: int = Query(1200, ge=50, le=3000)):
    async def event_generator():
        last_fingerprint = None

        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    payload = {"error": "job not found", "job_id": job_id}
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                    return
                payload = _job_payload(job, tail)

            fingerprint = (
                payload["status"],
                payload["current_domain"],
                payload["completed_at"],
                len(payload["logs"]),
                len(payload["results"]),
            )

            if fingerprint != last_fingerprint:
                last_fingerprint = fingerprint
                yield f"event: snapshot\ndata: {json.dumps(payload)}\n\n"

            if payload["status"] in {"completed", "failed"}:
                done_payload = {"job_id": job_id, "status": payload["status"]}
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                return

            await asyncio.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/jobs/{job_id}/report/{domain}")
def get_report(job_id: str, domain: str, format: str = Query("json")):
    if format not in {"json", "md"}:
        raise HTTPException(status_code=400, detail="format must be json or md")

    result = _domain_result_or_404(job_id, domain)

    report_path = Path(result["report_json"] if format == "json" else result["report_md"]).resolve()
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report file missing")

    if format == "json":
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return JSONResponse(payload)
    return PlainTextResponse(report_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/generated-tests/{domain}")
def list_generated_tests(job_id: str, domain: str) -> Dict[str, Any]:
    result = _domain_result_or_404(job_id, domain)
    generated = _resolve_generated_tests(result)

    output_dir_raw = str(result.get("output_dir", "")).strip()
    output_dir = _canonical_path(output_dir_raw) if output_dir_raw else None
    items: List[Dict[str, Any]] = []
    for kind in sorted(generated.keys()):
        raw_path = generated[kind]
        resolved = _canonical_path(raw_path)
        is_within_output = bool(output_dir and resolved.is_relative_to(output_dir))
        exists = resolved.exists()
        size_bytes = resolved.stat().st_size if exists else None
        items.append(
            {
                "kind": kind,
                "path": raw_path,
                "exists": exists,
                "size_bytes": size_bytes,
                "safe_to_read": bool(is_within_output),
            }
        )

    return {
        "job_id": job_id,
        "domain": domain,
        "count": len(items),
        "generated_tests": items,
    }


@app.get("/api/jobs/{job_id}/generated-tests/{domain}/{kind}")
def get_generated_test_script(job_id: str, domain: str, kind: str):
    result = _domain_result_or_404(job_id, domain)
    script_path = _resolve_safe_generated_script_path(result, kind)
    return PlainTextResponse(script_path.read_text(encoding="utf-8"))


@app.get("/api/ping")
def api_ping() -> Dict[str, str]:
    return {
        "backend": "fastapi",
        "service": "qa_customer_api",
        "status": "ok",
    }


@app.post("/api/runs")
def create_run(req: RunRequest) -> Dict[str, Any]:
    response = create_job(req)
    return {"run_id": response.get("job_id"), "status": response.get("status", "queued")}


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    rows = list_jobs()
    return [
        {
            "run_id": item.get("id"),
            "status": item.get("status"),
            "created_at": item.get("created_at"),
            "started_at": item.get("started_at"),
            "completed_at": item.get("completed_at"),
            "current_domain": item.get("current_domain"),
            "domains": item.get("domains", []),
        }
        for item in rows
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, tail: int = Query(500, ge=50, le=3000)) -> Dict[str, Any]:
    payload = get_job(run_id, tail=tail)
    payload["run_id"] = payload.pop("id")
    return payload


@app.get("/api/runs/{run_id}/report")
def get_run_report(
    run_id: str,
    domain: Optional[str] = Query(default=None),
    format: str = Query("json"),
):
    if format not in {"json", "md"}:
        raise HTTPException(status_code=400, detail="format must be json or md")
    resolved_domain = _resolve_domain_for_run(run_id, domain)
    return get_report(run_id, resolved_domain, format=format)


@app.get("/api/runs/{run_id}/scripts")
def get_run_script(
    run_id: str,
    language: str = Query("python"),
    domain: Optional[str] = Query(default=None),
):
    resolved_domain = _resolve_domain_for_run(run_id, domain)
    normalized = str(language or "").strip().lower()
    kind_map = {
        "python": "python_pytest",
        "javascript": "javascript_jest",
        "js": "javascript_jest",
        "java": "java_restassured",
        "curl": "curl_script",
    }
    kind = kind_map.get(normalized, normalized)
    if kind not in SUPPORTED_SCRIPT_KINDS:
        raise HTTPException(status_code=400, detail="unsupported language")
    return get_generated_test_script(run_id, resolved_domain, kind)


@app.get("/api/runs/{run_id}/gam-context")
def get_run_gam_context(run_id: str, domain: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    resolved_domain = _resolve_domain_for_run(run_id, domain)
    result = _domain_result_or_404(run_id, resolved_domain)
    report_json = _canonical_path(result.get("report_json", ""))
    payload = _read_report_payload(report_json)
    return {
        "run_id": run_id,
        "domain": resolved_domain,
        "gam_context_pack": payload.get("gam_context_pack", {}),
        "gam_diagnostics": (payload.get("gam", {}) or {}).get("diagnostics", {}),
        "research_engine": (payload.get("gam", {}) or {}).get("research_engine", {}),
    }


@app.get("/api/runs/{run_id}/rl-decisions")
def get_run_rl_decisions(run_id: str, domain: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    resolved_domain = _resolve_domain_for_run(run_id, domain)
    result = _domain_result_or_404(run_id, resolved_domain)
    report_json = _canonical_path(result.get("report_json", ""))
    payload = _read_report_payload(report_json)
    return {
        "run_id": run_id,
        "domain": resolved_domain,
        "selection_decision_trace": payload.get("selection_decision_trace", []),
        "mutation_decision_trace": payload.get("mutation_decision_trace", []),
        "selection_policy": payload.get("selection_policy", {}),
        "mutation_policy": payload.get("mutation_policy", {}),
        "learning_feedback": (payload.get("learning", {}) or {}).get("feedback", {}),
    }


@app.get("/api/runs/{run_id}/learning-delta")
def get_run_learning_delta(
    run_id: str,
    from_run: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    resolved_domain = _resolve_domain_for_run(run_id, domain)
    result = _domain_result_or_404(run_id, resolved_domain)
    current_report = _read_report_payload(_canonical_path(result.get("report_json", "")))
    learning = current_report.get("learning", {}) or {}
    summary = current_report.get("summary", {}) or {}
    weak_deltas = current_report.get("weak_pattern_deltas", {}) or {}
    base_payload: Dict[str, Any] = {
        "run_id": run_id,
        "domain": resolved_domain,
        "learning_delta_status": learning.get("learning_delta_status", "unchanged"),
        "improvement_deltas": (learning.get("state_snapshot", {}) or {}).get("improvement_deltas", {}),
        "weak_pattern_deltas": weak_deltas,
        "summary": {
            "pass_rate": summary.get("pass_rate"),
            "meets_quality_gate": summary.get("meets_quality_gate"),
            "total_scenarios": summary.get("total_scenarios"),
        },
    }

    if from_run:
        from_domain = _resolve_domain_for_run(from_run, resolved_domain)
        from_result = _domain_result_or_404(from_run, from_domain)
        from_report = _read_report_payload(_canonical_path(from_result.get("report_json", "")))
        from_summary = from_report.get("summary", {}) or {}
        base_payload["comparison"] = {
            "from_run": from_run,
            "from_domain": from_domain,
            "pass_rate_delta": round(
                float(summary.get("pass_rate", 0.0)) - float(from_summary.get("pass_rate", 0.0)),
                4,
            ),
            "failed_scenarios_delta": int(summary.get("failed_scenarios", 0))
            - int(from_summary.get("failed_scenarios", 0)),
        }

    return base_payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qa_customer_api:app", host="127.0.0.1", port=8787, reload=False)
