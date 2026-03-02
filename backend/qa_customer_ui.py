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


APP_TITLE = "SpecTestPilot Customer QA UI"
SUPPORTED_DOMAINS = ["ecommerce", "healthcare", "logistics", "hr"]
REPO_ROOT = Path(__file__).resolve().parent
JOB_ROOT = Path("/tmp/qa_ui_runs")
CHECKPOINT_ROOT = Path("/tmp/qa_ui_checkpoints")
JOB_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_LOG_LINES = 6000

app = FastAPI(title=APP_TITLE)
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "QA_UI_ALLOWED_ORIGINS",
        "http://localhost:3001,http://127.0.0.1:3001,http://localhost:3000,http://127.0.0.1:3000",
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


class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    domains: List[str] = Field(default_factory=lambda: ["ecommerce"])
    tenant_id: str = Field(default="customer_default", alias="tenantId")
    prompt: Optional[str] = None
    max_scenarios: int = Field(default=16, alias="maxScenarios")
    pass_threshold: float = Field(default=0.70, alias="passThreshold")
    base_url: str = Field(default="http://localhost:8000", alias="baseUrl")
    customer_mode: bool = Field(default=True, alias="customerMode")
    verify_persistence: bool = Field(default=True, alias="verifyPersistence")
    customer_root: str = Field(default=str(Path.home() / ".spec_test_pilot"), alias="customerRoot")

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, value: List[str]) -> List[str]:
        normalized = [str(v).strip().lower() for v in value if str(v).strip()]
        if not normalized:
            raise ValueError("Select at least one domain")
        invalid = [v for v in normalized if v not in SUPPORTED_DOMAINS]
        if invalid:
            raise ValueError(f"Unsupported domains: {', '.join(invalid)}")
        return list(dict.fromkeys(normalized))

    @field_validator("max_scenarios")
    @classmethod
    def validate_max_scenarios(cls, value: int) -> int:
        if value < 1:
            return 1
        if value > 500:
            return 500
        return value

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
        "rl_training_steps": training_stats.get("rl_training_steps"),
        "rl_buffer_size": training_stats.get("rl_buffer_size"),
        "selection_algorithm": payload.get("selection_policy", {}).get("algorithm"),
        "selection_selected_count": payload.get("selection_policy", {}).get("selected_count"),
        "selection_candidate_count": payload.get("selection_policy", {}).get("candidate_count"),
        "run_reward": payload.get("learning", {}).get("feedback", {}).get("run_reward"),
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


def _resolve_safe_generated_script_path(result: Dict[str, Any], kind: str) -> Path:
    generated = _resolve_generated_tests(result)
    script_path_raw = generated.get(kind)
    if not script_path_raw:
        raise HTTPException(status_code=404, detail=f"generated test script kind not found: {kind}")

    script_path = Path(script_path_raw).resolve()
    output_dir_raw = str(result.get("output_dir", "")).strip()
    if not output_dir_raw:
        raise HTTPException(status_code=500, detail="missing output directory for this run result")
    output_dir = Path(output_dir_raw).resolve()
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

        cmd = [
            "bash",
            str(REPO_ROOT / "run_qa_domain.sh"),
            "--domain",
            domain,
            "--action",
            "both",
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
            "--rl-checkpoint",
            str(checkpoint),
        ]

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

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
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
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint),
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
  <title>SpecTestPilot Customer QA UI</title>
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
    input[type=text], input[type=number], textarea {
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
    <h1>SpecTestPilot Customer QA Runner</h1>
  </header>

  <div class=\"wrap\">
    <section class=\"card\">
      <h2>Run Configuration</h2>
      <div class=\"field\">
        <label>Domains</label>
        <div class=\"domains\">
          <label class=\"domain-item\"><input type=\"checkbox\" name=\"domain\" value=\"ecommerce\" checked> ecommerce</label>
          <label class=\"domain-item\"><input type=\"checkbox\" name=\"domain\" value=\"healthcare\"> healthcare</label>
          <label class=\"domain-item\"><input type=\"checkbox\" name=\"domain\" value=\"logistics\"> logistics</label>
          <label class=\"domain-item\"><input type=\"checkbox\" name=\"domain\" value=\"hr\"> hr</label>
        </div>
      </div>

      <div class=\"field\">
        <label>Tenant ID</label>
        <input id=\"tenant\" type=\"text\" value=\"customer_default\" />
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
      return Array.from(document.querySelectorAll('input[name="domain"]:checked')).map(i => i.value);
    }

    async function startRun() {
      const domains = selectedDomains();
      if (!domains.length) {
        alert('Select at least one domain.');
        return;
      }

      const body = {
        domains,
        tenant_id: document.getElementById('tenant').value.trim() || 'customer_default',
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
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "current_domain": None,
        "request": req.model_dump(),
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
    output_dir = Path(output_dir_raw).resolve() if output_dir_raw else None
    items: List[Dict[str, Any]] = []
    for kind in sorted(generated.keys()):
        raw_path = generated[kind]
        resolved = Path(raw_path).resolve()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qa_customer_ui:app", host="127.0.0.1", port=8787, reload=False)
