'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

const DOMAINS = ['ecommerce', 'healthcare', 'logistics', 'hr'];
const DOMAIN_LABELS = {
  ecommerce: 'E-commerce',
  healthcare: 'Healthcare',
  logistics: 'Logistics',
  hr: 'HR'
};
const STEP_MARKERS = [
  { name: 'Spec Prepared', marker: '[OK] OpenAPI spec written' },
  { name: 'Run Started', marker: '[RUN] QA specialist agent' },
  { name: 'RL Session Started', marker: 'Started observability session' },
  { name: 'RL Training Executed', marker: 'RL training executed' },
  { name: 'QA Run Complete', marker: 'QA specialist run complete' },
  { name: 'Reports Written', marker: 'JSON report:' }
];
const FLOW_STEPS = [
  {
    id: 'request_accepted',
    name: '1) Request Accepted',
    marker: '[RUN] QA specialist agent',
    input: 'domains, tenant, baseUrl, thresholds',
    output: 'job_id and queued execution'
  },
  {
    id: 'openapi_prepared',
    name: '2) OpenAPI Prepared',
    marker: '[OK] OpenAPI spec written',
    input: 'domain prompt + template',
    output: 'openapi_under_test.yaml'
  },
  {
    id: 'scenario_selection',
    name: '3) Scenario Selection',
    marker: 'selection algorithm',
    input: 'candidate scenarios + uncertainty scores',
    output: 'selected scenarios for this run'
  },
  {
    id: 'isolated_execution',
    name: '4) Isolated Execution',
    marker: 'Dynamic Mock Server initialized',
    input: 'selected scenarios + mock server',
    output: 'scenario_results[] with actual status/time'
  },
  {
    id: 'rl_training',
    name: '5) RL Training',
    marker: 'RL training executed',
    input: 'decision signals + rewards/penalties',
    output: 'updated policy and checkpoint'
  },
  {
    id: 'reports_emitted',
    name: '6) Reports Emitted',
    marker: 'qa_execution_report.json',
    input: 'summary + learning state + traces',
    output: 'JSON + Markdown report files'
  }
];
const CUSTOMER_APIS = [
  {
    method: 'POST',
    path: '/api/jobs',
    purpose: 'Start one multi-domain agent run',
    body: '{ domains[], tenantId, maxScenarios, passThreshold, verifyPersistence, ... }'
  },
  {
    method: 'GET',
    path: '/api/jobs',
    purpose: 'List jobs and status',
    body: '-'
  },
  {
    method: 'GET',
    path: '/api/jobs/{jobId}?tail=1200',
    purpose: 'Read current job snapshot and logs',
    body: '-'
  },
  {
    method: 'GET',
    path: '/api/jobs/{jobId}/events',
    purpose: 'Realtime SSE snapshots (interactive UI stream)',
    body: '-'
  },
  {
    method: 'GET',
    path: '/api/jobs/{jobId}/report/{domain}?format=json|md',
    purpose: 'Fetch domain report',
    body: '-'
  },
  {
    method: 'GET',
    path: '/api/jobs/{jobId}/generated-tests/{domain}',
    purpose: 'List generated test scripts for one domain',
    body: '-'
  },
  {
    method: 'GET',
    path: '/api/jobs/{jobId}/generated-tests/{domain}/{kind}',
    purpose: 'Fetch generated script content',
    body: '-'
  }
];
const API_BASE = (process.env.NEXT_PUBLIC_BACKEND_BASE_URL || '').replace(/\/$/, '');
const API_URL = (path) => `${API_BASE}${path}`;
const CONNECTION_MODE = API_BASE ? 'Direct FastAPI Backend' : 'Next.js API Proxy';
const SCRIPT_KIND_LABELS = {
  python_pytest: 'Python / Pytest',
  javascript_jest: 'JavaScript / Jest',
  curl_script: 'cURL Script',
  java_restassured: 'Java / RestAssured'
};

function getField(obj, keys, fallback = null) {
  if (!obj) {
    return fallback;
  }
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, key) && obj[key] !== undefined) {
      return obj[key];
    }
  }
  return fallback;
}

function toPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 'n/a';
  }
  return `${(n * 100).toFixed(1)}%`;
}

function toNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeGeneratedTestItems(input) {
  if (!input) {
    return [];
  }
  if (Array.isArray(input)) {
    return input.map((item) => ({
      kind: String(getField(item, ['kind'], 'unknown')),
      path: String(getField(item, ['path'], '')),
      exists: Boolean(getField(item, ['exists'], false)),
      safe_to_read: Boolean(getField(item, ['safe_to_read'], false)),
      size_bytes: getField(item, ['size_bytes'], null)
    }));
  }
  if (typeof input === 'object') {
    return Object.entries(input).map(([kind, filePath]) => ({
      kind: String(kind),
      path: String(filePath || ''),
      exists: true,
      safe_to_read: true,
      size_bytes: null
    }));
  }
  return [];
}

export default function HomePage() {
  const [domains, setDomains] = useState(['ecommerce']);
  const [tenantId, setTenantId] = useState('customer_default');
  const [prompt, setPrompt] = useState('');
  const [maxScenarios, setMaxScenarios] = useState(16);
  const [passThreshold, setPassThreshold] = useState(0.7);
  const [baseUrl, setBaseUrl] = useState('http://localhost:8000');
  const [customerRoot, setCustomerRoot] = useState('~/.spec_test_pilot');
  const [customerMode, setCustomerMode] = useState(true);
  const [verifyPersistence, setVerifyPersistence] = useState(true);

  const [job, setJob] = useState(null);
  const [reportText, setReportText] = useState('Select a domain report to inspect.');
  const [selectedReportDomain, setSelectedReportDomain] = useState('');
  const [selectedReportFormat, setSelectedReportFormat] = useState('json');
  const [reportJson, setReportJson] = useState(null);
  const [scenarioFilter, setScenarioFilter] = useState('all');
  const [scenarioSearch, setScenarioSearch] = useState('');
  const [generatedTests, setGeneratedTests] = useState([]);
  const [generatedTestsDomain, setGeneratedTestsDomain] = useState('');
  const [selectedScriptKind, setSelectedScriptKind] = useState('');
  const [scriptText, setScriptText] = useState('Select a generated test script to preview.');
  const [outputDomain, setOutputDomain] = useState('');
  const [flashMessage, setFlashMessage] = useState('');
  const [running, setRunning] = useState(false);

  const timerRef = useRef(null);
  const eventSourceRef = useRef(null);
  const autoLoadedJobRef = useRef('');
  const flashTimerRef = useRef(null);

  const logText = useMemo(() => (job?.logs || []).join('\n'), [job]);
  const results = job?.results || {};
  const currentDomain = getField(job, ['currentDomain', 'current_domain'], 'none');
  const startedAt = getField(job, ['startedAt', 'started_at'], 'n/a');
  const completedAt = getField(job, ['completedAt', 'completed_at'], 'n/a');
  const jobId = getField(job, ['id'], '');
  const reportSummary = reportJson?.summary || null;
  const trainingStats = reportJson?.agent_lightning?.training_stats || {};
  const learningFeedback = reportJson?.learning?.feedback || {};
  const selectionPolicy = reportJson?.selection_policy || {};
  const stateSnapshot = reportJson?.learning?.state_snapshot || {};
  const scenarioResults = Array.isArray(reportJson?.scenario_results) ? reportJson.scenario_results : [];
  const topDecisions = Array.isArray(selectionPolicy?.top_decisions) ? selectionPolicy.top_decisions : [];
  const weakestPatterns = Array.isArray(stateSnapshot?.weakest_patterns) ? stateSnapshot.weakest_patterns : [];
  const rewardBreakdown = learningFeedback?.reward_breakdown || {};
  const resultSummaries = Object.values(results).map((r) => r?.summary || {});
  const resultDomains = Object.keys(results);
  const selectedDomainLabel = selectedReportDomain ? DOMAIN_LABELS[selectedReportDomain] || selectedReportDomain : 'none';
  const passedScenarioCount = scenarioResults.filter((row) => !!row?.passed).length;
  const failedScenarioCount = Math.max(0, scenarioResults.length - passedScenarioCount);
  const successDomainCount = Object.values(results).filter((result) => {
    const code = Number(getField(result, ['exitCode', 'return_code'], 1));
    return code === 0;
  }).length;
  const overallPassRate = reportJson ? toPct(getField(reportSummary, ['pass_rate'], null)) : 'n/a';

  const filteredScenarios = useMemo(() => {
    const needle = scenarioSearch.trim().toLowerCase();
    return scenarioResults.filter((row) => {
      const passed = !!row?.passed;
      if (scenarioFilter === 'pass' && !passed) {
        return false;
      }
      if (scenarioFilter === 'fail' && passed) {
        return false;
      }
      if (!needle) {
        return true;
      }
      const blob = [
        row?.name,
        row?.test_type,
        row?.method,
        row?.endpoint_template,
        row?.endpoint_resolved,
        String(row?.actual_status ?? ''),
        String(row?.expected_status ?? '')
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return blob.includes(needle);
    });
  }, [scenarioFilter, scenarioResults, scenarioSearch]);

  const reportSelectionSelected = toNumberOrNull(getField(selectionPolicy, ['selected_count', 'selectedCount'], null));
  const reportSelectionCandidates = toNumberOrNull(getField(selectionPolicy, ['candidate_count', 'candidateCount'], null));
  const summarySelectionSelected = resultSummaries
    .map((s) => toNumberOrNull(getField(s, ['selection_selected_count', 'selectionSelectedCount'], null)))
    .find((v) => v !== null);
  const summarySelectionCandidates = resultSummaries
    .map((s) => toNumberOrNull(getField(s, ['selection_candidate_count', 'selectionCandidateCount'], null)))
    .find((v) => v !== null);
  const selectionSelected = reportSelectionSelected ?? summarySelectionSelected;
  const selectionCandidates = reportSelectionCandidates ?? summarySelectionCandidates;
  const completedDomains = Object.keys(results).length;
  const reportReadyDomains = Object.values(results).filter((r) => {
    const jsonPath = getField(r, ['report_json', 'reportJsonPath'], '');
    const mdPath = getField(r, ['report_md', 'reportMdPath'], '');
    return Boolean(jsonPath || mdPath);
  }).length;
  const rlStepFromReport = toNumberOrNull(getField(trainingStats, ['rl_training_steps', 'rlTrainingSteps'], null));
  const rlStepFromSummary = resultSummaries
    .map((s) => toNumberOrNull(getField(s, ['rl_training_steps', 'rlTrainingSteps'], null)))
    .reduce((acc, value) => (value !== null && (acc === null || value > acc) ? value : acc), null);
  const rlStepValue = rlStepFromReport ?? rlStepFromSummary;

  function isFlowStepDone(stepId) {
    if (stepId === 'request_accepted') {
      return Boolean(jobId);
    }
    if (stepId === 'openapi_prepared') {
      return logText.includes('[OK] OpenAPI spec written') || completedDomains > 0;
    }
    if (stepId === 'scenario_selection') {
      return (
        (selectionSelected !== null && selectionCandidates !== null) ||
        logText.includes('selection_policy') ||
        logText.toLowerCase().includes('selected scenarios')
      );
    }
    if (stepId === 'isolated_execution') {
      return (
        logText.includes('Dynamic Mock Server initialized') ||
        completedDomains > 0 ||
        scenarioResults.length > 0
      );
    }
    if (stepId === 'rl_training') {
      return logText.includes('RL training executed') || (rlStepValue !== null && rlStepValue > 0);
    }
    if (stepId === 'reports_emitted') {
      return logText.includes('qa_execution_report.json') || reportReadyDomains > 0;
    }
    return false;
  }

  function flowStepActual(stepId) {
    if (stepId === 'scenario_selection') {
      if (selectionSelected !== null && selectionCandidates !== null) {
        return `actual: selected=${selectionSelected} / candidates=${selectionCandidates}`;
      }
      return 'actual: waiting for selection metrics';
    }
    if (stepId === 'isolated_execution') {
      return `actual: domains_completed=${completedDomains}`;
    }
    if (stepId === 'rl_training') {
      return `actual: rl_steps=${rlStepValue === null ? 'n/a' : rlStepValue}`;
    }
    if (stepId === 'reports_emitted') {
      return `actual: report_ready_domains=${reportReadyDomains}`;
    }
    return '';
  }

  function closeRealtimeConnection() {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }

  async function pollJob(jobId) {
    try {
      const reqUrl = API_URL(`/api/jobs/${jobId}?tail=1200`);
      const res = await fetch(reqUrl, { cache: 'no-store' });
      if (!res.ok) {
        return;
      }
      const payload = await res.json();
      setJob(payload);

      if (payload.status === 'running' || payload.status === 'queued') {
        timerRef.current = setTimeout(() => pollJob(jobId), 1500);
      } else {
        setRunning(false);
        closeRealtimeConnection();
      }
    } catch {
      timerRef.current = setTimeout(() => pollJob(jobId), 2000);
    }
  }

  function connectRealtime(jobId) {
    closeRealtimeConnection();

    const source = new EventSource(API_URL(`/api/jobs/${jobId}/events`));
    eventSourceRef.current = source;

    source.addEventListener('snapshot', (event) => {
      try {
        const payload = JSON.parse(event.data);
        setJob(payload);
        if (payload.status === 'completed' || payload.status === 'failed') {
          setRunning(false);
          closeRealtimeConnection();
        }
      } catch {
        // Ignore malformed snapshot payload.
      }
    });

    source.addEventListener('done', () => {
      setRunning(false);
      closeRealtimeConnection();
    });

    source.onerror = () => {
      source.close();
      eventSourceRef.current = null;
      // Fallback to polling if SSE stream disconnects during active run.
      if (jobId && running) {
        timerRef.current = setTimeout(() => pollJob(jobId), 1500);
      }
    };
  }

  async function onRun() {
    if (!domains.length) {
      alert('Select at least one domain.');
      return;
    }

    setRunning(true);
    setReportText('Select a domain report to inspect.');
    setReportJson(null);
    setSelectedReportDomain('');
    setSelectedReportFormat('json');
    setGeneratedTests([]);
    setGeneratedTestsDomain('');
    setSelectedScriptKind('');
    setScriptText('Select a generated test script to preview.');
    autoLoadedJobRef.current = '';

    const body = {
      domains,
      tenantId,
      prompt: prompt.trim() || null,
      maxScenarios,
      passThreshold,
      baseUrl,
      customerMode,
      verifyPersistence,
      customerRoot
    };

    const reqUrl = API_URL('/api/jobs');
    const res = await fetch(reqUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });

    const payload = await res.json();
    if (!res.ok) {
      setRunning(false);
      alert(payload.error || 'Failed to start run');
      return;
    }

    const startedJobId = getField(payload, ['jobId', 'job_id'], '');
    if (!startedJobId) {
      setRunning(false);
      alert('Backend response missing job id');
      return;
    }
    connectRealtime(startedJobId);
  }

  async function openReport(domain, format) {
    if (!job?.id) {
      return;
    }
    setSelectedReportDomain(domain);
    setSelectedReportFormat(format);

    const reqUrl = API_URL(`/api/jobs/${job.id}/report/${domain}?format=${format}`);
    const res = await fetch(reqUrl, {
      cache: 'no-store'
    });
    if (!res.ok) {
      setReportJson(null);
      setReportText(`Failed to open ${format.toUpperCase()} report for ${domain}`);
      return;
    }

    if (format === 'json') {
      const payload = await res.json();
      setReportJson(payload);
      setReportText(JSON.stringify(payload, null, 2));
      const fromReport = normalizeGeneratedTestItems(payload?.generated_test_files);
      setGeneratedTests(fromReport);
      setGeneratedTestsDomain(domain);
      setSelectedScriptKind('');
      setScriptText('Select a generated test script to preview.');
      await loadGeneratedTests(domain, fromReport);
      return;
    }
    setReportText(await res.text());
  }

  async function loadGeneratedTests(domain, fallbackItems = []) {
    if (!job?.id) {
      return;
    }
    const reqUrl = API_URL(`/api/jobs/${job.id}/generated-tests/${domain}`);
    try {
      const res = await fetch(reqUrl, { cache: 'no-store' });
      if (!res.ok) {
        if (fallbackItems.length > 0) {
          setGeneratedTests(fallbackItems);
          setGeneratedTestsDomain(domain);
        }
        return;
      }
      const payload = await res.json();
      const items = normalizeGeneratedTestItems(getField(payload, ['generated_tests', 'generatedTests'], []));
      setGeneratedTests(items);
      setGeneratedTestsDomain(domain);
    } catch {
      if (fallbackItems.length > 0) {
        setGeneratedTests(fallbackItems);
        setGeneratedTestsDomain(domain);
      }
    }
  }

  async function openGeneratedScript(domain, kind) {
    if (!job?.id) {
      return;
    }
    setSelectedScriptKind(kind);
    const reqUrl = API_URL(`/api/jobs/${job.id}/generated-tests/${domain}/${kind}`);
    try {
      const res = await fetch(reqUrl, { cache: 'no-store' });
      if (!res.ok) {
        const errText = await res.text();
        setScriptText(`Failed to load script ${kind}: ${errText}`);
        return;
      }
      const raw = await res.text();
      setScriptText(raw || `Script ${kind} is empty.`);
    } catch (error) {
      setScriptText(`Failed to load script ${kind}: ${String(error)}`);
    }
  }

  function setFlash(text) {
    setFlashMessage(text);
    if (flashTimerRef.current) {
      window.clearTimeout(flashTimerRef.current);
    }
    flashTimerRef.current = window.setTimeout(() => {
      setFlashMessage('');
    }, 1800);
  }

  async function copyText(label, value) {
    const content = String(value || '').trim();
    if (!content) {
      setFlash(`No ${label} content to copy`);
      return;
    }
    try {
      await navigator.clipboard.writeText(content);
      setFlash(`${label} copied`);
    } catch {
      setFlash(`Failed to copy ${label}`);
    }
  }

  async function openSelectedDomainOutput(format = 'json') {
    if (!outputDomain) {
      setFlash('Select a domain first');
      return;
    }
    await openReport(outputDomain, format);
  }

  function toggleDomain(domain) {
    setDomains((prev) => {
      if (prev.includes(domain)) {
        return prev.filter((d) => d !== domain);
      }
      return [...prev, domain];
    });
  }

  useEffect(() => {
    if (!jobId || selectedReportDomain || resultDomains.length === 0) {
      return;
    }
    if (autoLoadedJobRef.current === jobId) {
      return;
    }
    autoLoadedJobRef.current = jobId;
    const firstDomain = resultDomains[0];
    void openReport(firstDomain, 'json');
  }, [jobId, selectedReportDomain, resultDomains.length]);

  useEffect(() => {
    if (resultDomains.length === 0) {
      if (outputDomain) {
        setOutputDomain('');
      }
      return;
    }
    if (!outputDomain || !resultDomains.includes(outputDomain)) {
      setOutputDomain(resultDomains[0]);
    }
  }, [resultDomains, outputDomain]);

  useEffect(() => {
    return () => {
      closeRealtimeConnection();
      if (flashTimerRef.current) {
        window.clearTimeout(flashTimerRef.current);
      }
    };
  }, []);

  return (
    <main className="page">
      <header className="header">
        <div className="header-row">
          <div>
            <h1>SpecTestPilot QA Workspace</h1>
            <p className="header-subtitle">Run QA agent, review scripts, inspect tested cases, and deliver reports.</p>
          </div>
          <div className="header-badges">
            <span className="header-badge">status={job?.status || 'idle'}</span>
            <span className="header-badge">domains={resultDomains.length}</span>
            <span className="header-badge">success={successDomainCount}</span>
            <span className="header-badge">pass_rate={overallPassRate}</span>
          </div>
        </div>
        {flashMessage && <div className="toast">{flashMessage}</div>}
      </header>

      <div className="layout">
        <section className="card">
          <h2>Run Configuration</h2>

          <div className="field">
            <label>Domains</label>
            <div className="domains">
              {DOMAINS.map((domain) => (
                <label className="domain" key={domain}>
                  <input
                    type="checkbox"
                    checked={domains.includes(domain)}
                    onChange={() => toggleDomain(domain)}
                  />{' '}
                  {domain}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Tenant ID</label>
            <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
          </div>

          <div className="field">
            <label>Max Scenarios</label>
            <input
              type="number"
              min={1}
              max={500}
              value={maxScenarios}
              onChange={(e) => setMaxScenarios(Number(e.target.value || 16))}
            />
          </div>

          <div className="field">
            <label>Pass Threshold (0..1)</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={passThreshold}
              onChange={(e) => setPassThreshold(Number(e.target.value || 0.7))}
            />
          </div>

          <div className="field">
            <label>Base URL</label>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
          </div>

          <div className="field">
            <label>Customer Root</label>
            <input value={customerRoot} onChange={(e) => setCustomerRoot(e.target.value)} />
          </div>

          <div className="field">
            <label>Prompt (optional)</label>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </div>

          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={customerMode}
                onChange={(e) => setCustomerMode(e.target.checked)}
              />{' '}
              customer mode (persistent workspace/checkpoint)
            </label>
            <label>
              <input
                type="checkbox"
                checked={verifyPersistence}
                onChange={(e) => setVerifyPersistence(e.target.checked)}
              />{' '}
              verify persistence (auto second pass)
            </label>
          </div>

          <button className="primary" disabled={running} onClick={onRun}>
            {running ? 'Running...' : 'Run QA Agent'}
          </button>
        </section>

        <section className="right">
          <div className="card">
            <h2>Runtime Status</h2>
            <div className="status">
              <span className={`dot ${job?.status || ''}`} />
              <strong>{job?.status || 'idle'}</strong>
            </div>
            <div className="meta">
              {job
                ? `job=${jobId} | current_domain=${currentDomain} | started=${startedAt} | completed=${completedAt}`
                : 'No run started yet.'}
            </div>
            <div className="steps">
              {STEP_MARKERS.map((step) => {
                const done = logText.includes(step.marker);
                return (
                  <div key={step.name} className={`step ${done ? 'done' : ''}`}>
                    {done ? '✓' : '•'} {step.name}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <details className="advanced">
              <summary>How It Works (API + Agent Flow)</summary>
              <div className="small">
                mode={CONNECTION_MODE}
                <br />
                backend={API_BASE || 'same-origin Next.js API routes'}
              </div>
              <div className="api-list">
                {CUSTOMER_APIS.map((item) => (
                  <div key={`${item.method}-${item.path}`} className="api-item">
                    <div className="api-head">
                      <span className={`method ${item.method.toLowerCase()}`}>{item.method}</span>
                      <code>{item.path}</code>
                    </div>
                    <div className="small">{item.purpose}</div>
                    <div className="small">payload: {item.body}</div>
                  </div>
                ))}
              </div>
              <div className="flow-list">
                {FLOW_STEPS.map((step) => {
                  const done = isFlowStepDone(step.id);
                  const actual = flowStepActual(step.id);
                  return (
                    <div key={step.name} className={`flow-item ${done ? 'done' : ''}`}>
                      <div className="flow-name">{step.name}</div>
                      <div className="small">input: {step.input}</div>
                      <div className="small">output: {step.output}</div>
                      {actual && <div className="small">{actual}</div>}
                    </div>
                  );
                })}
              </div>
            </details>
          </div>

          <div className="card">
            <h2>Domain Results</h2>
            <div className="results">
              {Object.keys(results).length === 0 && <div className="small">No domain results yet.</div>}
              {Object.entries(results).map(([domain, result]) => {
                const code = Number(getField(result, ['exitCode', 'return_code'], 1));
                const ok = code === 0;
                const s = result.summary || {};
                const generatedCount = normalizeGeneratedTestItems(
                  getField(result, ['generated_tests', 'generatedTests'], [])
                ).length;
                const passRate = getField(s, ['passRate', 'pass_rate'], null);
                const totalScenarios = getField(s, ['totalScenarios', 'total_scenarios'], 'n/a');
                const passedScenarios = getField(s, ['passedScenarios', 'passed_scenarios'], 'n/a');
                const failedScenarios = getField(s, ['failedScenarios', 'failed_scenarios'], 'n/a');
                const rlSteps = getField(s, ['rlTrainingSteps', 'rl_training_steps'], 'n/a');
                const rlBuffer = getField(s, ['rlBufferSize', 'rl_buffer_size'], 'n/a');
                return (
                  <div className="result" key={domain}>
                    <h3>
                      {DOMAIN_LABELS[domain] || domain}
                      <span className={`pill ${ok ? 'ok' : 'bad'}`}>{ok ? 'ok' : 'failed'}</span>
                    </h3>
                    <div className="small">
                      pass_rate={toPct(passRate)}
                      <br />
                      total={String(totalScenarios)} passed={String(passedScenarios)} failed={String(failedScenarios)}
                      <br />
                      rl_steps={String(rlSteps)} rl_buffer={String(rlBuffer)}
                      <br />
                      return_code={String(code)}
                      <br />
                      generated_scripts={String(generatedCount)}
                    </div>
                    <button onClick={() => openReport(domain, 'json')}>Open Customer Output</button>
                    <button onClick={() => openReport(domain, 'md')}>Open Markdown Report</button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <h2>Customer Output</h2>
            <div className="small">
              selected_domain={selectedDomainLabel}
              <br />
              output_flow=1_scripts {'->'} 2_cases_tested {'->'} 3_final_report
            </div>
            <div className="output-toolbar">
              <select
                value={outputDomain}
                onChange={(e) => setOutputDomain(e.target.value)}
                disabled={resultDomains.length === 0}
              >
                {resultDomains.length === 0 && <option value="">No domains yet</option>}
                {resultDomains.map((domain) => (
                  <option key={domain} value={domain}>
                    {DOMAIN_LABELS[domain] || domain}
                  </option>
                ))}
              </select>
              <button disabled={!outputDomain} onClick={() => openSelectedDomainOutput('json')}>
                Load Output
              </button>
              <button disabled={!outputDomain} onClick={() => openSelectedDomainOutput('md')}>
                Load Markdown
              </button>
              <button onClick={() => copyText('report', reportText)}>Copy Report Text</button>
            </div>
            {!selectedReportDomain && resultDomains.length > 0 && (
              <button onClick={() => openReport(resultDomains[0], 'json')}>Load First Domain Output</button>
            )}
          </div>

          <div className="card">
            <h2>1) Test Scripts Generated By Agent</h2>
            <div className="small">
              selected_domain={generatedTestsDomain || 'none'} | scripts={generatedTests.length}
            </div>
            {generatedTests.length === 0 && (
              <div className="small">Open customer output for a domain to view generated scripts.</div>
            )}
            {generatedTests.length > 0 && (
              <div className="script-list">
                {generatedTests.map((item) => (
                  <div className="script-item" key={`${generatedTestsDomain}-${item.kind}`}>
                    <div className="script-head">
                      <strong>{SCRIPT_KIND_LABELS[item.kind] || item.kind}</strong>
                      <span className={`pill ${item.exists ? 'ok' : 'bad'}`}>
                        {item.exists ? 'ready' : 'missing'}
                      </span>
                    </div>
                    <div className="small">kind={item.kind}</div>
                    <div className="small">path={item.path || 'n/a'}</div>
                    <div className="small">
                      size={item.size_bytes ?? 'n/a'} | safe_to_read={String(item.safe_to_read)}
                    </div>
                    <button
                      disabled={!item.exists || !item.safe_to_read || !generatedTestsDomain}
                      onClick={() => openGeneratedScript(generatedTestsDomain, item.kind)}
                    >
                      Preview Script
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="small">selected_script={selectedScriptKind || 'none'}</div>
            <div className="inline-actions">
              <button onClick={() => copyText('script', scriptText)}>Copy Script</button>
            </div>
            <pre>{scriptText}</pre>
          </div>

          <div className="card">
            <h2>2) Cases Tested By Agent</h2>
            <div className="small">
              selected_domain={selectedDomainLabel} | total={scenarioResults.length} | passed={passedScenarioCount} |
              failed={failedScenarioCount}
            </div>
            {!reportJson && (
              <div className="small">Open customer output for a domain to view tested cases.</div>
            )}
            {reportJson && (
              <div className="scenario-panel">
                <div className="scenario-head">
                  <h3>Scenario Results</h3>
                  <div className="scenario-controls">
                    <select value={scenarioFilter} onChange={(e) => setScenarioFilter(e.target.value)}>
                      <option value="all">all</option>
                      <option value="pass">passed</option>
                      <option value="fail">failed</option>
                    </select>
                    <input
                      placeholder="Search scenario name, endpoint, method..."
                      value={scenarioSearch}
                      onChange={(e) => setScenarioSearch(e.target.value)}
                    />
                  </div>
                </div>
                <div className="inline-actions">
                  <button onClick={() => setScenarioFilter('all')}>Show All</button>
                  <button onClick={() => setScenarioFilter('pass')}>Show Passed</button>
                  <button onClick={() => setScenarioFilter('fail')}>Show Failed</button>
                </div>
                <div className="small">showing {filteredScenarios.length} / {scenarioResults.length}</div>
                <div className="scenario-table-wrap">
                  <table className="scenario-table">
                    <thead>
                      <tr>
                        <th>status</th>
                        <th>name</th>
                        <th>type</th>
                        <th>endpoint</th>
                        <th>expected</th>
                        <th>actual</th>
                        <th>ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredScenarios.map((row) => (
                        <tr key={String(row.name)}>
                          <td>{row.passed ? 'pass' : 'fail'}</td>
                          <td>{row.name}</td>
                          <td>{row.test_type}</td>
                          <td>{`${row.method || ''} ${row.endpoint_template || ''}`.trim()}</td>
                          <td>{String(row.expected_status ?? 'n/a')}</td>
                          <td>{String(row.actual_status ?? 'n/a')}</td>
                          <td>{String(row.duration_ms ?? 'n/a')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <h2>3) Final QA Report</h2>
            <div className="small">
              selected_domain={selectedDomainLabel} | format={selectedReportFormat}
            </div>
            {selectedReportDomain && (
              <div className="inline-actions">
                <button onClick={() => openReport(selectedReportDomain, 'json')}>JSON View</button>
                <button onClick={() => openReport(selectedReportDomain, 'md')}>Markdown View</button>
                <button onClick={() => copyText('report', reportText)}>Copy Current Report</button>
              </div>
            )}
            {reportJson && (
              <div className="report-grid">
                <div className="report-metric">
                  <span>Total</span>
                  <strong>{String(getField(reportSummary, ['total_scenarios'], 'n/a'))}</strong>
                </div>
                <div className="report-metric">
                  <span>Pass Rate</span>
                  <strong>{toPct(getField(reportSummary, ['pass_rate'], null))}</strong>
                </div>
                <div className="report-metric">
                  <span>Quality Gate</span>
                  <strong>{String(getField(reportSummary, ['meets_quality_gate'], 'n/a'))}</strong>
                </div>
                <div className="report-metric">
                  <span>RL Steps</span>
                  <strong>{String(getField(trainingStats, ['rl_training_steps'], 'n/a'))}</strong>
                </div>
                <div className="report-metric">
                  <span>RL Buffer</span>
                  <strong>{String(getField(trainingStats, ['rl_buffer_size'], 'n/a'))}</strong>
                </div>
                <div className="report-metric">
                  <span>Run Reward</span>
                  <strong>{String(getField(learningFeedback, ['run_reward'], 'n/a'))}</strong>
                </div>
              </div>
            )}
            <pre>{reportText}</pre>
          </div>

          <div className="card">
            <h2>Live Process Log</h2>
            <pre>{logText || 'No logs yet.'}</pre>
          </div>

          <div className="card">
            <details className="advanced">
              <summary>Advanced Agent R&D (optional)</summary>
              {!reportJson && (
                <div className="small">Load a JSON report to inspect decision policy and learning internals.</div>
              )}
              {reportJson && (
                <>
                <div className="report-grid">
                  <div className="report-metric">
                    <span>Policy</span>
                    <strong>{String(getField(selectionPolicy, ['algorithm'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Candidates</span>
                    <strong>{String(getField(selectionPolicy, ['candidate_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Selected</span>
                    <strong>{String(getField(selectionPolicy, ['selected_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Uncertain Selected</span>
                    <strong>{String(getField(selectionPolicy, ['uncertain_selected_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Rewarded Decisions</span>
                    <strong>{String(getField(learningFeedback, ['rewarded_decisions'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Penalized Decisions</span>
                    <strong>{String(getField(learningFeedback, ['penalized_decisions'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Run Count</span>
                    <strong>{String(getField(stateSnapshot, ['run_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Tracked Patterns</span>
                    <strong>{String(getField(stateSnapshot, ['scenario_patterns_tracked'], 'n/a'))}</strong>
                  </div>
                </div>

                <div className="reward-grid">
                  <div className="reward-box">
                    <span>Pass Rate Component</span>
                    <strong>{String(getField(rewardBreakdown, ['pass_rate_component'], 'n/a'))}</strong>
                  </div>
                  <div className="reward-box">
                    <span>Coverage Component</span>
                    <strong>{String(getField(rewardBreakdown, ['coverage_component'], 'n/a'))}</strong>
                  </div>
                  <div className="reward-box">
                    <span>Failure Component</span>
                    <strong>{String(getField(rewardBreakdown, ['failure_component'], 'n/a'))}</strong>
                  </div>
                  <div className="reward-box">
                    <span>Latency Penalty</span>
                    <strong>{String(getField(rewardBreakdown, ['latency_penalty_component'], 'n/a'))}</strong>
                  </div>
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>Top Selection Decisions</h3>
                  </div>
                  <div className="small">showing {Math.min(topDecisions.length, 15)} / {topDecisions.length}</div>
                  <div className="scenario-table-wrap">
                    <table className="scenario-table">
                      <thead>
                        <tr>
                          <th>name</th>
                          <th>type</th>
                          <th>reason</th>
                          <th>score</th>
                          <th>uncertainty</th>
                          <th>expected_reward</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topDecisions.slice(0, 15).map((row) => (
                          <tr key={String(row.name)}>
                            <td>{row.name}</td>
                            <td>{row.test_type}</td>
                            <td>{row.selection_reason}</td>
                            <td>{String(row.score ?? 'n/a')}</td>
                            <td>{String(row.uncertainty ?? 'n/a')}</td>
                            <td>{String(row.expected_reward ?? 'n/a')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>Weakest Patterns (Needs Improvement)</h3>
                  </div>
                  <div className="small">showing {Math.min(weakestPatterns.length, 12)} / {weakestPatterns.length}</div>
                  <div className="scenario-table-wrap">
                    <table className="scenario-table">
                      <thead>
                        <tr>
                          <th>fingerprint</th>
                          <th>failure_rate</th>
                          <th>attempts</th>
                          <th>avg_reward</th>
                        </tr>
                      </thead>
                      <tbody>
                        {weakestPatterns.slice(0, 12).map((row) => (
                          <tr key={String(row.fingerprint)}>
                            <td>{row.fingerprint}</td>
                            <td>{String(row.failure_rate ?? 'n/a')}</td>
                            <td>{String(row.attempts ?? 'n/a')}</td>
                            <td>{String(row.avg_reward ?? 'n/a')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                </>
              )}
            </details>
          </div>
        </section>
      </div>
    </main>
  );
}
