'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';

const STEP_MARKERS = [
  { name: 'Spec Prepared', marker: '[OK] OpenAPI spec written' },
  { name: 'Run Started', marker: '[RUN] QA specialist agent' },
  { name: 'Adaptive Session Started', marker: 'Started observability session' },
  { name: 'Signals Buffered', marker: 'buffered_only' },
  { name: 'QA Run Complete', marker: 'QA specialist run complete' },
  { name: 'Reports Written', marker: 'JSON report:' }
];
const FLOW_STEPS = [
  {
    id: 'request_accepted',
    name: '1) Request Accepted',
    marker: '[RUN] QA specialist agent',
    input: 'domains, tenant, baseUrl, thresholds, script kind',
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
    id: 'learning_buffer',
    name: '5) Signal Buffering',
    marker: 'buffered_only',
    input: 'decision signals + rewards',
    output: 'signals buffered; scheduled trainer updates model later'
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
    body: '{ domains[], specPaths{domain:path}, tenantId, scriptKind, maxScenarios, passThreshold, verifyPersistence, ... }'
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
const DEFAULT_TEST_BASE_URL = (process.env.NEXT_PUBLIC_TEST_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const API_URL = (path) => `${API_BASE}${path}`;
const CONNECTION_MODE = API_BASE ? 'Direct FastAPI Backend' : 'Next.js API Proxy';
const SCRIPT_KIND_LABELS = {
  python_pytest: 'Python / Pytest',
  javascript_jest: 'JavaScript / Jest',
  curl_script: 'cURL Script',
  java_restassured: 'Java / RestAssured'
};
const SCRIPT_KINDS = ['python_pytest', 'javascript_jest', 'curl_script', 'java_restassured'];
const DOMAIN_PRESET_OPTIONS = [
  { value: 'ecommerce', label: 'E-commerce API (preset)' },
  { value: 'healthcare', label: 'Healthcare Appointments API (preset)' },
  { value: 'logistics', label: 'Logistics Shipment API (preset)' },
  { value: 'hr', label: 'HR Recruitment API (preset)' }
];
const OPENAPI_SOURCE_OPTIONS = [
  { value: 'preset', label: 'Use preset OpenAPI template (auto generated)' },
  { value: 'custom_path', label: 'Use custom OpenAPI file path' }
];

function sanitizeDomainToken(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64);
}

function parseDomainList(raw) {
  const tokens = String(raw || '')
    .split(/[\n,]/)
    .map((item) => sanitizeDomainToken(item))
    .filter(Boolean);
  return [...new Set(tokens)];
}

function parseSpecPathsMap(raw) {
  const out = {};
  const lines = String(raw || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    const eq = line.indexOf('=');
    if (eq <= 0) {
      continue;
    }
    const domain = sanitizeDomainToken(line.slice(0, eq));
    const specPath = line.slice(eq + 1).trim();
    if (!domain || !specPath) {
      continue;
    }
    out[domain] = specPath;
  }
  return out;
}

function formatDomainLabel(domain) {
  const clean = String(domain || '').trim();
  if (!clean) {
    return 'n/a';
  }
  return clean
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function toTitleWords(value) {
  const clean = String(value || '')
    .trim()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ');
  if (!clean) {
    return '';
  }
  return clean
    .split(' ')
    .map((token) => {
      if (/^\d+$/.test(token)) {
        return token;
      }
      if (token === token.toUpperCase() && token.length > 1) {
        return token;
      }
      return token.charAt(0).toUpperCase() + token.slice(1).toLowerCase();
    })
    .join(' ');
}

function normalizeFieldToken(token) {
  const clean = String(token || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!clean) {
    return '';
  }
  if (clean.endsWith('id') && clean.length > 2) {
    return `${clean.slice(0, -2)} id`;
  }
  return clean.replace(/_/g, ' ');
}

function extractRlTagsFromScenarioName(name) {
  const raw = String(name || '').trim();
  if (!raw.includes('_rl_')) {
    return [];
  }
  const pieces = raw.split('_rl_').slice(1);
  return pieces
    .map((item) => String(item || '').trim())
    .filter(Boolean);
}

function humanizeMutationTag(tag) {
  const normalized = String(tag || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!normalized) {
    return '';
  }
  if (normalized === 'missing_auth') {
    return 'missing auth token';
  }
  if (normalized === 'invalid_auth') {
    return 'invalid auth token';
  }
  if (normalized === 'query_fuzz') {
    return 'unexpected query parameter payload';
  }
  if (normalized === 'path_not_found') {
    return 'path not found';
  }
  if (normalized === 'adaptive_status_hypothesis') {
    return 'nonexistent resource id check';
  }
  if (normalized.startsWith('missing_required_')) {
    return `missing required field: ${normalizeFieldToken(normalized.slice('missing_required_'.length))}`;
  }
  if (normalized.startsWith('learned_missing_required_')) {
    return `missing required field: ${normalizeFieldToken(normalized.slice('learned_missing_required_'.length))}`;
  }
  if (normalized.startsWith('learned_missing_')) {
    return `missing field: ${normalizeFieldToken(normalized.slice('learned_missing_'.length))}`;
  }
  if (normalized.startsWith('missing_')) {
    return `missing field: ${normalizeFieldToken(normalized.slice('missing_'.length))}`;
  }
  if (normalized.startsWith('learned_below_min_')) {
    return `${normalizeFieldToken(normalized.slice('learned_below_min_'.length))} below minimum`;
  }
  if (normalized.startsWith('below_min_')) {
    return `${normalizeFieldToken(normalized.slice('below_min_'.length))} below minimum`;
  }
  if (normalized.startsWith('history_seed_')) {
    return `${toTitleWords(normalized.slice('history_seed_'.length))} (history seeded)`;
  }
  return toTitleWords(normalized);
}

function humanizeScenarioName(row) {
  const displayFromBackend = String(row?.display_name || row?.displayName || '').trim();
  const rawNameFromBackend = String(row?.name_raw || row?.nameRaw || row?.name || '').trim();
  if (displayFromBackend) {
    return {
      label: displayFromBackend,
      rawName: rawNameFromBackend
    };
  }
  const method = String(row?.method || '').trim().toUpperCase();
  const endpoint = String(row?.endpoint_template || row?.endpoint || row?.endpoint_resolved || '').trim();
  const rawName = rawNameFromBackend;
  const cleanedName = rawName.replace(/^test_/i, '').trim();
  const baseName = cleanedName.split('_rl_')[0] || cleanedName;
  const tags = extractRlTagsFromScenarioName(cleanedName);
  const mutationFromBackend = String(row?.mutation_strategy || row?.mutationStrategy || '').trim();
  if (mutationFromBackend && mutationFromBackend.toLowerCase() !== 'unknown' && !tags.length) {
    tags.push(mutationFromBackend);
  }
  const historySeeded = tags.some((item) => item.startsWith('history_seed_'));
  const mutationTag = [...tags].reverse().find((item) => !item.startsWith('history_seed_')) || '';
  const mutationLabel = humanizeMutationTag(mutationTag);
  const fallbackIntent = toTitleWords(String(row?.test_type || '').replace(/_/g, ' ')) || toTitleWords(baseName);
  let intent = mutationLabel || fallbackIntent || 'Scenario';
  if (historySeeded && !intent.toLowerCase().includes('history seeded')) {
    intent = `${intent} (history seeded)`;
  }
  intent = intent.charAt(0).toUpperCase() + intent.slice(1);
  const endpointLabel = [method, endpoint].filter(Boolean).join(' ').trim();
  const label = endpointLabel ? `${endpointLabel} - ${intent}` : intent;
  return {
    label,
    rawName
  };
}

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
  return new Intl.NumberFormat(undefined, {
    style: 'percent',
    maximumFractionDigits: 1
  }).format(n);
}

function toNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatBytes(value) {
  const size = Number(value);
  if (!Number.isFinite(size) || size < 0) {
    return 'n/a';
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDelta(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 'n/a';
  }
  const sign = n > 0 ? '+' : '';
  return `${sign}${new Intl.NumberFormat(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(n)}`;
}

function formatDateTime(value) {
  const raw = String(value || '').trim();
  if (!raw || raw.toLowerCase() === 'n/a') {
    return 'n/a';
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short'
  }).format(parsed);
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

function findLastLogMatch(lines, regex) {
  for (let idx = lines.length - 1; idx >= 0; idx -= 1) {
    const line = String(lines[idx] || '');
    const match = line.match(regex);
    if (match) {
      return match;
    }
  }
  return null;
}

function toJsonString(payload) {
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return String(payload);
  }
}

function isScriptLoadError(text) {
  const value = String(text || '').toLowerCase();
  return value.startsWith('[error]') || value.startsWith('[unavailable]');
}

function pickFirstUsableScriptKind(items, contents) {
  const list = Array.isArray(items) ? items : [];
  for (const item of list) {
    const kind = String(item?.kind || '').trim();
    if (!kind) {
      continue;
    }
    const content = String(contents?.[kind] || '');
    if (content.trim() && !isScriptLoadError(content)) {
      return kind;
    }
  }
  return '';
}

function parsePythonScriptInsights(text) {
  const lines = String(text || '').split(/\r?\n/).filter((line) => line.trim().length > 0);
  const imports = [];
  const importRegex = /^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)/gm;
  for (const match of text.matchAll(importRegex)) {
    imports.push(match[1]);
  }

  const testNames = [];
  const testRegex = /^\s*def\s+(test_[a-zA-Z0-9_]+)\s*\(/gm;
  for (const match of text.matchAll(testRegex)) {
    testNames.push(match[1]);
  }

  const endpointPaths = [];
  const endpointRegex = /BASE_URL\s*\+\s*"([^"]+)"/g;
  for (const match of text.matchAll(endpointRegex)) {
    endpointPaths.push(match[1]);
  }

  const placeholders = endpointPaths.filter((path) => /\{[^}]+\}/.test(path));
  const placeholderNames = [];
  for (const path of placeholders) {
    const tokenMatch = path.match(/\{([^}]+)\}/g) || [];
    for (const token of tokenMatch) {
      placeholderNames.push(token);
    }
  }

  const methodCounts = { get: 0, post: 0, put: 0, patch: 0, delete: 0 };
  const methodRegex = /requests\.(get|post|put|patch|delete)\(/g;
  for (const match of text.matchAll(methodRegex)) {
    const key = String(match[1] || '').toLowerCase();
    if (Object.prototype.hasOwnProperty.call(methodCounts, key)) {
      methodCounts[key] += 1;
    }
  }
  const requestCount = Object.values(methodCounts).reduce((sum, value) => sum + value, 0);

  const statusCodes = [];
  const statusRegex = /assert\s+response\.status_code\s*==\s*(\d{3})/g;
  for (const match of text.matchAll(statusRegex)) {
    const code = Number(match[1]);
    if (Number.isFinite(code)) {
      statusCodes.push(code);
    }
  }
  const statusBuckets = { success_2xx: 0, client_4xx: 0, server_5xx: 0, other: 0 };
  for (const code of statusCodes) {
    if (code >= 200 && code < 300) {
      statusBuckets.success_2xx += 1;
    } else if (code >= 400 && code < 500) {
      statusBuckets.client_4xx += 1;
    } else if (code >= 500 && code < 600) {
      statusBuckets.server_5xx += 1;
    } else {
      statusBuckets.other += 1;
    }
  }

  let focus = 'Mixed coverage.';
  if (statusBuckets.client_4xx > 0 && statusBuckets.success_2xx === 0 && statusBuckets.server_5xx === 0) {
    focus = 'Negative testing focused (auth, validation, and error handling).';
  } else if (statusBuckets.success_2xx > 0 && statusBuckets.client_4xx === 0 && statusBuckets.server_5xx === 0) {
    focus = 'Happy-path focused.';
  } else if (statusBuckets.success_2xx > 0 && statusBuckets.client_4xx > 0) {
    focus = 'Mixed happy-path + negative testing.';
  }

  const warnings = [];
  if (placeholders.length > 0) {
    warnings.push(
      `Path template placeholders detected (${placeholders.length}): ${[...new Set(placeholderNames)].join(', ')}`
    );
  }
  if (imports.includes('pytest') && !/\bpytest\./.test(text) && !/@pytest\b/.test(text)) {
    warnings.push("`import pytest` appears unused in this script.");
  }
  if (imports.includes('json') && !/\bjson\./.test(text)) {
    warnings.push("`import json` appears unused in this script.");
  }
  if (statusBuckets.success_2xx === 0 && statusBuckets.client_4xx > 0) {
    warnings.push('No happy-path 2xx assertions found in current selected script.');
  }

  return {
    language: 'python_pytest',
    lineCount: lines.length,
    testCount: testNames.length,
    requestCount,
    endpointCount: endpointPaths.length,
    methodCounts,
    statusBuckets,
    focus,
    warnings,
    sampleEndpoints: endpointPaths.slice(0, 4)
  };
}

function parseGeneratedScriptInsights(kind, text) {
  const raw = String(text || '').trim();
  if (!raw || raw.startsWith('Select a generated test script')) {
    return null;
  }
  if (kind === 'python_pytest') {
    return parsePythonScriptInsights(raw);
  }

  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const genericWarnings = [];
  if (/\{[^}]+\}/.test(raw)) {
    genericWarnings.push('Path template placeholders detected (example: {id}).');
  }
  return {
    language: kind || 'unknown',
    lineCount: lines.length,
    testCount: 0,
    requestCount: 0,
    endpointCount: 0,
    methodCounts: {},
    statusBuckets: {},
    focus: 'Preview this script content directly below.',
    warnings: genericWarnings,
    sampleEndpoints: []
  };
}

function BarListChart({ title, rows = [], valueFormatter = (v) => String(v), emptyText = 'No data.' }) {
  if (!rows.length) {
    return (
      <div className="chart-card">
        <h3>{title}</h3>
        <div className="small">{emptyText}</div>
      </div>
    );
  }
  const max = Math.max(...rows.map((row) => Number(row?.value || 0)), 0.0001);
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <div className="chart-rows">
        {rows.map((row) => {
          const value = Number(row?.value || 0);
          const pct = Math.max(0, Math.min(100, (value / max) * 100));
          return (
            <div className="chart-row" key={String(row?.label)}>
              <div className="chart-label">{String(row?.label)}</div>
              <div className="chart-bar">
                <div className="chart-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="chart-value">{valueFormatter(value)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrendSpark({ title, points = [], valueFormatter = (v) => String(v), emptyText = 'No trend data yet.' }) {
  if (!points.length) {
    return (
      <div className="chart-card">
        <h3>{title}</h3>
        <div className="small">{emptyText}</div>
      </div>
    );
  }
  const min = Math.min(...points.map((row) => Number(row?.value || 0)));
  const max = Math.max(...points.map((row) => Number(row?.value || 0)));
  const range = max - min || 1;
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <div className="spark-wrap">
        {points.map((row, idx) => {
          const value = Number(row?.value || 0);
          const h = 16 + ((value - min) / range) * 84;
          return (
            <div className="spark-col" key={`${String(row?.label)}-${idx}`} title={`${row?.label}: ${valueFormatter(value)}`}>
              <div className="spark-bar" style={{ height: `${h}px` }} />
              <div className="spark-label">{String(row?.label)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StackedPassFailChart({ title, rows = [], emptyText = 'No test-type coverage yet.' }) {
  if (!rows.length) {
    return (
      <div className="chart-card">
        <h3>{title}</h3>
        <div className="small">{emptyText}</div>
      </div>
    );
  }
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <div className="stacked-rows">
        {rows.map((row) => {
          const total = Number(row?.total || 0) || 1;
          const pass = Number(row?.passed || 0);
          const fail = Number(row?.failed || 0);
          const passPct = Math.max(0, Math.min(100, (pass / total) * 100));
          const failPct = Math.max(0, Math.min(100, (fail / total) * 100));
          return (
            <div className="stacked-row" key={String(row?.testType)}>
              <div className="stacked-label">{String(row?.testType)}</div>
              <div className="stacked-bar">
                <div className="stacked-pass" style={{ width: `${passPct}%` }} />
                <div className="stacked-fail" style={{ width: `${failPct}%` }} />
              </div>
              <div className="stacked-value">{pass}/{total}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function KpiRing({ label, value = 0, color = '#d45f1f' }) {
  const safe = Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) : 0;
  const pct = Math.round(safe * 100);
  return (
    <div className="kpi-ring">
      <div
        className="kpi-ring-visual"
        style={{
          background: `conic-gradient(${color} ${pct}%, #e6edf4 ${pct}% 100%)`
        }}
      >
        <div className="kpi-ring-core">{pct}%</div>
      </div>
      <div className="kpi-ring-label">{label}</div>
    </div>
  );
}

export default function HomePage() {
  const [domainsInput, setDomainsInput] = useState('');
  const [specPathsInput, setSpecPathsInput] = useState('');
  const [presetDomain, setPresetDomain] = useState('ecommerce');
  const [presetOpenapiSource, setPresetOpenapiSource] = useState('preset');
  const [presetSpecPath, setPresetSpecPath] = useState('');
  const [tenantId, setTenantId] = useState('customer_default');
  const [workspaceId, setWorkspaceId] = useState('customer_default');
  const [runScriptKind, setRunScriptKind] = useState('python_pytest');
  const [prompt, setPrompt] = useState('');
  const [maxScenarios, setMaxScenarios] = useState(16);
  const [maxRuntimeSec, setMaxRuntimeSec] = useState(0);
  const [llmTokenCap, setLlmTokenCap] = useState(0);
  const [environmentProfile, setEnvironmentProfile] = useState('mock');
  const [passThreshold, setPassThreshold] = useState(0.7);
  const [baseUrl, setBaseUrl] = useState(DEFAULT_TEST_BASE_URL);
  const [customerRoot, setCustomerRoot] = useState('~/.spec_test_pilot');
  const [customerMode, setCustomerMode] = useState(true);
  const [verifyPersistence, setVerifyPersistence] = useState(false);

  const [job, setJob] = useState(null);
  const [reportText, setReportText] = useState('Select a domain report to inspect.');
  const [selectedReportDomain, setSelectedReportDomain] = useState('');
  const [selectedReportFormat, setSelectedReportFormat] = useState('json');
  const [reportJson, setReportJson] = useState(null);
  const [scenarioFilter, setScenarioFilter] = useState('all');
  const [scenarioSearch, setScenarioSearch] = useState('');
  const [generatedTests, setGeneratedTests] = useState([]);
  const [generatedTestsDomain, setGeneratedTestsDomain] = useState('');
  const [generatedScriptContents, setGeneratedScriptContents] = useState({});
  const [generatedScriptsLoading, setGeneratedScriptsLoading] = useState(false);
  const [generatedScriptsLoadedDomain, setGeneratedScriptsLoadedDomain] = useState('');
  const [selectedScriptKind, setSelectedScriptKind] = useState('');
  const [scriptText, setScriptText] = useState('Select a generated test script to preview.');
  const [outputDomain, setOutputDomain] = useState('');
  const [flashMessage, setFlashMessage] = useState('');
  const [running, setRunning] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  const [simpleMode, setSimpleMode] = useState(true);
  const [reportViewTab, setReportViewTab] = useState('overview');
  const [trendMetric, setTrendMetric] = useState('run_reward');
  const [connectionProbe, setConnectionProbe] = useState({
    status: 'idle',
    backend: 'unknown',
    detail: ''
  });
  const rlTrainMode = 'periodic';

  const domains = useMemo(() => parseDomainList(domainsInput), [domainsInput]);
  const specPaths = useMemo(() => parseSpecPathsMap(specPathsInput), [specPathsInput]);
  const selectedPresetDomain = useMemo(() => sanitizeDomainToken(presetDomain), [presetDomain]);
  const submitDomains = useMemo(() => {
    const merged = new Set(domains);
    if (selectedPresetDomain) {
      merged.add(selectedPresetDomain);
    }
    return [...merged];
  }, [domains, selectedPresetDomain]);
  const submitSpecPaths = useMemo(() => {
    const merged = { ...specPaths };
    if (!selectedPresetDomain) {
      return merged;
    }
    if (presetOpenapiSource === 'custom_path') {
      const customPath = String(presetSpecPath || '').trim();
      if (customPath) {
        merged[selectedPresetDomain] = customPath;
      }
    } else {
      delete merged[selectedPresetDomain];
    }
    return merged;
  }, [specPaths, selectedPresetDomain, presetOpenapiSource, presetSpecPath]);
  const effectiveDomainCount = useMemo(() => {
    const merged = new Set(submitDomains);
    for (const domain of Object.keys(submitSpecPaths)) {
      merged.add(domain);
    }
    return merged.size;
  }, [submitDomains, submitSpecPaths]);

  const timerRef = useRef(null);
  const eventSourceRef = useRef(null);
  const autoLoadedJobRef = useRef('');
  const flashTimerRef = useRef(null);

  const logText = useMemo(() => (job?.logs || []).join('\n'), [job]);
  const results = job?.results || {};
  const currentDomain = getField(job, ['currentDomain', 'current_domain'], 'none');
  const jobScriptKind = String(
    getField(job, ['request'], {})?.scriptKind ||
      getField(job, ['request'], {})?.script_kind ||
      runScriptKind
  );
  const jobRlTrainMode = String(
    getField(job, ['request'], {})?.rlTrainMode ||
      getField(job, ['request'], {})?.rl_train_mode ||
      rlTrainMode
  );
  const startedAt = getField(job, ['startedAt', 'started_at'], 'n/a');
  const completedAt = getField(job, ['completedAt', 'completed_at'], 'n/a');
  const jobId = getField(job, ['id'], '');
  const reportSummary = reportJson?.summary || null;
  const reportMetadata = reportJson?.metadata || {};
  const generatedScriptExecution = reportJson?.generated_script_execution || {};
  const trainingStats = reportJson?.agent_lightning?.training_stats || {};
  const learningFeedback = reportJson?.learning?.feedback || {};
  const selectionPolicy = reportJson?.selection_policy || {};
  const mutationPolicy = reportJson?.mutation_policy || {};
  const gamData = reportJson?.gam || {};
  const scenarioContext = reportJson?.scenario_context || {};
  const scenarioContextCounts = scenarioContext?.counts || {};
  const scenarioContextSourceBreakdown = scenarioContext?.source_breakdown || {};
  const gamResearchEngine = gamData?.research_engine || {};
  const promptTrace = reportJson?.prompt_trace || {};
  const scenarioGeneration = promptTrace?.scenario_generation || {};
  const llmGenerationDiagnostics = scenarioGeneration?.llm_diagnostics || {};
  const llmParseDiagnostics = llmGenerationDiagnostics?.parse_diagnostics || {};
  const gamDiagnostics = gamData?.diagnostics || {};
  const gamPlan = Array.isArray(gamData?.research_plan) ? gamData.research_plan : [];
  const gamLearningSignalPageId = gamData?.learning_signal_page_id || null;
  const gamSpecContextPageId = gamData?.spec_context_page_id || null;
  const gamResearchExcerpts = Array.isArray(gamData?.research_excerpts) ? gamData.research_excerpts : [];
  const gamEngineIterations = Array.isArray(gamResearchEngine?.iterations) ? gamResearchEngine.iterations : [];
  const gamSourceBreakdown = gamData?.excerpt_source_breakdown || gamDiagnostics?.source_breakdown || {};
  const gamWarnings = Array.isArray(gamDiagnostics?.warnings) ? gamDiagnostics.warnings : [];
  const gamExcerptPreview = (Array.isArray(gamData?.excerpt_preview) && gamData.excerpt_preview.length > 0)
    ? gamData.excerpt_preview
    : (Array.isArray(promptTrace?.memory_excerpt_preview) ? promptTrace.memory_excerpt_preview : []);
  const stateSnapshot = reportJson?.learning?.state_snapshot || {};
  const scenarioResults = Array.isArray(reportJson?.scenario_results) ? reportJson.scenario_results : [];
  const topDecisions = Array.isArray(selectionPolicy?.top_decisions) ? selectionPolicy.top_decisions : [];
  const weakestPatterns = Array.isArray(stateSnapshot?.weakest_patterns) ? stateSnapshot.weakest_patterns : [];
  const decisionHistoryTail = Array.isArray(stateSnapshot?.decision_history_tail)
    ? stateSnapshot.decision_history_tail
    : [];
  const latestRunMetrics = stateSnapshot?.latest_run_metrics || {};
  const previousRunMetrics = stateSnapshot?.previous_run_metrics || {};
  const improvementDeltas = stateSnapshot?.improvement_deltas || {};
  const rewardBreakdown = learningFeedback?.reward_breakdown || {};
  const failureTaxonomyBreakdown = reportSummary?.failure_taxonomy_breakdown || {};
  const resultSummaries = Object.values(results).map((r) => r?.summary || {});
  const resultDomains = Object.keys(results);
  const selectedDomainLabel = selectedReportDomain ? formatDomainLabel(selectedReportDomain) : 'none';
  const passedScenarioCount = scenarioResults.filter((row) => !!row?.passed).length;
  const failedScenarioCount = Math.max(0, scenarioResults.length - passedScenarioCount);
  const successDomainCount = Object.values(results).filter((result) => {
    const code = Number(getField(result, ['exitCode', 'return_code'], 1));
    return code === 0;
  }).length;
  const overallPassRate = reportJson ? toPct(getField(reportSummary, ['pass_rate'], null)) : 'n/a';
  const summaryQualityGate = reportSummary ? Boolean(getField(reportSummary, ['meets_quality_gate'], false)) : null;
  const jobState = String(job?.status || 'idle');
  const [reportScenarioMode, setReportScenarioMode] = useState('critical');

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
      const scenarioLabel = humanizeScenarioName(row).label;
      const blob = [
        row?.name,
        scenarioLabel,
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

  const scenarioCoverageRows = useMemo(() => {
    const map = new Map();
    for (const row of scenarioResults) {
      const testType = String(row?.test_type || 'unknown');
      const current = map.get(testType) || { testType, total: 0, passed: 0, failed: 0 };
      current.total += 1;
      if (row?.passed) {
        current.passed += 1;
      } else {
        current.failed += 1;
      }
      map.set(testType, current);
    }
    return Array.from(map.values()).sort((a, b) => b.total - a.total);
  }, [scenarioResults]);

  const endpointCoverageCount = useMemo(() => {
    const unique = new Set();
    for (const row of scenarioResults) {
      const method = String(row?.method || '').toUpperCase();
      const endpoint = String(row?.endpoint_template || '');
      const key = `${method} ${endpoint}`.trim();
      if (key) {
        unique.add(key);
      }
    }
    return unique.size;
  }, [scenarioResults]);

  const rlExecutedScenarioCount = useMemo(() => {
    return scenarioResults.filter((row) => String(row?.name || '').includes('_rl_')).length;
  }, [scenarioResults]);

  const historySeedExecutedCount = useMemo(() => {
    return scenarioResults.filter((row) => String(row?.name || '').includes('rl_history_seed')).length;
  }, [scenarioResults]);

  const decisionSignals = Array.isArray(learningFeedback?.decision_signals)
    ? learningFeedback.decision_signals
    : [];

  const topImprovedScenarios = useMemo(() => {
    return [...decisionSignals]
      .filter((item) => Number(item?.reward) >= 0)
      .sort((a, b) => Number(b?.reward || 0) - Number(a?.reward || 0))
      .slice(0, 6);
  }, [decisionSignals]);

  const topFailedScenarios = useMemo(() => {
    return [...decisionSignals]
      .filter((item) => Number(item?.reward) < 0)
      .sort((a, b) => Number(a?.reward || 0) - Number(b?.reward || 0))
      .slice(0, 6);
  }, [decisionSignals]);

  const rlMutationExamples = useMemo(() => {
    const raw = Array.isArray(mutationPolicy?.applied_examples)
      ? mutationPolicy.applied_examples
      : [];
    return raw.slice(0, 8);
  }, [mutationPolicy]);
  const mutationStrategyBreakdown = useMemo(() => {
    const raw = mutationPolicy?.mutation_strategy_breakdown;
    if (!raw || typeof raw !== 'object') {
      return [];
    }
    return Object.entries(raw)
      .map(([strategy, count]) => [String(strategy), Number(count || 0)])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }, [mutationPolicy]);

  const executedScenarioSources = useMemo(() => {
    const contextSelected = scenarioContextSourceBreakdown?.selected;
    if (contextSelected && typeof contextSelected === 'object') {
      return {
        llmBase: Number(getField(contextSelected, ['llm_base'], 0)),
        heuristicBase: Number(getField(contextSelected, ['heuristic_base'], 0)),
        rlMutation: Number(getField(contextSelected, ['rl_mutation'], 0)),
        rlHistorySeed: Number(getField(contextSelected, ['rl_history_seed'], 0))
      };
    }
    const baseSourceHint = String(getField(promptTrace, ['scenario_generation', 'base_source'], 'llm_base'));
    let llmBase = 0;
    let heuristicBase = 0;
    let rlMutation = 0;
    let rlHistorySeed = 0;
    for (const row of scenarioResults) {
      const name = String(row?.name || '');
      if (name.includes('rl_history_seed')) {
        rlHistorySeed += 1;
      } else if (name.includes('_rl_')) {
        rlMutation += 1;
      } else {
        if (baseSourceHint === 'heuristic_base') {
          heuristicBase += 1;
        } else {
          llmBase += 1;
        }
      }
    }
    return {
      llmBase,
      heuristicBase,
      rlMutation,
      rlHistorySeed
    };
  }, [promptTrace, scenarioResults, scenarioContextSourceBreakdown]);

  const selectedScenarioContextRows = useMemo(() => {
    const raw = getField(scenarioContext, ['selected_scenarios'], []);
    if (!Array.isArray(raw)) {
      return [];
    }
    return raw.slice(0, 40);
  }, [scenarioContext]);

  const selectedScriptContent = useMemo(() => {
    if (selectedScriptKind && Object.prototype.hasOwnProperty.call(generatedScriptContents, selectedScriptKind)) {
      return String(generatedScriptContents[selectedScriptKind] || '');
    }
    return String(scriptText || '');
  }, [generatedScriptContents, scriptText, selectedScriptKind]);
  const canCopyScript = useMemo(() => {
    const content = String(selectedScriptContent || '').trim();
    return Boolean(content) && !isScriptLoadError(content);
  }, [selectedScriptContent]);

  const selectedScriptInsights = useMemo(() => {
    return parseGeneratedScriptInsights(selectedScriptKind, selectedScriptContent);
  }, [selectedScriptContent, selectedScriptKind]);

  const domainPassRateRows = useMemo(() => {
    return Object.entries(results)
      .map(([domain, result]) => ({
        label: formatDomainLabel(domain),
        value: Number(getField(result?.summary || {}, ['pass_rate', 'passRate'], 0) || 0)
      }))
      .sort((a, b) => b.value - a.value);
  }, [results]);
  const domainCoverageRows = useMemo(() => {
    return Object.entries(results)
      .map(([domain, result]) => {
        const summary = result?.summary || {};
        const total = Number(getField(summary, ['total_scenarios', 'totalScenarios'], 0) || 0);
        const passed = Number(getField(summary, ['passed_scenarios', 'passedScenarios'], 0) || 0);
        const failed = Math.max(0, total - passed);
        return {
          testType: formatDomainLabel(domain),
          total,
          passed,
          failed
        };
      })
      .sort((a, b) => b.total - a.total);
  }, [results]);

  const rewardBreakdownRows = useMemo(() => {
    return Object.entries(rewardBreakdown || {})
      .filter(([, value]) => Number.isFinite(Number(value)))
      .map(([label, value]) => ({
        label: String(label).replace(/_component$/, ''),
        value: Number(value)
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [rewardBreakdown]);

  const failureTaxonomyRows = useMemo(() => {
    return Object.entries(failureTaxonomyBreakdown || {})
      .map(([label, value]) => ({ label: String(label), value: Number(value || 0) }))
      .sort((a, b) => b.value - a.value);
  }, [failureTaxonomyBreakdown]);

  const trendRows = useMemo(() => {
    const source = Array.isArray(decisionHistoryTail) ? decisionHistoryTail : [];
    const metricKey = trendMetric === 'avg_decision_reward' ? 'average_decision_reward' : 'run_reward';
    return source.map((item, idx) => ({
      label: `r${idx + 1}`,
      value: Number(getField(item, [metricKey], 0) || 0)
    }));
  }, [decisionHistoryTail, trendMetric]);
  const scenarioStatusRows = useMemo(() => {
    return [
      { label: 'Passed', value: Number(passedScenarioCount || 0) },
      { label: 'Failed', value: Number(failedScenarioCount || 0) }
    ];
  }, [failedScenarioCount, passedScenarioCount]);
  const slowestScenarioRows = useMemo(() => {
    return [...scenarioResults]
      .map((row) => ({
        label: String(row?.name || 'unknown'),
        value: Number(row?.duration_ms || 0)
      }))
      .filter((row) => Number.isFinite(row.value))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
  }, [scenarioResults]);
  const failedByTypeRows = useMemo(() => {
    return scenarioCoverageRows
      .map((row) => ({
        label: String(row?.testType || 'unknown'),
        value: Number(row?.failed || 0)
      }))
      .filter((row) => row.value > 0)
      .sort((a, b) => b.value - a.value);
  }, [scenarioCoverageRows]);
  const stageMetricRows = useMemo(() => {
    const raw = getField(reportMetadata, ['stage_metrics_ms'], {});
    if (!raw || typeof raw !== 'object') {
      return [];
    }
    return Object.entries(raw)
      .map(([key, value]) => ({
        label: String(key).replace(/^stage_\d+_/, '').replace(/_/g, ' '),
        value: Number(value || 0)
      }))
      .filter((row) => Number.isFinite(row.value))
      .sort((a, b) => b.value - a.value);
  }, [reportMetadata]);
  const reportScenarioSpotlight = useMemo(() => {
    const rows = Array.isArray(scenarioResults) ? [...scenarioResults] : [];
    if (!rows.length) {
      return [];
    }
    const negativeTypes = new Set([
      'authentication',
      'authorization',
      'security',
      'input_validation',
      'error_handling',
      'boundary_testing',
      'edge_cases'
    ]);
    if (reportScenarioMode === 'critical') {
      return rows
        .sort((a, b) => {
          const aFail = a?.passed ? 0 : 1;
          const bFail = b?.passed ? 0 : 1;
          if (aFail !== bFail) {
            return bFail - aFail;
          }
          return Number(b?.duration_ms || 0) - Number(a?.duration_ms || 0);
        })
        .slice(0, 12);
    }
    if (reportScenarioMode === 'negative') {
      return rows
        .filter((row) => {
          const testType = String(row?.test_type || '').toLowerCase();
          const expectedStatus = Number(row?.expected_status || 0);
          return negativeTypes.has(testType) || expectedStatus >= 400;
        })
        .sort((a, b) => Number(b?.duration_ms || 0) - Number(a?.duration_ms || 0))
        .slice(0, 12);
    }
    if (reportScenarioMode === 'slow') {
      return rows
        .sort((a, b) => Number(b?.duration_ms || 0) - Number(a?.duration_ms || 0))
        .slice(0, 12);
    }
    return rows.slice(0, 12);
  }, [reportScenarioMode, scenarioResults]);

  const customerSummaryText = useMemo(() => {
    if (!reportJson || !reportSummary) {
      return '';
    }
    return [
      `Domain: ${selectedDomainLabel}`,
      `Total Scenarios: ${String(getField(reportSummary, ['total_scenarios'], 'n/a'))}`,
      `Pass Rate: ${toPct(getField(reportSummary, ['pass_rate'], null))}`,
      `Quality Gate: ${String(getField(reportSummary, ['meets_quality_gate'], 'n/a'))}`,
      `Failed Scenarios: ${String(getField(reportSummary, ['failed_scenarios'], 'n/a'))}`,
      `Flaky Ratio: ${String(getField(reportSummary, ['flaky_ratio'], 'n/a'))}`
    ].join('\n');
  }, [reportJson, reportSummary, selectedDomainLabel]);
  const failedExamples = useMemo(() => {
    const raw = getField(reportSummary, ['failed_examples'], []);
    return Array.isArray(raw) ? raw.slice(0, 6) : [];
  }, [reportSummary]);
  const qualityFailReasons = useMemo(() => {
    const raw = getField(reportSummary, ['quality_gate_fail_reasons'], []);
    return Array.isArray(raw) ? raw : [];
  }, [reportSummary]);
  const qualityWarnings = useMemo(() => {
    const raw = getField(reportSummary, ['quality_gate_warnings'], []);
    return Array.isArray(raw) ? raw : [];
  }, [reportSummary]);
  const reportRiskLevel = useMemo(() => {
    const pass = Number(getField(reportSummary, ['pass_rate'], 0) || 0);
    const failed = Number(getField(reportSummary, ['failed_scenarios'], 0) || 0);
    const flaky = Number(getField(reportSummary, ['flaky_ratio'], 0) || 0);
    if (pass >= 0.95 && failed === 0 && flaky <= 0.05) {
      return 'low';
    }
    if (pass >= 0.8 && failed <= 3 && flaky <= 0.15) {
      return 'moderate';
    }
    return 'high';
  }, [reportSummary]);
  const recommendedActions = useMemo(() => {
    const out = [];
    const pass = Number(getField(reportSummary, ['pass_rate'], 0) || 0);
    const flaky = Number(getField(reportSummary, ['flaky_ratio'], 0) || 0);
    if (pass < 0.9) {
      out.push('Re-run with persistence enabled and compare changed failures.');
    }
    if (flaky > 0.1) {
      out.push('Investigate flaky scenarios first before accepting the run.');
    }
    if (failedExamples.length > 0) {
      out.push('Prioritize top failed endpoints and publish a fix/retest checklist.');
    }
    if (out.length === 0) {
      out.push('No immediate action required. Report is ready to share.');
    }
    return out.slice(0, 4);
  }, [failedExamples.length, reportSummary]);

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
  const selectedStateDomain = selectedReportDomain || outputDomain || resultDomains[0] || '';
  const selectedDomainResult = selectedStateDomain ? results[selectedStateDomain] || null : null;
  const selectedDomainSummary = selectedDomainResult?.summary || {};
  const outputDirValue = getField(selectedDomainResult, ['outputDir', 'output_dir'], '');
  const checkpointValue = getField(selectedDomainResult, ['checkpointPath', 'checkpoint'], '');
  const reportJsonPathValue = getField(selectedDomainResult, ['reportJsonPath', 'report_json'], '');
  const reportMdPathValue = getField(selectedDomainResult, ['reportMdPath', 'report_md'], '');
  const firstPassReportJsonPath = getField(
    selectedDomainResult,
    ['firstPassReportJsonPath', 'first_pass_report_json'],
    ''
  );
  const secondPassReportJsonPath = getField(
    selectedDomainResult,
    ['secondPassReportJsonPath', 'second_pass_report_json'],
    ''
  );
  const openapiPathFromLog = findLastLogMatch(job?.logs || [], /OpenAPI spec written:\s*(.+)$/i)?.[1]?.trim() || '';
  const commandFromLog = findLastLogMatch(job?.logs || [], /\$\s+bash\s+(.+)$/i)?.[1]?.trim() || '';
  const reportJsonPathFromLog = findLastLogMatch(job?.logs || [], /JSON report:\s*(.+)$/i)?.[1]?.trim() || '';
  const reportMdPathFromLog = findLastLogMatch(job?.logs || [], /MD report:\s*(.+)$/i)?.[1]?.trim() || '';

  const stateTrace = useMemo(() => {
    return [
      {
        id: 'api_request',
        title: '1) API Request Payload',
        ready: Boolean(job?.request),
        payload: {
          request: job?.request || null,
          job_meta: {
            id: jobId || null,
            status: job?.status || null,
            created_at: getField(job, ['createdAt', 'created_at'], null),
            started_at: getField(job, ['startedAt', 'started_at'], null),
            completed_at: getField(job, ['completedAt', 'completed_at'], null),
            current_domain: currentDomain
          }
        }
      },
      {
        id: 'domain_command',
        title: '2) Domain Command + Runtime Paths',
        ready: Boolean(commandFromLog || selectedDomainResult),
        payload: {
          selected_domain: selectedStateDomain || null,
          command: commandFromLog || null,
          script_kind_request: jobScriptKind || null,
          script_kind_result:
            getField(selectedDomainResult, ['scriptKind', 'script_kind'], null) ||
            getField(selectedDomainSummary, ['scriptKind', 'script_kind'], null),
          output_dir: outputDirValue || null,
          checkpoint: checkpointValue || null,
          return_code: getField(selectedDomainResult, ['exitCode', 'return_code'], null),
          domain_summary: selectedDomainSummary
        }
      },
      {
        id: 'openapi_prepared',
        title: '3) OpenAPI Prepared',
        ready: Boolean(openapiPathFromLog || reportJson?.metadata),
        payload: {
          spec_path_from_log: openapiPathFromLog || null,
          spec_path_from_report: reportJson?.metadata?.spec_path || null,
          spec_title: reportJson?.metadata?.spec_title || null,
          spec_version: reportJson?.metadata?.spec_version || null
        }
      },
      {
        id: 'spec_intelligence',
        title: '4) Spec Intelligence + OSS Tooling',
        ready: Boolean(reportJson?.spec_intelligence || reportJson?.oss_tooling),
        payload: {
          spec_intelligence: reportJson?.spec_intelligence || null,
          oss_tooling: reportJson?.oss_tooling || null,
          metadata_stage_metrics_ms: reportJson?.metadata?.stage_metrics_ms || {}
        }
      },
      {
        id: 'prompt_trace',
        title: '5) Prompt Assembly Output',
        ready: Boolean(reportJson?.prompt_trace),
        payload: reportJson?.prompt_trace || null
      },
      {
        id: 'scenario_selection',
        title: '6) Scenario Selection Output',
        ready: Boolean(reportJson?.selection_policy),
        payload: reportJson?.selection_policy || null
      },
      {
        id: 'scenario_context',
        title: '7) Scenario Context (LLM/GAM/Mutation Influence)',
        ready: Boolean(reportJson?.scenario_context),
        payload: {
          scenario_context: reportJson?.scenario_context || null,
          prompt_gam_focus_points: reportJson?.prompt_trace?.gam_focus_points_used || [],
          prompt_rl_focus_points: reportJson?.prompt_trace?.rl_focus_points_used || [],
          mutation_policy: reportJson?.mutation_policy || null
        }
      },
      {
        id: 'generated_tests',
        title: '8) Generated Test Scripts (API Output)',
        ready: generatedTests.length > 0 || Boolean(reportJson?.generated_test_files),
        payload: {
          selected_domain: generatedTestsDomain || selectedStateDomain || null,
          scripts_loading: generatedScriptsLoading,
          scripts_loaded_domain: generatedScriptsLoadedDomain || null,
          generated_test_files_report: reportJson?.generated_test_files || {},
          generated_test_files_api: generatedTests,
          generated_script_contents: generatedScriptContents,
          generated_script_execution: generatedScriptExecution
        }
      },
      {
        id: 'scenario_execution',
        title: '9) Executed Scenario Results',
        ready: scenarioResults.length > 0,
        payload: {
          total: scenarioResults.length,
          passed: passedScenarioCount,
          failed: failedScenarioCount,
          scenarios: scenarioResults
        }
      },
      {
        id: 'gam_research',
        title: '10) GAM Deep Research State',
        ready: Boolean(reportJson?.gam),
        payload: reportJson?.gam || null
      },
      {
        id: 'learning_buffer',
        title: '11) Learning Buffer and Scheduled Trainer State',
        ready: Boolean(reportJson?.agent_lightning || reportJson?.learning),
        payload: {
          learning: reportJson?.learning || null,
          agent_lightning: reportJson?.agent_lightning || null
        }
      },
      {
        id: 'final_reports',
        title: '12) Final Report Paths + Payload',
        ready: Boolean(reportJson || reportJsonPathValue || reportMdPathValue),
        payload: {
          report_files: reportJson?.report_files || null,
          report_json_from_domain_result: reportJsonPathValue || null,
          report_md_from_domain_result: reportMdPathValue || null,
          report_json_first_pass: firstPassReportJsonPath || null,
          report_json_second_pass: secondPassReportJsonPath || null,
          report_json_from_log: reportJsonPathFromLog || null,
          report_md_from_log: reportMdPathFromLog || null
        }
      }
    ];
  }, [
    checkpointValue,
    commandFromLog,
    currentDomain,
    failedScenarioCount,
    generatedScriptContents,
    generatedScriptsLoadedDomain,
    generatedScriptsLoading,
    generatedTests,
    generatedTestsDomain,
    job,
    jobId,
    openapiPathFromLog,
    outputDirValue,
    passedScenarioCount,
    reportJson,
    reportJsonPathFromLog,
    reportJsonPathValue,
    reportMdPathFromLog,
    reportMdPathValue,
    firstPassReportJsonPath,
    secondPassReportJsonPath,
    generatedScriptExecution,
    scenarioResults,
    selectedDomainResult,
    selectedDomainSummary,
    selectedStateDomain
  ]);

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
    if (stepId === 'learning_buffer') {
      return scenarioResults.length > 0 || reportReadyDomains > 0 || jobState === 'completed';
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
    if (stepId === 'learning_buffer') {
      return `actual: mode=periodic (fixed) | model_steps=${rlStepValue === null ? 'n/a' : rlStepValue} | buffer=${String(getField(trainingStats, ['rl_buffer_size'], 'n/a'))}`;
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
      // Fallback to polling if SSE stream disconnects.
      // Do not gate on React state here; closures can hold stale `running=false`.
      if (jobId && !timerRef.current) {
        timerRef.current = setTimeout(() => pollJob(jobId), 1500);
      }
    };
  }

  async function onRun() {
    if (effectiveDomainCount === 0) {
      alert('Provide at least one domain or a spec path mapping.');
      return;
    }

    setRunning(true);
    setReportText('Select a domain report to inspect.');
    setReportJson(null);
    setSelectedReportDomain('');
    setSelectedReportFormat('json');
    setGeneratedTests([]);
    setGeneratedTestsDomain('');
    setGeneratedScriptContents({});
    setGeneratedScriptsLoadedDomain('');
    setGeneratedScriptsLoading(false);
    setSelectedScriptKind('');
    setScriptText('Select a generated test script to preview.');
    autoLoadedJobRef.current = '';

    const body = {
      domains: submitDomains,
      specPaths: submitSpecPaths,
      tenantId,
      workspaceId,
      scriptKind: runScriptKind,
      prompt: prompt.trim() || null,
      maxScenarios,
      maxRuntimeSec: Number(maxRuntimeSec) > 0 ? Number(maxRuntimeSec) : null,
      llmTokenCap: Number(llmTokenCap) > 0 ? Number(llmTokenCap) : null,
      environmentProfile,
      rlTrainMode,
      passThreshold,
      baseUrl,
      customerMode,
      verifyPersistence,
      customerRoot
    };

    const reqUrl = API_URL('/api/jobs');
    let res;
    try {
      res = await fetch(reqUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body)
      });
    } catch (error) {
      setRunning(false);
      const detail = error instanceof Error ? error.message : String(error);
      const modeHint = API_BASE
        ? 'Start backend with ./backend/start-backend.sh, or run UI in local proxy mode: QA_UI_MODE=full_next ./frontend/run_customer_ui_next.sh'
        : 'Check that Next.js API routes are running in this UI process.';
      alert(`Failed to reach API: ${reqUrl}\n${modeHint}\n\n${detail}`);
      return;
    }

    let payload = {};
    const rawPayload = await res.text();
    try {
      payload = rawPayload ? JSON.parse(rawPayload) : {};
    } catch {
      payload = { error: rawPayload || 'Unexpected non-JSON response from API' };
    }
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
    // Safety-net polling starts immediately and continues until terminal status.
    // This keeps UI state moving even if SSE is interrupted or browser-filtered.
    if (!timerRef.current) {
      timerRef.current = setTimeout(() => pollJob(startedJobId), 1500);
    }
  }

  async function loadAllGeneratedScripts(domain, items) {
    if (!job?.id || !domain) {
      return;
    }
    setGeneratedScriptsLoading(true);
    const next = {};
    for (const item of items) {
      const kind = String(item?.kind || '').trim();
      if (!kind) {
        continue;
      }
      if (item.exists === false || item.safe_to_read === false) {
        next[kind] = '[unavailable] file is missing or blocked by safety check';
        continue;
      }
      const reqUrl = API_URL(`/api/jobs/${job.id}/generated-tests/${domain}/${kind}`);
      try {
        const res = await fetch(reqUrl, { cache: 'no-store' });
        if (!res.ok) {
          const errText = await res.text();
          next[kind] = `[error] failed to fetch script: ${errText}`;
          continue;
        }
        next[kind] = await res.text();
      } catch (error) {
        next[kind] = `[error] failed to fetch script: ${String(error)}`;
      }
    }
    setGeneratedScriptContents(next);
    setGeneratedScriptsLoadedDomain(domain);
    setGeneratedScriptsLoading(false);
    const firstUsableKind = pickFirstUsableScriptKind(items, next);
    if (firstUsableKind) {
      setSelectedScriptKind(firstUsableKind);
      setScriptText(String(next[firstUsableKind] || ''));
      return;
    }
    const firstAnyKind = String(items?.[0]?.kind || '');
    if (firstAnyKind) {
      setSelectedScriptKind(firstAnyKind);
      setScriptText(String(next[firstAnyKind] || 'Script is unavailable for preview.'));
    } else {
      setSelectedScriptKind('');
      setScriptText('No generated scripts are available for this domain.');
    }
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
      setGeneratedScriptContents({});
      setGeneratedScriptsLoadedDomain('');
      setGeneratedScriptsLoading(false);
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
          await loadAllGeneratedScripts(domain, fallbackItems);
        }
        return;
      }
      const payload = await res.json();
      const items = normalizeGeneratedTestItems(getField(payload, ['generated_tests', 'generatedTests'], []));
      setGeneratedTests(items);
      setGeneratedTestsDomain(domain);
      await loadAllGeneratedScripts(domain, items);
    } catch {
      if (fallbackItems.length > 0) {
        setGeneratedTests(fallbackItems);
        setGeneratedTestsDomain(domain);
        await loadAllGeneratedScripts(domain, fallbackItems);
      }
    }
  }

  async function openGeneratedScript(domain, kind) {
    if (!job?.id) {
      return;
    }
    setSelectedScriptKind(kind);
    if (generatedScriptsLoadedDomain === domain && Object.prototype.hasOwnProperty.call(generatedScriptContents, kind)) {
      setScriptText(generatedScriptContents[kind] || `Script ${kind} is empty.`);
      return;
    }
    const reqUrl = API_URL(`/api/jobs/${job.id}/generated-tests/${domain}/${kind}`);
    try {
      const res = await fetch(reqUrl, { cache: 'no-store' });
      if (!res.ok) {
        const errText = await res.text();
        setScriptText(`Failed to load script ${kind}: ${errText}`);
        return;
      }
      const raw = await res.text();
      setGeneratedScriptContents((prev) => ({ ...prev, [kind]: raw }));
      setGeneratedScriptsLoadedDomain(domain);
      setScriptText(raw || `Script ${kind} is empty.`);
    } catch (error) {
      setScriptText(`Failed to load script ${kind}: ${String(error)}`);
    }
  }

  async function downloadScript() {
    const content = String(selectedScriptContent || '').trim();
    if (!content || isScriptLoadError(content)) {
      setFlash('No valid script selected');
      return;
    }
    const domainPart = sanitizeDomainToken(generatedTestsDomain || selectedReportDomain || 'domain') || 'domain';
    const kindPart = sanitizeDomainToken(selectedScriptKind || 'script') || 'script';
    const fileName = `${domainPart}_${kindPart}.txt`;
    try {
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      setFlash(`Downloaded ${fileName}`);
    } catch {
      setFlash('Failed to download script');
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
    let active = true;
    setConnectionProbe({
      status: 'checking',
      backend: 'unknown',
      detail: ''
    });

    const probe = async () => {
      try {
        const res = await fetch(API_URL('/api/ping'), { cache: 'no-store' });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = await res.json();
        if (!active) {
          return;
        }
        setConnectionProbe({
          status: 'ok',
          backend: String(payload?.backend || 'unknown'),
          detail: String(payload?.service || '')
        });
      } catch (error) {
        if (!active) {
          return;
        }
        setConnectionProbe({
          status: 'error',
          backend: 'unreachable',
          detail: String(error?.message || error || 'unknown_error')
        });
      }
    };

    void probe();
    return () => {
      active = false;
    };
  }, [API_BASE]);

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
            <p className="header-kicker">SPECFORGE STUDIO</p>
            <Typography variant="h4" component="h1">Customer QA Command Center</Typography>
            <p className="header-subtitle">Start a run, monitor quality, and deliver a clear share-ready report.</p>
          </div>
          <Stack direction="row" flexWrap="wrap" gap={1} justifyContent="flex-end" className="header-badges">
            <Chip size="small" label={`status=${job?.status || 'idle'}`} variant="outlined" />
            <Chip size="small" label={`domains=${resultDomains.length}`} variant="outlined" />
            <Chip size="small" label={`success=${successDomainCount}`} variant="outlined" />
            <Chip size="small" label={`pass_rate=${overallPassRate}`} variant="outlined" />
            <Chip
              size="small"
              label={`runtime=${API_BASE ? 'fastapi' : 'next_proxy'}`}
              color={API_BASE ? 'success' : 'warning'}
              variant="outlined"
            />
            <Chip size="small" label={`backend=${API_BASE || 'same-origin'}`} variant="outlined" />
            <Chip
              size="small"
              label={`probe=${connectionProbe.status}:${connectionProbe.backend}`}
              color={connectionProbe.status === 'ok' && connectionProbe.backend === 'fastapi' ? 'success' : 'warning'}
              variant="outlined"
            />
            <Chip
              size="small"
              label={simpleMode ? 'view=simple' : 'view=advanced'}
              color={simpleMode ? 'primary' : 'default'}
              variant={simpleMode ? 'filled' : 'outlined'}
              onClick={() => setSimpleMode((v) => !v)}
              clickable
            />
          </Stack>
        </div>
        {flashMessage && <Alert severity="info" className="toast">{flashMessage}</Alert>}
      </header>

      <Box className="quick-nav">
        <Button size="small" variant="outlined" href="#run-config">Run Setup</Button>
        <Button size="small" variant="outlined" href="#runtime-status">Live Status</Button>
        <Button size="small" variant="outlined" href="#domain-results">Domain Health</Button>
        {!simpleMode && <Button size="small" variant="outlined" href="#generated-scripts">Generated Scripts</Button>}
        {!simpleMode && <Button size="small" variant="outlined" href="#cases-tested">Test Results</Button>}
        <Button size="small" variant="outlined" href="#final-report">Customer Report</Button>
      </Box>

      <section className="overview-strip">
        <div className={`overview-pill status-${jobState}`}>
          <span>Run Status</span>
          <strong>{jobState}</strong>
        </div>
        <div className="overview-pill">
          <span>Completed Domains</span>
          <strong>{String(completedDomains)}</strong>
        </div>
        <div className="overview-pill">
          <span>Successful Domains</span>
          <strong>{String(successDomainCount)}</strong>
        </div>
        <div className="overview-pill">
          <span>Overall Pass Rate</span>
          <strong>{overallPassRate}</strong>
        </div>
        <div className={`overview-pill ${summaryQualityGate === true ? 'ok' : summaryQualityGate === false ? 'bad' : ''}`}>
          <span>Quality Gate</span>
          <strong>{summaryQualityGate === null ? 'n/a' : summaryQualityGate ? 'pass' : 'fail'}</strong>
        </div>
      </section>

      <div className="layout">
        <section className="card card-config" id="run-config">
          <h2>Start New QA Run</h2>

          <div className="field">
            <label>Domain (predefined)</label>
            <select value={presetDomain} onChange={(e) => setPresetDomain(e.target.value)}>
              {DOMAIN_PRESET_OPTIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>OpenAPI spec source</label>
            <select value={presetOpenapiSource} onChange={(e) => setPresetOpenapiSource(e.target.value)}>
              {OPENAPI_SOURCE_OPTIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            {presetOpenapiSource === 'custom_path' && (
              <input
                value={presetSpecPath}
                onChange={(e) => setPresetSpecPath(e.target.value)}
                placeholder="~/specs/openapi_custom.yaml"
              />
            )}
            <div className="small">
              selected_domain={selectedPresetDomain || 'none'} | openapi_source=
              {presetOpenapiSource === 'preset' ? 'preset_template' : 'custom_path'}
            </div>
          </div>

          <details className="advanced">
            <summary>Advanced domain/spec overrides (optional)</summary>
            <div className="field">
              <label>Additional Domains (comma/newline list)</label>
              <textarea
                value={domainsInput}
                onChange={(e) => setDomainsInput(e.target.value)}
                placeholder={'payments\npartner_api'}
              />
            </div>
            <div className="field">
              <label>Spec Paths (optional, one per line)</label>
              <textarea
                value={specPathsInput}
                onChange={(e) => setSpecPathsInput(e.target.value)}
                placeholder={'payments=/tmp/openapi_payments.yaml\npartner_api=~/specs/partner.yaml'}
              />
            </div>
            <div className="small">
              format: domain=/absolute/or/home-relative/path/to/openapi.yaml | manual_domains=
              {domains.join(', ') || 'none'}
            </div>
          </details>

          <div className="small">
            effective_domains={String(effectiveDomainCount)} | submit_domains={submitDomains.join(', ') || 'none'}
          </div>

          <div className="field">
            <label>Tenant ID</label>
            <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
          </div>

          <div className="field">
            <label>Workspace ID</label>
            <input value={workspaceId} onChange={(e) => setWorkspaceId(e.target.value)} />
          </div>

          <div className="field">
            <label>Environment Profile</label>
            <select value={environmentProfile} onChange={(e) => setEnvironmentProfile(e.target.value)}>
              <option value="mock">mock</option>
              <option value="staging">staging</option>
              <option value="prod_safe">prod_safe</option>
            </select>
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

          <details className="advanced">
            <summary>Advanced run controls</summary>
            <div className="field">
              <label>Script Language</label>
              <select value={runScriptKind} onChange={(e) => setRunScriptKind(e.target.value)}>
                {SCRIPT_KINDS.map((kind) => (
                  <option key={kind} value={kind}>
                    {SCRIPT_KIND_LABELS[kind] || kind}
                  </option>
                ))}
              </select>
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
              <label>Max Runtime Seconds (optional)</label>
              <input
                type="number"
                min={0}
                max={7200}
                value={maxRuntimeSec}
                onChange={(e) => setMaxRuntimeSec(Number(e.target.value || 0))}
              />
            </div>

            <div className="field">
              <label>LLM Token Cap (optional)</label>
              <input
                type="number"
                min={0}
                max={16000}
                value={llmTokenCap}
                onChange={(e) => setLlmTokenCap(Number(e.target.value || 0))}
              />
            </div>

            <div className="field">
              <label>Base URL</label>
              <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
              <div className="small">
                Used as `BASE_URL` inside generated test scripts. Default from `NEXT_PUBLIC_TEST_BASE_URL`.
              </div>
            </div>

            <div className="field">
              <label>Customer Root</label>
              <input value={customerRoot} onChange={(e) => setCustomerRoot(e.target.value)} />
            </div>

            <div className="field">
              <label>Prompt (optional)</label>
              <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </div>
          </details>

          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={customerMode}
                onChange={(e) => setCustomerMode(e.target.checked)}
              />{' '}
              save workspace and model checkpoint
            </label>
            <label>
              <input
                type="checkbox"
                checked={verifyPersistence}
                onChange={(e) => setVerifyPersistence(e.target.checked)}
              />{' '}
              run a second pass for consistency check
            </label>
          </div>

          <details className="advanced">
            <summary>Developer diagnostics and UI controls</summary>
            <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={showTechnical}
                onChange={(e) => setShowTechnical(e.target.checked)}
              />{' '}
              show advanced diagnostics (engine internals)
            </label>
            <label>
              <input
                type="checkbox"
                checked={simpleMode}
                onChange={(e) => setSimpleMode(e.target.checked)}
              />{' '}
              simple UI mode (recommended)
            </label>
            </div>
          </details>

          <button className="primary" disabled={running} onClick={onRun}>
            {running ? 'Running...' : 'Start QA Run'}
          </button>
        </section>

        <section className="right">
          <div className="card" id="runtime-status">
            <h2>Run Progress</h2>
            <div className="status">
              <span className={`dot ${job?.status || ''}`} />
              <strong>{job?.status || 'idle'}</strong>
            </div>
            <div className="meta">
              {job
                ? `job=${jobId} | current_domain=${currentDomain} | script_kind=${jobScriptKind} | started=${formatDateTime(startedAt)} | completed=${formatDateTime(completedAt)}`
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

          {simpleMode && (
          <div className="card">
            <h2>Simple View</h2>
            <div className="small">
              Showing essentials only: Run Progress, Domain Health, and Customer Report.
              Switch to advanced view to inspect generated scripts and detailed scenario analytics.
            </div>
            <div className="inline-actions">
              <Button size="small" variant="contained" onClick={() => setSimpleMode(false)}>
                Open Advanced View
              </Button>
            </div>
          </div>
          )}

          {!simpleMode && (
          <div className="card">
            <h2>Customer Checklist</h2>
            <div className="report-grid">
              <div className="report-metric">
                <span>1. Run Started</span>
                <strong>{jobId ? 'done' : 'pending'}</strong>
              </div>
              <div className="report-metric">
                <span>2. Domain Outputs Ready</span>
                <strong>{reportReadyDomains > 0 ? 'done' : 'pending'}</strong>
              </div>
              <div className="report-metric">
                <span>3. Report Selected</span>
                <strong>{selectedReportDomain ? 'done' : 'pending'}</strong>
              </div>
              <div className="report-metric">
                <span>4. Quality Gate</span>
                <strong>{summaryQualityGate === null ? 'pending' : summaryQualityGate ? 'pass' : 'fail'}</strong>
              </div>
              <div className="report-metric">
                <span>5. Shareable Output</span>
                <strong>{selectedReportDomain ? selectedReportFormat.toUpperCase() : 'pending'}</strong>
              </div>
            </div>
            <div className="small">
              tip: open <b>Domain Health</b>, choose a domain, then load JSON/Markdown report for handoff.
            </div>
          </div>
          )}

          {showTechnical && (
          <div className="card">
            <h2>Model Buffer and Policy Progress</h2>
            {!reportJson && (
              <div className="small">Open a domain JSON report to see learning progress per run.</div>
            )}
            {reportJson && (
              <>
                <div className="report-grid">
                  <div className="report-metric">
                    <span>Learning Runs</span>
                    <strong>{String(getField(stateSnapshot, ['run_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Model Steps</span>
                    <strong>{String(getField(trainingStats, ['rl_training_steps'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Buffer Size</span>
                    <strong>{String(getField(trainingStats, ['rl_buffer_size'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Run Reward Δ</span>
                    <strong>{formatDelta(getField(improvementDeltas, ['run_reward_delta'], null), 4)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Avg Decision Reward Δ</span>
                    <strong>{formatDelta(getField(improvementDeltas, ['avg_decision_reward_delta'], null), 4)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Penalized Decisions Δ</span>
                    <strong>{formatDelta(getField(improvementDeltas, ['penalized_decisions_delta'], null), 0)}</strong>
                  </div>
                </div>
                <div className="small">
                  latest_run_reward={String(getField(latestRunMetrics, ['run_reward'], 'n/a'))} | previous_run_reward=
                  {String(getField(previousRunMetrics, ['run_reward'], 'n/a'))}
                </div>
                <div className="small">
                  first_pass_report={firstPassReportJsonPath || 'n/a'}
                  <br />
                  second_pass_report={secondPassReportJsonPath || 'n/a'}
                </div>
                {decisionHistoryTail.length > 0 && (
                  <div className="scenario-table-wrap">
                    <table className="scenario-table">
                      <thead>
                        <tr>
                          <th>run</th>
                          <th>run_reward</th>
                          <th>avg_decision_reward</th>
                          <th>rewarded</th>
                          <th>penalized</th>
                          <th>timestamp</th>
                        </tr>
                      </thead>
                      <tbody>
                        {decisionHistoryTail.map((item, idx) => (
                          <tr key={`${String(item.timestamp || '')}-${idx}`}>
                            <td>{String(idx + 1)}</td>
                            <td>{String(getField(item, ['run_reward'], 'n/a'))}</td>
                            <td>{String(getField(item, ['average_decision_reward'], 'n/a'))}</td>
                            <td>{String(getField(item, ['rewarded_decisions'], 'n/a'))}</td>
                            <td>{String(getField(item, ['penalized_decisions'], 'n/a'))}</td>
                            <td>{String(getField(item, ['timestamp'], 'n/a'))}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
          )}

          {showTechnical && (
          <div className="card">
            <h2>Scenario Influence Map (Generator vs Memory vs Mutation)</h2>
            {!reportJson && (
              <div className="small">Open a domain JSON report to inspect how each component influenced the run.</div>
            )}
            {reportJson && (
              <>
                <div className="report-grid">
                  <div className="report-metric">
                    <span>Base Candidates</span>
                    <strong>{String(getField(selectionPolicy, ['base_candidate_count'], getField(scenarioContextCounts, ['base_generated'], 'n/a')))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Mutation Added Candidates</span>
                    <strong>{String(getField(mutationPolicy, ['mutated_candidates_added'], 0))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Selected For Execution</span>
                    <strong>{String(getField(selectionPolicy, ['selected_count'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>GAM Excerpts Used</span>
                    <strong>{String(getField(gamDiagnostics, ['total_excerpts'], getField(gamData, ['research_excerpt_count'], 0)))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>GAM Plan Items</span>
                    <strong>{String(getField(gamData, ['research_plan_count'], gamPlan.length))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>GAM Planner Mode</span>
                    <strong>{String((getField(gamResearchEngine, ['plan_modes'], []).slice(-1)[0]) || 'heuristic')}</strong>
                  </div>
                  <div className="report-metric">
                    <span>GAM Reflector Mode</span>
                    <strong>{String((getField(gamResearchEngine, ['reflect_modes'], []).slice(-1)[0]) || 'heuristic')}</strong>
                  </div>
                  <div className="report-metric">
                    <span>GAM Quality Score</span>
                    <strong>{String(getField(gamDiagnostics, ['quality_score'], 'n/a'))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Training Mode</span>
                    <strong>{String(getField(trainingStats, ['train_mode'], jobRlTrainMode || 'periodic'))}</strong>
                  </div>
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>Executed Scenarios By Source</h3>
                  </div>
                  <div className="report-grid">
                    <div className="report-metric">
                      <span>Base Generated</span>
                      <strong>{String(getField(scenarioContextCounts, ['base_generated'], getField(selectionPolicy, ['base_candidate_count'], 'n/a')))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Candidate Pool</span>
                      <strong>{String(getField(scenarioContextCounts, ['candidate_total'], getField(selectionPolicy, ['candidate_count'], 'n/a')))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Selected Total</span>
                      <strong>{String(getField(scenarioContextCounts, ['selected_total'], getField(selectionPolicy, ['selected_count'], 'n/a')))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>LLM Base Executed</span>
                      <strong>{String(executedScenarioSources.llmBase)}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Heuristic Base Executed</span>
                      <strong>{String(executedScenarioSources.heuristicBase)}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Mutation Executed</span>
                      <strong>{String(executedScenarioSources.rlMutation)}</strong>
                    </div>
                    <div className="report-metric">
                      <span>History-Seed Executed</span>
                      <strong>{String(executedScenarioSources.rlHistorySeed)}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Selected New vs History</span>
                      <strong>{String(getField(scenarioContextCounts, ['selected_new_vs_history'], 'n/a'))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Selected Weak Historical</span>
                      <strong>{String(getField(scenarioContextCounts, ['selected_historical_weak_patterns'], 'n/a'))}</strong>
                    </div>
                  </div>
                  <div className="small">
                    interpretation: LLM creates base candidates, GAM enriches planning context, mutation policies diversify scenarios, and model checkpoint updates happen on periodic jobs.
                  </div>
                  {selectedScenarioContextRows.length > 0 && (
                    <div className="scenario-table-wrap">
                      <table className="scenario-table">
                        <thead>
                          <tr>
                            <th>name</th>
                            <th>source</th>
                            <th>strategy</th>
                            <th>reason</th>
                            <th>hist_seen</th>
                            <th>hist_fail_rate</th>
                            <th>expected</th>
                            <th>actual</th>
                            <th>passed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedScenarioContextRows.map((item, idx) => {
                            const scenarioInfo = humanizeScenarioName(item);
                            return (
                              <tr key={`${String(item?.fingerprint || item?.name || 'row')}-${idx}`}>
                                <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                  <div>{scenarioInfo.label}</div>
                                  <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                                </td>
                                <td>{String(getField(item, ['source'], 'n/a'))}</td>
                                <td>{String(getField(item, ['mutation_strategy'], '')) || '-'}</td>
                                <td>{String(getField(item, ['selection_reason'], 'n/a'))}</td>
                                <td>{String(getField(item, ['historical_seen_before'], 'n/a'))}</td>
                                <td>{String(getField(item, ['historical_failure_rate_before'], 'n/a'))}</td>
                                <td>{String(getField(item, ['expected_status'], 'n/a'))}</td>
                                <td>{String(getField(item, ['actual_status'], 'n/a'))}</td>
                                <td>{String(getField(item, ['passed'], 'n/a'))}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>GAM Research Influence</h3>
                  </div>
                  <div className="report-grid">
                    <div className="report-metric">
                      <span>Convention Excerpts</span>
                      <strong>{String(getField(gamDiagnostics, ['convention_excerpts'], 'n/a'))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Non-Convention Excerpts</span>
                      <strong>{String(getField(gamDiagnostics, ['non_convention_excerpts'], 'n/a'))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Actionable Excerpts</span>
                      <strong>{String(getField(gamDiagnostics, ['actionable_excerpt_count'], 'n/a'))}</strong>
                    </div>
                    <div className="report-metric">
                      <span>Machine-Like Excerpts</span>
                      <strong>{String(getField(gamDiagnostics, ['machine_like_excerpt_count'], 'n/a'))}</strong>
                    </div>
                  </div>
                  <div className="small">
                    warnings: {gamWarnings.length > 0 ? gamWarnings.join(' | ') : 'none'}
                  </div>
                  <div className="small">
                    plan: {gamPlan.length > 0
                      ? gamPlan.join(' | ')
                      : 'n/a'}
                  </div>
                  <div className="small">
                    reflection: {String(getField(gamData, ['research_reflection'], 'n/a'))}
                  </div>
                  <div className="small">
                    gam_llm_enabled: {String(Boolean(getField(gamResearchEngine, ['llm_enabled'], false)))} |
                    gam_llm_model: {String(getField(gamResearchEngine, ['llm_model'], 'n/a'))}
                  </div>
                  <div className="small">
                    gam_llm_stats: plan_success=
                    {String(getField(gamResearchEngine, ['llm_stats', 'plan_success'], 0))} /
                    {String(getField(gamResearchEngine, ['llm_stats', 'plan_calls'], 0))}, reflect_success=
                    {String(getField(gamResearchEngine, ['llm_stats', 'reflect_success'], 0))} /
                    {String(getField(gamResearchEngine, ['llm_stats', 'reflect_calls'], 0))}
                  </div>
                  <div className="small">
                    learning_signal_page_id: {String(gamLearningSignalPageId || 'none')}
                  </div>
                  <div className="small">
                    spec_context_page_id: {String(gamSpecContextPageId || 'none')}
                  </div>
                  <details className="advanced">
                    <summary>GAM Planner/Reflector Iteration Trace</summary>
                    <pre>{toJsonString(gamEngineIterations)}</pre>
                  </details>
                  <details className="advanced">
                    <summary>GAM Excerpt Source Breakdown</summary>
                    <pre>{toJsonString(gamSourceBreakdown)}</pre>
                  </details>
                  <details className="advanced">
                    <summary>GAM Excerpt Preview</summary>
                    <pre>{toJsonString(gamExcerptPreview)}</pre>
                  </details>
                  <details className="advanced">
                    <summary>GAM Retrieved Excerpts (Raw)</summary>
                    <pre>{toJsonString(gamResearchExcerpts)}</pre>
                  </details>
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>Mutation Influence (Examples)</h3>
                  </div>
                  {mutationStrategyBreakdown.length > 0 && (
                    <div className="small">
                      strategy_mix: {mutationStrategyBreakdown.map(([name, count]) => `${name}=${count}`).join(' | ')}
                    </div>
                  )}
                  {rlMutationExamples.length === 0 && (
                    <div className="small">No mutation examples recorded in this run.</div>
                  )}
                  {rlMutationExamples.length > 0 && (
                    <div className="scenario-table-wrap">
                      <table className="scenario-table">
                        <thead>
                          <tr>
                            <th>from</th>
                            <th>to</th>
                            <th>strategy</th>
                            <th>test_type</th>
                            <th>expected_status</th>
                            <th>priority</th>
                            <th>budget</th>
                            <th>op_fail_rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rlMutationExamples.map((item, idx) => (
                            <tr key={`${String(item.to || 'row')}-${idx}`}>
                              <td>{String(item.from || item.from_fingerprint || 'history_seed')}</td>
                              <td>{String(item.to || 'n/a')}</td>
                              <td>{String(item.strategy || item.source || 'n/a')}</td>
                              <td>{String(item.test_type || 'n/a')}</td>
                              <td>{String(item.expected_status ?? 'n/a')}</td>
                              <td>{String(item.priority ?? 'n/a')}</td>
                              <td>{String(item.mutation_budget ?? 'n/a')}</td>
                              <td>{String(item.operation_failure_rate ?? 'n/a')}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
          )}

          {showTechnical && (
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
          )}

          <div className="card" id="domain-results">
            <h2>Domain Health</h2>
            {Object.keys(results).length > 0 && (
              <div className="charts-grid">
                <BarListChart
                  title="All Domain Pass Rates"
                  rows={domainPassRateRows}
                  valueFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                />
                <StackedPassFailChart
                  title="Domain Scenario Outcomes (Pass/Fail)"
                  rows={domainCoverageRows}
                />
              </div>
            )}
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
                const scriptKindValue =
                  getField(result, ['scriptKind', 'script_kind'], null) ||
                  getField(s, ['scriptKind', 'script_kind'], null) ||
                  runScriptKind;
                return (
                  <div className="result" key={domain}>
                    <h3>
                      {formatDomainLabel(domain)}
                      <span className={`pill ${ok ? 'ok' : 'bad'}`}>{ok ? 'ok' : 'failed'}</span>
                    </h3>
                    <div className="small">
                      pass_rate={toPct(passRate)}
                      <br />
                      total={String(totalScenarios)} passed={String(passedScenarios)} failed={String(failedScenarios)}
                      <br />
                      script_kind={String(scriptKindValue)}
                      <br />
                      return_code={String(code)}
                      <br />
                      generated_scripts={String(generatedCount)}
                    </div>
                  <button onClick={() => openReport(domain, 'json')}>Open Interactive Report</button>
                    <button onClick={() => openReport(domain, 'md')}>Open Shareable Markdown</button>
                  </div>
                );
              })}
            </div>
          </div>

          {!simpleMode && (
          <div className="card">
            <details className="advanced">
              <summary>Customer Output (Click to Open/Close)</summary>
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
                      {formatDomainLabel(domain)}
                    </option>
                  ))}
                </select>
                <button disabled={!outputDomain} onClick={() => openSelectedDomainOutput('json')}>
                  Load Interactive Report
                </button>
                <button disabled={!outputDomain} onClick={() => openSelectedDomainOutput('md')}>
                  Load Markdown
                </button>
                <button onClick={() => copyText('report', reportText)}>Copy Report Text</button>
              </div>
              {!selectedReportDomain && resultDomains.length > 0 && (
                <button onClick={() => openReport(resultDomains[0], 'json')}>Load First Domain Output</button>
              )}
            </details>
          </div>
          )}

          {showTechnical && (
          <div className="card">
            <details className="advanced">
              <summary>Prompt Used By Agent (Click to Open/Close)</summary>
              {!reportJson && (
                <div className="small">
                  Load customer output first to inspect the exact prompt payload used by the agent.
                </div>
              )}
              {reportJson && (
                <>
                  <div className="small">
                    base_prompt_length={String((promptTrace?.base_prompt || '').length)} | effective_prompt_length=
                    {String((promptTrace?.effective_prompt || '').length)} | gam_focus_points=
                    {String(promptTrace?.gam_focus_points_used_count || 0)} | adaptive_focus_points=
                    {String(promptTrace?.rl_focus_points_used_count || 0)} | gam_enriched=
                    {String(Boolean(promptTrace?.prompt_was_enriched_by_gam))} | adaptive_enriched=
                    {String(Boolean(promptTrace?.prompt_was_enriched_by_rl))}
                  </div>
                  <div className="small">
                    base_prompt_supplied_by_user={String(Boolean(promptTrace?.base_prompt_supplied_by_user))} |
                    prompt_seeded_from_default={String(Boolean(promptTrace?.prompt_seeded_from_default))}
                  </div>
                  <div className="small">
                    scenario_engine={String(getField(scenarioGeneration, ['engine'], 'n/a'))} | base_source=
                    {String(getField(scenarioGeneration, ['base_source'], 'n/a'))} | llm_response_mode=
                    {String(getField(llmGenerationDiagnostics, ['response_mode'], 'n/a'))}
                  </div>
                  <div className="small">
                    llm_parse_status={String(getField(llmParseDiagnostics, ['status'], 'n/a'))} | parse_attempts=
                    {String(getField(llmParseDiagnostics, ['parse_attempts'], 'n/a'))} | parse_failures=
                    {String(getField(llmParseDiagnostics, ['parse_failures'], 'n/a'))} | schema_rejections=
                    {String(getField(scenarioGeneration?.llm_stats, ['scenario_schema_rejections'], 0))}
                  </div>
                  <details className="advanced" open={Boolean(promptTrace?.scenario_generation)}>
                    <summary>Scenario Generator Engine</summary>
                    <pre>{toJsonString(promptTrace?.scenario_generation || {})}</pre>
                  </details>
                  <details className="advanced" open>
                    <summary>Base Prompt Input</summary>
                    <pre>{promptTrace?.base_prompt || 'n/a'}</pre>
                  </details>
                  <details
                    className="advanced"
                    open={Boolean((promptTrace?.gam_focus_points_used || promptTrace?.focus_points_used || []).length)}
                  >
                    <summary>
                      GAM Excerpts Added Into Prompt (
                      {String((promptTrace?.gam_focus_points_used || promptTrace?.focus_points_used || []).length)})
                    </summary>
                    <pre>{toJsonString(promptTrace?.gam_focus_points_used || promptTrace?.focus_points_used || [])}</pre>
                  </details>
                  <details className="advanced" open={Boolean((promptTrace?.rl_focus_points_used || []).length)}>
                    <summary>
                      Adaptive Focus Points Added Into Prompt ({String((promptTrace?.rl_focus_points_used || []).length)})
                    </summary>
                    <pre>{toJsonString(promptTrace?.rl_focus_points_used || [])}</pre>
                  </details>
                  <details className="advanced">
                    <summary>Memory Excerpt Preview (What Was Available)</summary>
                    <pre>{toJsonString(promptTrace?.memory_excerpt_preview || [])}</pre>
                  </details>
                  <details className="advanced">
                    <summary>Why GAM Excerpts May Repeat</summary>
                    <pre>{toJsonString(promptTrace?.notes || {})}</pre>
                  </details>
                  <details className="advanced" open>
                    <summary>Final Effective Prompt Sent To Scenario Generator</summary>
                    <pre>{promptTrace?.effective_prompt || promptTrace?.base_prompt || 'n/a'}</pre>
                  </details>
                </>
              )}
            </details>
          </div>
          )}

          {showTechnical && (
          <div className="card">
            <details className="advanced">
              <summary>Where Adaptive Optimization Is Used (Click to Open/Close)</summary>
              {!reportJson && (
                <div className="small">Load customer output first to view runtime optimization usage locations.</div>
              )}
              {reportJson && (
                <pre>{toJsonString({
                  adaptive_prompt_influence: {
                    prompt_was_enriched_by_rl: Boolean(promptTrace?.prompt_was_enriched_by_rl),
                    rl_focus_points_used_count: Number(promptTrace?.rl_focus_points_used_count || 0),
                    rl_focus_points_used: promptTrace?.rl_focus_points_used || []
                  },
                  mutation_stage: {
                    mutated_candidates_added: Number(getField(mutationPolicy, ['mutated_candidates_added'], 0)),
                    direct_mutation_candidates_added: Number(getField(mutationPolicy, ['direct_mutation_candidates_added'], 0)),
                    history_seed_candidates_added: Number(getField(mutationPolicy, ['history_seed_candidates_added'], 0)),
                    top_targets: getField(mutationPolicy, ['top_targets'], []).slice(0, 5)
                  },
                  selection_stage: {
                    algorithm: getField(selectionPolicy, ['algorithm'], 'n/a'),
                    candidate_count: Number(getField(selectionPolicy, ['candidate_count'], 0)),
                    selected_count: Number(getField(selectionPolicy, ['selected_count'], 0)),
                    uncertain_selected_count: Number(getField(selectionPolicy, ['uncertain_selected_count'], 0)),
                    top_decisions: topDecisions.slice(0, 5)
                  },
                  periodic_training_stage: {
                    training_enabled: Boolean(getField(trainingStats, ['training_enabled'], false)),
                    rl_training_steps: Number(getField(trainingStats, ['rl_training_steps'], 0)),
                    rl_buffer_size: Number(getField(trainingStats, ['rl_buffer_size'], 0)),
                    rewarded_decisions: Number(getField(learningFeedback, ['rewarded_decisions'], 0)),
                    penalized_decisions: Number(getField(learningFeedback, ['penalized_decisions'], 0))
                  },
                  note: "Mutation and selection policies optimize coverage during runs, while model updates happen periodically in background mode. GAM supports LLM-driven planning/reflection (see GAM Research Influence for plan_mode/reflect_mode and llm_stats)."
                })}</pre>
              )}
            </details>
          </div>
          )}

          {showTechnical && (
          <div className="card">
            <details className="advanced" open>
              <summary>State-by-State API Output (Glass Box) (Click to Open/Close)</summary>
              <div className="small">
                selected_domain={selectedStateDomain || 'none'} | states={stateTrace.length} | scripts_loading=
                {String(generatedScriptsLoading)}
              </div>
              <div className="state-trace-list">
                {stateTrace.map((state) => (
                  <details key={state.id} className={`state-trace-item ${state.ready ? 'done' : ''}`} open={state.ready}>
                    <summary>
                      <span className="state-trace-title">{state.ready ? '✓' : '•'} {state.title}</span>
                      <span className={`pill ${state.ready ? 'ok' : 'bad'}`}>{state.ready ? 'ready' : 'waiting'}</span>
                    </summary>
                    <pre>{toJsonString(state.payload)}</pre>
                  </details>
                ))}
              </div>
            </details>
          </div>
          )}

          {!simpleMode && (
          <div className="card" id="generated-scripts">
            <h2>Generated Test Scripts</h2>
            <div className="script-summary-grid">
              <div className="script-summary-item">
                <span>Selected Domain</span>
                <strong>{generatedTestsDomain || 'none'}</strong>
              </div>
              <div className="script-summary-item">
                <span>Scripts Returned</span>
                <strong>{String(generatedTests.length)}</strong>
              </div>
              <div className="script-summary-item">
                <span>Loading</span>
                <strong>{generatedScriptsLoading ? 'yes' : 'no'}</strong>
              </div>
              <div className="script-summary-item">
                <span>Execution Status</span>
                <strong>{String(getField(generatedScriptExecution, ['status'], 'n/a'))}</strong>
              </div>
              <div className="script-summary-item">
                <span>Executed Tests</span>
                <strong>{String(getField(generatedScriptExecution, ['total_tests'], 'n/a'))}</strong>
              </div>
              <div className="script-summary-item">
                <span>Passed / Failed</span>
                <strong>
                  {String(getField(generatedScriptExecution, ['passed_tests'], 'n/a'))} /{' '}
                  {String(getField(generatedScriptExecution, ['failed_tests'], 'n/a'))}
                </strong>
              </div>
            </div>
            {generatedTests.length === 0 && (
              <div className="small">Load customer output for a domain to view generated scripts.</div>
            )}
            {generatedTests.length > 0 && (
              <div className="inline-actions">
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => loadGeneratedTests(generatedTestsDomain || selectedStateDomain, generatedTests)}
                  disabled={generatedScriptsLoading || !(generatedTestsDomain || selectedStateDomain)}
                >
                  Reload Script Bundle From API
                </Button>
              </div>
            )}
            {generatedTests.length > 0 && (
              <div className="script-table-wrap">
                <table className="script-table">
                  <thead>
                    <tr>
                      <th>Language</th>
                      <th>Kind</th>
                      <th>Path</th>
                      <th>Status</th>
                      <th>Readable</th>
                      <th>Size</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {generatedTests.map((item) => (
                      <tr key={`${generatedTestsDomain}-${item.kind}`}>
                        <td>{SCRIPT_KIND_LABELS[item.kind] || item.kind}</td>
                        <td><code>{item.kind}</code></td>
                        <td className="script-path-cell"><code>{item.path || 'n/a'}</code></td>
                        <td>{item.exists ? 'ready' : 'missing'}</td>
                        <td>{item.safe_to_read ? 'yes' : 'no'}</td>
                        <td>{formatBytes(item.size_bytes)}</td>
                        <td>
                          <Button
                            size="small"
                            variant="outlined"
                            disabled={!item.exists || !item.safe_to_read || !generatedTestsDomain}
                            onClick={() => openGeneratedScript(generatedTestsDomain, item.kind)}
                          >
                            Preview
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {selectedScriptInsights && (
              <div className="script-insight-panel">
                <div className="script-insight-head">
                  <h3>Script Explanation</h3>
                  <span className="small">language={SCRIPT_KIND_LABELS[selectedScriptInsights.language] || selectedScriptInsights.language}</span>
                </div>
                <div className="script-summary-grid">
                  <div className="script-summary-item">
                    <span>Non-empty Lines</span>
                    <strong>{String(selectedScriptInsights.lineCount)}</strong>
                  </div>
                  <div className="script-summary-item">
                    <span>Test Functions</span>
                    <strong>{String(selectedScriptInsights.testCount)}</strong>
                  </div>
                  <div className="script-summary-item">
                    <span>HTTP Calls</span>
                    <strong>{String(selectedScriptInsights.requestCount)}</strong>
                  </div>
                  <div className="script-summary-item">
                    <span>Endpoint Mentions</span>
                    <strong>{String(selectedScriptInsights.endpointCount)}</strong>
                  </div>
                  <div className="script-summary-item">
                    <span>2xx / 4xx / 5xx</span>
                    <strong>
                      {String(selectedScriptInsights.statusBuckets.success_2xx || 0)} /{' '}
                      {String(selectedScriptInsights.statusBuckets.client_4xx || 0)} /{' '}
                      {String(selectedScriptInsights.statusBuckets.server_5xx || 0)}
                    </strong>
                  </div>
                  <div className="script-summary-item">
                    <span>Primary Focus</span>
                    <strong>{selectedScriptInsights.focus}</strong>
                  </div>
                </div>
                {Object.keys(selectedScriptInsights.methodCounts || {}).length > 0 && (
                  <div className="small">
                    methods: GET={String(selectedScriptInsights.methodCounts.get || 0)} POST=
                    {String(selectedScriptInsights.methodCounts.post || 0)} PUT=
                    {String(selectedScriptInsights.methodCounts.put || 0)} PATCH=
                    {String(selectedScriptInsights.methodCounts.patch || 0)} DELETE=
                    {String(selectedScriptInsights.methodCounts.delete || 0)}
                  </div>
                )}
                {selectedScriptInsights.sampleEndpoints.length > 0 && (
                  <div className="small script-endpoints">
                    endpoint samples: {selectedScriptInsights.sampleEndpoints.join(' | ')}
                  </div>
                )}
                {selectedScriptInsights.warnings.length > 0 && (
                  <div className="script-warning-box">
                    {selectedScriptInsights.warnings.map((warning) => (
                      <div key={warning} className="small">
                        ! {warning}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div className="script-preview-head">
              <div className="small">Selected Script: {selectedScriptKind || 'none'}</div>
              <div className="inline-actions">
                <Button size="small" variant="outlined" onClick={() => copyText('script', selectedScriptContent)} disabled={!canCopyScript}>
                  Copy Script
                </Button>
                <Button size="small" variant="contained" onClick={downloadScript} disabled={!canCopyScript}>
                  Download Script
                </Button>
              </div>
            </div>
            {!canCopyScript && (
              <Alert severity="warning" sx={{ mt: 1 }}>
                Script preview is unavailable. Select a readable script from the table.
              </Alert>
            )}
            <pre className="script-preview">{selectedScriptContent || 'Select a generated test script to preview.'}</pre>
          </div>
          )}

          {!simpleMode && (
          <div className="card" id="cases-tested">
            <h2>Test Results</h2>
            <div className="small">
              selected_domain={selectedDomainLabel} | total={scenarioResults.length} | passed={passedScenarioCount} |
              failed={failedScenarioCount}
            </div>
            {!reportJson && (
              <div className="small">Open customer output for a domain to view tested cases.</div>
            )}
            {reportJson && (
              <>
                <div className="report-grid">
                  <div className="report-metric">
                    <span>Endpoints Covered</span>
                    <strong>{String(endpointCoverageCount)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Test Types Covered</span>
                    <strong>{String(scenarioCoverageRows.length)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Mutation Scenarios Executed</span>
                    <strong>{String(rlExecutedScenarioCount)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>History-Seeded Executed</span>
                    <strong>{String(historySeedExecutedCount)}</strong>
                  </div>
                  <div className="report-metric">
                    <span>Direct Mutations Added</span>
                    <strong>{String(getField(mutationPolicy, ['direct_mutation_candidates_added'], 0))}</strong>
                  </div>
                  <div className="report-metric">
                    <span>History-Seed Mutations Added</span>
                    <strong>{String(getField(mutationPolicy, ['history_seed_candidates_added'], 0))}</strong>
                  </div>
                </div>

                <div className="charts-grid">
                  <BarListChart
                    title="Scenario Outcome Mix"
                    rows={scenarioStatusRows}
                    valueFormatter={(v) => String(Math.round(v))}
                  />
                  <BarListChart
                    title="Slowest Scenarios (ms)"
                    rows={slowestScenarioRows}
                    valueFormatter={(v) => `${v.toFixed(1)} ms`}
                    emptyText="No scenario duration data."
                  />
                </div>

                <div className="charts-grid">
                  <StackedPassFailChart
                    title="Coverage By Test Type"
                    rows={scenarioCoverageRows}
                  />
                  <BarListChart
                    title="Failed Scenarios By Test Type"
                    rows={failedByTypeRows}
                    valueFormatter={(v) => String(Math.round(v))}
                    emptyText="No failed test types."
                  />
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>Coverage By Test Type</h3>
                  </div>
                  <details className="advanced">
                    <summary>Open detailed table</summary>
                    <div className="scenario-table-wrap">
                      <table className="scenario-table">
                        <thead>
                          <tr>
                            <th>test_type</th>
                            <th>total</th>
                            <th>passed</th>
                            <th>failed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {scenarioCoverageRows.map((row) => (
                            <tr key={row.testType}>
                              <td>{row.testType}</td>
                              <td>{String(row.total)}</td>
                              <td>{String(row.passed)}</td>
                              <td>{String(row.failed)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                </div>

                <div className="scenario-panel">
                  <div className="scenario-head">
                    <h3>What Improved Due To Adaptive Optimization</h3>
                  </div>
                  <div className="small">
                    rewarded_decisions={String(getField(learningFeedback, ['rewarded_decisions'], 'n/a'))} | penalized_decisions=
                    {String(getField(learningFeedback, ['penalized_decisions'], 'n/a'))}
                  </div>
                  {rlMutationExamples.length > 0 && (
                    <div className="small rl-examples">
                      mutation_examples: {rlMutationExamples
                        .map((item) => `${String(item.from || item.from_fingerprint || 'seed')} -> ${String(item.to || 'n/a')}`)
                        .join(' | ')}
                    </div>
                  )}
                  <details className="advanced">
                    <summary>Open improved scenarios table</summary>
                    <div className="scenario-table-wrap">
                      <table className="scenario-table">
                        <thead>
                          <tr>
                            <th>top_improved_scenario</th>
                            <th>reward</th>
                            <th>type</th>
                            <th>endpoint</th>
                          </tr>
                        </thead>
                        <tbody>
                          {topImprovedScenarios.length === 0 && (
                            <tr>
                              <td colSpan={4}>No rewarded decision signals yet.</td>
                            </tr>
                          )}
                          {topImprovedScenarios.map((row) => {
                            const scenarioInfo = humanizeScenarioName(row);
                            return (
                              <tr key={`improved-${String(row.name)}`}>
                                <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                  <div>{scenarioInfo.label}</div>
                                  <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                                </td>
                                <td>{String(row.reward)}</td>
                                <td>{String(row.test_type)}</td>
                                <td>{`${String(row.method || '')} ${String(row.endpoint_template || '')}`.trim()}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </details>
                  {topFailedScenarios.length > 0 && (
                    <details className="advanced">
                      <summary>Open needs-improvement scenarios table</summary>
                      <div className="scenario-table-wrap">
                        <table className="scenario-table">
                          <thead>
                            <tr>
                              <th>needs_improvement</th>
                              <th>reward</th>
                              <th>type</th>
                              <th>endpoint</th>
                            </tr>
                          </thead>
                          <tbody>
                            {topFailedScenarios.map((row) => {
                              const scenarioInfo = humanizeScenarioName(row);
                              return (
                                <tr key={`regress-${String(row.name)}`}>
                                  <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                    <div>{scenarioInfo.label}</div>
                                    <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                                  </td>
                                  <td>{String(row.reward)}</td>
                                  <td>{String(row.test_type)}</td>
                                  <td>{`${String(row.method || '')} ${String(row.endpoint_template || '')}`.trim()}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  )}
                </div>

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
                  <details className="advanced">
                    <summary>Open detailed scenario table</summary>
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
                          {filteredScenarios.map((row) => {
                            const scenarioInfo = humanizeScenarioName(row);
                            return (
                              <tr key={String(row.name)}>
                                <td>{row.passed ? 'pass' : 'fail'}</td>
                                <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                  <div>{scenarioInfo.label}</div>
                                  <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                                </td>
                                <td>{row.test_type}</td>
                                <td>{`${row.method || ''} ${row.endpoint_template || ''}`.trim()}</td>
                                <td>{String(row.expected_status ?? 'n/a')}</td>
                                <td>{String(row.actual_status ?? 'n/a')}</td>
                                <td>{String(row.duration_ms ?? 'n/a')}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </details>
                </div>
              </>
            )}
          </div>
          )}

          <div className="card" id="final-report">
            <h2>Customer Report</h2>
            <div className="small">
              selected_domain={selectedDomainLabel} | format={selectedReportFormat}
            </div>
            {selectedReportDomain && (
              <div className="inline-actions">
                <Button size="small" variant="outlined" onClick={() => openReport(selectedReportDomain, 'json')}>JSON View</Button>
                <Button size="small" variant="outlined" onClick={() => openReport(selectedReportDomain, 'md')}>Markdown View</Button>
                <Button size="small" variant="outlined" onClick={() => copyText('report', reportText)}>Copy Current Report</Button>
                <Button size="small" variant="contained" onClick={() => copyText('customer summary', customerSummaryText)} disabled={!customerSummaryText}>
                  Copy Summary Snapshot
                </Button>
              </div>
            )}
            {reportJson && (
              <>
                <Box className="report-hero">
                  <div>
                    <Typography variant="h6">Executive Summary</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {summaryQualityGate === true
                        ? 'Quality gate passed. This run is suitable for customer sharing.'
                        : 'Quality gate did not pass. Review failures before sharing externally.'}
                    </Typography>
                  </div>
                  <Stack direction="row" gap={1} flexWrap="wrap">
                    <Chip
                      label={`Risk: ${reportRiskLevel}`}
                      color={reportRiskLevel === 'low' ? 'success' : reportRiskLevel === 'moderate' ? 'warning' : 'error'}
                    />
                    <Chip label={`Pass: ${toPct(getField(reportSummary, ['pass_rate'], null))}`} />
                    <Chip label={`Failed: ${String(getField(reportSummary, ['failed_scenarios'], 0))}`} />
                    <Chip label={`Flaky: ${String(getField(reportSummary, ['flaky_ratio'], 0))}`} />
                  </Stack>
                </Box>

                {(qualityFailReasons.length > 0 || qualityWarnings.length > 0) && (
                  <Stack gap={1} sx={{ mb: 1 }}>
                    {qualityFailReasons.length > 0 && (
                      <Alert severity="error">
                        Quality gate failed because: {qualityFailReasons.join(', ')}
                      </Alert>
                    )}
                    {qualityWarnings.length > 0 && (
                      <Alert severity="warning">
                        Warnings: {qualityWarnings.join(', ')}
                      </Alert>
                    )}
                  </Stack>
                )}

                <div className="report-kpi-rings">
                  <KpiRing label="Pass Rate" value={Number(getField(reportSummary, ['pass_rate'], 0) || 0)} color="#1f7a58" />
                  <KpiRing label="Selection Rate" value={(Number(selectionSelected || 0) > 0 && Number(selectionCandidates || 0) > 0) ? Number(selectionSelected) / Number(selectionCandidates) : 0} color="#0d5e7a" />
                  <KpiRing label="Negative Mix" value={scenarioResults.length ? Math.min(1, (scenarioCoverageRows.filter((row) => Number(row?.failed || 0) > 0).length + failedScenarioCount) / scenarioResults.length) : 0} color="#d47b1f" />
                </div>

                <div className="charts-grid">
                  <div className="chart-card">
                    <h3>Top Findings</h3>
                    {failedExamples.length === 0 && <div className="small">No failed examples in this report.</div>}
                    {failedExamples.length > 0 && (
                      <ul className="report-list">
                        {failedExamples.map((item, idx) => {
                          const scenarioInfo = humanizeScenarioName(item);
                          return (
                            <li key={`${String(item?.name || 'item')}-${idx}`} title={scenarioInfo.rawName || scenarioInfo.label}>
                              <b>{scenarioInfo.label}</b>:
                              {' '}expected {String(getField(item, ['expected_status'], 'n/a'))}, got {String(getField(item, ['actual_status'], 'n/a'))}
                              {scenarioInfo.rawName && (
                                <> | raw={scenarioInfo.rawName}</>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                  <div className="chart-card">
                    <h3>Recommended Actions</h3>
                    <ul className="report-list">
                      {recommendedActions.map((item, idx) => (
                        <li key={`${item}-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="scenario-panel report-spotlight">
                  <div className="scenario-head">
                    <h3>Interactive Scenario Spotlight</h3>
                    <div className="spotlight-controls">
                      <Chip
                        size="small"
                        label="Critical"
                        clickable
                        color={reportScenarioMode === 'critical' ? 'primary' : 'default'}
                        variant={reportScenarioMode === 'critical' ? 'filled' : 'outlined'}
                        onClick={() => setReportScenarioMode('critical')}
                      />
                      <Chip
                        size="small"
                        label="Negative Focus"
                        clickable
                        color={reportScenarioMode === 'negative' ? 'warning' : 'default'}
                        variant={reportScenarioMode === 'negative' ? 'filled' : 'outlined'}
                        onClick={() => setReportScenarioMode('negative')}
                      />
                      <Chip
                        size="small"
                        label="Slowest"
                        clickable
                        color={reportScenarioMode === 'slow' ? 'error' : 'default'}
                        variant={reportScenarioMode === 'slow' ? 'filled' : 'outlined'}
                        onClick={() => setReportScenarioMode('slow')}
                      />
                    </div>
                  </div>
                  <div className="small">Mode: {reportScenarioMode} | rows: {reportScenarioSpotlight.length}</div>
                  <div className="scenario-table-wrap">
                    <table className="scenario-table">
                      <thead>
                        <tr>
                          <th>verdict</th>
                          <th>scenario</th>
                          <th>test_type</th>
                          <th>endpoint</th>
                          <th>expected</th>
                          <th>actual</th>
                          <th>duration_ms</th>
                        </tr>
                      </thead>
                      <tbody>
                        {reportScenarioSpotlight.map((row) => {
                          const scenarioInfo = humanizeScenarioName(row);
                          return (
                            <tr key={`${String(row?.name || 'scenario')}-${String(row?.endpoint_template || '')}`}>
                              <td>{row?.passed ? 'pass' : 'fail'}</td>
                              <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                <div>{scenarioInfo.label}</div>
                                <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                              </td>
                              <td>{String(row?.test_type || 'n/a')}</td>
                              <td>{`${String(row?.method || '')} ${String(row?.endpoint_template || '')}`.trim()}</td>
                              <td>{String(row?.expected_status ?? 'n/a')}</td>
                              <td>{String(row?.actual_status ?? 'n/a')}</td>
                              <td>{String(row?.duration_ms ?? 'n/a')}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <Tabs
                  value={reportViewTab}
                  onChange={(_, value) => setReportViewTab(value)}
                  className="report-tabs"
                  variant="scrollable"
                  allowScrollButtonsMobile
                >
                  <Tab label={jobState === 'running' ? 'Overview (Live)' : 'Overview'} value="overview" />
                  <Tab label="Quality" value="quality" />
                  <Tab label="Trends" value="learning" />
                  <Tab label="Raw" value="raw" />
                </Tabs>

                {reportViewTab === 'overview' && (
                  <>
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
                        <span>Failed</span>
                        <strong>{String(getField(reportSummary, ['failed_scenarios'], 'n/a'))}</strong>
                      </div>
                      <div className="report-metric">
                        <span>Flaky Ratio</span>
                        <strong>{String(getField(reportSummary, ['flaky_ratio'], 'n/a'))}</strong>
                      </div>
                      <div className="report-metric">
                        <span>Run Reward</span>
                        <strong>{String(getField(learningFeedback, ['run_reward'], 'n/a'))}</strong>
                      </div>
                    </div>

                    <div className="charts-grid">
                      <BarListChart
                        title="Domain Pass Rate Comparison"
                        rows={domainPassRateRows}
                        valueFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                        emptyText="Run more than one domain to compare."
                      />
                      <StackedPassFailChart
                        title="Test Type Coverage (Pass/Fail)"
                        rows={scenarioCoverageRows}
                      />
                      <BarListChart
                        title="Stage Runtime (ms)"
                        rows={stageMetricRows}
                        valueFormatter={(v) => `${v.toFixed(1)} ms`}
                        emptyText="No stage metrics available."
                      />
                      <BarListChart
                        title="Failed Scenarios by Type"
                        rows={failedByTypeRows}
                        valueFormatter={(v) => String(Math.round(v))}
                        emptyText="No failed test types."
                      />
                    </div>
                  </>
                )}

                {reportViewTab === 'quality' && (
                  <div className="charts-grid">
                    <BarListChart
                      title="Failure Taxonomy Breakdown"
                      rows={failureTaxonomyRows}
                      valueFormatter={(v) => String(Math.round(v))}
                      emptyText="No taxonomy failures in this report."
                    />
                    <BarListChart
                      title="Reward Breakdown Components"
                      rows={rewardBreakdownRows}
                      valueFormatter={(v) => v.toFixed(4)}
                    />
                  </div>
                )}

                {reportViewTab === 'learning' && (
                  <>
                    <div className="inline-actions">
                      <label className="small">
                        Trend Metric
                      </label>
                      <select value={trendMetric} onChange={(e) => setTrendMetric(e.target.value)}>
                        <option value="run_reward">Run Reward</option>
                        <option value="avg_decision_reward">Average Decision Reward</option>
                      </select>
                    </div>
                    <div className="charts-grid">
                      <TrendSpark
                        title="Learning Trend by Run"
                        points={trendRows}
                        valueFormatter={(v) => v.toFixed(4)}
                      />
                      <div className="chart-card">
                        <h3>Learning Snapshot</h3>
                        <div className="report-grid">
                          <div className="report-metric">
                            <span>Model Steps</span>
                            <strong>{String(getField(trainingStats, ['rl_training_steps'], 'n/a'))}</strong>
                          </div>
                          <div className="report-metric">
                            <span>Buffer Size</span>
                            <strong>{String(getField(trainingStats, ['rl_buffer_size'], 'n/a'))}</strong>
                          </div>
                          <div className="report-metric">
                            <span>Rewarded</span>
                            <strong>{String(getField(learningFeedback, ['rewarded_decisions'], 'n/a'))}</strong>
                          </div>
                          <div className="report-metric">
                            <span>Penalized</span>
                            <strong>{String(getField(learningFeedback, ['penalized_decisions'], 'n/a'))}</strong>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {reportViewTab === 'raw' && (
                  <details className="advanced" open>
                    <summary>Raw Report Payload</summary>
                    <pre>{reportText}</pre>
                  </details>
                )}
              </>
            )}
            {!reportJson && (
              <details className="advanced">
                <summary>Raw Report Payload</summary>
                <pre>{reportText}</pre>
              </details>
            )}
          </div>

          {showTechnical && (
          <div className="card">
            <h2>Live Process Log</h2>
            <pre>{logText || 'No logs yet.'}</pre>
          </div>
          )}

          {showTechnical && (
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
                        {topDecisions.slice(0, 15).map((row) => {
                          const scenarioInfo = humanizeScenarioName(row);
                          return (
                            <tr key={String(row.name)}>
                              <td title={scenarioInfo.rawName || scenarioInfo.label}>
                                <div>{scenarioInfo.label}</div>
                                <div className="small">raw: {scenarioInfo.rawName || 'n/a'}</div>
                              </td>
                              <td>{row.test_type}</td>
                              <td>{row.selection_reason}</td>
                              <td>{String(row.score ?? 'n/a')}</td>
                              <td>{String(row.uncertainty ?? 'n/a')}</td>
                              <td>{String(row.expected_reward ?? 'n/a')}</td>
                            </tr>
                          );
                        })}
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
          )}
        </section>
      </div>
    </main>
  );
}
