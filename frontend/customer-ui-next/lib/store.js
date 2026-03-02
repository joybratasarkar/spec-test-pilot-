const MAX_LOG_LINES = 7000;
const STORE_KEY = '__QA_UI_NEXT_STORE__';

function getStore() {
  if (!globalThis[STORE_KEY]) {
    globalThis[STORE_KEY] = {
      jobs: new Map(),
      listeners: new Map()
    };
  }
  return globalThis[STORE_KEY];
}

function sanitizeToken(value, fallback = 'default') {
  const token = String(value ?? '').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
  return token || fallback;
}

export function normalizeRequest(input = {}) {
  const rawDomains = Array.isArray(input.domains) ? input.domains : ['ecommerce'];
  const domains = [...new Set(rawDomains.map((d) => sanitizeToken(d)).filter(Boolean))].filter((d) =>
    ['ecommerce', 'healthcare', 'logistics', 'hr'].includes(d)
  );

  return {
    domains: domains.length ? domains : ['ecommerce'],
    tenantId: sanitizeToken(input.tenantId ?? input.tenant_id ?? 'customer_default'),
    prompt: typeof input.prompt === 'string' && input.prompt.trim() ? input.prompt.trim() : null,
    maxScenarios: Math.min(500, Math.max(1, Number(input.maxScenarios ?? input.max_scenarios ?? 16) || 16)),
    passThreshold: Math.min(1, Math.max(0, Number(input.passThreshold ?? input.pass_threshold ?? 0.7) || 0.7)),
    baseUrl: typeof input.baseUrl === 'string' && input.baseUrl.trim() ? input.baseUrl.trim() : 'http://localhost:8000',
    customerMode: input.customerMode !== false && input.customer_mode !== false,
    verifyPersistence: input.verifyPersistence !== false && input.verify_persistence !== false,
    customerRoot:
      typeof input.customerRoot === 'string' && input.customerRoot.trim()
        ? input.customerRoot.trim()
        : '~/.spec_test_pilot'
  };
}

export function createJob(request) {
  const store = getStore();
  const id = Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
  const now = new Date().toISOString();

  const job = {
    id,
    status: 'queued',
    createdAt: now,
    startedAt: null,
    completedAt: null,
    currentDomain: null,
    request,
    logs: [],
    results: {},
    error: null
  };

  store.jobs.set(id, job);
  emitJob(id);
  return job;
}

export function getJob(jobId) {
  return getStore().jobs.get(jobId) || null;
}

export function listJobs() {
  return Array.from(getStore().jobs.values()).map((job) => ({
    id: job.id,
    status: job.status,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt,
    currentDomain: job.currentDomain,
    domains: job.request.domains
  }));
}

export function updateJob(jobId, patch) {
  const job = getJob(jobId);
  if (!job) {
    return null;
  }
  Object.assign(job, patch);
  emitJob(jobId);
  return job;
}

export function appendJobLog(jobId, line) {
  const job = getJob(jobId);
  if (!job) {
    return;
  }
  job.logs.push(String(line).replace(/\n$/, ''));
  if (job.logs.length > MAX_LOG_LINES) {
    job.logs.splice(0, job.logs.length - MAX_LOG_LINES);
  }
  emitJob(jobId);
}

export function setDomainResult(jobId, domain, result) {
  const job = getJob(jobId);
  if (!job) {
    return;
  }
  job.results[domain] = result;
  emitJob(jobId);
}

export function snapshotJob(jobId, tail = 800) {
  const job = getJob(jobId);
  if (!job) {
    return null;
  }

  const logTail = Math.max(50, Math.min(3000, Number(tail) || 800));
  return {
    id: job.id,
    status: job.status,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt,
    currentDomain: job.currentDomain,
    request: job.request,
    results: job.results,
    logs: job.logs.slice(-logTail),
    error: job.error
  };
}

export function subscribeJob(jobId, callback) {
  const store = getStore();
  if (!store.listeners.has(jobId)) {
    store.listeners.set(jobId, new Set());
  }
  const bucket = store.listeners.get(jobId);
  bucket.add(callback);
  return () => {
    bucket.delete(callback);
    if (bucket.size === 0) {
      store.listeners.delete(jobId);
    }
  };
}

function emitJob(jobId) {
  const store = getStore();
  const bucket = store.listeners.get(jobId);
  if (!bucket || bucket.size === 0) {
    return;
  }
  const payload = snapshotJob(jobId, 1500);
  for (const cb of bucket) {
    try {
      cb(payload);
    } catch {
      // Listener errors must not break store updates.
    }
  }
}
