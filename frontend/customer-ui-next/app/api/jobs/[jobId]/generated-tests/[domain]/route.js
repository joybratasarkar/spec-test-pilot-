export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { promises as fs } from 'node:fs';
import path from 'node:path';

import { getJob } from '@/lib/store';

async function loadGeneratedTestsFromReport(reportJsonPath) {
  try {
    const raw = await fs.readFile(reportJsonPath, 'utf-8');
    const payload = JSON.parse(raw);
    const generated = payload?.generated_test_files;
    if (!generated || typeof generated !== 'object') {
      return {};
    }
    const out = {};
    for (const [kind, filePath] of Object.entries(generated)) {
      if (typeof filePath === 'string' && filePath.trim()) {
        out[String(kind)] = filePath;
      }
    }
    return out;
  } catch {
    return {};
  }
}

function isPathWithin(parentPath, childPath) {
  const rel = path.relative(parentPath, childPath);
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}

export async function GET(_request, { params }) {
  const job = getJob(params.jobId);
  if (!job) {
    return Response.json({ error: 'job not found' }, { status: 404 });
  }

  const result = job.results?.[params.domain];
  if (!result) {
    return Response.json({ error: 'domain result not found' }, { status: 404 });
  }

  const outputDir = path.resolve(String(result.outputDir || ''));
  const generated = await loadGeneratedTestsFromReport(result.reportJsonPath);
  const items = [];

  for (const kind of Object.keys(generated).sort()) {
    const filePathRaw = generated[kind];
    const resolved = path.resolve(filePathRaw);
    const safeToRead = isPathWithin(outputDir, resolved);
    let exists = false;
    let sizeBytes = null;

    try {
      const stat = await fs.stat(resolved);
      exists = stat.isFile();
      sizeBytes = exists ? stat.size : null;
    } catch {
      exists = false;
      sizeBytes = null;
    }

    items.push({
      kind,
      path: filePathRaw,
      exists,
      size_bytes: sizeBytes,
      safe_to_read: safeToRead
    });
  }

  return Response.json({
    job_id: params.jobId,
    domain: params.domain,
    count: items.length,
    generated_tests: items
  });
}
