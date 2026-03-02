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

  const generated = await loadGeneratedTestsFromReport(result.reportJsonPath);
  const scriptPathRaw = generated[params.kind];
  if (!scriptPathRaw) {
    return Response.json({ error: `generated test script kind not found: ${params.kind}` }, { status: 404 });
  }

  const outputDir = path.resolve(String(result.outputDir || ''));
  const scriptPath = path.resolve(scriptPathRaw);
  if (!isPathWithin(outputDir, scriptPath)) {
    return Response.json({ error: 'generated script path is outside output directory' }, { status: 400 });
  }

  try {
    const raw = await fs.readFile(scriptPath, 'utf-8');
    return new Response(raw, {
      headers: {
        'content-type': 'text/plain; charset=utf-8'
      }
    });
  } catch {
    return Response.json({ error: 'generated script file missing' }, { status: 404 });
  }
}
