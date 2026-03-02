export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { promises as fs } from 'node:fs';

import { getJob } from '@/lib/store';

export async function GET(request, { params }) {
  const { searchParams } = new URL(request.url);
  const format = (searchParams.get('format') || 'json').toLowerCase();
  if (!['json', 'md'].includes(format)) {
    return Response.json({ error: 'format must be json or md' }, { status: 400 });
  }

  const job = getJob(params.jobId);
  if (!job) {
    return Response.json({ error: 'job not found' }, { status: 404 });
  }

  const result = job.results?.[params.domain];
  if (!result) {
    return Response.json({ error: 'domain result not found' }, { status: 404 });
  }

  const filePath = format === 'json' ? result.reportJsonPath : result.reportMdPath;
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    if (format === 'json') {
      return Response.json(JSON.parse(raw));
    }
    return new Response(raw, {
      headers: {
        'content-type': 'text/plain; charset=utf-8'
      }
    });
  } catch {
    return Response.json({ error: 'report file missing' }, { status: 404 });
  }
}
