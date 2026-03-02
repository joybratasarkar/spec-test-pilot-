export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { createJob, listJobs, normalizeRequest } from '@/lib/store';
import { markJobFailed, runJob } from '@/lib/runner';

export async function GET() {
  return Response.json(listJobs());
}

export async function POST(request) {
  let body = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  let normalized;
  try {
    normalized = normalizeRequest(body);
  } catch (error) {
    return Response.json({ error: String(error?.message || error) }, { status: 400 });
  }

  const job = createJob(normalized);

  runJob(job.id).catch((error) => {
    void markJobFailed(job.id, String(error?.message || error));
  });

  return Response.json({ jobId: job.id, status: 'queued' });
}
