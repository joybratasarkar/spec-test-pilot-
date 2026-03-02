export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { getJob, snapshotJob, subscribeJob } from '@/lib/store';

const encoder = new TextEncoder();

function sseChunk(event, payload) {
  return encoder.encode(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
}

export async function GET(request, { params }) {
  const jobId = params.jobId;
  if (!getJob(jobId)) {
    return Response.json({ error: 'job not found' }, { status: 404 });
  }

  let unsubscribe = null;
  let heartbeat = null;

  const stream = new ReadableStream({
    start(controller) {
      const sendSnapshot = (payload) => {
        if (!payload) {
          return;
        }
        controller.enqueue(sseChunk('snapshot', payload));
      };

      // Initial snapshot
      sendSnapshot(snapshotJob(jobId, 1500));

      unsubscribe = subscribeJob(jobId, (payload) => {
        sendSnapshot(payload);
        if (payload && (payload.status === 'completed' || payload.status === 'failed')) {
          controller.enqueue(sseChunk('done', { status: payload.status }));
        }
      });

      // Keep-alive heartbeat
      heartbeat = setInterval(() => {
        controller.enqueue(sseChunk('ping', { ts: Date.now() }));
      }, 15000);

      const close = () => {
        if (heartbeat) {
          clearInterval(heartbeat);
          heartbeat = null;
        }
        if (unsubscribe) {
          unsubscribe();
          unsubscribe = null;
        }
        try {
          controller.close();
        } catch {
          // Stream may already be closed.
        }
      };

      request.signal.addEventListener('abort', close, { once: true });
    },
    cancel() {
      if (heartbeat) {
        clearInterval(heartbeat);
        heartbeat = null;
      }
      if (unsubscribe) {
        unsubscribe();
        unsubscribe = null;
      }
    }
  });

  return new Response(stream, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no'
    }
  });
}
