import { NextRequest } from 'next/server';
import { NextResponse, NextRequest } from 'next/server';
import { createParser } from 'eventsource-parser';

// Extremely simple file-based embedding store in memory for demo
// In production, use a vector DB (Pinecone, pgvector, etc.)
let cachedDocs: { id: string; text: string; embedding: number[] }[] | null = null;

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const LlamaHost = process.env.LLAMA_DEPLOY_HOST || 'http://localhost:4501';

export async function POST(req: NextRequest) {
  // Parse messages array from Vercel AI SDK
  let messages: any[] = [];
  try {
    const raw = await req.text();
    if (raw) messages = JSON.parse(raw)?.messages ?? [];
  } catch {
    try {
      const form = await req.formData();
      const m = form.get('messages');
      if (typeof m === 'string') messages = JSON.parse(m);
    } catch {}
  }

  const userMsg: string = messages?.[messages.length - 1]?.content ?? '';
  const body = {
    input: JSON.stringify({ user_msg: userMsg }),
    service_id: 'workflow',
  };

  const deployUrl = `${LlamaHost}/deployments/chat/tasks/create`;
  const resp = await fetch(deployUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    return NextResponse.json({ error: txt || 'Backend error' }, { status: 500 });
  }

  // For simplicity, expect a text response; if streaming, adapt to SSE
  const data = await resp.json().catch(async () => ({ text: await resp.text() }));
  const text = data?.output?.response ?? data?.text ?? JSON.stringify(data);
  return NextResponse.json({ id: 'assistant', role: 'assistant', content: text });
}
