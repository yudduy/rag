import { NextRequest } from 'next/server';
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import path from 'node:path';
import fs from 'node:fs/promises';

// Extremely simple file-based embedding store in memory for demo
// In production, use a vector DB (Pinecone, pgvector, etc.)
let cachedDocs: { id: string; text: string; embedding: number[] }[] | null = null;

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function embedText(input: string): Promise<number[]> {
  // Cheap embedding via text-embedding-3-small is supported through OpenAI embed API.
  // Here we approximate with a naive hash->vector for demo to avoid extra provider setup.
  // Replace with a real embedding call if needed.
  const vec = new Array(128).fill(0);
  for (let i = 0; i < input.length; i++) {
    vec[i % vec.length] += input.charCodeAt(i) / 255;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

function cosine(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length && i < b.length; i++) s += a[i] * b[i];
  return s;
}

async function loadDocs(baseDir: string) {
  if (cachedDocs) return cachedDocs;
  const entries = await fs.readdir(baseDir, { withFileTypes: true });
  const docs: { id: string; text: string; embedding: number[] }[] = [];
  for (const e of entries) {
    if (!e.isFile()) continue;
    if (!e.name.match(/\.(txt|md)$/i)) continue;
    const filePath = path.join(baseDir, e.name);
    const text = await fs.readFile(filePath, 'utf8');
    const embedding = await embedText(text);
    docs.push({ id: e.name, text, embedding });
  }
  cachedDocs = docs;
  return docs;
}

export async function POST(req: NextRequest) {
  // Robust body parsing: JSON first, then formData fallback
  let messages: any[] = [];
  try {
    const raw = await req.text();
    if (raw) {
      const parsed = JSON.parse(raw);
      messages = parsed?.messages ?? [];
    }
  } catch {
    try {
      const form = await req.formData();
      const m = form.get('messages');
      if (typeof m === 'string') {
        messages = JSON.parse(m);
      }
    } catch {}
  }

  const userMsg = messages?.[messages.length - 1]?.content ?? '';

  const dataDir = path.join(process.cwd(), 'data');
  // Ensure data directory exists and load documents safely
  let docs: { id: string; text: string; embedding: number[] }[] = [];
  try {
    await fs.mkdir(dataDir, { recursive: true });
    docs = await loadDocs(dataDir);
  } catch (_) {
    docs = [];
  }

  // Retrieve top-k docs by cosine similarity to user question
  const qEmb = await embedText(userMsg);
  const ranked = docs
    .map((d) => ({ d, score: cosine(qEmb, d.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  const context = ranked.length
    ? ranked.map(({ d, score }) => `# ${d.id} (score ${score.toFixed(3)})\n${d.text}`).join('\n\n---\n\n')
    : 'No documents available.';

  try {
    const result = await streamText({
      model: openai('gpt-4o'),
      system: 'You are a helpful RAG assistant. Use the provided CONTEXT to answer. If unsure, say so.',
      messages: [
        { role: 'system', content: 'Answer with citations like [source:filename]' },
        { role: 'user', content: `CONTEXT:\n${context}\n\nQUESTION: ${userMsg}` },
      ],
    });

    return result.toUIMessageStreamResponse();
  } catch (err: any) {
    const msg = err?.message || 'Unknown error';
    return new Response(JSON.stringify({ error: msg }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}
