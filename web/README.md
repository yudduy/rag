# RAG Chat (Vercel AI SDK)

A minimal Next.js app using Vercel AI SDK with a simple file-based RAG API.

## Setup

1. Node 18+
2. Install deps:

```bash
cd web
npm install
```

3. Add data files:

Place `.txt` or `.md` files into `web/data`. For example, copy from repo root:

```bash
cp ../ui/data/*.txt ../ui/data/*.md ./data/ 2>/dev/null || true
```

4. Env var:

Set `OPENAI_API_KEY` in your environment (or Vercel project settings).

## Run locally

```bash
npm run dev
```

Open `http://localhost:3000`. Ask questions about your uploaded docs.

## Deploy to Vercel

- Push this folder and use Vercel dashboard to import `web/` as a project
- Set `OPENAI_API_KEY` in Project Settings → Environment Variables
- Deploy

## Notes

- The demo uses a naive embedding stub for simplicity. Swap `embedText` with a real embedding call to OpenAI’s `text-embedding-3-small` or a vector DB (Pinecone, pgvector).
- For large doc sets, precompute embeddings and store in a DB.
