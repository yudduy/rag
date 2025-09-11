'use client';

import { useChat } from '@ai-sdk/react';
import { useState, useRef } from 'react';

export default function Page() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({ api: '/api/chat' });
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  return (
    <main>
      <h1>RAG Chat</h1>
      <p>Ask a question about the uploaded documents.</p>

      <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, minHeight: 240 }}>
        {messages.map((m) => (
          <div key={m.id} style={{ marginBottom: 12 }}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
        {isLoading && <div>Thinking…</div>}
      </div>

      <form onSubmit={handleSubmit} style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Ask about your docs…"
          aria-label="message"
          style={{ flex: 1, padding: 10, border: '1px solid #ddd', borderRadius: 6 }}
        />
        <button type="submit" disabled={isLoading} style={{ padding: '10px 14px' }}>
          Send
        </button>
      </form>

      <section style={{ marginTop: 24 }}>
        <h3>Ingest documents</h3>
        <input ref={fileRef} type="file" multiple />
        <button
          disabled={uploading}
          style={{ marginLeft: 8, padding: '6px 10px' }}
          onClick={async () => {
            if (!fileRef.current?.files?.length) return;
            setUploading(true);
            try {
              const form = new FormData();
              for (const f of Array.from(fileRef.current.files)) form.append('files', f);
              const resp = await fetch(process.env.NEXT_PUBLIC_RAG_API_INGEST || 'http://localhost:4501/ingest', {
                method: 'POST',
                body: form,
              });
              if (!resp.ok) alert('Ingest failed');
            } finally {
              setUploading(false);
            }
          }}
        >
          {uploading ? 'Uploading…' : 'Upload & Index'}
        </button>
      </section>
    </main>
  );
}
