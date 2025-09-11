'use client';

import { useChat } from '@ai-sdk/react';
import { useState } from 'react';

export default function Page() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({ api: '/api/chat' });
  const [question, setQuestion] = useState('');

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
    </main>
  );
}
