import React from 'react';

interface AgentInputProps {
  event: {
    input?: string;
    data?: any;
    timestamp?: string;
  };
}

export default function AgentInput({ event }: AgentInputProps) {
  return (
    <div className="event-container agent-input-event">
      <div className="event-header">
        <h4>📝 Processing Input</h4>
        <span className="timestamp">{event.timestamp}</span>
      </div>
      <div className="event-content">
        <p><strong>Input received:</strong> {event.input || event.data}</p>
        <p className="status">Analyzing query and preparing response...</p>
      </div>
      <style jsx>{`
        .event-container {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        .agent-input-event {
          border-left: 4px solid #ffc107;
        }
        .event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .event-header h4 {
          margin: 0;
          color: #ffc107;
          font-size: 16px;
        }
        .timestamp {
          font-size: 12px;
          color: #666;
        }
        .event-content p {
          margin: 4px 0;
          color: #333;
        }
        .status {
          font-style: italic;
          color: #666;
        }
      `}</style>
    </div>
  );
}