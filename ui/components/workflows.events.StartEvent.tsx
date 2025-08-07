import React from 'react';

interface StartEventProps {
  event: {
    input?: string;
    data?: any;
    timestamp?: string;
  };
}

export default function StartEvent({ event }: StartEventProps) {
  return (
    <div className="event-container start-event">
      <div className="event-header">
        <h4>🚀 Query Started</h4>
        <span className="timestamp">{event.timestamp}</span>
      </div>
      <div className="event-content">
        <p><strong>Query:</strong> {event.input || event.data}</p>
      </div>
      <style jsx>{`
        .event-container {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        .start-event {
          border-left: 4px solid #28a745;
        }
        .event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .event-header h4 {
          margin: 0;
          color: #28a745;
          font-size: 16px;
        }
        .timestamp {
          font-size: 12px;
          color: #666;
        }
        .event-content p {
          margin: 0;
          color: #333;
        }
      `}</style>
    </div>
  );
}