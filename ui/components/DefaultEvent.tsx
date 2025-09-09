import React from 'react';

interface DefaultEventProps {
  event: any;
  eventType?: string;
}

export default function DefaultEvent({ event, eventType }: DefaultEventProps) {
  return (
    <div className="event-container default-event">
      <div className="event-header">
        <h4>📋 {eventType || 'Unknown Event'}</h4>
        <span className="timestamp">{event.timestamp}</span>
      </div>
      <div className="event-content">
        <details>
          <summary>Event Details</summary>
          <pre>{JSON.stringify(event, null, 2)}</pre>
        </details>
      </div>
      <style jsx>{`
        .event-container {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        .default-event {
          border-left: 4px solid #6c757d;
        }
        .event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .event-header h4 {
          margin: 0;
          color: #6c757d;
          font-size: 16px;
        }
        .timestamp {
          font-size: 12px;
          color: #666;
        }
        details {
          margin: 8px 0;
        }
        summary {
          cursor: pointer;
          font-weight: 500;
          color: #495057;
        }
        pre {
          background: white;
          padding: 12px;
          border-radius: 4px;
          margin-top: 8px;
          border: 1px solid #e1e5e9;
          font-size: 12px;
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
}