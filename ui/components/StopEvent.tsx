import React from 'react';

export default function StopEvent({ event }: { event: any }) {
  return (
    <div className="p-4 bg-green-50 rounded-lg border border-green-200">
      <h3 className="font-semibold text-green-800 mb-2">Workflow Complete</h3>
      <div className="text-sm text-gray-700">
        {event.result && (
          <div className="whitespace-pre-wrap">
            {event.result}
          </div>
        )}
      </div>
    </div>
  );
}