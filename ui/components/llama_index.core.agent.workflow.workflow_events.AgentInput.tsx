export default function AgentInput({ event }: { event: any }) {
  // Extract input content from ChatMessage objects or fallback to string representation
  const getInputContent = () => {
    if (!event.input) return null;
    
    // If input is an array of ChatMessage objects
    if (Array.isArray(event.input)) {
      return event.input
        .map((msg: any) => {
          // Handle ChatMessage objects with content property
          if (typeof msg === 'object' && msg.content) {
            return msg.content;
          }
          // Handle string messages
          return String(msg);
        })
        .join('\n');
    }
    
    // If input is a simple string
    if (typeof event.input === 'string') {
      return event.input;
    }
    
    // Fallback to JSON representation
    return JSON.stringify(event.input);
  };

  const inputContent = getInputContent();

  return (
    <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
      <h3 className="font-semibold text-blue-800 mb-2">Agent Input</h3>
      <div className="text-sm text-gray-700">
        {inputContent && (
          <div className="mb-2">
            <strong>Input:</strong> 
            <div className="mt-1 whitespace-pre-wrap">{inputContent}</div>
          </div>
        )}
        {event.current_agent_name && (
          <div className="mb-2">
            <strong>Agent:</strong> {event.current_agent_name}
          </div>
        )}
        {event.data && (
          <div className="mb-2">
            <strong>Data:</strong> 
            <pre className="mt-1 text-xs bg-gray-100 p-2 rounded overflow-x-auto">
              {JSON.stringify(event.data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
} 