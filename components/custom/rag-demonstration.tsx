"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { 
  RAGDemonstrationSession, 
  RAGDemonstrationEvent,
  RAGDemonstrationConfig 
} from "@/lib/rag-demonstration-types";
import { QueryEmbeddingStepComponent } from "./rag-steps/query-embedding-step";
import { DocumentRetrievalStepComponent } from "./rag-steps/document-retrieval-step";
import { ContextAssemblyStepComponent } from "./rag-steps/context-assembly-step";
import { ResponseGenerationStepComponent } from "./rag-steps/response-generation-step";
import { RAGMetricsPanel } from "./rag-steps/rag-metrics-panel";

interface RAGDemonstrationProps {
  isVisible: boolean;
  onClose: () => void;
}

export function RAGDemonstration({ isVisible, onClose }: RAGDemonstrationProps) {
  const [currentSession, setCurrentSession] = useState<RAGDemonstrationSession | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [config, setConfig] = useState<RAGDemonstrationConfig>({
    showEmbeddingVectors: true,
    showDetailedMetrics: true,
    showDocumentContent: true,
    maxDocumentsDisplay: 5,
    updateInterval: 100,
    enableRealTimeUpdates: true,
  });

  // Server-Sent Events connection for real-time updates
  useEffect(() => {
    if (!isVisible || !config.enableRealTimeUpdates) return;

    let eventSource: EventSource | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connectSSE = () => {
      try {
        eventSource = new EventSource('/api/rag-demonstration/events');

        eventSource.onopen = () => {
          console.log('🔗 RAG Demonstration SSE connected');
          setIsConnected(true);
        };

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Handle different message types
            if (data.type === 'connection' || data.type === 'heartbeat') {
              // Connection/heartbeat messages - just log
              return;
            }
            
            // Handle RAG demonstration events
            const ragEvent: RAGDemonstrationEvent = data;
            handleRAGEvent(ragEvent);
          } catch (error) {
            console.error('Error parsing SSE event:', error);
          }
        };

        eventSource.onerror = (error) => {
          console.log('🔌 RAG Demonstration SSE disconnected');
          setIsConnected(false);
          eventSource?.close();
          
          // Attempt to reconnect after 3 seconds
          reconnectTimeout = setTimeout(connectSSE, 3000);
        };

      } catch (error) {
        console.error('Failed to create SSE connection:', error);
        setIsConnected(false);
        
        // Retry connection after 5 seconds
        reconnectTimeout = setTimeout(connectSSE, 5000);
      }
    };

    connectSSE();

    return () => {
      if (eventSource) {
        eventSource.close();
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, [isVisible, config.enableRealTimeUpdates]);

  const handleRAGEvent = useCallback((event: RAGDemonstrationEvent) => {
    switch (event.type) {
      case 'session_start':
        setCurrentSession(event.data);
        break;
      
      case 'step_start':
      case 'step_progress':
      case 'step_complete':
        setCurrentSession(prev => {
          if (!prev || prev.sessionId !== event.sessionId) return prev;
          
          const updatedSession = { ...prev };
          const stepKey = event.stepId as keyof typeof updatedSession.steps;
          
          if (updatedSession.steps[stepKey]) {
            updatedSession.steps[stepKey] = {
              ...updatedSession.steps[stepKey],
              ...event.data,
            };
          }
          
          return updatedSession;
        });
        break;
      
      case 'session_complete':
        setCurrentSession(prev => {
          if (!prev || prev.sessionId !== event.sessionId) return prev;
          return {
            ...prev,
            status: 'completed',
            totalDuration: event.data.totalDuration,
            citations: event.data.citations,
          };
        });
        break;
      
      case 'error':
        setCurrentSession(prev => {
          if (!prev || prev.sessionId !== event.sessionId) return prev;
          return {
            ...prev,
            status: 'error',
          };
        });
        break;
    }
  }, []);

  const clearSession = () => {
    setCurrentSession(null);
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-background rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold">RAG Pipeline Demonstration</h2>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {currentSession && (
              <Button
                variant="outline"
                size="sm"
                onClick={clearSession}
              >
                🗑️ Clear
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {!currentSession ? (
            <div className="flex flex-col items-center justify-center h-full p-8 text-center">
              <div className="text-6xl mb-4">🔍</div>
              <h3 className="text-lg font-medium mb-2">RAG Pipeline Ready</h3>
              <p className="text-muted-foreground mb-4 max-w-md">
                Start a conversation in the chat to see the real-time RAG pipeline in action. 
                Watch how your queries are embedded, documents are retrieved, and responses are generated.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="p-3 border rounded-lg">
                  <div className="font-medium">🧠 Query Embedding</div>
                  <div className="text-muted-foreground">Vector generation</div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="font-medium">📚 Document Retrieval</div>
                  <div className="text-muted-foreground">Similarity search</div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="font-medium">🔗 Context Assembly</div>
                  <div className="text-muted-foreground">Information fusion</div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="font-medium">✨ Response Generation</div>
                  <div className="text-muted-foreground">AI completion</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6 space-y-6">
              {/* Session Header */}
              <div className="bg-muted/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">Current Query</h3>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <span>Session: {currentSession.sessionId.slice(-8)}</span>
                    {currentSession.totalDuration && (
                      <span>• Total: {currentSession.totalDuration}ms</span>
                    )}
                  </div>
                </div>
                <p className="text-sm bg-background rounded p-2 font-mono">
                  "{currentSession.query}"
                </p>
              </div>

              {/* Pipeline Steps */}
              <div className="space-y-4">
                <QueryEmbeddingStepComponent 
                  step={currentSession.steps.queryEmbedding}
                  config={config}
                />
                
                <DocumentRetrievalStepComponent 
                  step={currentSession.steps.documentRetrieval}
                  config={config}
                />
                
                <ContextAssemblyStepComponent 
                  step={currentSession.steps.contextAssembly}
                  config={config}
                />
                
                <ResponseGenerationStepComponent 
                  step={currentSession.steps.responseGeneration}
                  config={config}
                />
              </div>

              {/* Citations */}
              {currentSession.citations && currentSession.citations.length > 0 && (
                <div className="border rounded-lg p-4">
                  <h3 className="font-medium mb-3">📎 Citations & Sources</h3>
                  <div className="grid gap-3">
                    {currentSession.citations.slice(0, config.maxDocumentsDisplay).map((doc, idx) => (
                      <div key={doc.id} className="bg-muted/30 rounded p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">[{idx + 1}] {doc.source}</span>
                          <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                            {(doc.relevanceScore * 100).toFixed(1)}% match
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {doc.snippet}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer with Metrics */}
        {currentSession && (
          <div className="border-t p-4">
            <RAGMetricsPanel session={currentSession} />
          </div>
        )}
      </motion.div>
    </div>
  );
}
