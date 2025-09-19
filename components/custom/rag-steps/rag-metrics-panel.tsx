"use client";

import { RAGDemonstrationSession } from "@/lib/rag-demonstration-types";

interface RAGMetricsPanelProps {
  session: RAGDemonstrationSession;
}

export function RAGMetricsPanel({ session }: RAGMetricsPanelProps) {
  const totalTime = session.totalDuration || (Date.now() - session.timestamp);
  
  const getStepTime = (stepName: keyof typeof session.steps) => {
    const step = session.steps[stepName];
    if (!step || !step.endTime || !step.startTime) return null;
    return step.endTime - step.startTime;
  };

  return (
    <div className="bg-card border rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3">RAG Metrics</h3>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-muted-foreground">Total Time:</p>
          <p className="font-mono">{totalTime}ms</p>
        </div>
        
        <div>
          <p className="text-muted-foreground">Status:</p>
          <p className="capitalize">{session.status}</p>
        </div>
        
        {getStepTime('queryEmbedding') && (
          <div>
            <p className="text-muted-foreground">Embedding Time:</p>
            <p className="font-mono">{getStepTime('queryEmbedding')}ms</p>
          </div>
        )}
        
        {getStepTime('documentRetrieval') && (
          <div>
            <p className="text-muted-foreground">Retrieval Time:</p>
            <p className="font-mono">{getStepTime('documentRetrieval')}ms</p>
          </div>
        )}
        
        {getStepTime('contextAssembly') && (
          <div>
            <p className="text-muted-foreground">Assembly Time:</p>
            <p className="font-mono">{getStepTime('contextAssembly')}ms</p>
          </div>
        )}
        
        {getStepTime('responseGeneration') && (
          <div>
            <p className="text-muted-foreground">Generation Time:</p>
            <p className="font-mono">{getStepTime('responseGeneration')}ms</p>
          </div>
        )}
      </div>
      
      {session.steps.documentRetrieval?.data?.documents && (
        <div className="mt-3 pt-3 border-t">
          <p className="text-sm text-muted-foreground">Documents Retrieved:</p>
          <p className="text-sm">{session.steps.documentRetrieval.data.documents.length}</p>
        </div>
      )}
    </div>
  );
}
