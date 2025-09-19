"use client";

import { StepContainer } from "./step-container";
import { DocumentRetrievalStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";

interface DocumentRetrievalStepProps {
  step: DocumentRetrievalStep;
  isActive: boolean;
  config: RAGDemonstrationConfig;
}

export function DocumentRetrievalStepComponent({ step, isActive, config }: DocumentRetrievalStepProps) {
  return (
    <StepContainer
      title="Document Retrieval"
      status={step.status}
      isActive={isActive}
      startTime={step.startTime}
      endTime={step.endTime}
    >
      <div className="space-y-3">
        {step.data?.documents && (
          <div>
            <p className="text-sm text-muted-foreground">
              Retrieved {step.data.documents.length} documents:
            </p>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {step.data.documents.slice(0, config.maxDocumentsDisplay || 5).map((doc, idx) => (
                <div key={idx} className="text-xs bg-muted p-2 rounded">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium">Score: {doc.relevanceScore.toFixed(4)}</span>
                    <span className="text-muted-foreground">ID: {doc.id}</span>
                  </div>
                  {config.showDocumentContent && (
                    <p className="text-muted-foreground line-clamp-3">
                      {doc.content.substring(0, 150)}...
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {step.data?.totalResults !== undefined && (
          <div>
            <p className="text-sm text-muted-foreground">Total Results:</p>
            <p className="text-sm">{step.data.totalResults}</p>
          </div>
        )}
        
        {step.data?.filteredResults !== undefined && (
          <div>
            <p className="text-sm text-muted-foreground">Filtered Results:</p>
            <p className="text-sm">{step.data.filteredResults}</p>
          </div>
        )}
        
        {step.data?.searchParams?.threshold && (
          <div>
            <p className="text-sm text-muted-foreground">Similarity Threshold:</p>
            <p className="text-sm">{step.data.searchParams.threshold}</p>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
