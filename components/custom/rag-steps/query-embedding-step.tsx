"use client";

import { StepContainer } from "./step-container";
import { QueryEmbeddingStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";

interface QueryEmbeddingStepProps {
  step: QueryEmbeddingStep;
  isActive: boolean;
  config: RAGDemonstrationConfig;
}

export function QueryEmbeddingStepComponent({ step, isActive, config }: QueryEmbeddingStepProps) {
  return (
    <StepContainer
      title="Query Embedding"
      status={step.status}
      isActive={isActive}
      startTime={step.startTime}
      endTime={step.endTime}
    >
      <div className="space-y-3">
        <div>
          <p className="text-sm text-muted-foreground">Original Query:</p>
          <p className="text-sm font-mono bg-muted p-2 rounded">
            {step.data?.originalQuery || "Processing..."}
          </p>
        </div>
        
        {step.data?.processedQuery && (
          <div>
            <p className="text-sm text-muted-foreground">Processed Query:</p>
            <p className="text-sm font-mono bg-muted p-2 rounded">
              {step.data.processedQuery}
            </p>
          </div>
        )}
        
        {step.data?.embeddingVector && config.showEmbeddingVectors && (
          <div>
            <p className="text-sm text-muted-foreground">Embedding Vector (first 10 dimensions):</p>
            <p className="text-xs font-mono bg-muted p-2 rounded">
              [{step.data.embeddingVector.slice(0, 10).map(v => v.toFixed(4)).join(", ")}...]
            </p>
          </div>
        )}
        
        {step.data?.embeddingPreview && config.showEmbeddingVectors && (
          <div>
            <p className="text-sm text-muted-foreground">Embedding Preview:</p>
            <p className="text-xs font-mono bg-muted p-2 rounded">
              {step.data.embeddingPreview}
            </p>
          </div>
        )}
        
        {step.data?.embeddingModel && (
          <div>
            <p className="text-sm text-muted-foreground">Model:</p>
            <p className="text-sm">{step.data.embeddingModel}</p>
          </div>
        )}
        
        {step.data?.embeddingDimensions && (
          <div>
            <p className="text-sm text-muted-foreground">Dimensions:</p>
            <p className="text-sm">{step.data.embeddingDimensions}</p>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
