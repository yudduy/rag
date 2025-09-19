"use client";

import { StepContainer } from "./step-container";
import { ContextAssemblyStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";

interface ContextAssemblyStepProps {
  step: ContextAssemblyStep;
  isActive: boolean;
  config: RAGDemonstrationConfig;
}

export function ContextAssemblyStepComponent({ step, isActive, config }: ContextAssemblyStepProps) {
  return (
    <StepContainer
      title="Context Assembly"
      status={step.status}
      isActive={isActive}
      startTime={step.startTime}
      endTime={step.endTime}
    >
      <div className="space-y-3">
        {step.data?.contextPreview && (
          <div>
            <p className="text-sm text-muted-foreground">Context Preview:</p>
            <div className="text-xs font-mono bg-muted p-2 rounded max-h-32 overflow-y-auto">
              {step.data.contextPreview.substring(0, 500)}
              {step.data.contextPreview.length > 500 && "..."}
            </div>
          </div>
        )}
        
        {step.data?.contextLength && (
          <div>
            <p className="text-sm text-muted-foreground">Context Length:</p>
            <p className="text-sm">{step.data.contextLength}</p>
          </div>
        )}
        
        {step.data?.selectedDocuments !== undefined && (
          <div>
            <p className="text-sm text-muted-foreground">Documents Used:</p>
            <p className="text-sm">{step.data.selectedDocuments.length}</p>
          </div>
        )}
        
        {step.data?.assemblyStrategy && (
          <div>
            <p className="text-sm text-muted-foreground">Assembly Strategy:</p>
            <p className="text-sm">{step.data.assemblyStrategy}</p>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
