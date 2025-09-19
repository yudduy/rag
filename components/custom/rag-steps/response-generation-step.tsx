"use client";

import { StepContainer } from "./step-container";
import { ResponseGenerationStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";

interface ResponseGenerationStepProps {
  step: ResponseGenerationStep;
  isActive: boolean;
  config: RAGDemonstrationConfig;
}

export function ResponseGenerationStepComponent({ step, isActive, config }: ResponseGenerationStepProps) {
  return (
    <StepContainer
      title="Response Generation"
      status={step.status}
      isActive={isActive}
      startTime={step.startTime}
      endTime={step.endTime}
    >
      <div className="space-y-3">
        {step.data?.response && (
          <div>
            <p className="text-sm text-muted-foreground">Generated Response:</p>
            <div className="text-sm bg-muted p-2 rounded max-h-32 overflow-y-auto">
              {step.data.response}
            </div>
          </div>
        )}
        
        {step.data?.model && (
          <div>
            <p className="text-sm text-muted-foreground">Model:</p>
            <p className="text-sm">{step.data.model}</p>
          </div>
        )}
        
        {step.data?.tokenUsage && config.showDetailedMetrics && (
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Prompt Tokens:</p>
              <p>{step.data.tokenUsage.prompt}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Completion Tokens:</p>
              <p>{step.data.tokenUsage.completion}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Total Tokens:</p>
              <p>{step.data.tokenUsage.total}</p>
            </div>
          </div>
        )}
        
        {step.data?.finishReason && config.showDetailedMetrics && (
          <div>
            <p className="text-sm text-muted-foreground">Finish Reason:</p>
            <p className="text-sm">{step.data.finishReason}</p>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
