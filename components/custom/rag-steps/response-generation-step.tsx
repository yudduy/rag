"use client";

import { motion } from "framer-motion";
import { ResponseGenerationStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";
import { StepContainer } from "./step-container";

interface ResponseGenerationStepProps {
  step: ResponseGenerationStep;
  config: RAGDemonstrationConfig;
}

export function ResponseGenerationStepComponent({ step, config }: ResponseGenerationStepProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'pending': return '⏳';
      case 'processing': return '✨';
      case 'completed': return '🎯';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  return (
    <StepContainer
      title="Response Generation"
      icon={getStatusIcon()}
      status={step.status}
      duration={step.duration}
      isProcessing={step.status === 'processing'}
    >
      <div className="space-y-3">
        {/* Model Information */}
        {step.data?.model && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Model:</span>
            <span className="font-medium">{step.data.model}</span>
          </div>
        )}

        {/* Token Usage Statistics */}
        {config.showDetailedMetrics && step.data && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {step.data.promptLength && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Prompt:</span>
                <span className="font-medium">{step.data.promptLength.toLocaleString()} chars</span>
              </div>
            )}
            
            {step.data.contextLength && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Context:</span>
                <span className="font-medium">{step.data.contextLength.toLocaleString()} chars</span>
              </div>
            )}
            
            {step.data.responseLength && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Response:</span>
                <span className="font-medium">{step.data.responseLength.toLocaleString()} chars</span>
              </div>
            )}
            
            {step.data.tokenUsage?.total && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Tokens:</span>
                <span className="font-medium">{step.data.tokenUsage.total.toLocaleString()}</span>
              </div>
            )}
          </div>
        )}

        {/* Detailed Token Breakdown */}
        {config.showDetailedMetrics && step.data?.tokenUsage && (
          <div className="bg-muted/30 rounded p-3">
            <h4 className="text-sm font-medium mb-2">Token Usage Breakdown</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="text-muted-foreground">Prompt</div>
                <div className="font-medium text-blue-600 dark:text-blue-400">
                  {step.data.tokenUsage.prompt?.toLocaleString() || 'N/A'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-muted-foreground">Completion</div>
                <div className="font-medium text-green-600 dark:text-green-400">
                  {step.data.tokenUsage.completion?.toLocaleString() || 'N/A'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-muted-foreground">Total</div>
                <div className="font-medium">
                  {step.data.tokenUsage.total?.toLocaleString() || 'N/A'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Prompt Preview */}
        {config.showDocumentContent && step.data?.prompt && (
          <div>
            <h4 className="text-sm font-medium mb-2">System Prompt Preview</h4>
            <div className="bg-muted/30 rounded p-3 max-h-32 overflow-y-auto">
              <div className="font-mono text-xs text-muted-foreground whitespace-pre-wrap">
                {step.data.prompt.length > 500 
                  ? `${step.data.prompt.substring(0, 500)}...\n\n(${(step.data.prompt.length - 500).toLocaleString()} more characters)`
                  : step.data.prompt
                }
              </div>
            </div>
          </div>
        )}

        {/* Processing Animation */}
        {step.status === 'processing' && (
          <motion.div
            className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="flex gap-1"
            >
              {['✨', '🤖', '💭'].map((emoji, i) => (
                <motion.span
                  key={i}
                  animate={{ 
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity, 
                    delay: i * 0.3 
                  }}
                >
                  {emoji}
                </motion.span>
              ))}
            </motion.div>
            <span>Generating AI response with context...</span>
          </motion.div>
        )}

        {/* Success State */}
        {step.status === 'completed' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-green-50 dark:bg-green-950/50 border border-green-200 dark:border-green-800 rounded p-3"
          >
            <div className="flex items-center gap-2 text-sm text-green-700 dark:text-green-300">
              <span>🎉</span>
              <strong>Response Generated Successfully!</strong>
            </div>
            <div className="text-xs text-green-600 dark:text-green-400 mt-1">
              The AI has processed your query with the retrieved context and generated a comprehensive response.
            </div>
          </motion.div>
        )}

        {/* Error State */}
        {step.status === 'error' && step.error && (
          <div className="bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded p-3">
            <div className="text-sm text-red-600 dark:text-red-400">
              <strong>Generation Error:</strong> {step.error}
            </div>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
