"use client";

import { motion } from "framer-motion";
import { QueryEmbeddingStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";
import { StepContainer } from "./step-container";

interface QueryEmbeddingStepProps {
  step: QueryEmbeddingStep;
  config: RAGDemonstrationConfig;
}

export function QueryEmbeddingStepComponent({ step, config }: QueryEmbeddingStepProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'pending': return '⏳';
      case 'processing': return '🧠';
      case 'completed': return '✅';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'pending': return 'text-muted-foreground';
      case 'processing': return 'text-blue-600 dark:text-blue-400';
      case 'completed': return 'text-green-600 dark:text-green-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <StepContainer
      title="Query Embedding"
      icon={getStatusIcon()}
      status={step.status}
      duration={step.duration}
      isProcessing={step.status === 'processing'}
    >
      <div className="space-y-3">
        {/* Query Processing */}
        {step.data?.originalQuery && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium mb-2">Original Query</h4>
              <div className="bg-muted/30 rounded p-2 text-sm font-mono">
                &quot;{step.data.originalQuery}&quot;
              </div>
            </div>
            
            {step.data.processedQuery && step.data.processedQuery !== step.data.originalQuery && (
              <div>
                <h4 className="text-sm font-medium mb-2">Processed Query</h4>
                <div className="bg-muted/30 rounded p-2 text-sm font-mono">
                  &quot;{step.data.processedQuery}&quot;
                </div>
              </div>
            )}
          </div>
        )}

        {/* Model Information */}
        {step.data?.embeddingModel && (
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Model:</span>
              <span className="font-medium">{step.data.embeddingModel}</span>
            </div>
            
            {step.data.embeddingDimensions && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Dimensions:</span>
                <span className="font-medium">{step.data.embeddingDimensions}</span>
              </div>
            )}
          </div>
        )}

        {/* Embedding Vector Preview */}
        {config.showEmbeddingVectors && step.data?.embeddingPreview && (
          <div>
            <h4 className="text-sm font-medium mb-2">Embedding Vector (preview)</h4>
            <div className="bg-muted/30 rounded p-3">
              <div className="font-mono text-xs break-all text-muted-foreground">
                [{step.data.embeddingPreview}...]
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Showing first 8 dimensions of {step.data.embeddingDimensions} total
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
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
            />
            <span>Generating embedding vector...</span>
          </motion.div>
        )}

        {/* Error State */}
        {step.status === 'error' && step.error && (
          <div className="bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded p-3">
            <div className="text-sm text-red-600 dark:text-red-400">
              <strong>Error:</strong> {step.error}
            </div>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
