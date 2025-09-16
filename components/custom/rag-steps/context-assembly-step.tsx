"use client";

import { motion } from "framer-motion";
import { ContextAssemblyStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";
import { StepContainer } from "./step-container";

interface ContextAssemblyStepProps {
  step: ContextAssemblyStep;
  config: RAGDemonstrationConfig;
}

export function ContextAssemblyStepComponent({ step, config }: ContextAssemblyStepProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'pending': return '⏳';
      case 'processing': return '🔗';
      case 'completed': return '📝';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  return (
    <StepContainer
      title="Context Assembly"
      icon={getStatusIcon()}
      status={step.status}
      duration={step.duration}
      isProcessing={step.status === 'processing'}
    >
      <div className="space-y-3">
        {/* Assembly Strategy */}
        {step.data?.assemblyStrategy && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Strategy:</span>
            <span className="font-medium">{step.data.assemblyStrategy}</span>
          </div>
        )}

        {/* Context Statistics */}
        {step.data && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            {step.data.selectedDocuments && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Documents:</span>
                <span className="font-medium">{step.data.selectedDocuments.length}</span>
              </div>
            )}
            
            {step.data.contextLength && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Context Size:</span>
                <span className="font-medium">{step.data.contextLength.toLocaleString()} chars</span>
              </div>
            )}
            
            {step.data.contextLength && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Est. Tokens:</span>
                <span className="font-medium">~{Math.ceil(step.data.contextLength / 4).toLocaleString()}</span>
              </div>
            )}
          </div>
        )}

        {/* Selected Documents */}
        {step.data?.selectedDocuments && step.data.selectedDocuments.length > 0 && (
          <div className="bg-muted/30 rounded p-3">
            <h4 className="text-sm font-medium mb-2">Selected for Context</h4>
            <div className="space-y-2">
              {step.data.selectedDocuments.map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center justify-between bg-background rounded p-2 border"
                >
                  <span className="text-sm truncate flex-1">
                    [{idx + 1}] {doc.source}
                  </span>
                  <div className="flex items-center gap-2 ml-2">
                    <span className="text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-1 rounded">
                      {doc.content.length} chars
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {(doc.relevanceScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Context Preview */}
        {config.showDocumentContent && step.data?.contextPreview && (
          <div>
            <h4 className="text-sm font-medium mb-2">Context Preview</h4>
            <div className="bg-muted/30 rounded p-3 max-h-32 overflow-y-auto">
              <div className="font-mono text-xs text-muted-foreground whitespace-pre-wrap">
                {step.data.contextPreview}
                {step.data.contextPreview.length < (step.data.contextLength || 0) && (
                  <span className="text-primary">
                    \n\n... ({((step.data.contextLength || 0) - step.data.contextPreview.length).toLocaleString()} more characters)
                  </span>
                )}
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
              initial={{ opacity: 0.3 }}
              animate={{ opacity: 1 }}
            >
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-2 h-2 bg-current rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ 
                    duration: 1.2, 
                    repeat: Infinity, 
                    delay: i * 0.2 
                  }}
                />
              ))}
            </motion.div>
            <span>Assembling context from selected documents...</span>
          </motion.div>
        )}

        {/* Error State */}
        {step.status === 'error' && step.error && (
          <div className="bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded p-3">
            <div className="text-sm text-red-600 dark:text-red-400">
              <strong>Assembly Error:</strong> {step.error}
            </div>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
