"use client";

import { motion } from "framer-motion";
import { DocumentRetrievalStep, RAGDemonstrationConfig } from "@/lib/rag-demonstration-types";
import { StepContainer } from "./step-container";

interface DocumentRetrievalStepProps {
  step: DocumentRetrievalStep;
  config: RAGDemonstrationConfig;
}

export function DocumentRetrievalStepComponent({ step, config }: DocumentRetrievalStepProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'pending': return '⏳';
      case 'processing': return '🔍';
      case 'completed': return '📚';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  return (
    <StepContainer
      title="Document Retrieval"
      icon={getStatusIcon()}
      status={step.status}
      duration={step.duration}
      isProcessing={step.status === 'processing'}
    >
      <div className="space-y-3">
        {/* Search Parameters */}
        {step.data?.searchParams && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Top K:</span>
              <span className="font-medium">{step.data.searchParams.topK}</span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Threshold:</span>
              <span className="font-medium">{step.data.searchParams.threshold}</span>
            </div>
            
            {step.data.namespace && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Namespace:</span>
                <span className="font-medium font-mono text-xs">{step.data.namespace}</span>
              </div>
            )}
            
            {step.data.totalResults !== undefined && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Found:</span>
                <span className="font-medium">{step.data.totalResults} docs</span>
              </div>
            )}
          </div>
        )}

        {/* Results Summary */}
        {step.data?.documents && step.data.documents.length > 0 && (
          <div className="bg-muted/30 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium">Retrieved Documents</h4>
              <span className="text-xs text-muted-foreground">
                {step.data.filteredResults || step.data.documents.length} relevant documents
              </span>
            </div>
            
            <div className="space-y-2">
              {step.data.documents.slice(0, config.maxDocumentsDisplay).map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="bg-background rounded p-2 border"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium truncate flex-1">
                      [{idx + 1}] {doc.source}
                    </span>
                    <div className="flex items-center gap-2 ml-2">
                      <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                        {(doc.relevanceScore * 100).toFixed(1)}%
                      </span>
                      {doc.metadata.page && (
                        <span className="text-xs text-muted-foreground">
                          p.{doc.metadata.page}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  {config.showDocumentContent && (
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {doc.snippet}
                    </p>
                  )}
                </motion.div>
              ))}
              
              {step.data.documents.length > config.maxDocumentsDisplay && (
                <div className="text-xs text-muted-foreground text-center py-1">
                  ... and {step.data.documents.length - config.maxDocumentsDisplay} more documents
                </div>
              )}
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
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-4 h-4 bg-current rounded-full opacity-60"
            />
            <span>Searching vector database...</span>
          </motion.div>
        )}

        {/* No Results */}
        {step.status === 'completed' && (!step.data?.documents || step.data.documents.length === 0) && (
          <div className="bg-yellow-50 dark:bg-yellow-950/50 border border-yellow-200 dark:border-yellow-800 rounded p-3">
            <div className="text-sm text-yellow-700 dark:text-yellow-300">
              <strong>No documents found</strong> - No relevant documents matched the similarity threshold
            </div>
          </div>
        )}

        {/* Error State */}
        {step.status === 'error' && step.error && (
          <div className="bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded p-3">
            <div className="text-sm text-red-600 dark:text-red-400">
              <strong>Retrieval Error:</strong> {step.error}
            </div>
          </div>
        )}
      </div>
    </StepContainer>
  );
}
