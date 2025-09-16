"use client";

import { motion } from "framer-motion";
import { RAGDemonstrationSession } from "@/lib/rag-demonstration-types";

interface RAGMetricsPanelProps {
  session: RAGDemonstrationSession;
}

export function RAGMetricsPanel({ session }: RAGMetricsPanelProps) {
  const getStepDuration = (stepName: keyof typeof session.steps) => {
    const step = session.steps[stepName];
    return step.duration || 0;
  };

  const getTotalStepsCompleted = () => {
    return Object.values(session.steps).filter(step => step.status === 'completed').length;
  };

  const getOverallStatus = () => {
    const hasError = Object.values(session.steps).some(step => step.status === 'error');
    if (hasError) return 'error';
    
    const allCompleted = Object.values(session.steps).every(step => step.status === 'completed');
    if (allCompleted) return 'completed';
    
    const hasProcessing = Object.values(session.steps).some(step => step.status === 'processing');
    if (hasProcessing) return 'processing';
    
    return 'pending';
  };

  const metrics = [
    {
      label: 'Query Embedding',
      value: getStepDuration('queryEmbedding'),
      unit: 'ms',
      color: 'text-blue-600 dark:text-blue-400'
    },
    {
      label: 'Document Retrieval',
      value: getStepDuration('documentRetrieval'),
      unit: 'ms',
      color: 'text-purple-600 dark:text-purple-400'
    },
    {
      label: 'Context Assembly',
      value: getStepDuration('contextAssembly'),
      unit: 'ms',
      color: 'text-orange-600 dark:text-orange-400'
    },
    {
      label: 'Response Generation',
      value: getStepDuration('responseGeneration'),
      unit: 'ms',
      color: 'text-green-600 dark:text-green-400'
    }
  ];

  const overallStatus = getOverallStatus();
  const completedSteps = getTotalStepsCompleted();

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-muted/30 rounded-lg p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-sm">Pipeline Metrics</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {completedSteps}/4 steps completed
          </span>
          {overallStatus === 'completed' && (
            <span className="text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-1 rounded">
              ✅ Complete
            </span>
          )}
          {overallStatus === 'processing' && (
            <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2 py-1 rounded">
              ⏳ Processing
            </span>
          )}
          {overallStatus === 'error' && (
            <span className="text-xs bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 px-2 py-1 rounded">
              ❌ Error
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map((metric, idx) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="text-center"
          >
            <div className="text-xs text-muted-foreground mb-1">
              {metric.label}
            </div>
            <div className={`font-medium ${metric.color}`}>
              {metric.value > 0 ? `${metric.value}${metric.unit}` : '-'}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Total Duration */}
      {session.totalDuration && (
        <div className="mt-3 pt-3 border-t border-muted-foreground/20">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Total Pipeline Duration:</span>
            <span className="font-medium">
              {session.totalDuration}ms
            </span>
          </div>
        </div>
      )}

      {/* Progress Bar */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
          <span>Progress</span>
          <span>{Math.round((completedSteps / 4) * 100)}%</span>
        </div>
        <div className="w-full bg-muted-foreground/20 rounded-full h-2">
          <motion.div
            className="bg-primary rounded-full h-2"
            initial={{ width: 0 }}
            animate={{ width: `${(completedSteps / 4) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
    </motion.div>
  );
}
