"use client";

// Removed framer-motion for minimalistic UI
import { ReactNode } from "react";

interface StepContainerProps {
  title: string;
  icon: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  duration?: number;
  isProcessing?: boolean;
  children: ReactNode;
}

export function StepContainer({ 
  title, 
  icon, 
  status, 
  duration, 
  isProcessing = false, 
  children 
}: StepContainerProps) {
  const getBorderColor = () => {
    switch (status) {
      case 'pending': return 'border-muted-foreground/20';
      case 'processing': return 'border-blue-500/50';
      case 'completed': return 'border-green-500/50';
      case 'error': return 'border-red-500/50';
      default: return 'border-muted-foreground/20';
    }
  };

  const getBackgroundColor = () => {
    switch (status) {
      case 'pending': return 'bg-muted/10';
      case 'processing': return 'bg-blue-50/50 dark:bg-blue-950/20';
      case 'completed': return 'bg-green-50/50 dark:bg-green-950/20';
      case 'error': return 'bg-red-50/50 dark:bg-red-950/20';
      default: return 'bg-muted/10';
    }
  };

  return (
    <div
      className={`border rounded-lg p-4 ${getBorderColor()} ${getBackgroundColor()}`}
    >
      {/* Step Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-xl">{icon}</span>
          <h3 className="font-semibold">{title}</h3>
          
          {/* Processing Indicator */}
          {isProcessing && (
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          )}
        </div>
        
        {/* Duration */}
        {duration && (
          <div className="text-sm text-muted-foreground">
{duration}ms
          </div>
        )}
      </div>

      {/* Step Content */}
      <div className="pl-8">
        {children}
      </div>
    </div>
  );
}
