"use client";

import { ReactNode } from "react";

interface StepContainerProps {
  title: string;
  status: "pending" | "processing" | "completed" | "error";
  isActive: boolean;
  startTime?: number;
  endTime?: number;
  children: ReactNode;
}

export function StepContainer({ 
  title, 
  status, 
  isActive, 
  startTime, 
  endTime, 
  children 
}: StepContainerProps) {
  const getStatusColor = () => {
    switch (status) {
      case "pending": return "text-muted-foreground";
      case "processing": return "text-blue-600";
      case "completed": return "text-green-600";
      case "error": return "text-red-600";
      default: return "text-muted-foreground";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "pending": return "⏳";
      case "processing": return "🔄";
      case "completed": return "✅";
      case "error": return "❌";
      default: return "⏳";
    }
  };

  const duration = startTime && endTime ? endTime - startTime : null;

  return (
    <div className={`border rounded-lg p-4 ${isActive ? "ring-2 ring-blue-500" : ""}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span>{getStatusIcon()}</span>
          <h3 className={`font-semibold ${getStatusColor()}`}>{title}</h3>
        </div>
        {duration && (
          <span className="text-xs text-muted-foreground font-mono">
            {duration}ms
          </span>
        )}
      </div>
      
      {status !== "pending" && children}
    </div>
  );
}
