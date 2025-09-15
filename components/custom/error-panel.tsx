"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";

export interface ErrorDetails {
  title: string;
  message: string;
  type: 'auth' | 'database' | 'upload' | 'network' | 'validation' | 'system';
  details?: {
    code?: string;
    timestamp?: string;
    userId?: string;
    sessionId?: string;
    endpoint?: string;
    stack?: string;
  };
  suggestions?: string[];
  technicalInfo?: {
    label: string;
    value: string;
  }[];
  canRetry?: boolean;
  canReport?: boolean;
}

interface ErrorPanelProps {
  error: ErrorDetails;
  onRetry?: () => void;
  onReport?: () => void;
  onDismiss?: () => void;
  isVisible: boolean;
}

const errorTypeConfig = {
  auth: {
    icon: "🔐",
    color: "border-red-500 bg-red-50",
    titleColor: "text-red-800",
    textColor: "text-red-700"
  },
  database: {
    icon: "🗄️",
    color: "border-orange-500 bg-orange-50", 
    titleColor: "text-orange-800",
    textColor: "text-orange-700"
  },
  upload: {
    icon: "📁",
    color: "border-blue-500 bg-blue-50",
    titleColor: "text-blue-800", 
    textColor: "text-blue-700"
  },
  network: {
    icon: "🌐",
    color: "border-purple-500 bg-purple-50",
    titleColor: "text-purple-800",
    textColor: "text-purple-700"
  },
  validation: {
    icon: "⚠️",
    color: "border-yellow-500 bg-yellow-50",
    titleColor: "text-yellow-800",
    textColor: "text-yellow-700"
  },
  system: {
    icon: "⚙️",
    color: "border-gray-500 bg-gray-50",
    titleColor: "text-gray-800", 
    textColor: "text-gray-700"
  }
};

export function ErrorPanel({ 
  error, 
  onRetry, 
  onReport, 
  onDismiss, 
  isVisible 
}: ErrorPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  
  const config = errorTypeConfig[error.type];

  const copyErrorDetails = async () => {
    const errorReport = {
      title: error.title,
      message: error.message,
      type: error.type,
      details: error.details,
      technicalInfo: error.technicalInfo,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    try {
      await navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2));
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy error details:', err);
    }
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className={`border-2 rounded-lg p-4 mb-4 ${config.color}`}
        >
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <span className="text-2xl">{config.icon}</span>
              <div className="flex-1">
                <h3 className={`font-semibold text-lg ${config.titleColor}`}>
                  {error.title}
                </h3>
                <p className={`mt-1 ${config.textColor}`}>
                  {error.message}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {error.details && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-xs"
                >
                  {isExpanded ? "Hide Details" : "Show Details"}
                </Button>
              )}
              {onDismiss && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onDismiss}
                  className="text-xs"
                >
                  ✕
                </Button>
              )}
            </div>
          </div>

          {/* Suggestions */}
          {error.suggestions && error.suggestions.length > 0 && (
            <div className="mt-4">
              <h4 className={`font-medium text-sm ${config.titleColor} mb-2`}>
                💡 What you can do:
              </h4>
              <ul className={`text-sm ${config.textColor} space-y-1`}>
                {error.suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-xs mt-0.5">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Expanded Details */}
          <AnimatePresence>
            {isExpanded && error.details && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="mt-4 border-t pt-4"
              >
                <h4 className={`font-medium text-sm ${config.titleColor} mb-3`}>
                  🔍 Technical Details:
                </h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  {error.details.code && (
                    <div>
                      <span className="font-medium">Error Code:</span>
                      <div className="font-mono bg-white/50 px-2 py-1 rounded mt-1">
                        {error.details.code}
                      </div>
                    </div>
                  )}
                  
                  {error.details.timestamp && (
                    <div>
                      <span className="font-medium">Timestamp:</span>
                      <div className="font-mono bg-white/50 px-2 py-1 rounded mt-1">
                        {error.details.timestamp}
                      </div>
                    </div>
                  )}
                  
                  {error.details.userId && (
                    <div>
                      <span className="font-medium">User ID:</span>
                      <div className="font-mono bg-white/50 px-2 py-1 rounded mt-1">
                        {error.details.userId}
                      </div>
                    </div>
                  )}
                  
                  {error.details.endpoint && (
                    <div>
                      <span className="font-medium">Endpoint:</span>
                      <div className="font-mono bg-white/50 px-2 py-1 rounded mt-1">
                        {error.details.endpoint}
                      </div>
                    </div>
                  )}
                </div>

                {/* Technical Info */}
                {error.technicalInfo && error.technicalInfo.length > 0 && (
                  <div className="mt-4">
                    <h5 className={`font-medium text-xs ${config.titleColor} mb-2`}>
                      System Information:
                    </h5>
                    <div className="space-y-2">
                      {error.technicalInfo.map((info, index) => (
                        <div key={index} className="flex justify-between text-xs">
                          <span className="font-medium">{info.label}:</span>
                          <span className="font-mono text-right">{info.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Stack Trace */}
                {error.details.stack && (
                  <div className="mt-4">
                    <h5 className={`font-medium text-xs ${config.titleColor} mb-2`}>
                      Stack Trace:
                    </h5>
                    <pre className="text-xs bg-white/50 p-2 rounded overflow-x-auto max-h-32">
                      {error.details.stack}
                    </pre>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 mt-4">
            {error.canRetry && onRetry && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRetry}
                className="text-xs"
              >
                🔄 Retry
              </Button>
            )}
            
            <Button
              variant="outline"
              size="sm"
              onClick={copyErrorDetails}
              className="text-xs"
            >
              {isCopied ? "✅ Copied" : "📋 Copy Details"}
            </Button>
            
            {error.canReport && onReport && (
              <Button
                variant="outline"
                size="sm"
                onClick={onReport}
                className="text-xs"
              >
                📤 Report Issue
              </Button>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Helper function to create error objects
export function createError(
  type: ErrorDetails['type'],
  title: string,
  message: string,
  options: Partial<ErrorDetails> = {}
): ErrorDetails {
  return {
    type,
    title,
    message,
    canRetry: true,
    canReport: true,
    ...options,
    details: {
      timestamp: new Date().toISOString(),
      ...options.details
    }
  };
}
