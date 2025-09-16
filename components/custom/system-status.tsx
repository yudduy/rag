"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  message: string;
  responseTime?: number;
  lastChecked?: string;
  details?: string;
}

interface SystemStatusProps {
  isVisible: boolean;
  onClose: () => void;
}

const statusConfig = {
  healthy: {
    icon: "✅",
    color: "text-green-600 dark:text-green-400",
    bgColor: "bg-green-50 dark:bg-green-950/50 border-green-200 dark:border-green-800"
  },
  warning: {
    icon: "⚠️",
    color: "text-yellow-600 dark:text-yellow-400", 
    bgColor: "bg-yellow-50 dark:bg-yellow-950/50 border-yellow-200 dark:border-yellow-800"
  },
  error: {
    icon: "❌",
    color: "text-red-600 dark:text-red-400",
    bgColor: "bg-red-50 dark:bg-red-950/50 border-red-200 dark:border-red-800"
  },
  unknown: {
    icon: "❓",
    color: "text-muted-foreground",
    bgColor: "bg-muted/50 border-border"
  }
};

export function SystemStatus({ isVisible, onClose }: SystemStatusProps) {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const checkSystemHealth = async () => {
    setIsLoading(true);
    const startTime = Date.now();

    try {
      // Check Authentication
      const authCheck = await fetch('/api/auth/session').catch(() => null);
      const authStatus: ServiceStatus = {
        name: "Authentication",
        status: authCheck?.ok ? 'healthy' : 'error',
        message: authCheck?.ok ? 'Session management working' : 'Authentication service unavailable',
        responseTime: authCheck ? Date.now() - startTime : undefined,
        lastChecked: new Date().toISOString()
      };

      // Check Database (via documents endpoint)
      const dbStart = Date.now();
      const dbCheck = await fetch('/api/documents/upload').catch(() => null);
      const dbStatus: ServiceStatus = {
        name: "Database",
        status: dbCheck?.ok ? 'healthy' : 'error',
        message: dbCheck?.ok ? 'Database connection active' : 'Database connection failed',
        responseTime: dbCheck ? Date.now() - dbStart : undefined,
        lastChecked: new Date().toISOString(),
        details: dbCheck ? `HTTP ${dbCheck.status}` : 'Connection timeout'
      };

      // Check RAG System (this would need a dedicated health endpoint)
      const ragStatus: ServiceStatus = {
        name: "RAG System",
        status: 'unknown',
        message: 'RAG system status not directly checkable',
        lastChecked: new Date().toISOString(),
        details: 'Pinecone and HuggingFace embeddings'
      };

      // Check Chat API
      const chatStart = Date.now();
      const chatCheck = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: 'health-check', messages: [] })
      }).catch(() => null);
      
      const chatStatus: ServiceStatus = {
        name: "Chat API",
        status: chatCheck?.status === 401 ? 'healthy' : (chatCheck?.ok ? 'healthy' : 'error'),
        message: chatCheck?.status === 401 ? 'Chat API responding (auth required)' : (chatCheck?.ok ? 'Chat API healthy' : 'Chat API error'),
        responseTime: chatCheck ? Date.now() - chatStart : undefined,
        lastChecked: new Date().toISOString(),
        details: chatCheck ? `HTTP ${chatCheck.status}` : 'No response'
      };

      setServices([authStatus, dbStatus, ragStatus, chatStatus]);

    } catch (error) {
      console.error('Health check error:', error);
      setServices([{
        name: "System Check",
        status: 'error',
        message: 'Failed to perform health check',
        lastChecked: new Date().toISOString(),
        details: error instanceof Error ? error.message : 'Unknown error'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isVisible) {
      checkSystemHealth();
    }
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-background rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] overflow-hidden"
      >
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-xl font-semibold">System Status</h2>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={checkSystemHealth}
              disabled={isLoading}
            >
              {isLoading ? "🔄 Checking..." : "🔄 Refresh"}
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          </div>
        </div>

        <div className="p-6">
          <div className="space-y-4">
            {services.length === 0 && !isLoading && (
              <div className="text-center py-8 text-muted-foreground">
                Click "Refresh" to check system status
              </div>
            )}

            {isLoading && (
              <div className="text-center py-8">
                <div className="animate-spin text-2xl mb-2">🔄</div>
                <p className="text-muted-foreground">Checking system health...</p>
              </div>
            )}

            {services.map((service, index) => {
              const config = statusConfig[service.status];
              return (
                <div
                  key={index}
                  className={`border rounded-lg p-4 ${config.bgColor}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <span className="text-xl">{config.icon}</span>
                      <div className="flex-1">
                        <h3 className={`font-semibold ${config.color}`}>
                          {service.name}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {service.message}
                        </p>
                        
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          {service.responseTime && (
                            <span>⚡ {service.responseTime}ms</span>
                          )}
                          {service.lastChecked && (
                            <span>🕒 {new Date(service.lastChecked).toLocaleTimeString()}</span>
                          )}
                        </div>
                        
                        {service.details && (
                          <p className="text-xs text-gray-500 mt-1 font-mono">
                            {service.details}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {services.length > 0 && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-sm mb-2">💡 Troubleshooting Tips:</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• If Authentication fails: Clear cookies and log in again</li>
                <li>• If Database errors: Check network connection and try again</li>
                <li>• If Chat API fails: Refresh the page and retry</li>
                <li>• For persistent issues: Contact support with these status details</li>
              </ul>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
