"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LoaderIcon, FileIcon, CheckCircle } from "./icons";
import { Button } from "@/components/ui/button";
import { truncateFilename } from "@/lib/string-utils";

export interface IndexingStatus {
  filename: string;
  status: 'parsing' | 'chunking' | 'embedding' | 'indexing' | 'completed';
  chunks?: number;
  pages?: number;
  processingTime?: number;
}

interface IndexingPillProps {
  status: IndexingStatus | null;
  onDismiss?: () => void;
}

const statusMessages = {
  parsing: 'Parsing document...',
  chunking: 'Creating chunks...',
  embedding: 'Generating embeddings...',
  indexing: 'Indexing to database...',
  completed: 'Indexing complete'
};

const statusIcons = {
  parsing: LoaderIcon,
  chunking: LoaderIcon,
  embedding: LoaderIcon,
  indexing: LoaderIcon,
  completed: CheckCircle
};

export function IndexingPill({ status, onDismiss }: IndexingPillProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [shouldDismiss, setShouldDismiss] = useState(false);

  useEffect(() => {
    if (status) {
      // Only show animation during active indexing states
      const activeStates = ['parsing', 'chunking', 'embedding', 'indexing'];
      const shouldShow = activeStates.includes(status.status) || status.status === 'completed';
      
      setIsVisible(shouldShow);
      setShouldDismiss(false);
      
      // Auto-dismiss after 5 seconds when completed
      if (status.status === 'completed') {
        const timer = setTimeout(() => {
          setShouldDismiss(true);
          setTimeout(() => {
            setIsVisible(false);
            onDismiss?.();
          }, 300);
        }, 5000);
        
        return () => clearTimeout(timer);
      }
    } else {
      setIsVisible(false);
    }
  }, [status, onDismiss]);

  if (!status || !isVisible) return null;

  const Icon = statusIcons[status.status];
  const isLoading = status.status !== 'completed';

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10, scale: 0.95 }}
        animate={{ 
          opacity: shouldDismiss ? 0 : 1, 
          y: shouldDismiss ? -10 : 0,
          scale: shouldDismiss ? 0.95 : 1
        }}
        exit={{ opacity: 0, y: -10, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        className="flex items-center gap-2 px-3 py-2 bg-muted/50 border rounded-lg text-sm mb-2"
      >
        <div className="flex items-center gap-2 flex-1">
          {isLoading ? (
            <div className="animate-spin text-blue-600">
              <Icon size={14} />
            </div>
          ) : (
            <div className="text-green-600">
              <Icon size={14} />
            </div>
          )}
          
          <div className="text-muted-foreground">
            <FileIcon size={14} />
          </div>
          
          <span className="font-medium">
            {truncateFilename(status.filename)}
          </span>
          
          <span className="text-muted-foreground">
            {statusMessages[status.status]}
          </span>
          
          {status.status === 'completed' && status.chunks && (
            <span className="text-xs text-muted-foreground">
              ({status.chunks} chunks)
            </span>
          )}
        </div>
        
        {status.status === 'completed' && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
            onClick={() => {
              setShouldDismiss(true);
              setTimeout(() => {
                setIsVisible(false);
                onDismiss?.();
              }, 300);
            }}
          >
            ×
          </Button>
        )}
      </motion.div>
    </AnimatePresence>
  );
}

// Hook to manage indexing status
export function useIndexingStatus() {
  const [status, setStatus] = useState<IndexingStatus | null>(null);

  const updateStatus = (newStatus: IndexingStatus) => {
    setStatus(newStatus);
  };

  const clearStatus = () => {
    setStatus(null);
  };

  return {
    status,
    updateStatus,
    clearStatus
  };
}
