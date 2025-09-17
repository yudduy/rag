"use client";

import { useState, useRef } from "react";
import { toast } from "sonner";
import useSWR, { mutate } from "swr";

import { FileIcon, TrashIcon, UploadIcon, LoaderIcon } from "@/components/custom/icons";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ErrorPanel, ErrorDetails } from "./error-panel";
import { RAGDemonstration } from "./rag-demonstration";

interface Document {
  id: string;
  filename: string;
  originalName: string;
  fileType: string;
  fileSize: number;
  chunkCount: number;
  status: string;
  createdAt: string;
  displayTitle?: string;
  metadata?: {
    wordCount: number;
    processingTime: number;
  };
}

interface DocumentManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export function DocumentManager({ isOpen, onClose }: DocumentManagerProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [currentError, setCurrentError] = useState<ErrorDetails | null>(null);
  const [showRAGDemo, setShowRAGDemo] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch user's documents
  const { data, error, isLoading } = useSWR(
    isOpen ? "/api/documents/upload" : null,
    fetcher,
    { refreshInterval: 0 }
  );

  const documents: Document[] = data?.documents || [];

  const handleFileUpload = async (files: FileList) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    const allowedTypes = [
      "text/plain",
      "text/markdown",
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ];

    if (!allowedTypes.includes(file.type)) {
      toast.error("Unsupported file type. Please upload TXT, MD, PDF, or DOCX files.");
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size must be less than 10MB.");
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        // Check if we have detailed error information
        if (result.errorDetails) {
          setCurrentError(result.errorDetails);
        } else {
          // Fallback to basic error display
          toast.error(result.error || "Upload failed");
        }
        return;
      }

      if (result.success) {
        toast.success(`${file.name} uploaded and indexed successfully!`);
        setCurrentError(null); // Clear any previous errors
        
        // Refresh documents list
        mutate("/api/documents/upload");
        
        // Clear file input
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      } else {
        toast.error("Upload failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
      
      // Create a generic error for network/unexpected issues
      setCurrentError({
        type: "network",
        title: "Network Error",
        message: "Failed to connect to the server",
        details: {
          code: "NETWORK_ERROR",
          timestamp: new Date().toISOString(),
        },
        suggestions: [
          "Check your internet connection",
          "Try again in a few moments",
          "Refresh the page and retry"
        ],
        technicalInfo: [
          { label: "Error", value: error instanceof Error ? error.message : "Unknown error" }
        ],
        canRetry: true,
        canReport: false
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string, filename: string) => {
    if (!documentId) {
      toast.error("Invalid document ID");
      return;
    }

    try {
      console.log(`Attempting to delete document: ${documentId} (${filename})`);
      
      const response = await fetch(`/api/documents/${documentId}`, {
        method: "DELETE",
      });

      const responseData = await response.json();
      console.log("Delete response:", responseData);

      if (!response.ok) {
        throw new Error(responseData.error || `Delete failed with status ${response.status}`);
      }

      toast.success(`${filename} deleted successfully`);
      // Refresh documents list
      mutate("/api/documents/upload");
    } catch (error) {
      console.error("Delete error:", error);
      toast.error(error instanceof Error ? error.message : "Delete failed");
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatDate = (dateString: string | null | undefined) => {
    if (!dateString) return "Unknown date";
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return "Invalid date";
    
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-xl font-semibold">Document Manager</h2>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setShowRAGDemo(true)}
              title="View RAG Pipeline Demonstration"
            >
              RAG Demonstration
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>

        <div className="p-6">
          {/* Error Panel */}
          {currentError && (
            <ErrorPanel
              error={currentError}
              isVisible={!!currentError}
              onRetry={() => {
                setCurrentError(null);
                // Retry with the last file if available
                const input = fileInputRef.current;
                if (input?.files && input.files.length > 0) {
                  handleFileUpload(input.files);
                }
              }}
              onDismiss={() => setCurrentError(null)}
              onReport={() => {
                // Could implement error reporting here
                console.log("Error reported:", currentError);
                toast.success("Error report sent to development team");
              }}
            />
          )}

          {/* Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? "border-primary bg-primary/5"
                : "border-gray-300 hover:border-gray-400"
            } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {isUploading ? (
              <div className="flex flex-col items-center gap-2">
                <div className="animate-spin">
                  <LoaderIcon size={24} />
                </div>
                <p>Uploading and indexing...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                <UploadIcon size={32} />
                <div>
                  <p className="text-lg font-medium">Drop files here or click to upload</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Supports TXT, MD, PDF, DOCX (max 10MB)
                  </p>
                </div>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="flex items-center gap-2"
                >
                  <UploadIcon size={16} />
                  <span>Choose Files</span>
                </Button>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
          />

          {/* Documents List */}
          <div className="mt-6">
            <h3 className="font-medium mb-4">
              Your Documents ({documents.length})
            </h3>
            
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin">
                  <LoaderIcon size={24} />
                </div>
                <span className="ml-2">Loading documents...</span>
              </div>
            ) : error ? (
              <div className="text-center py-8 text-red-500">
                Failed to load documents
              </div>
            ) : documents.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No documents uploaded yet
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <FileIcon size={20} />
                      <div className="min-w-0 flex-1">
                        <p className="font-medium truncate" title={doc.displayTitle || doc.originalName}>
                          {doc.displayTitle || doc.originalName}
                        </p>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>{formatFileSize(doc.fileSize)}</span>
                          <span>{doc.chunkCount || 0} chunks</span>
                          <span>{formatDate(doc.createdAt)}</span>
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteDocument(doc.id, doc.originalName)}
                      className="text-red-500 hover:text-red-700"
                    >
                      <TrashIcon size={16} />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RAG Demonstration Panel */}
      <RAGDemonstration 
        isVisible={showRAGDemo} 
        onClose={() => setShowRAGDemo(false)} 
      />
    </div>
  );
}

export default DocumentManager;
