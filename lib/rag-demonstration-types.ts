/**
 * TypeScript interfaces for RAG Demonstration System
 * Defines the data structures for real-time RAG pipeline visualization
 */

export interface RAGDemonstrationStep<T = any> {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  startTime?: number;
  endTime?: number;
  duration?: number;
  data?: T;
  error?: string;
}

export interface QueryEmbeddingData {
  originalQuery: string;
  processedQuery: string;
  embeddingModel: string;
  embeddingDimensions: number;
  embeddingVector?: number[]; // Truncated for display
  embeddingPreview: string; // First few dimensions as string
}

export interface QueryEmbeddingStep extends RAGDemonstrationStep<QueryEmbeddingData> {}

export interface DocumentRetrievalData {
  searchQuery: string;
  namespace: string;
  searchParams: {
    topK: number;
    threshold: number;
  };
  totalResults: number;
  filteredResults: number;
  documents: RetrievedDocument[];
}

export interface DocumentRetrievalStep extends RAGDemonstrationStep<DocumentRetrievalData> {}

export interface RetrievedDocument {
  id: string;
  source: string;
  content: string;
  snippet: string;
  relevanceScore: number;
  metadata: {
    page?: number;
    chunkId: string;
    fileType: string;
    [key: string]: any;
  };
}

export interface ContextAssemblyData {
  selectedDocuments: RetrievedDocument[];
  contextLength: number;
  contextPreview: string;
  assemblyStrategy: string;
}

export interface ContextAssemblyStep extends RAGDemonstrationStep<ContextAssemblyData> {}

export interface ResponseGeneration {
  model: string;
  prompt: string;
  promptLength: number;
  contextLength: number;
  response?: string;
  responseLength: number;
  finishReason?: string; // e.g., "stop" | "length" | ...
  tokenUsage?: {
    prompt: number;
    completion: number;
    total: number;
  };
}

export interface ResponseGenerationData {
  model?: string;
  prompt?: string;
  promptLength?: number;
  contextLength?: number;
  response?: string;
  responseLength?: number;
  finishReason?: string;
  tokenUsage?: {
    prompt: number;
    completion: number;
    total: number;
  };
}

export interface ResponseGenerationStep extends RAGDemonstrationStep<ResponseGenerationData> {}

export interface RAGDemonstrationSession {
  sessionId: string;
  userId: string;
  query: string;
  timestamp: number;
  status: 'active' | 'completed' | 'error';
  steps: {
    queryEmbedding: QueryEmbeddingStep;
    documentRetrieval: DocumentRetrievalStep;
    contextAssembly: ContextAssemblyStep;
    responseGeneration: ResponseGenerationStep;
  };
  totalDuration?: number;
  citations: RetrievedDocument[];
}

export interface RAGDemonstrationEvent {
  type: 'session_start' | 'step_start' | 'step_update' | 'step_progress' | 'step_complete' | 'step_error' | 'session_complete' | 'session_error';
  sessionId: string;
  stepId?: string;
  data: any;
  timestamp: number;
}

export interface RAGMetrics {
  averageEmbeddingTime: number;
  averageRetrievalTime: number;
  averageContextTime: number;
  averageResponseTime: number;
  totalQueries: number;
  successRate: number;
  averageRelevanceScore: number;
}

export interface RAGDemonstrationConfig {
  showEmbeddingVectors: boolean;
  showDetailedMetrics: boolean;
  showDocumentContent: boolean;
  maxDocumentsDisplay: number;
  updateInterval: number;
  enableRealTimeUpdates: boolean;
}
