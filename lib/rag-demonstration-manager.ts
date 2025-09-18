/**
 * RAG Demonstration Manager
 * Manages real-time RAG pipeline events and WebSocket communications
 */

import { 
  RAGDemonstrationSession, 
  RAGDemonstrationEvent,
  QueryEmbeddingStep,
  QueryEmbeddingData,
  DocumentRetrievalStep,
  DocumentRetrievalData,
  ContextAssemblyStep,
  ContextAssemblyData,
  ResponseGenerationStep,
  ResponseGenerationData,
  RetrievedDocument
} from './rag-demonstration-types';

class RAGDemonstrationManager {
  private sessions = new Map<string, RAGDemonstrationSession>();
  private subscribers = new Map<string, Set<(event: RAGDemonstrationEvent) => void>>();

  /**
   * Create a new RAG demonstration session
   */
  createSession(sessionId: string, userId: string, query: string): RAGDemonstrationSession {
    const session: RAGDemonstrationSession = {
      sessionId,
      userId,
      query,
      timestamp: Date.now(),
      status: 'active',
      steps: {
        queryEmbedding: {
          id: 'query-embedding',
          name: 'Query Embedding',
          status: 'pending'
        },
        documentRetrieval: {
          id: 'document-retrieval',
          name: 'Document Retrieval',
          status: 'pending'
        },
        contextAssembly: {
          id: 'context-assembly',
          name: 'Context Assembly',
          status: 'pending'
        },
        responseGeneration: {
          id: 'response-generation',
          name: 'Response Generation',
          status: 'pending'
        }
      },
      citations: []
    };

    this.sessions.set(sessionId, session);
    
    // Emit session start event
    this.emitEvent({
      type: 'session_start',
      sessionId,
      data: session,
      timestamp: Date.now()
    });

    return session;
  }

  /**
   * Update query embedding step
   */
  updateQueryEmbeddingStep(
    sessionId: string, 
    status: 'processing' | 'completed' | 'error',
    data?: Partial<QueryEmbeddingStep['data']>,
    error?: string
  ) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const { startTime, endTime, duration } = this.calculateTiming(session.steps.queryEmbedding, status);

    session.steps.queryEmbedding = {
      ...session.steps.queryEmbedding,
      status,
      startTime: session.steps.queryEmbedding.startTime || startTime,
      endTime,
      duration,
      data: data ? { ...(session.steps.queryEmbedding.data ?? {}), ...data } as QueryEmbeddingData : session.steps.queryEmbedding.data,
      error
    };

    this.emitEvent({
      type: status === 'processing' ? 'step_start' : 'step_complete',
      sessionId,
      stepId: 'queryEmbedding',
      data: session.steps.queryEmbedding,
      timestamp: Date.now()
    });
  }

  /**
   * Update document retrieval step
   */
  updateDocumentRetrievalStep(
    sessionId: string,
    status: 'processing' | 'completed' | 'error',
    data?: Partial<DocumentRetrievalStep['data']>,
    error?: string
  ) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const { startTime, endTime, duration } = this.calculateTiming(session.steps.documentRetrieval, status);

    session.steps.documentRetrieval = {
      ...session.steps.documentRetrieval,
      status,
      startTime: session.steps.documentRetrieval.startTime || startTime,
      endTime,
      duration,
      data: data ? { ...(session.steps.documentRetrieval.data ?? {}), ...data } as DocumentRetrievalData : session.steps.documentRetrieval.data,
      error
    };

    this.emitEvent({
      type: status === 'processing' ? 'step_start' : 'step_complete',
      sessionId,
      stepId: 'documentRetrieval',
      data: session.steps.documentRetrieval,
      timestamp: Date.now()
    });
  }

  /**
   * Update context assembly step
   */
  updateContextAssemblyStep(
    sessionId: string,
    status: 'processing' | 'completed' | 'error',
    data?: Partial<ContextAssemblyStep['data']>,
    error?: string
  ) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const { startTime, endTime, duration } = this.calculateTiming(session.steps.contextAssembly, status);

    session.steps.contextAssembly = {
      ...session.steps.contextAssembly,
      status,
      startTime: session.steps.contextAssembly.startTime || startTime,
      endTime,
      duration,
      data: data ? { ...(session.steps.contextAssembly.data ?? {}), ...data } as ContextAssemblyData : session.steps.contextAssembly.data,
      error
    };

    this.emitEvent({
      type: status === 'processing' ? 'step_start' : 'step_complete',
      sessionId,
      stepId: 'contextAssembly',
      data: session.steps.contextAssembly,
      timestamp: Date.now()
    });
  }

  /**
   * Update response generation step
   */
  updateResponseGenerationStep(
    sessionId: string,
    status: 'processing' | 'completed' | 'error',
    data?: Partial<ResponseGenerationStep['data']>,
    error?: string
  ) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const { startTime, endTime, duration } = this.calculateTiming(session.steps.responseGeneration, status);

    session.steps.responseGeneration = {
      ...session.steps.responseGeneration,
      status,
      startTime: session.steps.responseGeneration.startTime || startTime,
      endTime,
      duration,
      data: data ? { ...(session.steps.responseGeneration.data ?? {}), ...(data ?? {}) } : session.steps.responseGeneration.data,
      error
    };

    this.emitEvent({
      type: status === 'processing' ? 'step_start' : 'step_complete',
      sessionId,
      stepId: 'responseGeneration',
      data: session.steps.responseGeneration,
      timestamp: Date.now()
    });
  }

  /**
   * Complete a session
   */
  completeSession(sessionId: string, citations: RetrievedDocument[] = []) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const sessionStartTime = session.timestamp;
    const totalDuration = Date.now() - sessionStartTime;

    session.status = 'completed';
    session.totalDuration = totalDuration;
    session.citations = citations;

    this.emitEvent({
      type: 'session_complete',
      sessionId,
      data: {
        totalDuration,
        citations
      },
      timestamp: Date.now()
    });
  }

  /**
   * Mark session as error
   */
  errorSession(sessionId: string, error: string) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    session.status = 'error';

    this.emitEvent({
      type: 'session_error',
      sessionId,
      data: { error },
      timestamp: Date.now()
    });
  }

  /**
   * Subscribe to events for a user
   */
  subscribe(userId: string, callback: (event: RAGDemonstrationEvent) => void) {
    if (!this.subscribers.has(userId)) {
      this.subscribers.set(userId, new Set());
    }
    this.subscribers.get(userId)!.add(callback);

    // Return unsubscribe function
    return () => {
      const userSubscribers = this.subscribers.get(userId);
      if (userSubscribers) {
        userSubscribers.delete(callback);
        if (userSubscribers.size === 0) {
          this.subscribers.delete(userId);
        }
      }
    };
  }

  private calculateTiming(currentStep: any, status: string) {
    const startTime = currentStep.startTime || Date.now();
    const endTime = status === 'completed' || status === 'error' ? Date.now() : undefined;
    const duration = endTime ? endTime - startTime : undefined;
    
    return { startTime, endTime, duration };
  }

  /**
   * Emit event to all subscribers of the session's user
   */
  private emitEvent(event: RAGDemonstrationEvent) {
    const session = this.sessions.get(event.sessionId);
    if (!session) return;

    const userSubscribers = this.subscribers.get(session.userId);
    if (userSubscribers) {
      userSubscribers.forEach(callback => {
        try {
          callback(event);
        } catch (error) {
          console.error('Error in RAG demonstration event callback:', error);
        }
      });
    }
  }

  /**
   * Get session by ID
   */
  getSession(sessionId: string): RAGDemonstrationSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Clean up old sessions (older than 1 hour)
   */
  cleanup() {
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    
    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.timestamp < oneHourAgo) {
        this.sessions.delete(sessionId);
      }
    }
  }

  /**
   * Get all active sessions for a user
   */
  getUserSessions(userId: string): RAGDemonstrationSession[] {
    return Array.from(this.sessions.values()).filter(session => session.userId === userId);
  }
}

// Global instance
export const ragDemoManager = new RAGDemonstrationManager();

// Clean up old sessions every 10 minutes
let cleanupInterval: NodeJS.Timeout | undefined;

// Clean up old sessions every 10 minutes
if (typeof setInterval !== 'undefined') {
  cleanupInterval = setInterval(() => {
    ragDemoManager.cleanup();
  }, 10 * 60 * 1000);
}

// Export a function to clear the interval if needed
export function stopCleanup() {
  if (cleanupInterval) {
    clearInterval(cleanupInterval);
    cleanupInterval = undefined;
  }
}
