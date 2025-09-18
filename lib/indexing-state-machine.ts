/**
 * Indexing State Machine
 * Provides clear state transitions and validation boundaries for document processing
 */

export type IndexingState = 
  | 'idle'
  | 'uploading'
  | 'parsing' 
  | 'chunking'
  | 'embedding'
  | 'indexing'
  | 'ready'
  | 'error';

export type IndexingEvent =
  | 'START_UPLOAD'
  | 'UPLOAD_COMPLETE'
  | 'PARSING_COMPLETE'
  | 'CHUNKING_COMPLETE'
  | 'EMBEDDING_COMPLETE'
  | 'INDEXING_COMPLETE'
  | 'ERROR'
  | 'RETRY'
  | 'RESET';

export interface IndexingContext {
  filename: string;
  fileSize: number;
  fileType: string;
  progress: {
    chunks?: number;
    pages?: number;
    processingTime?: number;
  };
  error?: {
    stage: IndexingState;
    message: string;
    code: string;
    timestamp: string;
    retryable: boolean;
  };
}

export interface StateTransition {
  from: IndexingState;
  event: IndexingEvent;
  to: IndexingState;
  condition?: (context: IndexingContext) => boolean;
  action?: (context: IndexingContext) => IndexingContext;
}

// Define valid state transitions
export const INDEXING_TRANSITIONS: StateTransition[] = [
  // Happy path
  { from: 'idle', event: 'START_UPLOAD', to: 'uploading' },
  { from: 'uploading', event: 'UPLOAD_COMPLETE', to: 'parsing' },
  { from: 'parsing', event: 'PARSING_COMPLETE', to: 'chunking' },
  { from: 'chunking', event: 'CHUNKING_COMPLETE', to: 'embedding' },
  { from: 'embedding', event: 'EMBEDDING_COMPLETE', to: 'indexing' },
  { from: 'indexing', event: 'INDEXING_COMPLETE', to: 'ready' },
  
  // Error transitions from any state
  { from: 'uploading', event: 'ERROR', to: 'error' },
  { from: 'parsing', event: 'ERROR', to: 'error' },
  { from: 'chunking', event: 'ERROR', to: 'error' },
  { from: 'embedding', event: 'ERROR', to: 'error' },
  { from: 'indexing', event: 'ERROR', to: 'error' },
  
  // Recovery transitions
  { from: 'error', event: 'RETRY', to: 'uploading', condition: (ctx) => ctx.error?.retryable === true },
  { from: 'error', event: 'RESET', to: 'idle' },
  { from: 'ready', event: 'RESET', to: 'idle' },
];

export class IndexingStateMachine {
  private state: IndexingState = 'idle';
  private context: IndexingContext;

  constructor(filename: string, fileSize: number, fileType: string) {
    this.context = {
      filename,
      fileSize,
      fileType,
      progress: {}
    };
  }

  getCurrentState(): IndexingState {
    return this.state;
  }

  getContext(): IndexingContext {
    return { ...this.context };
  }

  canTransition(event: IndexingEvent): boolean {
    return INDEXING_TRANSITIONS.some(
      t => t.from === this.state && 
           t.event === event && 
           (!t.condition || t.condition(this.context))
    );
  }

  transition(event: IndexingEvent, payload?: Partial<IndexingContext>): boolean {
    const transition = INDEXING_TRANSITIONS.find(
      t => t.from === this.state && 
           t.event === event && 
           (!t.condition || t.condition(this.context))
    );

    if (!transition) {
      console.warn(`Invalid transition: ${this.state} -> ${event}`);
      return false;
    }

    // Update context with payload
    if (payload) {
      this.context = { ...this.context, ...payload };
    }

    // Execute action if defined
    if (transition.action) {
      this.context = transition.action(this.context);
    }

    // Transition to new state
    this.state = transition.to;
    
    return true;
  }

  // Convenience methods for common transitions
  startUpload(): boolean {
    return this.transition('START_UPLOAD');
  }

  completeUpload(): boolean {
    return this.transition('UPLOAD_COMPLETE');
  }

  completeParsing(chunks?: number, pages?: number): boolean {
    return this.transition('PARSING_COMPLETE', {
      progress: { ...this.context.progress, chunks, pages }
    });
  }

  completeChunking(chunks: number): boolean {
    return this.transition('CHUNKING_COMPLETE', {
      progress: { ...this.context.progress, chunks }
    });
  }

  completeEmbedding(): boolean {
    return this.transition('EMBEDDING_COMPLETE');
  }

  completeIndexing(processingTime: number): boolean {
    return this.transition('INDEXING_COMPLETE', {
      progress: { ...this.context.progress, processingTime }
    });
  }

  error(stage: IndexingState, message: string, code: string, retryable: boolean = true): boolean {
    return this.transition('ERROR', {
      error: {
        stage,
        message,
        code,
        timestamp: new Date().toISOString(),
        retryable
      }
    });
  }

  retry(): boolean {
    return this.transition('RETRY');
  }

  reset(): boolean {
    this.context.progress = {};
    this.context.error = undefined;
    return this.transition('RESET');
  }

  // UI state helpers
  isActive(): boolean {
    return ['uploading', 'parsing', 'chunking', 'embedding', 'indexing'].includes(this.state);
  }

  isComplete(): boolean {
    return this.state === 'ready';
  }

  hasError(): boolean {
    return this.state === 'error';
  }

  canRetry(): boolean {
    return this.state === 'error' && this.context.error?.retryable === true;
  }

  getProgressMessage(): string {
    const messages = {
      idle: 'Ready to upload',
      uploading: 'Uploading file...',
      parsing: 'Parsing document...',
      chunking: 'Creating chunks...',
      embedding: 'Generating embeddings...',
      indexing: 'Indexing to database...',
      ready: 'Indexing complete',
      error: this.context.error?.message || 'An error occurred'
    };
    
    return messages[this.state];
  }
}

// Validation boundaries for each stage
export const VALIDATION_SCHEMAS = {
  upload: {
    maxSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: ['txt', 'md', 'pdf', 'docx']
  },
  parsing: {
    minContentLength: 10,
    maxContentLength: 1000000 // 1MB of text
  },
  chunking: {
    minChunks: 1,
    maxChunks: 1000
  },
  embedding: {
    maxEmbeddingDimensions: 1536
  }
};

export function validateStage(stage: IndexingState, data: any): { valid: boolean; error?: string } {
  switch (stage) {
    case 'uploading':
      if (!data.fileSize || data.fileSize > VALIDATION_SCHEMAS.upload.maxSize) {
        return { valid: false, error: 'File size exceeds limit' };
      }
      if (!data.fileType || !VALIDATION_SCHEMAS.upload.allowedTypes.includes(data.fileType)) {
        return { valid: false, error: 'Unsupported file type' };
      }
      break;
      
    case 'parsing':
      if (!data.content || data.content.length < VALIDATION_SCHEMAS.parsing.minContentLength) {
        return { valid: false, error: 'No text content extracted' };
      }
      if (data.content.length > VALIDATION_SCHEMAS.parsing.maxContentLength) {
        return { valid: false, error: 'Content too large' };
      }
      break;
      
    case 'chunking':
      if (typeof data.chunks !== 'number' || data.chunks < VALIDATION_SCHEMAS.chunking.minChunks) {
        return { valid: false, error: 'No chunks created' };
      }
      if (data.chunks > VALIDATION_SCHEMAS.chunking.maxChunks) {
        return { valid: false, error: 'Too many chunks created' };
      }
      break;
  }
  
  return { valid: true };
}
