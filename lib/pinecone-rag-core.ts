/**
 * Pinecone-based RAG Core implementation
 * Cost-optimized for serverless free tier with namespace-based user isolation
 */

import { Pinecone } from '@pinecone-database/pinecone';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { z } from "zod";
import { HuggingFaceEmbeddings } from './huggingface-embeddings';

// Types
export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    source: string;
    chunk_index: number;
    total_chunks: number;
    file_type: string;
    upload_date: string;
    user_id: string;
    deleted?: boolean;
  };
}

export interface RAGQueryResult {
  answer: string;
  sources: Array<{
    content: string;
    source: string;
    relevance_score: number;
  }>;
  query_expansion?: string[];
}

// Configuration
const RAG_CONFIG = {
  chunk_size: parseInt(process.env.RAG_CHUNK_SIZE || "1000"),
  chunk_overlap: parseInt(process.env.RAG_CHUNK_OVERLAP || "200"),
  max_retrieval_docs: parseInt(process.env.RAG_MAX_DOCS || "5"),
  similarity_threshold: parseFloat(process.env.RAG_SIMILARITY_THRESHOLD || "0.7"),
  query_expansion_enabled: process.env.RAG_QUERY_EXPANSION === "true",
  reranking_enabled: process.env.RAG_RERANKING === "true",
};

// Schema for document validation
export const DocumentUploadSchema = z.object({
  filename: z.string().min(1),
  content: z.string().min(1),
  file_type: z.enum(["txt", "md", "pdf", "docx"]),
  user_id: z.string().uuid(),
});

export class PineconeRAGCore {
  private pinecone: Pinecone;
  private embeddings: HuggingFaceEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private indexName: string;
  private index: any = null;

  constructor() {
    if (!process.env.PINECONE_API_KEY) {
      throw new Error("PINECONE_API_KEY is required for RAG functionality");
    }

    this.pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    this.embeddings = new HuggingFaceEmbeddings();
    this.indexName = process.env.PINECONE_INDEX_NAME || 'rag-documents';

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: RAG_CONFIG.chunk_size,
      chunkOverlap: RAG_CONFIG.chunk_overlap,
      separators: ["\n\n", "\n", ". ", " ", ""],
    });

    this.initializeIndex();
  }

  /**
   * Initialize Pinecone index
   */
  private async initializeIndex(): Promise<void> {
    try {
      // Check if index exists
      const indexList = await this.pinecone.listIndexes();
      const indexExists = indexList.indexes?.some(index => index.name === this.indexName);

      if (!indexExists) {
        console.log(`Creating Pinecone index: ${this.indexName}`);
        await this.pinecone.createIndex({
          name: this.indexName,
          dimension: this.embeddings.getDimensions(), // 384 for all-MiniLM-L6-v2
          metric: 'cosine',
          spec: {
            serverless: {
              cloud: 'aws',
              region: 'us-east-1', // Free tier region
            },
          },
        });

        // Wait for index to be ready
        console.log('Waiting for index to be ready...');
        await this.waitForIndexReady();
      }

      this.index = this.pinecone.index(this.indexName);
      console.log(`Connected to Pinecone index: ${this.indexName}`);
    } catch (error) {
      console.error('Error initializing Pinecone index:', error);
      throw error;
    }
  }

  /**
   * Wait for index to be ready
   */
  private async waitForIndexReady(maxWaitTime = 60000): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      try {
        const indexDescription = await this.pinecone.describeIndex(this.indexName);
        if (indexDescription.status?.ready) {
          return;
        }
      } catch (error) {
        // Index might not be fully created yet
      }
      
      await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    }
    
    throw new Error(`Index ${this.indexName} did not become ready within ${maxWaitTime}ms`);
  }

  /**
   * Get user namespace
   */
  private getUserNamespace(userId: string): string {
    return `user_${userId}`;
  }

  /**
   * Process and index a document
   */
  async indexDocument(
    filename: string,
    content: string,
    fileType: string,
    userId: string
  ): Promise<{ success: boolean; chunks: number; error?: string }> {
    try {
      // Validate input
      const validation = DocumentUploadSchema.safeParse({
        filename,
        content,
        file_type: fileType,
        user_id: userId,
      });

      if (!validation.success) {
        return {
          success: false,
          chunks: 0,
          error: `Validation failed: ${validation.error.message}`,
        };
      }

      // Ensure index is ready
      if (!this.index) {
        await this.initializeIndex();
      }

      // Split document into chunks
      const docs = await this.textSplitter.createDocuments([content], [
        {
          source: filename,
          file_type: fileType,
          user_id: userId,
          upload_date: new Date().toISOString(),
        },
      ]);

      // Generate embeddings for all chunks
      const texts = docs.map(doc => doc.pageContent);
      const embeddings = await this.embeddings.embedTexts(texts);

      // Prepare vectors for Pinecone
      const vectors = docs.map((doc, index) => {
        // Estimate page number for PDFs (rough approximation)
        const estimatedPage = fileType === 'pdf' 
          ? Math.floor((index / docs.length) * 10) + 1 // Simple estimation
          : null;

        return {
          id: `${userId}_${filename}_${index}`,
          values: embeddings[index],
          metadata: {
            content: doc.pageContent,
            source: filename,
            chunk_index: index,
            total_chunks: docs.length,
            file_type: fileType,
            user_id: userId,
            upload_date: new Date().toISOString(),
            page: estimatedPage,
          },
        };
      });

      // Upsert to Pinecone with user namespace
      const namespace = this.getUserNamespace(userId);
      await this.index.namespace(namespace).upsert(vectors);

      return {
        success: true,
        chunks: vectors.length,
      };
    } catch (error) {
      console.error("Error indexing document:", error);
      return {
        success: false,
        chunks: 0,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Query expansion using simple keyword extraction and synonyms
   */
  private expandQuery(query: string): string[] {
    if (!RAG_CONFIG.query_expansion_enabled) {
      return [query];
    }

    // Simple query expansion - can be enhanced with more sophisticated methods
    const expansions = [query];
    
    // Add variations based on common patterns
    const words = query.toLowerCase().split(/\s+/);
    
    // Add question variations
    if (!query.includes("what") && !query.includes("how") && !query.includes("why")) {
      if (words.some(w => ["define", "definition", "meaning"].includes(w))) {
        expansions.push(`What is ${query}?`);
      }
      if (words.some(w => ["process", "steps", "procedure"].includes(w))) {
        expansions.push(`How to ${query}?`);
      }
    }

    return expansions.slice(0, 3); // Limit to 3 variations
  }

  /**
   * Retrieve relevant documents for a query
   */
  async retrieveDocuments(
    query: string,
    userId: string,
    options: {
      maxDocs?: number;
      threshold?: number;
    } = {}
  ): Promise<Array<{
    content: string;
    source: string;
    relevance_score: number;
    metadata: any;
    snippet: string;
    page: number | null;
    chunkId: string;
  }>> {
    if (!this.index) {
      await this.initializeIndex();
    }

    try {
      const maxDocs = options.maxDocs || RAG_CONFIG.max_retrieval_docs;
      const threshold = options.threshold || RAG_CONFIG.similarity_threshold;

      // Generate query embedding
      const queryEmbedding = await this.embeddings.embedText(query);

      // Query Pinecone with user namespace
      const namespace = this.getUserNamespace(userId);
      const queryResponse = await this.index.namespace(namespace).query({
        vector: queryEmbedding,
        topK: maxDocs,
        includeMetadata: true,
        includeValues: false,
      });

      // Process and filter results
      const results = queryResponse.matches
        ?.filter((match: any) => {
          const score = match.score || 0;
          const metadata = match.metadata || {};
          return score >= threshold && !metadata.deleted;
        })
        .map((match: any) => ({
          content: match.metadata?.content || '',
          source: match.metadata?.source || '',
          relevance_score: match.score || 0,
          metadata: match.metadata || {},
          // Add citation-specific fields
          snippet: this.createSnippet(match.metadata?.content || ''),
          page: match.metadata?.page || null,
          chunkId: match.id || '',
        })) || [];

      return results.sort((a: any, b: any) => b.relevance_score - a.relevance_score);

    } catch (error) {
      console.error("Error retrieving documents:", error);
      return [];
    }
  }

  /**
   * Create a snippet from content for citation display
   */
  private createSnippet(content: string, maxLength: number = 120): string {
    if (!content) return '';
    
    // Remove excessive whitespace and newlines
    const cleaned = content.replace(/\s+/g, ' ').trim();
    
    // Truncate to maxLength and add ellipsis if needed
    if (cleaned.length <= maxLength) return cleaned;
    
    // Find last complete word within limit
    const truncated = cleaned.substring(0, maxLength);
    const lastSpace = truncated.lastIndexOf(' ');
    
    if (lastSpace > maxLength * 0.8) {
      return truncated.substring(0, lastSpace) + '...';
    }
    
    return truncated + '...';
  }

  /**
   * Get user's indexed documents
   */
  async getUserDocuments(userId: string): Promise<Array<{
    filename: string;
    chunks: number;
    file_type: string;
    upload_date: string;
  }>> {
    if (!this.index) {
      await this.initializeIndex();
    }

    try {
      const namespace = this.getUserNamespace(userId);
      
      // Query all vectors in user's namespace
      const queryResponse = await this.index.namespace(namespace).query({
        vector: new Array(this.embeddings.getDimensions()).fill(0), // Dummy vector
        topK: 10000, // Large number to get all documents
        includeMetadata: true,
        includeValues: false,
      });

      // Group by filename and aggregate
      const documentMap = new Map<string, {
        filename: string;
        chunks: number;
        file_type: string;
        upload_date: string;
      }>();

      queryResponse.matches?.forEach((match: any) => {
        const metadata = match.metadata;
        if (metadata && !metadata.deleted) {
          const filename = metadata.source;
          
          if (!documentMap.has(filename)) {
            documentMap.set(filename, {
              filename,
              chunks: 0,
              file_type: metadata.file_type,
              upload_date: metadata.upload_date,
            });
          }
          
          const doc = documentMap.get(filename)!;
          doc.chunks++;
        }
      });

      return Array.from(documentMap.values())
        .sort((a, b) => new Date(b.upload_date).getTime() - new Date(a.upload_date).getTime());

    } catch (error) {
      console.error("Error getting user documents:", error);
      return [];
    }
  }

  /**
   * Delete user's document
   */
  async deleteDocument(filename: string, userId: string): Promise<boolean> {
    if (!this.index) {
      await this.initializeIndex();
    }

    try {
      const namespace = this.getUserNamespace(userId);
      
      // Find all vectors for this document
      const queryResponse = await this.index.namespace(namespace).query({
        vector: new Array(this.embeddings.getDimensions()).fill(0), // Dummy vector
        topK: 10000, // Large number to get all chunks
        includeMetadata: true,
        includeValues: false,
        filter: {
          source: filename,
          user_id: userId,
        },
      });

      // Extract vector IDs to delete
      const vectorIds = queryResponse.matches?.map((match: any) => match.id) || [];
      
      if (vectorIds.length === 0) {
        return false; // Document not found
      }

      // Delete vectors from Pinecone
      await this.index.namespace(namespace).deleteMany(vectorIds);
      
      return true;
    } catch (error) {
      console.error("Error deleting document:", error);
      return false;
    }
  }

  /**
   * Get system status
   */
  async getStatus() {
    try {
      const indexStats = this.index ? await this.index.describeIndexStats() : null;
      
      return {
        pinecone_connected: !!this.index,
        index_name: this.indexName,
        embedding_model: this.embeddings.getModelName(),
        embedding_dimensions: this.embeddings.getDimensions(),
        total_vectors: indexStats?.totalVectorCount || 0,
        namespaces: indexStats?.namespaces || {},
        config: RAG_CONFIG,
      };
    } catch (error) {
      return {
        pinecone_connected: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        config: RAG_CONFIG,
      };
    }
  }

  /**
   * Test the complete pipeline
   */
  async testPipeline(): Promise<{
    success: boolean;
    embeddings_test?: any;
    pinecone_test?: any;
    error?: string;
  }> {
    try {
      // Test embeddings
      const embeddingsTest = await this.embeddings.test();
      if (!embeddingsTest.success) {
        return {
          success: false,
          embeddings_test: embeddingsTest,
          error: 'Embeddings test failed',
        };
      }

      // Test Pinecone connection
      const pineconeTest = await this.getStatus();
      
      return {
        success: pineconeTest.pinecone_connected,
        embeddings_test: embeddingsTest,
        pinecone_test: pineconeTest,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}

// Singleton instance
let pineconeRagCore: PineconeRAGCore | null = null;

export function getPineconeRAGCore(): PineconeRAGCore {
  if (!pineconeRagCore) {
    pineconeRagCore = new PineconeRAGCore();
  }
  return pineconeRagCore;
}

export default getPineconeRAGCore;
