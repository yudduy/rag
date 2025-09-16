/**
 * HuggingFace Embeddings implementation for cost-effective RAG
 * Uses HuggingFace Inference API free tier
 */

import { HfInference } from '@huggingface/inference';

export interface EmbeddingResult {
  embedding: number[];
  text: string;
}

export class HuggingFaceEmbeddings {
  private hf: HfInference;
  private model: string;
  private maxRetries: number;
  private retryDelay: number;

  constructor(options: {
    apiKey?: string;
    model?: string;
    maxRetries?: number;
    retryDelay?: number;
  } = {}) {
    this.hf = new HfInference(options.apiKey || process.env.HUGGINGFACE_API_KEY);
    this.model = options.model || 'sentence-transformers/all-MiniLM-L6-v2';
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000;

    if (!process.env.HUGGINGFACE_API_KEY && !options.apiKey) {
      console.warn('No HuggingFace API key provided. Using public inference endpoint with rate limits.');
    }
  }

  /**
   * Generate embeddings for a single text
   */
  async embedText(text: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error('Text cannot be empty');
    }

    // Truncate text if too long (model-specific limits)
    const truncatedText = this.truncateText(text, 512);

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.hf.featureExtraction({
          model: this.model,
          inputs: truncatedText,
        });

        // Handle different response formats
        let embedding: number[];
        if (Array.isArray(response)) {
          if (Array.isArray(response[0])) {
            embedding = response[0] as number[];
          } else {
            embedding = response as number[];
          }
        } else {
          throw new Error('Unexpected response format from HuggingFace API');
        }

        // Validate embedding
        if (!embedding || embedding.length === 0) {
          throw new Error('Empty embedding returned');
        }

        return embedding;
      } catch (error) {
        console.error(`HuggingFace API attempt ${attempt} failed:`, error);
        
        if (attempt === this.maxRetries) {
          throw new Error(`Failed to generate embedding after ${this.maxRetries} attempts: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }

        // Wait before retry with exponential backoff
        await this.sleep(this.retryDelay * Math.pow(2, attempt - 1));
      }
    }

    throw new Error('Max retries exceeded');
  }

  /**
   * Generate embeddings for multiple texts in batch
   */
  async embedTexts(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) {
      return [];
    }

    // Process in smaller batches to avoid rate limits
    const batchSize = 10;
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => this.embedText(text));
      
      try {
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
      } catch (error) {
        console.error(`Batch processing failed for texts ${i}-${i + batch.length}:`, error);
        throw error;
      }

      // Add delay between batches to respect rate limits
      if (i + batchSize < texts.length) {
        await this.sleep(200); // 200ms delay between batches
      }
    }

    return results;
  }

  /**
   * Get embedding dimensions for this model
   */
  getDimensions(): number {
    // all-MiniLM-L6-v2 produces 384-dimensional embeddings
    switch (this.model) {
      case 'sentence-transformers/all-MiniLM-L6-v2':
        return 384;
      case 'sentence-transformers/all-mpnet-base-v2':
        return 768;
      case 'sentence-transformers/all-distilroberta-v1':
        return 768;
      default:
        return 384; // Default fallback
    }
  }

  /**
   * Get the model name being used
   */
  getModelName(): string {
    return this.model;
  }

  /**
   * Truncate text to fit model's token limit
   */
  private truncateText(text: string, maxTokens: number): string {
    // Simple approximation: ~4 characters per token
    const maxChars = maxTokens * 4;
    
    if (text.length <= maxChars) {
      return text;
    }

    // Truncate at word boundary when possible
    const truncated = text.substring(0, maxChars);
    const lastSpace = truncated.lastIndexOf(' ');
    
    return lastSpace > maxChars * 0.8 ? truncated.substring(0, lastSpace) : truncated;
  }

  /**
   * Sleep utility for retries and rate limiting
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Test the embeddings service
   */
  async test(): Promise<{ success: boolean; dimensions: number; error?: string }> {
    try {
      const testText = "This is a test sentence for embedding generation.";
      const embedding = await this.embedText(testText);
      
      return {
        success: true,
        dimensions: embedding.length,
      };
    } catch (error) {
      return {
        success: false,
        dimensions: 0,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}

// Default instance
let defaultEmbeddings: HuggingFaceEmbeddings | null = null;

export function getHuggingFaceEmbeddings(): HuggingFaceEmbeddings {
  if (!defaultEmbeddings) {
    defaultEmbeddings = new HuggingFaceEmbeddings();
  }
  return defaultEmbeddings;
}

export default HuggingFaceEmbeddings;
