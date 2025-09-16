/**
 * Document processing utilities for various file formats
 */

import mammoth from "mammoth";
import pdf from "pdf-parse";
import { z } from "zod";

export const SupportedFileTypes = {
  "text/plain": "txt",
  "text/markdown": "md", 
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
} as const;

export type SupportedMimeType = keyof typeof SupportedFileTypes;
export type SupportedFileExtension = typeof SupportedFileTypes[SupportedMimeType];

export const DocumentProcessorSchema = z.object({
  filename: z.string().min(1),
  mimeType: z.string().refine((type): type is SupportedMimeType => 
    type in SupportedFileTypes, 
    { message: "Unsupported file type" }
  ),
  buffer: z.instanceof(Buffer),
  maxSizeBytes: z.number().default(10 * 1024 * 1024), // 10MB default
});

export interface ProcessedDocument {
  filename: string;
  originalName: string;
  fileType: SupportedFileExtension;
  fileSize: number;
  content: string;
  metadata: {
    pageCount?: number;
    wordCount: number;
    processingTime: number;
  };
}

export class DocumentProcessor {
  /**
   * Process uploaded file and extract text content
   */
  static async processFile(
    filename: string,
    mimeType: string,
    buffer: Buffer,
    options: { maxSizeBytes?: number } = {}
  ): Promise<ProcessedDocument> {
    const startTime = Date.now();

    // Validate input
    const validation = DocumentProcessorSchema.safeParse({
      filename,
      mimeType,
      buffer,
      maxSizeBytes: options.maxSizeBytes,
    });

    if (!validation.success) {
      throw new Error(`Validation failed: ${validation.error.message}`);
    }

    const { maxSizeBytes = 10 * 1024 * 1024 } = options;
    
    if (buffer.length > maxSizeBytes) {
      throw new Error(`File size ${buffer.length} bytes exceeds maximum ${maxSizeBytes} bytes`);
    }

    const fileType = SupportedFileTypes[mimeType as SupportedMimeType];
    let content: string;
    let pageCount: number | undefined;

    try {
      switch (fileType) {
        case "txt":
        case "md":
          content = buffer.toString("utf-8");
          break;
        
        case "pdf":
          const pdfData = await pdf(buffer);
          content = pdfData.text;
          pageCount = pdfData.numpages;
          break;
        
        case "docx":
          const docxResult = await mammoth.extractRawText({ buffer });
          content = docxResult.value;
          break;
        
        default:
          throw new Error(`Unsupported file type: ${fileType}`);
      }

      // Clean up content
      content = this.cleanContent(content);
      
      if (!content.trim()) {
        throw new Error("No text content could be extracted from the file");
      }

      const processingTime = Date.now() - startTime;
      const wordCount = this.countWords(content);

      return {
        filename: this.sanitizeFilename(filename),
        originalName: filename,
        fileType,
        fileSize: buffer.length,
        content,
        metadata: {
          pageCount,
          wordCount,
          processingTime,
        },
      };

    } catch (error) {
      throw new Error(
        `Failed to process ${fileType} file: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Clean and normalize extracted text content
   */
  private static cleanContent(content: string): string {
    return content
      // Remove excessive whitespace
      .replace(/\s+/g, " ")
      // Remove excessive line breaks
      .replace(/\n\s*\n\s*\n/g, "\n\n")
      // Trim whitespace
      .trim();
  }

  /**
   * Count words in content
   */
  private static countWords(content: string): number {
    return content.trim().split(/\s+/).filter(word => word.length > 0).length;
  }

  /**
   * Sanitize filename for safe storage
   */
  private static sanitizeFilename(filename: string): string {
    // Remove path components and dangerous characters
    const sanitized = filename
      .replace(/[^a-zA-Z0-9._-]/g, "_")
      .replace(/_{2,}/g, "_")
      .replace(/^_+|_+$/g, "");
    
    // Ensure filename is not empty and has reasonable length
    const finalName = sanitized || "document";
    return finalName.length > 100 ? finalName.substring(0, 100) : finalName;
  }

  /**
   * Validate file type by extension and mime type
   */
  static isValidFileType(filename: string, mimeType: string): boolean {
    const extension = filename.toLowerCase().split('.').pop();
    const supportedExtensions = Object.values(SupportedFileTypes);
    
    return (
      mimeType in SupportedFileTypes &&
      extension !== undefined &&
      supportedExtensions.includes(extension as SupportedFileExtension)
    );
  }

  /**
   * Get file type from mime type
   */
  static getFileType(mimeType: string): SupportedFileExtension | null {
    return SupportedFileTypes[mimeType as SupportedMimeType] || null;
  }

  /**
   * Estimate processing time based on file size and type
   */
  static estimateProcessingTime(fileSize: number, fileType: SupportedFileExtension): number {
    const baseTime = 100; // 100ms base
    const sizeMultiplier = Math.ceil(fileSize / (1024 * 1024)); // Per MB
    
    const typeMultipliers = {
      txt: 1,
      md: 1,
      pdf: 3, // PDF parsing is more intensive
      docx: 2, // DOCX parsing is moderately intensive
    };
    
    return baseTime * sizeMultiplier * typeMultipliers[fileType];
  }
}

export default DocumentProcessor;
