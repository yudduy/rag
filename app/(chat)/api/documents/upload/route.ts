import { NextResponse } from "next/server";

import { auth } from "@/app/(auth)/auth";
import { createDocument, updateDocumentChunkCount } from "@/db/queries";
import { DocumentProcessor } from "@/lib/document-processor";
import { getPineconeRAGCore } from "@/lib/pinecone-rag-core";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = [
  "text/plain",
  "text/markdown", 
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
];

export async function POST(request: Request) {
  const startTime = Date.now();
  try {
    // Check authentication
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ 
        error: "Authentication required",
        errorDetails: {
          type: "auth",
          title: "Authentication Required",
          message: "You must be logged in to upload documents",
          details: {
            code: "AUTH_REQUIRED",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload"
          },
          suggestions: [
            "Please log in to your account",
            "If you're already logged in, try refreshing the page",
            "Clear your browser cookies and log in again"
          ],
          technicalInfo: [
            { label: "Session Status", value: session ? "Exists but incomplete" : "Not found" },
            { label: "User ID", value: session?.user?.id || "Missing" }
          ],
          canRetry: true,
          canReport: false
        }
      }, { status: 401 });
    }

    // Parse form data
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ 
        error: "No file provided",
        errorDetails: {
          type: "validation",
          title: "No File Selected",
          message: "Please select a file to upload",
          details: {
            code: "NO_FILE",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload"
          },
          suggestions: [
            "Click 'Choose Files' or drag and drop a file",
            "Make sure the file is properly selected before uploading"
          ],
          canRetry: true,
          canReport: false
        }
      }, { status: 400 });
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json({
        error: "File too large",
        errorDetails: {
          type: "validation",
          title: "File Size Too Large", 
          message: `File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds the ${MAX_FILE_SIZE / 1024 / 1024}MB limit`,
          details: {
            code: "FILE_TOO_LARGE",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload"
          },
          suggestions: [
            "Try compressing your document",
            "Split large documents into smaller files",
            "Use a PDF compression tool to reduce file size"
          ],
          technicalInfo: [
            { label: "File Size", value: `${(file.size / 1024 / 1024).toFixed(2)} MB` },
            { label: "Maximum Allowed", value: `${MAX_FILE_SIZE / 1024 / 1024} MB` },
            { label: "File Name", value: file.name }
          ],
          canRetry: true,
          canReport: false
        }
      }, { status: 400 });
    }

    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      return NextResponse.json({
        error: "Unsupported file type",
        errorDetails: {
          type: "validation",
          title: "Unsupported File Type",
          message: `File type "${file.type}" is not supported. Please use TXT, MD, PDF, or DOCX files.`,
          details: {
            code: "UNSUPPORTED_FILE_TYPE",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload"
          },
          suggestions: [
            "Convert your file to PDF format",
            "Save as a .txt or .md file",
            "Use Microsoft Word to save as .docx format"
          ],
          technicalInfo: [
            { label: "Detected Type", value: file.type || "Unknown" },
            { label: "File Name", value: file.name },
            { label: "Supported Types", value: ALLOWED_TYPES.join(", ") }
          ],
          canRetry: true,
          canReport: false
        }
      }, { status: 400 });
    }

    // Validate file extension matches MIME type
    if (!DocumentProcessor.isValidFileType(file.name, file.type)) {
      return NextResponse.json({
        error: "File extension mismatch",
        errorDetails: {
          type: "validation",
          title: "File Extension Mismatch",
          message: "The file extension doesn't match the detected file type",
          details: {
            code: "EXTENSION_MISMATCH",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload"
          },
          suggestions: [
            "Rename the file with the correct extension",
            "Re-save the file in the correct format",
            "Check if the file was corrupted during transfer"
          ],
          technicalInfo: [
            { label: "File Name", value: file.name },
            { label: "Detected MIME Type", value: file.type },
            { label: "Expected Extension", value: "Should match MIME type" }
          ],
          canRetry: true,
          canReport: true
        }
      }, { status: 400 });
    }

    // Process file
    const buffer = Buffer.from(await file.arrayBuffer());
    let processedDoc;
    
    try {
      processedDoc = await DocumentProcessor.processFile(
        file.name,
        file.type,
        buffer,
        { maxSizeBytes: MAX_FILE_SIZE }
      );
    } catch (processingError) {
      console.error("Document processing error:", processingError);
      const errorMessage = processingError instanceof Error ? processingError.message : "Unknown error";
      
      return NextResponse.json({
        error: "Document processing failed",
        errorDetails: {
          type: "upload",
          title: "Document Processing Failed",
          message: `Failed to process your ${file.type} document: ${errorMessage}`,
          details: {
            code: "PROCESSING_FAILED",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload",
            userId: session.user.id
          },
          suggestions: [
            "Try re-saving the document in the same format",
            "Check if the document is corrupted or password-protected",
            "For PDFs: ensure the file is not secured or encrypted",
            "For Word docs: try saving as a newer .docx format"
          ],
          technicalInfo: [
            { label: "File Name", value: file.name },
            { label: "File Type", value: file.type },
            { label: "File Size", value: `${(file.size / 1024).toFixed(1)} KB` },
            { label: "Processing Time", value: `${Date.now() - startTime}ms` },
            { label: "Error Details", value: errorMessage }
          ],
          canRetry: true,
          canReport: true
        }
      }, { status: 422 });
    }

    // Save to database
    let savedDocument;
    try {
      savedDocument = await createDocument({
        filename: processedDoc.filename,
        originalName: processedDoc.originalName,
        fileType: processedDoc.fileType,
        fileSize: processedDoc.fileSize,
        content: processedDoc.content,
        userId: session.user.id,
      });
    } catch (dbError) {
      console.error("Database error:", dbError);
      const errorMessage = dbError instanceof Error ? dbError.message : "Unknown database error";
      
      return NextResponse.json({
        error: "Database save failed",
        errorDetails: {
          type: "database",
          title: "Failed to Save Document",
          message: "The document was processed successfully but couldn't be saved to the database",
          details: {
            code: "DB_SAVE_FAILED",
            timestamp: new Date().toISOString(),
            endpoint: "/api/documents/upload",
            userId: session.user.id
          },
          suggestions: [
            "Try uploading the document again",
            "Check if you have sufficient storage space",
            "Contact support if the problem persists"
          ],
          technicalInfo: [
            { label: "File Name", value: processedDoc.filename },
            { label: "User ID", value: session.user.id },
            { label: "Content Length", value: `${processedDoc.content.length} characters` },
            { label: "Processing Time", value: `${Date.now() - startTime}ms` },
            { label: "Database Error", value: errorMessage }
          ],
          canRetry: true,
          canReport: true
        }
      }, { status: 500 });
    }

    // Index document with RAG
    let indexingResult;
    try {
      const ragCore = getPineconeRAGCore();
      indexingResult = await ragCore.indexDocument(
        processedDoc.filename,
        processedDoc.content,
        processedDoc.fileType,
        session.user.id
      );

      if (!indexingResult.success) {
        console.error("RAG indexing failed:", indexingResult.error);
        // Don't fail the entire request, but log the error
      } else {
        // Update chunk count in database
        await updateDocumentChunkCount({
          id: savedDocument.id,
          chunkCount: indexingResult.chunks,
        });
      }
    } catch (ragError) {
      console.error("RAG indexing error:", ragError);
      // Continue without failing - document is still saved
      // But we'll include this in the response for transparency
    }

    // Return success response
    return NextResponse.json({
      success: true,
      document: {
        id: savedDocument.id,
        filename: savedDocument.filename,
        originalName: savedDocument.originalName,
        fileType: savedDocument.fileType,
        fileSize: savedDocument.fileSize,
        chunkCount: indexingResult?.chunks || 0,
        status: savedDocument.status,
        createdAt: savedDocument.createdAt,
        metadata: processedDoc.metadata,
      },
      indexing: {
        success: indexingResult?.success || false,
        chunks: indexingResult?.chunks || 0,
        error: indexingResult?.error,
      }
    });

  } catch (error) {
    console.error("Unexpected error during document upload:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    
    return NextResponse.json({
      error: "Unexpected upload error",
      errorDetails: {
        type: "system",
        title: "Unexpected System Error",
        message: "An unexpected error occurred while processing your upload",
        details: {
          code: "SYSTEM_ERROR",
          timestamp: new Date().toISOString(),
          endpoint: "/api/documents/upload",
          sessionId: "unknown"
        },
        suggestions: [
          "Try uploading the document again",
          "Wait a few minutes and retry",
          "Check your internet connection",
          "Contact support if the problem persists"
        ],
        technicalInfo: [
          { label: "Error Type", value: error?.constructor?.name || "Unknown" },
          { label: "Error Message", value: errorMessage },
          { label: "Processing Time", value: `${Date.now() - startTime}ms` },
          { label: "Timestamp", value: new Date().toISOString() }
        ],
        canRetry: true,
        canReport: true
      }
    }, { status: 500 });
  }
}

// GET endpoint to list user's documents
export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Get RAG core status and user documents
    const ragCore = getPineconeRAGCore();
    const ragDocuments = await ragCore.getUserDocuments(session.user.id);
    const ragStatus = await ragCore.getStatus();

    return NextResponse.json({
      documents: ragDocuments,
      status: ragStatus,
    });

  } catch (error) {
    console.error("Error fetching documents:", error);
    return NextResponse.json(
      { error: "Failed to fetch documents" },
      { status: 500 }
    );
  }
}
