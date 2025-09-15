import { NextResponse } from "next/server";

import { auth } from "@/app/(auth)/auth";
import { createDocument, updateDocumentChunkCount, getUser } from "@/db/queries";
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
  try {
    // Check authentication
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Parse form data
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Validate file
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: `File size exceeds ${MAX_FILE_SIZE / 1024 / 1024}MB limit` },
        { status: 400 }
      );
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
      return NextResponse.json(
        { 
          error: "Unsupported file type. Supported types: TXT, MD, PDF, DOCX",
          supportedTypes: ALLOWED_TYPES
        },
        { status: 400 }
      );
    }

    if (!DocumentProcessor.isValidFileType(file.name, file.type)) {
      return NextResponse.json(
        { error: "File extension doesn't match MIME type" },
        { status: 400 }
      );
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
      return NextResponse.json(
        { 
          error: "Failed to process document",
          details: processingError instanceof Error ? processingError.message : "Unknown error"
        },
        { status: 422 }
      );
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
      return NextResponse.json(
        { error: "Failed to save document to database" },
        { status: 500 }
      );
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
    console.error("Document upload error:", error);
    return NextResponse.json(
      { 
        error: "Internal server error",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
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
