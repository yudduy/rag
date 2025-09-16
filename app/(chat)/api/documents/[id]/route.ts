import { NextResponse } from "next/server";

import { auth } from "@/app/(auth)/auth";
import { getDocumentById, deleteDocumentById } from "@/db/queries";
import { getPineconeRAGCore } from "@/lib/pinecone-rag-core";

export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Check authentication
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const documentId = params.id;
    
    // Get document to verify ownership
    const document = await getDocumentById({ id: documentId });
    
    if (!document) {
      return NextResponse.json({ error: "Document not found" }, { status: 404 });
    }

    if (document.userId !== session.user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    // Delete from RAG system
    try {
      const ragCore = getPineconeRAGCore();
      const ragDeleted = await ragCore.deleteDocument(document.filename, session.user.id);
      console.log(`RAG deletion result for ${document.filename}:`, ragDeleted);
    } catch (ragError) {
      console.error("RAG deletion error:", ragError);
      // Continue with database deletion even if RAG deletion fails
    }

    // Delete from database
    await deleteDocumentById({ id: documentId });

    return NextResponse.json({
      success: true,
      message: "Document deleted successfully"
    });

  } catch (error) {
    console.error("Document deletion error:", error);
    return NextResponse.json(
      { error: "Failed to delete document" },
      { status: 500 }
    );
  }
}

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Check authentication
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const documentId = params.id;
    
    // Get document
    const document = await getDocumentById({ id: documentId });
    
    if (!document) {
      return NextResponse.json({ error: "Document not found" }, { status: 404 });
    }

    if (document.userId !== session.user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    // Return document info (without full content for privacy)
    return NextResponse.json({
      id: document.id,
      filename: document.filename,
      originalName: document.originalName,
      fileType: document.fileType,
      fileSize: document.fileSize,
      chunkCount: document.chunkCount,
      status: document.status,
      createdAt: document.createdAt,
      updatedAt: document.updatedAt,
      // Don't return content for security reasons
    });

  } catch (error) {
    console.error("Error fetching document:", error);
    return NextResponse.json(
      { error: "Failed to fetch document" },
      { status: 500 }
    );
  }
}
