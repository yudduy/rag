import { NextResponse } from "next/server";
import { auth } from "@/app/(auth)/auth";
import { getPineconeRAGCore } from "@/lib/pinecone-rag-core";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const ragCore = getPineconeRAGCore();
    
    // Debug RAG for authenticated user
    
    // Test 1: Get user documents (from Pinecone metadata)
    const userDocs = await ragCore.getUserDocuments(session.user.id);
    // Retrieved user documents from Pinecone
    
    // Test 2: Try a very generic query with low threshold
    const testQueries = [
      "document",
      "text",
      "content",
      "the"
    ];
    
    const results: Record<string, any> = {};
    
    for (const query of testQueries) {
      try {
        const docs = await ragCore.retrieveDocuments(
          query,
          session.user.id,
          { maxDocs: 10, threshold: 0.1 } // Very low threshold
        );
        results[query] = {
          count: docs.length,
          docs: docs.map(d => ({
            source: d.source,
            score: d.relevance_score,
            contentPreview: d.content.substring(0, 100)
          }))
        };
      } catch (error) {
        results[query] = { error: error instanceof Error ? error.message : 'Unknown error' };
      }
    }
    
    return NextResponse.json({
      userId: session.user.id,
      namespace: `user_${session.user.id}`,
      userDocuments: userDocs,
      testQueries: results
    });

  } catch (error) {
    console.error("Debug RAG error:", error);
    return NextResponse.json(
      { error: "Debug failed", details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
