import { convertToCoreMessages, Message, streamText } from "ai";
import { z } from "zod";

import { geminiFlashModel } from "@/ai";
import { auth } from "@/app/(auth)/auth";
import {
  deleteChatById,
  getChatById,
  saveChat,
} from "@/db/queries";
import { getPineconeRAGCore } from "@/lib/pinecone-rag-core";
import { generateUUID } from "@/lib/utils";

export async function POST(request: Request) {
  const { id, messages }: { id: string; messages: Array<Message> } =
    await request.json();

  const session = await auth();

  if (!session) {
    return new Response("Unauthorized", { status: 401 });
  }

  const coreMessages = convertToCoreMessages(messages).filter(
    (message) => message.content.length > 0,
  );

  // Get the latest user message for RAG processing
  const latestUserMessage = messages.filter(m => m.role === 'user').pop()?.content || '';
  
  // Retrieve relevant documents using RAG
  let ragContext = '';
  let ragSources: Array<{content: string, source: string, relevance_score: number}> = [];
  
  if (session.user?.id && latestUserMessage) {
    try {
      const ragCore = getPineconeRAGCore();
      const retrievedDocs = await ragCore.retrieveDocuments(
        latestUserMessage,
        session.user.id,
        { maxDocs: 3, threshold: 0.7 }
      );
      
      if (retrievedDocs.length > 0) {
        ragSources = retrievedDocs;
        ragContext = retrievedDocs
          .map((doc, idx) => `[Source ${idx + 1}: ${doc.source}]\n${doc.content}`)
          .join('\n\n');
      }
    } catch (ragError) {
      console.error('RAG retrieval error:', ragError);
      // Continue without RAG context
    }
  }

  // Check if user is asking about research papers or document analysis
  const isResearchPaperQuery = /research\s+papers?|analyze.*papers?|paper.*analysis|document.*analysis|analyze.*research|study.*papers?/i.test(latestUserMessage);

  // Build system prompt with RAG context
  const systemPrompt = ragContext 
    ? `You are an intelligent AI assistant with access to the user's uploaded documents. 
       
       DOCUMENT CONTEXT:
       ${ragContext}
       
       INSTRUCTIONS:
       - Use the document context above to answer questions when relevant
       - Always cite your sources when using information from documents (e.g., "According to [Source 1: filename.pdf]...")
       - If the user's question cannot be answered from the provided documents, clearly state this
       - Be conversational and helpful
       - Keep responses concise but informative
       - If no relevant documents are found, answer based on your general knowledge but mention that you don't have specific documents to reference
       - Today's date is ${new Date().toLocaleDateString()}
       
       You can also help with general tasks using the available tools when appropriate.`
    : `You are an intelligent AI assistant.
       
       INSTRUCTIONS:
       - Be helpful and conversational
       - Answer questions based on your knowledge
       - Keep responses concise but informative
       - You can help with various tasks using available tools
       - Today's date is ${new Date().toLocaleDateString()}
       
       SPECIAL HANDLING FOR RESEARCH PAPER REQUESTS:
       ${isResearchPaperQuery ? `- The user is asking about research papers or document analysis, but no documents are currently uploaded
       - Politely ask them to upload their research papers first using the Document Manager (📁 icon in the navbar)
       - Explain that once they upload their papers, you can help analyze them, extract key findings, compare studies, and answer specific questions about their content
       - Do not attempt to provide general information about research papers - focus on getting them to upload their specific documents for analysis` : `- If users want to upload documents for context, let them know they can use the document upload feature`}`;

  const result = await streamText({
    model: geminiFlashModel,
    system: systemPrompt,
    messages: coreMessages,
    tools: {
      searchDocuments: {
        description: "Search through the user's uploaded documents for specific information",
        parameters: z.object({
          query: z.string().describe("The search query to find relevant information in documents"),
          maxResults: z.number().optional().describe("Maximum number of results to return (default: 3)"),
        }),
        execute: async ({ query, maxResults = 3 }) => {
          if (!session.user?.id) {
            return { error: "User not authenticated" };
          }

          try {
            const ragCore = getPineconeRAGCore();
            const results = await ragCore.retrieveDocuments(
              query,
              session.user.id,
              { maxDocs: maxResults, threshold: 0.6 }
            );

            if (results.length === 0) {
              return { 
                message: "No relevant documents found for this query.",
                results: [],
                query 
              };
            }

            return {
              query,
              results: results.map((doc, idx) => ({
                source: doc.source,
                content: doc.content.substring(0, 500) + (doc.content.length > 500 ? '...' : ''),
                relevanceScore: doc.relevance_score,
                index: idx + 1
              })),
              totalFound: results.length,
              // Add citation metadata for UI
              citations: results.map((doc, idx) => ({
                id: doc.chunkId || `${doc.source}_${idx}`,
                filename: doc.source,
                snippet: doc.snippet || doc.content.substring(0, 120) + '...',
                page: doc.page,
                index: idx + 1
              }))
            };
          } catch (error) {
            console.error("Document search error:", error);
            return { 
              error: "Failed to search documents",
              query 
            };
          }
        },
      },
      getWeather: {
        description: "Get the current weather at a location",
        parameters: z.object({
          latitude: z.number().describe("Latitude coordinate"),
          longitude: z.number().describe("Longitude coordinate"),
        }),
        execute: async ({ latitude, longitude }) => {
          const response = await fetch(
            `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m&hourly=temperature_2m&daily=sunrise,sunset&timezone=auto`,
          );

          const weatherData = await response.json();
          return weatherData;
        },
      },
    },
    onFinish: async ({ responseMessages }) => {
      if (session.user && session.user.id) {
        try {
          await saveChat({
            id,
            messages: [...coreMessages, ...responseMessages],
            userId: session.user.id,
          });
        } catch (error) {
          console.error("Failed to save chat");
        }
      }
    },
    experimental_telemetry: {
      isEnabled: true,
      functionId: "stream-text",
    },
  });

  return result.toDataStreamResponse({});
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new Response("Not Found", { status: 404 });
  }

  const session = await auth();

  if (!session || !session.user) {
    return new Response("Unauthorized", { status: 401 });
  }

  try {
    const chat = await getChatById({ id });

    if (chat.userId !== session.user.id) {
      return new Response("Unauthorized", { status: 401 });
    }

    await deleteChatById({ id });

    return new Response("Chat deleted", { status: 200 });
  } catch (error) {
    return new Response("An error occurred while processing your request", {
      status: 500,
    });
  }
}
