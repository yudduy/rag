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
import { ragDemoManager } from "@/lib/rag-demonstration-manager";

export async function POST(request: Request) {
  const { id, messages }: { id: string; messages: Array<Message> } =
    await request.json();

  const session = await auth();

  if (!session) {
    return new Response(JSON.stringify({
      error: "Authentication required",
      errorDetails: {
        type: "auth",
        title: "Authentication Required",
        message: "Please log in to use the chat feature",
        details: {
          code: "AUTH_REQUIRED",
          timestamp: new Date().toISOString(),
          endpoint: "/api/chat"
        },
        suggestions: [
          "Please log in to your account",
          "If you're already logged in, try refreshing the page",
          "Check if your session has expired"
        ],
        canRetry: false,
        canReport: false
      }
    }), { 
      status: 401,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const coreMessages = convertToCoreMessages(messages).filter(
    (message) => message.content.length > 0,
  );

  // Critical: Ensure we have at least one message for Gemini
  if (coreMessages.length === 0) {
    console.error("No valid messages after filtering", { originalMessages: messages });
    return new Response(JSON.stringify({
      error: "No valid messages provided",
      errorDetails: {
        type: "validation",
        title: "Invalid Message Content",
        message: "No valid messages were provided for processing. Please ensure your message contains text content.",
        details: {
          code: "EMPTY_MESSAGES",
          timestamp: new Date().toISOString(),
          endpoint: "/api/chat",
          userId: session.user?.id
        },
        suggestions: [
          "Try typing a text message",
          "Ensure your message isn't empty",
          "If uploading documents, wait for indexing to complete first"
        ],
        technicalInfo: [
          { label: "Original Messages", value: messages.length.toString() },
          { label: "Filtered Messages", value: coreMessages.length.toString() }
        ],
        canRetry: true,
        canReport: true
      }
    }), { 
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  // Get the latest user message for RAG processing
  const latestUserMessage = messages.filter(m => m.role === 'user').pop()?.content || '';
  
  // Create RAG demonstration session
  const ragSessionId = generateUUID();
  let ragDemoSession = null;
  
  if (session.user?.id && latestUserMessage) {
    ragDemoSession = ragDemoManager.createSession(ragSessionId, session.user.id, latestUserMessage);
  }
  
  // Retrieve relevant documents using RAG with demonstration tracking
  let ragContext = '';
  let ragSources: Array<{content: string, source: string, relevance_score: number}> = [];
  
  if (session.user?.id && latestUserMessage) {
    try {
      const ragCore = getPineconeRAGCore();
      const modelInfo = ragCore.getEmbeddingModelInfo();
      
      // Step 1: Query Embedding
      ragDemoManager.updateQueryEmbeddingStep(ragSessionId, 'processing', {
        originalQuery: latestUserMessage,
        processedQuery: latestUserMessage,
        embeddingModel: modelInfo.modelName,
        embeddingDimensions: modelInfo.dimensions
      });
      
      // Generate embedding (this is already done internally, but we track it)
      const embeddingStartTime = Date.now();
      const queryEmbedding = await ragCore.generateEmbedding(latestUserMessage);
      const embeddingDuration = Date.now() - embeddingStartTime;
      
      ragDemoManager.updateQueryEmbeddingStep(ragSessionId, 'completed', {
        originalQuery: latestUserMessage,
        processedQuery: latestUserMessage,
        embeddingModel: modelInfo.modelName,
        embeddingDimensions: modelInfo.dimensions,
        embeddingVector: queryEmbedding.slice(0, 8), // First 8 dimensions for preview
        embeddingPreview: queryEmbedding.slice(0, 8).map(v => v.toFixed(4)).join(', ')
      });

      // Step 2: Document Retrieval
      ragDemoManager.updateDocumentRetrievalStep(ragSessionId, 'processing', {
        searchQuery: latestUserMessage,
        namespace: `user-${session.user.id}`,
        searchParams: {
          topK: 5,
          threshold: 0.1
        }
      });

      const retrievalStartTime = Date.now();
      const retrievedDocs = await ragCore.retrieveDocuments(
        latestUserMessage,
        session.user.id,
        { maxDocs: 5, threshold: 0.1 }
      );
      const retrievalDuration = Date.now() - retrievalStartTime;
      
      console.log(`📄 Retrieved ${retrievedDocs.length} documents:`, 
        retrievedDocs.map(d => ({ source: d.source, score: d.relevance_score, contentLength: d.content.length }))
      );

      ragDemoManager.updateDocumentRetrievalStep(ragSessionId, 'completed', {
        searchQuery: latestUserMessage,
        namespace: `user-${session.user.id}`,
        searchParams: {
          topK: 5,
          threshold: 0.1
        },
        totalResults: retrievedDocs.length,
        filteredResults: retrievedDocs.length,
        documents: retrievedDocs.map(doc => ({
          id: doc.chunkId || generateUUID(),
          source: doc.source,
          content: doc.content,
          snippet: doc.snippet,
          relevanceScore: doc.relevance_score,
          metadata: {
            page: doc.page,
            chunkId: doc.chunkId || generateUUID(),
            fileType: doc.metadata?.file_type || 'unknown',
            ...doc.metadata
          }
        }))
      });

      // Step 3: Context Assembly
      if (retrievedDocs.length > 0) {
        ragDemoManager.updateContextAssemblyStep(ragSessionId, 'processing', {
          selectedDocuments: retrievedDocs.map(doc => ({
            id: doc.chunkId || generateUUID(),
            source: doc.source,
            content: doc.content,
            snippet: doc.snippet,
            relevanceScore: doc.relevance_score,
            metadata: {
              page: doc.page,
              chunkId: doc.chunkId || generateUUID(),
              fileType: doc.metadata?.file_type || 'unknown',
              ...doc.metadata
            }
          })),
          assemblyStrategy: 'Relevance-based concatenation'
        });

        const assemblyStartTime = Date.now();
        ragSources = retrievedDocs;
        ragContext = retrievedDocs
          .map((doc, idx) => `[Source ${idx + 1}: ${doc.source}]\n${doc.content}`)
          .join('\n\n');
        const assemblyDuration = Date.now() - assemblyStartTime;

        ragDemoManager.updateContextAssemblyStep(ragSessionId, 'completed', {
          selectedDocuments: retrievedDocs.map(doc => ({
            id: doc.chunkId || generateUUID(),
            source: doc.source,
            content: doc.content,
            snippet: doc.snippet,
            relevanceScore: doc.relevance_score,
            metadata: {
              page: doc.page,
              chunkId: doc.chunkId || generateUUID(),
              fileType: doc.metadata?.file_type || 'unknown',
              ...doc.metadata
            }
          })),
          contextLength: ragContext.length,
          contextPreview: ragContext.substring(0, 500),
          assemblyStrategy: 'Relevance-based concatenation'
        });
        
        console.log(`✅ RAG context created: ${ragContext.length} characters`);
      } else {
        ragDemoManager.updateContextAssemblyStep(ragSessionId, 'completed', {
          selectedDocuments: [],
          contextLength: 0,
          contextPreview: '',
          assemblyStrategy: 'No documents selected'
        });
        console.log(`❌ No documents retrieved for query: "${latestUserMessage}"`);
      }
    } catch (ragError) {
      console.error('RAG retrieval error:', ragError);
      
      // Mark current step as error
      ragDemoManager.updateDocumentRetrievalStep(ragSessionId, 'error', undefined, 
        ragError instanceof Error ? ragError.message : 'Unknown RAG error'
      );
      
      // Continue without RAG context - this enables progressive enhancement
    }
  }

  // Check if user is asking about research papers or document analysis
  const isResearchPaperQuery = /research\s+papers?|analyze.*papers?|paper.*analysis|document.*analysis|analyze.*research|study.*papers?/i.test(latestUserMessage);

  // Debug RAG context
  console.log(`🔍 RAG Context Debug:`, {
    hasContext: !!ragContext,
    contextLength: ragContext.length,
    sourcesCount: ragSources.length,
    contextPreview: ragContext.substring(0, 200) + '...'
  });

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
       - Do not attempt to provide general information about research papers - focus on getting them to upload their specific documents for analysis` : `- If users want to upload documents for context, let them know they can use the document upload feature`}
       
       CONCURRENT OPERATIONS HANDLING:
       - If documents are currently being indexed, inform the user that some documents may still be processing
       - You can still answer questions based on already-indexed documents
       - Let users know that more comprehensive answers will be available once all documents are fully indexed`;

  console.log(`🤖 System prompt being sent to LLM:`, {
    hasRagContext: systemPrompt.includes('DOCUMENT CONTEXT:'),
    promptLength: systemPrompt.length,
    promptPreview: systemPrompt.substring(0, 300) + '...'
  });

  // Handle RAG context injection for Gemini
  let finalMessages = coreMessages;
  let finalSystemPrompt = systemPrompt;
  
  if (ragContext && coreMessages.length > 0) {
    // Gemini has issues with large contexts in system prompts
    // So we inject the context directly into the last user message
    const lastMessage = coreMessages[coreMessages.length - 1];
    if (lastMessage.role === 'user') {
      // Create detailed citation information for each source
      const citationInfo = ragSources.map((source, idx) => {
        return `[${idx + 1}] ${source.source} (Relevance: ${(source.relevance_score * 100).toFixed(1)}%)
Content: ${source.content.substring(0, 150)}${source.content.length > 150 ? '...' : ''}`;
      }).join('\n\n');

      const enhancedContent = `CONTEXT FROM YOUR UPLOADED DOCUMENTS:

${ragContext}

CITATION REFERENCE:
${citationInfo}

---

USER QUESTION: ${lastMessage.content}

Please answer using the document context provided above. When referencing information, use the citation format [1], [2], etc. corresponding to the sources listed in the citation reference.`;

      finalMessages = [
        ...coreMessages.slice(0, -1),
        {
          ...lastMessage,
          content: enhancedContent
        }
      ];
      
      // Use simpler system prompt without the large context
      finalSystemPrompt = `You are an intelligent AI assistant. When provided with document context, use it to answer questions accurately and cite your sources using the numbered format [1], [2], etc. Be helpful and conversational.`;
      
      console.log(`🔄 Injected RAG context with citations (${enhancedContent.length} characters)`);
    }
  }

  // Step 4: Response Generation
  if (ragDemoSession) {
    ragDemoManager.updateResponseGenerationStep(ragSessionId, 'processing', {
      model: 'gemini-1.5-flash',
      promptLength: finalSystemPrompt.length,
      contextLength: ragContext.length
    });
  }

  const responseStartTime = Date.now();
  const result = await streamText({
    model: geminiFlashModel,
    system: finalSystemPrompt,
    messages: finalMessages,
    // Add RAG sources as metadata for citation extraction
    experimental_providerMetadata: {
      ragSources: ragSources.map((source, idx) => ({
        id: `rag-${idx + 1}`,
        filename: source.source,
        snippet: source.content.substring(0, 200) + (source.content.length > 200 ? '...' : ''),
        relevanceScore: source.relevance_score,
        index: idx + 1,
        fullContent: source.content
      })) as any
    } as any,
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
      // Complete RAG demonstration tracking
      if (ragDemoSession) {
        const responseDuration = Date.now() - responseStartTime;
        const responseContent = responseMessages.map(msg => msg.content).join('');
        
        ragDemoManager.updateResponseGenerationStep(ragSessionId, 'completed', {
          model: 'gemini-1.5-flash',
          promptLength: finalSystemPrompt.length,
          contextLength: ragContext.length,
          responseLength: responseContent.length,
          tokenUsage: {
            prompt: Math.ceil(finalSystemPrompt.length / 4), // Rough token estimate
            completion: Math.ceil(responseContent.length / 4),
            total: Math.ceil((finalSystemPrompt.length + responseContent.length) / 4)
          }
        });

        // Complete the entire session
        ragDemoManager.completeSession(ragSessionId, ragSources.map(source => ({
          id: generateUUID(),
          source: source.source,
          content: source.content,
          snippet: source.content.substring(0, 120),
          relevanceScore: source.relevance_score,
          metadata: {
            chunkId: generateUUID(),
            fileType: 'unknown'
          }
        })));
      }

      if (session.user && session.user.id) {
        try {
          // Add RAG sources to the response message metadata
          const messagesWithRAG = responseMessages.map(msg => {
            if (msg.role === 'assistant' && ragSources.length > 0) {
              return {
                ...msg,
                experimental_attachments: [
                  ...((msg as any).experimental_attachments || []),
                  {
                    name: 'rag-sources',
                    contentType: 'application/json',
                    url: `data:application/json;base64,${Buffer.from(JSON.stringify(ragSources)).toString('base64')}`
                  }
                ]
              } as any;
            }
            return msg;
          });

          await saveChat({
            id,
            messages: [...coreMessages, ...messagesWithRAG],
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
