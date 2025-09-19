"use client";

import { Attachment, ToolInvocation } from "ai";
import { motion } from "framer-motion";
import { ReactNode } from "react";

import { BotIcon, UserIcon } from "./icons";
import { Markdown } from "./markdown";
import { PreviewAttachment } from "./preview-attachment";
import { CitationList, Citation } from "./citation-badge";
import { RAGCitations, RAGSource } from "./rag-citations";

// Extract citations from tool invocations
function extractCitations(toolInvocations: Array<ToolInvocation> | undefined): Citation[] {
  if (!toolInvocations) return [];
  
  const citations: Citation[] = [];
  
  toolInvocations.forEach((invocation) => {
    if (invocation.toolName === "searchDocuments" && invocation.state === "result") {
      const result = invocation.result as any;
      if (result.citations && Array.isArray(result.citations)) {
        citations.push(...result.citations);
      }
    }
  });

  return citations;
}

// Extract RAG citations from message content (for direct RAG injection)
function extractRAGCitations(content: string, chatId: string): Citation[] {
  const citations: Citation[] = [];
  
  // Look for citation patterns like [1], [2], etc. in the content
  const citationMatches = content.match(/\[(\d+)\]/g);
  if (!citationMatches) return [];
  
  // For now, create placeholder citations - in a full implementation,
  // we'd need to pass the RAG sources to the message component
  const uniqueNumbers = [...new Set(citationMatches.map(match => parseInt(match.replace(/[\[\]]/g, ''))))];
  
  uniqueNumbers.forEach(num => {
    citations.push({
      id: `rag-${chatId}-${num}`,
      filename: `Source ${num}`,
      snippet: `Citation ${num} from RAG context`,
      page: null,
      index: num
    });
  });
  
  return citations;
}

// Extract RAG sources from attachments
function extractRAGSources(attachments?: Array<Attachment>): RAGSource[] {
  if (!attachments) return [];
  
  const ragAttachment = attachments.find(att => att.name === 'rag-sources');
  if (!ragAttachment || !ragAttachment.url.startsWith('data:application/json;base64,')) {
    return [];
  }
  
  try {
    const base64Data = ragAttachment.url.split(',')[1];
    
    // Browser-safe base64 decoding
    let jsonData: string;
    if (typeof globalThis.atob !== 'undefined') {
      // Browser environment
      const binaryString = globalThis.atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      jsonData = new TextDecoder('utf-8').decode(bytes);
    } else {
      // Node.js environment fallback
      jsonData = Buffer.from(base64Data, 'base64').toString('utf-8');
    }
    
    return JSON.parse(jsonData);
  } catch (error) {
    console.error('Error parsing RAG sources:', error);
    return [];
  }
}

export const Message = ({
  chatId,
  role,
  content,
  toolInvocations,
  attachments,
  ragSources,
}: {
  chatId: string;
  role: string;
  content: string | ReactNode;
  toolInvocations: Array<ToolInvocation> | undefined;
  attachments?: Array<Attachment>;
  ragSources?: RAGSource[];
}) => {
  return (
    <motion.div
      className={`flex flex-row gap-4 px-4 w-full md:w-[500px] md:px-0 first-of-type:pt-20`}
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
    >
      <div className="size-[24px] border rounded-sm p-1 flex flex-col justify-center items-center shrink-0 text-zinc-500">
        {role === "assistant" ? <BotIcon /> : <UserIcon />}
      </div>

      <div className="flex flex-col gap-2 w-full">
        {content && typeof content === "string" && (
          <div className="text-zinc-800 dark:text-zinc-300 flex flex-col gap-4">
            <Markdown>{content}</Markdown>
            {/* Show citations for assistant messages */}
            {role === "assistant" && (
              <>
                {/* Traditional tool citations */}
                <div className="mt-2">
                  <CitationList citations={extractCitations(toolInvocations)} />
                </div>
                
                {/* RAG citations with full source information */}
                <RAGCitations 
                  content={content} 
                  ragSources={ragSources || extractRAGSources(attachments)} 
                />
              </>
            )}
          </div>
        )}

        {toolInvocations && (
          <div className="flex flex-col gap-4">
            {toolInvocations.map((toolInvocation) => {
              const { toolName, toolCallId, state } = toolInvocation;

              if (state === "result") {
                const { result } = toolInvocation;

                return (
                  <div key={toolCallId}>
                    {toolName === "getWeather" ? (
                      <div className="text-sm text-gray-600">Weather data: {JSON.stringify(result)}</div>
                    ) : (
                      <div>{JSON.stringify(result, null, 2)}</div>
                    )}
                  </div>
                );
              } else {
                return (
                  <div key={toolCallId} className="skeleton">
                    {toolName === "getWeather" ? (
                      <Weather />
                    ) : null}
                  </div>
                );
              }
            })}
          </div>
        )}

        {attachments && (
          <div className="flex flex-row gap-2">
            {attachments.map((attachment) => (
              <PreviewAttachment key={attachment.url} attachment={attachment} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};
