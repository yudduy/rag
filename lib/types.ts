import { Message } from "ai";

export interface RAGSource {
  source: string;
  content: string;
  relevance_score: number;
}

// Extended Message interface that includes ragSources
export interface ExtendedMessage extends Message {
  ragSources?: RAGSource[];
}

// Type guard to check if a message has ragSources
export function hasRAGSources(message: Message): message is ExtendedMessage {
  return 'ragSources' in message && Array.isArray((message as any).ragSources);
}
