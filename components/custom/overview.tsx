import { motion } from "framer-motion";
import Link from "next/link";

import { LogoGoogle, MessageIcon, VercelIcon } from "./icons";

export const Overview = () => {
  return (
    <motion.div
      key="overview"
      className="max-w-[500px] mt-20 mx-4 md:mx-0"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="border-none bg-muted/50 rounded-2xl p-6 flex flex-col gap-4 text-zinc-500 text-sm dark:text-zinc-400 dark:border-zinc-700">
        <p className="flex flex-row justify-center gap-4 items-center text-zinc-900 dark:text-zinc-50">
          <VercelIcon />
          <span>+</span>
          <MessageIcon />
        </p>
        <p>
          This is an experimental RAG (Retrieval-Augmented Generation) chatbot that combines document intelligence with general AI assistance. Upload your documents (PDF, DOCX, MD, TXT) and chat with them using semantic search, powered by{" "}
          <code className="rounded-sm bg-muted-foreground/15 px-1.5 py-0.5">
            Pinecone
          </code>{" "}
          vector storage,{" "}
          <code className="rounded-sm bg-muted-foreground/15 px-1.5 py-0.5">
            HuggingFace
          </code>{" "}
          embeddings, and{" "}
          <code className="rounded-sm bg-muted-foreground/15 px-1.5 py-0.5">
            Google Gemini
          </code>
          .
        </p>
        <p>
          Start by uploading documents using the{" "}
          <code className="rounded-sm bg-muted-foreground/15 px-1.5 py-0.5">
            📁 Document Manager
          </code>{" "}
          in the navbar, then ask questions about your content. The AI can also help with general tasks.
        </p>
      </div>
    </motion.div>
  );
};
