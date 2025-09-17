"use client";

import { RAGDemonstration } from "@/components/custom/rag-demonstration";
import { useRouter } from "next/navigation";

export default function DemonstrationPage() {
  const router = useRouter();

  return (
    <div className="fixed inset-0 z-50 bg-background">
      <RAGDemonstration 
        isVisible={true} 
        onClose={() => {
          // Navigate back to chat
          router.back();
        }} 
      />
    </div>
  );
}
