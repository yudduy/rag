import { RAGDemonstration } from "@/components/custom/rag-demonstration";

export default function DemonstrationPage() {
  return (
    <div className="fixed inset-0 z-50 bg-background">
      <RAGDemonstration 
        isVisible={true} 
        onClose={() => {
          // Navigate back to chat
          window.history.back();
        }} 
      />
    </div>
  );
}
