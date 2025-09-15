"use client";

import { useState } from "react";

import { FileIcon } from "@/components/custom/icons";
import { Button } from "@/components/ui/button";

import { DocumentManager } from "./document-manager";

interface NavbarClientProps {
  isAuthenticated: boolean;
}

export function NavbarClient({ isAuthenticated }: NavbarClientProps) {
  const [isDocumentManagerOpen, setIsDocumentManagerOpen] = useState(false);

  if (!isAuthenticated) {
    return null;
  }

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsDocumentManagerOpen(true)}
        className="flex items-center gap-2"
        title="Manage Documents"
      >
        <FileIcon size={16} />
        <span className="hidden sm:inline">Documents</span>
      </Button>

      <DocumentManager
        isOpen={isDocumentManagerOpen}
        onClose={() => setIsDocumentManagerOpen(false)}
      />
    </>
  );
}

export default NavbarClient;
