"use client";

import { useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { FileIcon } from "./icons";

export interface Citation {
  id: string;
  filename: string;
  snippet: string;
  page?: number | null;
  index: number;
}

interface CitationBadgeProps {
  citation: Citation;
}

export function CitationBadge({ citation }: CitationBadgeProps) {
  const [isOpen, setIsOpen] = useState(false);

  const truncateFilename = (filename: string, maxLength: number = 24) => {
    if (filename.length <= maxLength) return filename;
    const ext = filename.split('.').pop();
    const name = filename.substring(0, filename.lastIndexOf('.'));
    const truncatedName = name.substring(0, maxLength - ext!.length - 4) + '...';
    return `${truncatedName}.${ext}`;
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="inline-flex h-5 px-1 py-0 text-xs font-medium text-blue-600 hover:text-blue-800 hover:bg-blue-50 dark:text-blue-400 dark:hover:text-blue-300 dark:hover:bg-blue-950/50 rounded-sm mx-0.5"
          onClick={() => setIsOpen(!isOpen)}
        >
          [{citation.index}]
        </Button>
      </PopoverTrigger>
      <PopoverContent 
        className="w-80 p-3 text-sm"
        side="top"
        align="start"
        sideOffset={4}
      >
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <FileIcon size={16} />
            <span className="font-medium text-foreground">
              {truncateFilename(citation.filename)}
            </span>
            {citation.page && (
              <span className="text-xs text-muted-foreground">
                Page {citation.page}
              </span>
            )}
          </div>
          
          <div className="text-muted-foreground leading-relaxed">
            {citation.snippet}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

interface CitationListProps {
  citations: Citation[];
  maxVisible?: number;
}

export function CitationList({ citations, maxVisible = 3 }: CitationListProps) {
  const [showAll, setShowAll] = useState(false);
  
  if (citations.length === 0) return null;

  const visibleCitations = showAll ? citations : citations.slice(0, maxVisible);
  const hasMore = citations.length > maxVisible;

  return (
    <span className="inline-flex items-center gap-0.5">
      {visibleCitations.map((citation) => (
        <CitationBadge key={citation.id} citation={citation} />
      ))}
      
      {hasMore && !showAll && (
        <Button
          variant="ghost"
          size="sm"
          className="inline-flex h-5 px-1 py-0 text-xs text-muted-foreground hover:text-foreground rounded-sm"
          onClick={() => setShowAll(true)}
        >
          +{citations.length - maxVisible}
        </Button>
      )}
    </span>
  );
}
