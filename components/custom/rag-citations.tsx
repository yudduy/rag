"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { FileIcon } from "./icons";

export interface RAGSource {
  source: string;
  content: string;
  relevance_score: number;
}

interface RAGCitationsProps {
  content: string;
  ragSources: RAGSource[];
}

interface CitationPopupProps {
  citationNumber: number;
  source: RAGSource;
}

function CitationPopup({ citationNumber, source }: CitationPopupProps) {
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
          className="inline-flex h-5 px-1 py-0 text-xs font-medium text-blue-600 hover:text-blue-800 hover:bg-blue-50 dark:text-blue-400 dark:hover:text-blue-300 dark:hover:bg-blue-950/50 rounded-sm mx-0.5 border border-blue-200 dark:border-blue-800"
          onClick={() => setIsOpen(!isOpen)}
        >
          [{citationNumber}]
        </Button>
      </PopoverTrigger>
      <PopoverContent 
        className="w-96 p-4 text-sm"
        side="top"
        align="start"
        sideOffset={4}
      >
        <div className="space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b">
            <FileIcon size={16} />
            <span className="font-semibold text-foreground">
              {truncateFilename(source.source)}
            </span>
            <span className="text-xs bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 px-2 py-0.5 rounded-full">
              {(source.relevance_score * 100).toFixed(1)}% match
            </span>
          </div>
          
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Content Preview
            </div>
            <div className="text-muted-foreground leading-relaxed bg-muted/30 p-2 rounded text-xs max-h-32 overflow-y-auto">
              {source.content.length > 300 
                ? source.content.substring(0, 300) + '...' 
                : source.content
              }
            </div>
          </div>
          
          <div className="text-xs text-muted-foreground pt-2 border-t">
            Click to expand • Source: {source.source}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function RAGCitations({ content, ragSources }: RAGCitationsProps) {
  // Extract citation numbers from content
  const citationMatches = content.match(/\[(\d+)\]/g);
  if (!citationMatches || ragSources.length === 0) return null;

  const uniqueNumbers = [...new Set(citationMatches.map(match => parseInt(match.replace(/[\[\]]/g, ''))))];
  
  // Filter to only show citations that have corresponding sources
  const validCitations = uniqueNumbers.filter(num => num <= ragSources.length);
  
  if (validCitations.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-muted/50">
      <div className="text-xs font-medium text-muted-foreground mb-2">
        Sources referenced in this response:
      </div>
      <div className="flex flex-wrap gap-1">
        {validCitations.map(citationNum => {
          const source = ragSources[citationNum - 1]; // Convert to 0-based index
          return (
            <CitationPopup
              key={citationNum}
              citationNumber={citationNum}
              source={source}
            />
          );
        })}
      </div>
      <div className="text-xs text-muted-foreground mt-2">
        {validCitations.length} source{validCitations.length !== 1 ? 's' : ''} • 
        Click on citation numbers for details
      </div>
    </div>
  );
}



