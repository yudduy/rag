"use client";

import React, { useState, useRef, useEffect } from 'react';
import { 
  FileText, 
  Image as ImageIcon, 
  ExternalLink, 
  Download, 
  Eye, 
  BookOpen,
  Quote,
  Hash,
  Calendar,
  User,
  MapPin,
  ChevronDown,
  ChevronUp,
  Copy,
  Check
} from 'lucide-react';

interface Citation {
  id: string;
  type: 'text' | 'image' | 'document' | 'web';
  title?: string;
  content?: string;
  url?: string;
  image_path?: string;
  metadata?: {
    author?: string;
    date?: string;
    page_number?: number;
    source_type?: string;
    confidence_score?: number;
    relevance_score?: number;
    file_name?: string;
    extracted_text?: string;
    image_description?: string;
    word_count?: number;
    last_modified?: string;
  };
}

interface EnhancedCitationsProps {
  citations: Citation[];
  responseText: string;
  className?: string;
  showPreview?: boolean;
  groupBySources?: boolean;
}

export default function EnhancedCitations({ 
  citations = [], 
  responseText = "",
  className = "",
  showPreview = true,
  groupBySources = false
}: EnhancedCitationsProps) {
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set());
  const [hoveredCitation, setHoveredCitation] = useState<string | null>(null);
  const [copiedCitation, setCopiedCitation] = useState<string | null>(null);
  const [highlightedCitation, setHighlightedCitation] = useState<string | null>(null);
  
  const citationRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const copyTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Extract citation IDs from response text
  const extractCitationIds = (text: string): string[] => {
    const citationPattern = /\[citation:(\d+)\]/g;
    const matches = [];
    let match;
    
    while ((match = citationPattern.exec(text)) !== null) {
      matches.push(match[1]);
    }
    
    return [...new Set(matches)]; // Remove duplicates
  };

  const citationIds = extractCitationIds(responseText);
  const referencedCitations = citations.filter(citation => 
    citationIds.includes(citation.id)
  );

  // Group citations by source if requested
  const groupedCitations = groupBySources ? 
    referencedCitations.reduce((groups, citation) => {
      const sourceType = citation.metadata?.source_type || citation.type;
      if (!groups[sourceType]) {
        groups[sourceType] = [];
      }
      groups[sourceType].push(citation);
      return groups;
    }, {} as Record<string, Citation[]>) : 
    { all: referencedCitations };

  const toggleCitationExpansion = (citationId: string) => {
    const newExpanded = new Set(expandedCitations);
    if (newExpanded.has(citationId)) {
      newExpanded.delete(citationId);
    } else {
      newExpanded.add(citationId);
    }
    setExpandedCitations(newExpanded);
  };

  const scrollToCitation = (citationId: string) => {
    const element = citationRefs.current.get(citationId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setHighlightedCitation(citationId);
      setTimeout(() => setHighlightedCitation(null), 2000);
    }
  };

  const copyCitationText = async (citation: Citation) => {
    const citationText = formatCitationForCopy(citation);
    
    try {
      await navigator.clipboard.writeText(citationText);
      setCopiedCitation(citation.id);
      
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current);
      }
      
      copyTimeoutRef.current = setTimeout(() => {
        setCopiedCitation(null);
      }, 2000);
    } catch (error) {
      console.error('Failed to copy citation:', error);
    }
  };

  const formatCitationForCopy = (citation: Citation): string => {
    const parts = [];
    
    if (citation.metadata?.author) {
      parts.push(`Author: ${citation.metadata.author}`);
    }
    
    if (citation.title) {
      parts.push(`Title: ${citation.title}`);
    }
    
    if (citation.metadata?.date) {
      parts.push(`Date: ${citation.metadata.date}`);
    }
    
    if (citation.url) {
      parts.push(`URL: ${citation.url}`);
    }
    
    if (citation.metadata?.page_number) {
      parts.push(`Page: ${citation.metadata.page_number}`);
    }
    
    return parts.join('\n');
  };

  const downloadCitationContent = async (citation: Citation) => {
    try {
      if (citation.type === 'image' && citation.image_path) {
        // Download image
        const response = await fetch(citation.image_path);
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = citation.metadata?.file_name || `image-${citation.id}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else if (citation.content) {
        // Download text content
        const blob = new Blob([citation.content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `citation-${citation.id}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to download citation:', error);
    }
  };

  const formatDate = (dateString?: string): string => {
    if (!dateString) return 'Unknown date';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString(undefined, { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      });
    } catch {
      return dateString;
    }
  };

  const getConfidenceColor = (score?: number): string => {
    if (!score) return '#6c757d';
    if (score >= 0.8) return '#28a745';
    if (score >= 0.6) return '#ffc107';
    return '#dc3545';
  };

  const getCitationIcon = (citation: Citation) => {
    switch (citation.type) {
      case 'image':
        return <ImageIcon className="size-4" />;
      case 'document':
        return <FileText className="size-4" />;
      case 'web':
        return <ExternalLink className="size-4" />;
      default:
        return <BookOpen className="size-4" />;
    }
  };

  const renderCitationPreview = (citation: Citation) => {
    if (citation.type === 'image') {
      return (
        <div className="image-preview">
          <img 
            src={citation.image_path} 
            alt={citation.title || 'Citation image'}
            className="preview-image"
            loading="lazy"
          />
          {citation.metadata?.image_description && (
            <p className="image-description">{citation.metadata.image_description}</p>
          )}
          {citation.metadata?.extracted_text && (
            <div className="extracted-text">
              <strong>Extracted Text:</strong>
              <p>{citation.metadata.extracted_text}</p>
            </div>
          )}
        </div>
      );
    }
    
    return (
      <div className="text-preview">
        <Quote className="size-4 quote-icon" />
        <p className="preview-text">
          {citation.content ? 
            citation.content.length > 300 ? 
              `${citation.content.substring(0, 300)}...` : 
              citation.content :
            'No preview available'
          }
        </p>
      </div>
    );
  };

  const renderCitation = (citation: Citation) => {
    const isExpanded = expandedCitations.has(citation.id);
    const isCopied = copiedCitation === citation.id;
    const isHighlighted = highlightedCitation === citation.id;

    return (
      <div 
        key={citation.id}
        ref={(el) => {
          if (el) {
            citationRefs.current.set(citation.id, el);
          }
        }}
        className={`citation-item ${isHighlighted ? 'highlighted' : ''}`}
        onMouseEnter={() => setHoveredCitation(citation.id)}
        onMouseLeave={() => setHoveredCitation(null)}
      >
        <div className="citation-header">
          <div className="citation-main-info">
            <div className="citation-icon-title">
              {getCitationIcon(citation)}
              <span className="citation-id">[{citation.id}]</span>
              <h4 className="citation-title">
                {citation.title || `${citation.type} source`}
              </h4>
            </div>
            
            <div className="citation-scores">
              {citation.metadata?.confidence_score && (
                <div 
                  className="score-badge confidence"
                  style={{ backgroundColor: getConfidenceColor(citation.metadata.confidence_score) }}
                  title={`Confidence: ${Math.round(citation.metadata.confidence_score * 100)}%`}
                >
                  {Math.round(citation.metadata.confidence_score * 100)}%
                </div>
              )}
              
              {citation.metadata?.relevance_score && (
                <div 
                  className="score-badge relevance"
                  style={{ backgroundColor: getConfidenceColor(citation.metadata.relevance_score) }}
                  title={`Relevance: ${Math.round(citation.metadata.relevance_score * 100)}%`}
                >
                  R: {Math.round(citation.metadata.relevance_score * 100)}%
                </div>
              )}
            </div>
          </div>
          
          <div className="citation-controls">
            <button
              onClick={() => copyCitationText(citation)}
              className="control-btn"
              title="Copy citation"
              aria-label="Copy citation"
            >
              {isCopied ? <Check className="size-4" /> : <Copy className="size-4" />}
            </button>
            
            {(citation.content || citation.image_path) && (
              <button
                onClick={() => downloadCitationContent(citation)}
                className="control-btn"
                title="Download content"
                aria-label="Download content"
              >
                <Download className="size-4" />
              </button>
            )}
            
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="control-btn"
                title="Open source"
                aria-label="Open source"
              >
                <ExternalLink className="size-4" />
              </a>
            )}
            
            <button
              onClick={() => toggleCitationExpansion(citation.id)}
              className="expand-btn"
              aria-label={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            </button>
          </div>
        </div>
        
        <div className="citation-metadata">
          {citation.metadata?.author && (
            <div className="metadata-item">
              <User className="size-3" />
              <span>{citation.metadata.author}</span>
            </div>
          )}
          
          {citation.metadata?.date && (
            <div className="metadata-item">
              <Calendar className="size-3" />
              <span>{formatDate(citation.metadata.date)}</span>
            </div>
          )}
          
          {citation.metadata?.page_number && (
            <div className="metadata-item">
              <Hash className="size-3" />
              <span>Page {citation.metadata.page_number}</span>
            </div>
          )}
          
          {citation.metadata?.word_count && (
            <div className="metadata-item">
              <FileText className="size-3" />
              <span>{citation.metadata.word_count} words</span>
            </div>
          )}
        </div>
        
        {isExpanded && showPreview && (
          <div className="citation-preview">
            {renderCitationPreview(citation)}
          </div>
        )}
      </div>
    );
  };

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  if (referencedCitations.length === 0) {
    return null;
  }

  return (
    <div className={`enhanced-citations ${className}`}>
      <div className="citations-header">
        <div className="header-info">
          <BookOpen className="size-4" />
          <span>Sources ({referencedCitations.length})</span>
        </div>
        
        <div className="header-controls">
          <button
            onClick={() => {
              const allIds = new Set(referencedCitations.map(c => c.id));
              setExpandedCitations(
                expandedCitations.size === referencedCitations.length ? 
                new Set() : allIds
              );
            }}
            className="expand-all-btn"
          >
            {expandedCitations.size === referencedCitations.length ? 
              'Collapse All' : 'Expand All'
            }
          </button>
        </div>
      </div>
      
      <div className="citations-content">
        {groupBySources && Object.keys(groupedCitations).length > 1 ? (
          Object.entries(groupedCitations).map(([sourceType, sourceCitations]) => (
            <div key={sourceType} className="source-group">
              <h3 className="source-group-title">
                {sourceType.charAt(0).toUpperCase() + sourceType.slice(1)} Sources
              </h3>
              <div className="source-group-content">
                {sourceCitations.map(renderCitation)}
              </div>
            </div>
          ))
        ) : (
          <div className="citations-list">
            {referencedCitations.map(renderCitation)}
          </div>
        )}
      </div>
      
      {/* Clickable citations in response text would be handled by parent component */}
      <style jsx>{`
        .enhanced-citations {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 16px 0;
          background: #f8f9fa;
        }
        
        .citations-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 8px;
          border-bottom: 1px solid #e1e5e9;
        }
        
        .header-info {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #495057;
          font-weight: 600;
        }
        
        .expand-all-btn {
          padding: 4px 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
          font-size: 12px;
          color: #6c757d;
        }
        
        .expand-all-btn:hover {
          background: #f8f9fa;
        }
        
        .source-group {
          margin-bottom: 24px;
        }
        
        .source-group:last-child {
          margin-bottom: 0;
        }
        
        .source-group-title {
          margin: 0 0 12px 0;
          font-size: 16px;
          color: #495057;
          padding-bottom: 4px;
          border-bottom: 1px solid #dee2e6;
        }
        
        .citations-list, .source-group-content {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .citation-item {
          border: 1px solid #e1e5e9;
          border-radius: 6px;
          padding: 12px;
          background: white;
          transition: all 0.2s ease;
        }
        
        .citation-item:hover {
          border-color: #007bff;
          box-shadow: 0 2px 4px rgba(0,123,255,0.1);
        }
        
        .citation-item.highlighted {
          border-color: #ffc107;
          background: #fff3cd;
          box-shadow: 0 2px 4px rgba(255,193,7,0.2);
        }
        
        .citation-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 8px;
        }
        
        .citation-main-info {
          flex: 1;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-right: 12px;
        }
        
        .citation-icon-title {
          display: flex;
          align-items: center;
          gap: 8px;
          flex: 1;
        }
        
        .citation-id {
          color: #007bff;
          font-weight: 600;
          font-size: 14px;
        }
        
        .citation-title {
          margin: 0;
          font-size: 14px;
          color: #333;
          font-weight: 500;
          flex: 1;
        }
        
        .citation-scores {
          display: flex;
          gap: 4px;
          margin-left: 8px;
        }
        
        .score-badge {
          padding: 2px 6px;
          border-radius: 12px;
          color: white;
          font-size: 11px;
          font-weight: 500;
          white-space: nowrap;
        }
        
        .citation-controls {
          display: flex;
          gap: 4px;
          align-items: center;
        }
        
        .control-btn, .expand-btn {
          padding: 4px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #6c757d;
          text-decoration: none;
        }
        
        .control-btn:hover, .expand-btn:hover {
          background: #f8f9fa;
          color: #495057;
        }
        
        .citation-metadata {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin-bottom: 8px;
        }
        
        .metadata-item {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 12px;
          color: #6c757d;
        }
        
        .citation-preview {
          margin-top: 12px;
          padding: 12px;
          background: #f8f9fa;
          border-radius: 4px;
          border: 1px solid #e1e5e9;
        }
        
        .image-preview {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .preview-image {
          max-width: 200px;
          max-height: 150px;
          object-fit: cover;
          border-radius: 4px;
          border: 1px solid #e1e5e9;
        }
        
        .image-description, .extracted-text {
          font-size: 13px;
          color: #666;
        }
        
        .extracted-text {
          margin-top: 8px;
        }
        
        .extracted-text strong {
          color: #333;
        }
        
        .extracted-text p {
          margin: 4px 0 0 0;
          padding: 6px;
          background: white;
          border-radius: 3px;
          max-height: 80px;
          overflow-y: auto;
        }
        
        .text-preview {
          display: flex;
          gap: 8px;
          align-items: flex-start;
        }
        
        .quote-icon {
          color: #6c757d;
          margin-top: 2px;
          flex-shrink: 0;
        }
        
        .preview-text {
          margin: 0;
          font-size: 13px;
          color: #666;
          line-height: 1.4;
          font-style: italic;
        }
        
        @media (max-width: 768px) {
          .citations-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
          }
          
          .citation-header {
            flex-direction: column;
            gap: 8px;
          }
          
          .citation-main-info {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
            margin-right: 0;
          }
          
          .citation-scores {
            margin-left: 0;
          }
          
          .citation-metadata {
            flex-direction: column;
            gap: 4px;
          }
          
          .preview-image {
            max-width: 150px;
            max-height: 100px;
          }
        }
      `}</style>
    </div>
  );
}