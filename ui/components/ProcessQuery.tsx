import React, { useState, useEffect } from 'react';
import TTSControls from './TTSControls';
import MultimodalDisplay from './MultimodalDisplay';
import ResponseQuality from './ResponseQuality';
import EnhancedCitations from './EnhancedCitations';

interface ProcessQueryStepProps {
  event: {
    input?: string;
    output?: string;
    status?: string;
    timestamp?: string;
    // Enhanced SOTA features
    quality_metrics?: {
      confidence_score?: number;
      verification_status?: 'verified' | 'partial' | 'unverified' | 'failed';
      response_time?: number;
      cache_hit?: boolean;
      sources_found?: number;
      relevance_score?: number;
      factual_consistency?: number;
      hallucination_risk?: 'low' | 'medium' | 'high';
      quality_warnings?: string[];
      performance_metrics?: {
        retrieval_time?: number;
        generation_time?: number;
        total_tokens?: number;
        sources_retrieved?: number;
      };
    };
    images?: Array<{
      id: string;
      url: string;
      title?: string;
      description?: string;
      metadata?: {
        width?: number;
        height?: number;
        size?: number;
        format?: string;
        citation_id?: string;
        confidence_score?: number;
        extracted_text?: string;
      };
    }>;
    citations?: Array<{
      id: string;
      type: 'text' | 'image' | 'document' | 'web';
      title?: string;
      content?: string;
      url?: string;
      image_path?: string;
      metadata?: any;
    }>;
    tts_available?: boolean;
  };
}

export default function ProcessQueryStep({ event }: ProcessQueryStepProps) {
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [showEnhancedFeatures, setShowEnhancedFeatures] = useState(true);

  // Load TTS preference from localStorage
  useEffect(() => {
    const settings = localStorage.getItem('rag-ui-settings');
    if (settings) {
      try {
        const parsed = JSON.parse(settings);
        setTtsEnabled(parsed.ttsEnabled || false);
        setShowEnhancedFeatures(parsed.showConfidenceScores !== false);
      } catch (error) {
        console.error('Failed to load TTS settings:', error);
      }
    }
  }, []);

  return (
    <div className="event-container process-query-step">
      <div className="event-header">
        <h4>🔍 Processing Query</h4>
        <span className="timestamp">{event.timestamp}</span>
      </div>
      <div className="event-content">
        <div className="step-info">
          <p><strong>Step:</strong> process_query</p>
          <p><strong>Status:</strong> {event.status || 'Running'}</p>
          {event.input && <p><strong>Input:</strong> {event.input}</p>}
          {event.output && (
            <div className="output">
              <strong>Output:</strong>
              <div className="output-text">{event.output}</div>
              
              {/* Enhanced Response Features */}
              {event.output && showEnhancedFeatures && (
                <div className="response-enhancements">
                  {/* TTS Controls */}
                  {(event.tts_available || ttsEnabled) && (
                    <TTSControls
                      text={event.output}
                      enabled={ttsEnabled}
                      onToggleEnabled={setTtsEnabled}
                      className="response-tts"
                    />
                  )}
                  
                  {/* Quality Metrics */}
                  {event.quality_metrics && (
                    <ResponseQuality
                      metrics={event.quality_metrics}
                      className="response-quality"
                    />
                  )}
                  
                  {/* Multimodal Display */}
                  {event.images && event.images.length > 0 && (
                    <MultimodalDisplay
                      images={event.images}
                      className="response-images"
                      showCitations={true}
                    />
                  )}
                  
                  {/* Enhanced Citations */}
                  {event.citations && event.citations.length > 0 && (
                    <EnhancedCitations
                      citations={event.citations}
                      responseText={event.output}
                      className="response-citations"
                      showPreview={true}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <style jsx>{`
        .event-container {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        .process-query-step {
          border-left: 4px solid #6f42c1;
        }
        .event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .event-header h4 {
          margin: 0;
          color: #6f42c1;
          font-size: 16px;
        }
        .timestamp {
          font-size: 12px;
          color: #666;
        }
        .step-info p {
          margin: 4px 0;
          color: #333;
        }
        .output {
          margin: 8px 0;
        }
        .output-text {
          background: white;
          padding: 12px;
          border-radius: 4px;
          margin-top: 4px;
          border: 1px solid #e1e5e9;
          max-height: 200px;
          overflow-y: auto;
        }
        .response-enhancements {
          margin-top: 16px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .response-tts,
        .response-quality,
        .response-images,
        .response-citations {
          border-radius: 6px;
          overflow: hidden;
        }
        
        @media (max-width: 768px) {
          .event-container {
            padding: 12px;
          }
          .response-enhancements {
            margin-top: 12px;
            gap: 8px;
          }
        }
      `}</style>
    </div>
  );
}