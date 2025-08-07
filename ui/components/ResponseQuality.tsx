"use client";

import React, { useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  XCircle, 
  TrendingUp,
  BarChart3,
  Clock,
  Target,
  ChevronDown,
  ChevronUp,
  Shield,
  Zap
} from 'lucide-react';

interface QualityMetrics {
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
}

interface ResponseQualityProps {
  metrics: QualityMetrics;
  className?: string;
  showDetails?: boolean;
  compact?: boolean;
}

export default function ResponseQuality({ 
  metrics, 
  className = "", 
  showDetails = false,
  compact = false 
}: ResponseQualityProps) {
  const [isExpanded, setIsExpanded] = useState(showDetails);

  // Calculate overall quality score
  const calculateOverallQuality = (): number => {
    const scores = [
      metrics.confidence_score,
      metrics.relevance_score,
      metrics.factual_consistency
    ].filter(score => score !== undefined) as number[];
    
    if (scores.length === 0) return 0;
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  };

  const overallQuality = calculateOverallQuality();

  // Get quality level and color
  const getQualityLevel = (score: number): { level: string; color: string; icon: React.ReactElement } => {
    if (score >= 0.8) {
      return { 
        level: 'High', 
        color: '#28a745', 
        icon: <CheckCircle className="size-4" /> 
      };
    } else if (score >= 0.6) {
      return { 
        level: 'Medium', 
        color: '#ffc107', 
        icon: <Info className="size-4" /> 
      };
    } else if (score >= 0.3) {
      return { 
        level: 'Low', 
        color: '#fd7e14', 
        icon: <AlertTriangle className="size-4" /> 
      };
    } else {
      return { 
        level: 'Very Low', 
        color: '#dc3545', 
        icon: <XCircle className="size-4" /> 
      };
    }
  };

  const qualityInfo = getQualityLevel(overallQuality);

  // Get verification status info
  const getVerificationInfo = (status?: string) => {
    switch (status) {
      case 'verified':
        return { color: '#28a745', icon: <Shield className="size-4" />, text: 'Verified' };
      case 'partial':
        return { color: '#ffc107', icon: <AlertTriangle className="size-4" />, text: 'Partially Verified' };
      case 'failed':
        return { color: '#dc3545', icon: <XCircle className="size-4" />, text: 'Verification Failed' };
      default:
        return { color: '#6c757d', icon: <Info className="size-4" />, text: 'Unverified' };
    }
  };

  const verificationInfo = getVerificationInfo(metrics.verification_status);

  // Get hallucination risk info
  const getHallucinationRiskInfo = (risk?: string) => {
    switch (risk) {
      case 'low':
        return { color: '#28a745', text: 'Low Risk', icon: <CheckCircle className="size-4" /> };
      case 'medium':
        return { color: '#ffc107', text: 'Medium Risk', icon: <AlertTriangle className="size-4" /> };
      case 'high':
        return { color: '#dc3545', text: 'High Risk', icon: <XCircle className="size-4" /> };
      default:
        return { color: '#6c757d', text: 'Unknown', icon: <Info className="size-4" /> };
    }
  };

  const hallucinationInfo = getHallucinationRiskInfo(metrics.hallucination_risk);

  // Format time in milliseconds
  const formatTime = (ms?: number): string => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format percentage
  const formatPercentage = (score?: number): string => {
    if (score === undefined) return 'N/A';
    return `${Math.round(score * 100)}%`;
  };

  if (compact) {
    return (
      <div className={`response-quality compact ${className}`}>
        <div className="quality-summary">
          <div className="quality-indicator" style={{ color: qualityInfo.color }}>
            {qualityInfo.icon}
            <span>{qualityInfo.level}</span>
          </div>
          
          {metrics.cache_hit && (
            <div className="cache-indicator" title="Response served from cache">
              <Zap className="size-4" />
              <span>Cached</span>
            </div>
          )}
          
          {metrics.response_time && (
            <div className="time-indicator">
              <Clock className="size-4" />
              <span>{formatTime(metrics.response_time)}</span>
            </div>
          )}
        </div>
        
        <style jsx>{`
          .response-quality.compact {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 8px;
            background: #f8f9fa;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            font-size: 12px;
          }
          
          .quality-summary {
            display: flex;
            align-items: center;
            gap: 8px;
          }
          
          .quality-indicator, .cache-indicator, .time-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            color: #6c757d;
          }
          
          .cache-indicator {
            color: #007bff;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className={`response-quality ${className}`}>
      <div className="quality-header">
        <div className="main-metrics">
          <div className="overall-quality">
            <div className="quality-score" style={{ color: qualityInfo.color }}>
              {qualityInfo.icon}
              <span className="score-text">
                Quality: {qualityInfo.level} ({formatPercentage(overallQuality)})
              </span>
            </div>
            
            <div className="verification-status" style={{ color: verificationInfo.color }}>
              {verificationInfo.icon}
              <span>{verificationInfo.text}</span>
            </div>
          </div>
          
          <div className="performance-summary">
            {metrics.cache_hit && (
              <div className="cache-badge" title="Response served from cache">
                <Zap className="size-4" />
                <span>Cached</span>
              </div>
            )}
            
            {metrics.response_time && (
              <div className="response-time">
                <Clock className="size-4" />
                <span>{formatTime(metrics.response_time)}</span>
              </div>
            )}
            
            {metrics.sources_found !== undefined && (
              <div className="sources-count">
                <BarChart3 className="size-4" />
                <span>{metrics.sources_found} sources</span>
              </div>
            )}
          </div>
        </div>
        
        <button
          className="expand-toggle"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-label={isExpanded ? "Hide details" : "Show details"}
          title={isExpanded ? "Hide details" : "Show details"}
        >
          {isExpanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
        </button>
      </div>

      {/* Quality Warnings */}
      {metrics.quality_warnings && metrics.quality_warnings.length > 0 && (
        <div className="quality-warnings">
          <div className="warnings-header">
            <AlertTriangle className="size-4" style={{ color: '#ffc107' }} />
            <span>Quality Warnings</span>
          </div>
          <ul className="warnings-list">
            {metrics.quality_warnings.map((warning, index) => (
              <li key={index} className="warning-item">
                {warning}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Detailed Metrics */}
      {isExpanded && (
        <div className="detailed-metrics">
          <div className="metrics-grid">
            {metrics.confidence_score !== undefined && (
              <div className="metric-item">
                <div className="metric-header">
                  <Target className="size-4" />
                  <span>Confidence</span>
                </div>
                <div className="metric-value">
                  <div 
                    className="metric-bar"
                    style={{ 
                      width: `${metrics.confidence_score * 100}%`,
                      backgroundColor: getQualityLevel(metrics.confidence_score).color
                    }}
                  />
                  <span>{formatPercentage(metrics.confidence_score)}</span>
                </div>
              </div>
            )}

            {metrics.relevance_score !== undefined && (
              <div className="metric-item">
                <div className="metric-header">
                  <TrendingUp className="size-4" />
                  <span>Relevance</span>
                </div>
                <div className="metric-value">
                  <div 
                    className="metric-bar"
                    style={{ 
                      width: `${metrics.relevance_score * 100}%`,
                      backgroundColor: getQualityLevel(metrics.relevance_score).color
                    }}
                  />
                  <span>{formatPercentage(metrics.relevance_score)}</span>
                </div>
              </div>
            )}

            {metrics.factual_consistency !== undefined && (
              <div className="metric-item">
                <div className="metric-header">
                  <CheckCircle className="size-4" />
                  <span>Factual Consistency</span>
                </div>
                <div className="metric-value">
                  <div 
                    className="metric-bar"
                    style={{ 
                      width: `${metrics.factual_consistency * 100}%`,
                      backgroundColor: getQualityLevel(metrics.factual_consistency).color
                    }}
                  />
                  <span>{formatPercentage(metrics.factual_consistency)}</span>
                </div>
              </div>
            )}

            {metrics.hallucination_risk && (
              <div className="metric-item">
                <div className="metric-header">
                  {hallucinationInfo.icon}
                  <span>Hallucination Risk</span>
                </div>
                <div className="riREDACTED_SK_KEY" style={{ color: hallucinationInfo.color }}>
                  {hallucinationInfo.text}
                </div>
              </div>
            )}
          </div>

          {/* Performance Details */}
          {metrics.performance_metrics && (
            <div className="performance-details">
              <h4>Performance Metrics</h4>
              <div className="performance-grid">
                {metrics.performance_metrics.retrieval_time !== undefined && (
                  <div className="perf-metric">
                    <span className="perf-label">Retrieval Time:</span>
                    <span className="perf-value">{formatTime(metrics.performance_metrics.retrieval_time)}</span>
                  </div>
                )}
                
                {metrics.performance_metrics.generation_time !== undefined && (
                  <div className="perf-metric">
                    <span className="perf-label">Generation Time:</span>
                    <span className="perf-value">{formatTime(metrics.performance_metrics.generation_time)}</span>
                  </div>
                )}
                
                {metrics.performance_metrics.sources_retrieved !== undefined && (
                  <div className="perf-metric">
                    <span className="perf-label">Sources Retrieved:</span>
                    <span className="perf-value">{metrics.performance_metrics.sources_retrieved}</span>
                  </div>
                )}
                
                {metrics.performance_metrics.total_tokens !== undefined && (
                  <div className="perf-metric">
                    <span className="perf-label">Total Tokens:</span>
                    <span className="perf-value">{metrics.performance_metrics.total_tokens.toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .response-quality {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 12px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        
        .quality-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
        }
        
        .main-metrics {
          flex: 1;
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 12px;
        }
        
        .overall-quality {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        
        .quality-score {
          display: flex;
          align-items: center;
          gap: 6px;
          font-weight: 600;
        }
        
        .verification-status {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 14px;
        }
        
        .performance-summary {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 13px;
          color: #6c757d;
        }
        
        .cache-badge {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 2px 6px;
          background: #e3f2fd;
          color: #007bff;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }
        
        .response-time, .sources-count {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .expand-toggle {
          padding: 6px;
          border: 1px solid #ddd;
          border-radius: 6px;
          background: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .expand-toggle:hover {
          background: #f8f9fa;
        }
        
        .quality-warnings {
          margin-top: 12px;
          padding: 8px 12px;
          background: #fff3cd;
          border: 1px solid #ffeaa7;
          border-radius: 6px;
        }
        
        .warnings-header {
          display: flex;
          align-items: center;
          gap: 6px;
          font-weight: 600;
          margin-bottom: 6px;
        }
        
        .warnings-list {
          margin: 0;
          padding: 0 0 0 20px;
        }
        
        .warning-item {
          font-size: 13px;
          color: #856404;
          margin-bottom: 2px;
        }
        
        .detailed-metrics {
          margin-top: 16px;
          padding-top: 12px;
          border-top: 1px solid #e1e5e9;
        }
        
        .metrics-grid {
          display: grid;
          gap: 12px;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        }
        
        .metric-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        
        .metric-header {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 500;
          color: #495057;
        }
        
        .metric-value {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .metric-bar {
          height: 4px;
          background: #e1e5e9;
          border-radius: 2px;
          position: relative;
          flex: 1;
        }
        
        .riREDACTED_SK_KEY {
          font-weight: 500;
          font-size: 13px;
        }
        
        .performance-details {
          margin-top: 16px;
        }
        
        .performance-details h4 {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #495057;
        }
        
        .performance-grid {
          display: grid;
          gap: 8px;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }
        
        .perf-metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 13px;
        }
        
        .perf-label {
          color: #6c757d;
        }
        
        .perf-value {
          color: #495057;
          font-weight: 500;
        }
        
        @media (max-width: 768px) {
          .main-metrics {
            flex-direction: column;
            align-items: flex-start;
          }
          
          .overall-quality {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
          }
          
          .performance-summary {
            flex-wrap: wrap;
          }
          
          .metrics-grid {
            grid-template-columns: 1fr;
          }
          
          .performance-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}