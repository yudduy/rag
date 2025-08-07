"use client";

import React, { useState, useEffect, useRef } from 'react';
import { 
  ZoomIn, 
  ZoomOut, 
  Download, 
  ExternalLink, 
  X, 
  ChevronLeft, 
  ChevronRight,
  Image as ImageIcon,
  Eye,
  Grid,
  Maximize2,
  Info
} from 'lucide-react';

interface ImageSource {
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
}

interface MultimodalDisplayProps {
  images: ImageSource[];
  className?: string;
  showCitations?: boolean;
  maxImages?: number;
  layout?: 'grid' | 'carousel' | 'masonry';
}

export default function MultimodalDisplay({ 
  images = [], 
  className = "", 
  showCitations = true,
  maxImages = 20,
  layout = 'grid'
}: MultimodalDisplayProps) {
  const [selectedImage, setSelectedImage] = useState<ImageSource | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isZooming, setIsZooming] = useState(false);
  
  const lightboxRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const displayImages = images.slice(0, maxImages);

  // Handle keyboard navigation in lightbox
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isLightboxOpen) return;

      switch (event.key) {
        case 'Escape':
          closeLightbox();
          break;
        case 'ArrowLeft':
          navigateImage(-1);
          break;
        case 'ArrowRight':
          navigateImage(1);
          break;
        case '+':
        case '=':
          event.preventDefault();
          zoomIn();
          break;
        case '-':
          event.preventDefault();
          zoomOut();
          break;
        case '0':
          event.preventDefault();
          resetZoom();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isLightboxOpen, currentIndex, zoomLevel]);

  // Handle image loading states
  const handleImageLoad = (imageId: string) => {
    setLoadedImages(prev => new Set(prev).add(imageId));
  };

  const handleImageError = (imageId: string) => {
    setFailedImages(prev => new Set(prev).add(imageId));
  };

  // Lightbox navigation
  const openLightbox = (image: ImageSource, index: number) => {
    setSelectedImage(image);
    setCurrentIndex(index);
    setIsLightboxOpen(true);
    setZoomLevel(1);
    document.body.style.overflow = 'hidden';
  };

  const closeLightbox = () => {
    setIsLightboxOpen(false);
    setSelectedImage(null);
    setZoomLevel(1);
    document.body.style.overflow = '';
  };

  const navigateImage = (direction: number) => {
    if (displayImages.length === 0) return;
    
    const newIndex = (currentIndex + direction + displayImages.length) % displayImages.length;
    setCurrentIndex(newIndex);
    setSelectedImage(displayImages[newIndex]);
    setZoomLevel(1);
  };

  // Zoom controls
  const zoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 5));
  };

  const zoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.1));
  };

  const resetZoom = () => {
    setZoomLevel(1);
  };

  // Download image
  const downloadImage = async (image: ImageSource) => {
    try {
      const response = await fetch(image.url);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = image.title || `image-${image.id}.${image.metadata?.format || 'jpg'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download image:', error);
    }
  };

  // Format file size
  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  // Get confidence color
  const getConfidenceColor = (score?: number): string => {
    if (!score) return '#6c757d';
    if (score >= 0.8) return '#28a745';
    if (score >= 0.6) return '#ffc107';
    return '#dc3545';
  };

  const renderImageCard = (image: ImageSource, index: number) => {
    const isLoaded = loadedImages.has(image.id);
    const hasFailed = failedImages.has(image.id);

    return (
      <div 
        key={image.id} 
        className={`image-card ${viewMode}`}
        onClick={() => openLightbox(image, index)}
      >
        <div className="image-container">
          {!isLoaded && !hasFailed && (
            <div className="image-loading">
              <div className="loading-spinner"></div>
              <span>Loading...</span>
            </div>
          )}
          
          {hasFailed ? (
            <div className="image-failed">
              <ImageIcon className="size-8" />
              <span>Failed to load</span>
            </div>
          ) : (
            <img
              src={image.url}
              alt={image.title || image.description || 'Retrieved image'}
              onLoad={() => handleImageLoad(image.id)}
              onError={() => handleImageError(image.id)}
              className={`image ${!isLoaded ? 'loading' : ''}`}
              loading="lazy"
            />
          )}
          
          <div className="image-overlay">
            <div className="overlay-controls">
              <button
                className="overlay-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  downloadImage(image);
                }}
                aria-label="Download image"
                title="Download"
              >
                <Download className="size-4" />
              </button>
              <button
                className="overlay-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  openLightbox(image, index);
                }}
                aria-label="View full size"
                title="View full size"
              >
                <Maximize2 className="size-4" />
              </button>
            </div>
            
            {image.metadata?.confidence_score && (
              <div 
                className="confidence-badge"
                style={{ backgroundColor: getConfidenceColor(image.metadata.confidence_score) }}
                title={`Relevance: ${Math.round(image.metadata.confidence_score * 100)}%`}
              >
                {Math.round(image.metadata.confidence_score * 100)}%
              </div>
            )}
          </div>
        </div>
        
        <div className="image-info">
          {image.title && <h4 className="image-title">{image.title}</h4>}
          {image.description && <p className="image-description">{image.description}</p>}
          
          <div className="image-metadata">
            {image.metadata?.width && image.metadata?.height && (
              <span className="metadata-item">
                {image.metadata.width} × {image.metadata.height}
              </span>
            )}
            {image.metadata?.size && (
              <span className="metadata-item">
                {formatFileSize(image.metadata.size)}
              </span>
            )}
            {showCitations && image.metadata?.citation_id && (
              <span className="citation-badge">
                Citation: {image.metadata.citation_id}
              </span>
            )}
          </div>
          
          {image.metadata?.extracted_text && (
            <details className="extracted-text">
              <summary>Extracted Text</summary>
              <p>{image.metadata.extracted_text}</p>
            </details>
          )}
        </div>
      </div>
    );
  };

  if (displayImages.length === 0) {
    return null;
  }

  return (
    <div className={`multimodal-display ${className}`}>
      <div className="display-header">
        <div className="header-info">
          <ImageIcon className="size-4" />
          <span>{displayImages.length} image{displayImages.length !== 1 ? 's' : ''} retrieved</span>
          {images.length > maxImages && (
            <span className="overflow-indicator">
              (+{images.length - maxImages} more)
            </span>
          )}
        </div>
        
        <div className="display-controls">
          <button
            className={`view-toggle ${viewMode === 'grid' ? 'active' : ''}`}
            onClick={() => setViewMode('grid')}
            aria-label="Grid view"
            title="Grid view"
          >
            <Grid className="size-4" />
          </button>
          <button
            className={`view-toggle ${viewMode === 'list' ? 'active' : ''}`}
            onClick={() => setViewMode('list')}
            aria-label="List view"
            title="List view"
          >
            <Info className="size-4" />
          </button>
        </div>
      </div>

      <div className={`images-container ${layout} ${viewMode}-view`}>
        {displayImages.map((image, index) => renderImageCard(image, index))}
      </div>

      {/* Lightbox Modal */}
      {isLightboxOpen && selectedImage && (
        <div 
          className="lightbox-overlay"
          ref={lightboxRef}
          onClick={(e) => {
            if (e.target === lightboxRef.current) {
              closeLightbox();
            }
          }}
        >
          <div className="lightbox-content">
            <div className="lightbox-header">
              <div className="lightbox-info">
                <span>{currentIndex + 1} of {displayImages.length}</span>
                {selectedImage.title && <span>{selectedImage.title}</span>}
              </div>
              
              <div className="lightbox-controls">
                <button onClick={zoomOut} disabled={zoomLevel <= 0.1}>
                  <ZoomOut className="size-4" />
                </button>
                <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
                <button onClick={zoomIn} disabled={zoomLevel >= 5}>
                  <ZoomIn className="size-4" />
                </button>
                <button onClick={() => downloadImage(selectedImage)}>
                  <Download className="size-4" />
                </button>
                <button onClick={closeLightbox}>
                  <X className="size-4" />
                </button>
              </div>
            </div>
            
            <div className="lightbox-image-container">
              {displayImages.length > 1 && (
                <>
                  <button 
                    className="nav-btn prev"
                    onClick={() => navigateImage(-1)}
                    aria-label="Previous image"
                  >
                    <ChevronLeft className="size-6" />
                  </button>
                  <button 
                    className="nav-btn next"
                    onClick={() => navigateImage(1)}
                    aria-label="Next image"
                  >
                    <ChevronRight className="size-6" />
                  </button>
                </>
              )}
              
              <img
                ref={imageRef}
                src={selectedImage.url}
                alt={selectedImage.title || selectedImage.description || 'Full size image'}
                className="lightbox-image"
                style={{ 
                  transform: `scale(${zoomLevel})`,
                  transition: isZooming ? 'none' : 'transform 0.2s ease'
                }}
                onWheel={(e) => {
                  e.preventDefault();
                  if (e.deltaY < 0) {
                    zoomIn();
                  } else {
                    zoomOut();
                  }
                }}
              />
            </div>
            
            {selectedImage.description && (
              <div className="lightbox-description">
                <p>{selectedImage.description}</p>
              </div>
            )}
          </div>
        </div>
      )}

      <style jsx>{`
        .multimodal-display {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 16px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        
        .display-header {
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
        }
        
        .overflow-indicator {
          color: #6c757d;
          font-size: 0.9em;
        }
        
        .display-controls {
          display: flex;
          gap: 4px;
        }
        
        .view-toggle {
          padding: 6px 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .view-toggle:hover {
          background: #f8f9fa;
        }
        
        .view-toggle.active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }
        
        .images-container {
          display: grid;
          gap: 16px;
        }
        
        .images-container.grid-view {
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        }
        
        .images-container.list-view {
          grid-template-columns: 1fr;
        }
        
        .image-card {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          overflow: hidden;
          background: white;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .image-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .image-container {
          position: relative;
          aspect-ratio: 16/10;
          overflow: hidden;
        }
        
        .image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: opacity 0.3s ease;
        }
        
        .image.loading {
          opacity: 0;
        }
        
        .image-loading, .image-failed {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: #6c757d;
          background: #f8f9fa;
        }
        
        .loading-spinner {
          width: 24px;
          height: 24px;
          border: 2px solid #e1e5e9;
          border-top-color: #007bff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 8px;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .image-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0,0,0,0.5);
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          padding: 8px;
          opacity: 0;
          transition: opacity 0.2s ease;
        }
        
        .image-card:hover .image-overlay {
          opacity: 1;
        }
        
        .overlay-controls {
          display: flex;
          gap: 4px;
        }
        
        .overlay-btn {
          padding: 6px;
          border: none;
          border-radius: 4px;
          background: rgba(255,255,255,0.9);
          color: #333;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .overlay-btn:hover {
          background: white;
        }
        
        .confidence-badge {
          padding: 2px 6px;
          border-radius: 12px;
          color: white;
          font-size: 12px;
          font-weight: 500;
        }
        
        .image-info {
          padding: 12px;
        }
        
        .image-title {
          margin: 0 0 4px 0;
          font-size: 14px;
          font-weight: 600;
          color: #333;
        }
        
        .image-description {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: #666;
          line-height: 1.4;
        }
        
        .image-metadata {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 8px;
        }
        
        .metadata-item {
          font-size: 12px;
          color: #6c757d;
          padding: 2px 6px;
          background: #f1f3f4;
          border-radius: 4px;
        }
        
        .citation-badge {
          font-size: 12px;
          color: #007bff;
          padding: 2px 6px;
          background: #e3f2fd;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .extracted-text {
          margin-top: 8px;
        }
        
        .extracted-text summary {
          font-size: 12px;
          color: #007bff;
          cursor: pointer;
        }
        
        .extracted-text p {
          font-size: 12px;
          color: #666;
          margin-top: 4px;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 4px;
          max-height: 100px;
          overflow-y: auto;
        }
        
        /* Lightbox Styles */
        .lightbox-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }
        
        .lightbox-content {
          display: flex;
          flex-direction: column;
          max-width: 90vw;
          max-height: 90vh;
        }
        
        .lightbox-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px;
          background: rgba(0,0,0,0.8);
          color: white;
          border-radius: 8px 8px 0 0;
        }
        
        .lightbox-info {
          display: flex;
          gap: 16px;
          font-size: 14px;
        }
        
        .lightbox-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .lightbox-controls button {
          padding: 8px;
          border: none;
          border-radius: 4px;
          background: rgba(255,255,255,0.2);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .lightbox-controls button:hover {
          background: rgba(255,255,255,0.3);
        }
        
        .lightbox-controls button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .zoom-level {
          font-size: 12px;
          color: #ccc;
          min-width: 40px;
          text-align: center;
        }
        
        .lightbox-image-container {
          position: relative;
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          background: rgba(0,0,0,0.8);
        }
        
        .lightbox-image {
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
          transform-origin: center;
        }
        
        .nav-btn {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          padding: 16px;
          border: none;
          border-radius: 50%;
          background: rgba(0,0,0,0.6);
          color: white;
          cursor: pointer;
          z-index: 1;
        }
        
        .nav-btn:hover {
          background: rgba(0,0,0,0.8);
        }
        
        .nav-btn.prev {
          left: 20px;
        }
        
        .nav-btn.next {
          right: 20px;
        }
        
        .lightbox-description {
          padding: 16px;
          background: rgba(0,0,0,0.8);
          color: white;
          border-radius: 0 0 8px 8px;
          max-height: 100px;
          overflow-y: auto;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
          .images-container.grid-view {
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          }
          
          .display-header {
            flex-direction: column;
            gap: 12px;
            align-items: flex-start;
          }
          
          .lightbox-overlay {
            padding: 10px;
          }
          
          .lightbox-header {
            padding: 12px;
          }
          
          .lightbox-info {
            flex-direction: column;
            gap: 4px;
          }
          
          .nav-btn {
            padding: 12px;
          }
          
          .nav-btn.prev {
            left: 10px;
          }
          
          .nav-btn.next {
            right: 10px;
          }
        }
        
        @media (max-width: 480px) {
          .images-container.grid-view {
            grid-template-columns: 1fr;
          }
          
          .lightbox-controls {
            gap: 4px;
          }
          
          .lightbox-controls button {
            padding: 6px;
          }
        }
      `}</style>
    </div>
  );
}