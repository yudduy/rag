"use client";

import React, { useEffect, useState } from 'react';
import AccessibilityStyles from './AccessibilityStyles';

interface SOTASettings {
  ttsEnabled: boolean;
  multimodalEnabled: boolean;
  highContrast: boolean;
  reducedMotion: boolean;
  keyboardNavigation: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  theme: 'light' | 'dark' | 'system';
}

interface SOTAIntegrationProps {
  children: React.ReactNode;
  settings?: Partial<SOTASettings>;
}

const DEFAULT_SETTINGS: SOTASettings = {
  ttsEnabled: false,
  multimodalEnabled: true,
  highContrast: false,
  reducedMotion: false,
  keyboardNavigation: true,
  fontSize: 'medium',
  theme: 'system'
};

/**
 * SOTA Integration component that provides global state management and 
 * accessibility features for the enhanced RAG interface
 */
export default function SOTAIntegration({ 
  children, 
  settings: initialSettings = {} 
}: SOTAIntegrationProps) {
  const [settings, setSettings] = useState<SOTASettings>({
    ...DEFAULT_SETTINGS,
    ...initialSettings
  });

  const [isLoading, setIsLoading] = useState(true);

  // Load settings from localStorage and system preferences
  useEffect(() => {
    const loadSettings = async () => {
      try {
        // Load saved settings
        const savedSettings = localStorage.getItem('rag-ui-settings');
        if (savedSettings) {
          const parsed = JSON.parse(savedSettings);
          setSettings(prev => ({ ...prev, ...parsed }));
        }

        // Detect system preferences
        const systemPreferences: Partial<SOTASettings> = {};

        // Theme preference
        if (settings.theme === 'system') {
          systemPreferences.theme = window.matchMedia('(prefers-color-scheme: dark)').matches 
            ? 'dark' : 'light';
        }

        // Motion preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
          systemPreferences.reducedMotion = true;
        }

        // Contrast preference
        if (window.matchMedia('(prefers-contrast: high)').matches) {
          systemPreferences.highContrast = true;
        }

        // Apply system preferences
        if (Object.keys(systemPreferences).length > 0) {
          setSettings(prev => ({ ...prev, ...systemPreferences }));
        }

      } catch (error) {
        console.error('Failed to load SOTA settings:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadSettings();
  }, []);

  // Apply settings to document
  useEffect(() => {
    const applySettings = () => {
      const root = document.documentElement;
      
      // Theme
      if (settings.theme === 'system') {
        root.removeAttribute('data-theme');
      } else {
        root.setAttribute('data-theme', settings.theme);
      }
      
      // High contrast
      if (settings.highContrast) {
        root.setAttribute('data-high-contrast', 'true');
      } else {
        root.removeAttribute('data-high-contrast');
      }
      
      // Reduced motion
      if (settings.reducedMotion) {
        root.setAttribute('data-reduced-motion', 'true');
      } else {
        root.removeAttribute('data-reduced-motion');
      }
      
      // Keyboard navigation
      if (settings.keyboardNavigation) {
        root.setAttribute('data-keyboard-navigation', 'true');
      } else {
        root.removeAttribute('data-keyboard-navigation');
      }
      
      // Font size
      root.setAttribute('data-font-size', settings.fontSize);
      
      // Apply color scheme
      root.style.colorScheme = settings.theme === 'dark' ? 'dark' : 'light';
    };

    if (!isLoading) {
      applySettings();
    }
  }, [settings, isLoading]);

  // Listen for system preference changes
  useEffect(() => {
    const mediaQueries = [
      {
        query: window.matchMedia('(prefers-color-scheme: dark)'),
        handler: (e: MediaQueryListEvent) => {
          if (settings.theme === 'system') {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
          }
        }
      },
      {
        query: window.matchMedia('(prefers-reduced-motion: reduce)'),
        handler: (e: MediaQueryListEvent) => {
          setSettings(prev => ({ ...prev, reducedMotion: e.matches }));
        }
      },
      {
        query: window.matchMedia('(prefers-contrast: high)'),
        handler: (e: MediaQueryListEvent) => {
          setSettings(prev => ({ ...prev, highContrast: e.matches }));
        }
      }
    ];

    // Add event listeners
    mediaQueries.forEach(({ query, handler }) => {
      query.addListener(handler);
    });

    // Cleanup
    return () => {
      mediaQueries.forEach(({ query, handler }) => {
        query.removeListener(handler);
      });
    };
  }, [settings.theme]);

  // Keyboard navigation enhancement
  useEffect(() => {
    if (!settings.keyboardNavigation) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip links navigation
      if (e.key === 'Tab' && e.shiftKey === false) {
        const skipLinks = document.querySelectorAll('.skip-link');
        if (skipLinks.length > 0 && document.activeElement === document.body) {
          e.preventDefault();
          (skipLinks[0] as HTMLElement).focus();
        }
      }

      // Escape key to close modals/overlays
      if (e.key === 'Escape') {
        const modals = document.querySelectorAll('[role="dialog"][open], .modal.open');
        if (modals.length > 0) {
          const lastModal = modals[modals.length - 1] as HTMLElement;
          const closeBtn = lastModal.querySelector('[data-close], .close-btn') as HTMLElement;
          if (closeBtn) {
            closeBtn.click();
          }
        }
      }

      // Arrow key navigation for menu items
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        const activeElement = document.activeElement;
        if (activeElement?.closest('[role="menu"], .menu')) {
          e.preventDefault();
          // Menu navigation logic would go here
        }
      }
    };

    const handleFocusVisible = (e: Event) => {
      const target = e.target as HTMLElement;
      if (target && !target.matches(':focus-visible')) {
        target.classList.add('focus-visible');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('focusin', handleFocusVisible);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('focusin', handleFocusVisible);
    };
  }, [settings.keyboardNavigation]);

  // TTS global configuration
  useEffect(() => {
    if (settings.ttsEnabled) {
      // Configure TTS globally
      window.ragTTSEnabled = true;
      
      // Check for TTS support
      if ('speechSynthesis' in window) {
        console.log('TTS enabled and supported');
      } else {
        console.warn('TTS enabled but not supported by browser');
      }
    } else {
      window.ragTTSEnabled = false;
      
      // Stop any ongoing speech
      if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
      }
    }
  }, [settings.ttsEnabled]);

  // Provide skip links for screen readers
  const renderSkipLinks = () => (
    <nav aria-label="Skip links" className="skip-navigation">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <a href="#navigation" className="skip-link">
        Skip to navigation
      </a>
      <a href="#search" className="skip-link">
        Skip to search
      </a>
    </nav>
  );

  // Loading state with accessibility
  if (isLoading) {
    return (
      <div 
        className="loading-container"
        role="status"
        aria-live="polite"
        aria-label="Loading application"
      >
        <AccessibilityStyles />
        <div className="loading-spinner"></div>
        <span className="sr-only">Loading SOTA RAG Assistant...</span>
        
        <style jsx>{`
          .loading-container {
            position: fixed;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: var(--background-primary, #ffffff);
            color: var(--text-primary, #333333);
            z-index: 9999;
          }
          
          .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e1e5e9;
            border-top-color: #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
          }
          
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
          
          @media (prefers-reduced-motion: reduce) {
            .loading-spinner {
              animation: none;
              border-top-color: #007bff;
              opacity: 0.8;
            }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="sota-integration">
      <AccessibilityStyles />
      {renderSkipLinks()}
      
      {/* Main application content */}
      <main 
        id="main-content" 
        role="main"
        aria-label="SOTA RAG Assistant"
        className="custom-scrollbar content-spacing"
      >
        {children}
      </main>

      {/* Global announcements for screen readers */}
      <div 
        id="announcements" 
        aria-live="polite" 
        aria-atomic="false" 
        className="sr-only"
        role="status"
      />

      {/* Global error boundary */}
      <div 
        id="error-boundary" 
        aria-live="assertive" 
        className="sr-only"
        role="alert"
      />

      <style jsx>{`
        .sota-integration {
          min-height: 100vh;
          background: var(--background-primary, #ffffff);
          color: var(--text-primary, #333333);
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
                       'Helvetica Neue', Arial, sans-serif;
          line-height: 1.6;
        }
        
        .skip-navigation {
          position: absolute;
          top: 0;
          left: 0;
          z-index: 10000;
        }
        
        main {
          outline: none;
          min-height: calc(100vh - 60px);
          padding: 0;
        }
        
        /* Focus management for modals and overlays */
        :global(.modal-open) {
          overflow: hidden;
        }
        
        :global(.modal-open main) {
          filter: blur(2px);
          pointer-events: none;
        }
        
        /* Enhanced button styles */
        :global(button:disabled) {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        :global(.btn-group) {
          display: flex;
          gap: 4px;
        }
        
        :global(.btn-group button:first-child) {
          border-top-right-radius: 0;
          border-bottom-right-radius: 0;
        }
        
        :global(.btn-group button:last-child) {
          border-top-left-radius: 0;
          border-bottom-left-radius: 0;
        }
        
        :global(.btn-group button:not(:first-child):not(:last-child)) {
          border-radius: 0;
        }
        
        /* Status indicators */
        :global(.status-indicator) {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }
        
        :global(.status-online) {
          background: #d4edda;
          color: #155724;
        }
        
        :global(.status-offline) {
          background: #f8d7da;
          color: #721c24;
        }
        
        :global(.status-loading) {
          background: #fff3cd;
          color: #856404;
        }
        
        /* Progress indicators */
        :global(.progress) {
          height: 4px;
          background: #e9ecef;
          border-radius: 2px;
          overflow: hidden;
        }
        
        :global(.progress-bar) {
          height: 100%;
          background: #007bff;
          transition: width 0.3s ease;
        }
        
        :global(.progress-indeterminate .progress-bar) {
          width: 30%;
          animation: progress-indeterminate 2s infinite;
        }
        
        @keyframes progress-indeterminate {
          0% { margin-left: -30%; }
          100% { margin-left: 100%; }
        }
        
        /* Responsive utilities */
        @media (max-width: 768px) {
          .sota-integration {
            font-size: 16px; /* Prevent zoom on iOS */
          }
          
          :global(.hide-mobile) {
            display: none !important;
          }
          
          :global(.mobile-only) {
            display: block !important;
          }
        }
        
        @media (min-width: 769px) {
          :global(.hide-desktop) {
            display: none !important;
          }
          
          :global(.desktop-only) {
            display: block !important;
          }
        }
        
        /* Dark theme specific adjustments */
        :global([data-theme="dark"]) {
          background: #1a202c;
          color: #f7fafc;
        }
        
        :global([data-theme="dark"] .sota-integration) {
          background: #1a202c;
          color: #f7fafc;
        }
        
        :global([data-theme="dark"] input),
        :global([data-theme="dark"] select),
        :global([data-theme="dark"] textarea) {
          background: #2d3748;
          border-color: #4a5568;
          color: #f7fafc;
        }
        
        :global([data-theme="dark"] button) {
          background: #4a5568;
          border-color: #718096;
          color: #f7fafc;
        }
        
        :global([data-theme="dark"] button:hover) {
          background: #718096;
        }
      `}</style>
    </div>
  );
}

// Type declarations for global variables
declare global {
  interface Window {
    ragTTSEnabled?: boolean;
    ragMultimodalEnabled?: boolean;
    ragAccessibilityMode?: string;
  }
}