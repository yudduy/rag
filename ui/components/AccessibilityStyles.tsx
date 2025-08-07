"use client";

import React from 'react';

/**
 * Global accessibility styles component that provides WCAG 2.1 AA compliant styles
 * and enhanced accessibility features across the entire application.
 */
export default function AccessibilityStyles() {
  return (
    <style jsx global>{`
      /* WCAG 2.1 AA Compliance Styles */
      
      /* High contrast mode support */
      @media (prefers-contrast: high), [data-high-contrast="true"] {
        :root {
          --text-primary: #000000;
          --text-secondary: #000000;
          --background-primary: #ffffff;
          --background-secondary: #f0f0f0;
          --border-color: #000000;
          --focus-color: #0000ff;
          --link-color: #0000ff;
          --link-visited: #800080;
        }
        
        * {
          border-color: var(--border-color) !important;
          color: var(--text-primary) !important;
        }
        
        button, .btn, input, select, textarea {
          border: 2px solid var(--border-color) !important;
          background: var(--background-primary) !important;
          color: var(--text-primary) !important;
        }
        
        button:hover, .btn:hover {
          background: var(--background-secondary) !important;
          border-width: 3px !important;
        }
        
        a, .link {
          color: var(--link-color) !important;
          text-decoration: underline !important;
        }
        
        a:visited, .link:visited {
          color: var(--link-visited) !important;
        }
        
        .image-placeholder, .loading-placeholder {
          background: repeating-linear-gradient(
            45deg,
            #000,
            #000 2px,
            #fff 2px,
            #fff 8px
          ) !important;
        }
      }
      
      /* Reduced motion support */
      @media (prefers-reduced-motion: reduce), [data-reduced-motion="true"] {
        *,
        *::before,
        *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
          scroll-behavior: auto !important;
        }
        
        .loading-spinner {
          animation: none !important;
        }
        
        .fade-in, .slide-in, .zoom-in {
          opacity: 1 !important;
          transform: none !important;
        }
        
        .parallax {
          transform: none !important;
        }
      }
      
      /* Enhanced focus indicators for keyboard navigation */
      [data-keyboard-navigation="true"] *:focus,
      *:focus-visible {
        outline: 3px solid #005fcc !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 5px rgba(0, 95, 204, 0.1) !important;
        border-radius: 4px;
      }
      
      /* Skip links for screen readers */
      .skip-link {
        position: absolute;
        top: -40px;
        left: 6px;
        background: #000;
        color: #fff;
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        z-index: 9999;
        font-size: 14px;
      }
      
      .skip-link:focus {
        top: 6px;
        outline: 2px solid #fff;
        outline-offset: 2px;
      }
      
      /* Screen reader only content */
      .sr-only, .visually-hidden {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }
      
      .sr-only:focus, .visually-hidden:focus {
        position: static;
        width: auto;
        height: auto;
        padding: inherit;
        margin: inherit;
        overflow: visible;
        clip: auto;
        white-space: normal;
      }
      
      /* Enhanced font size support */
      [data-font-size="small"] {
        font-size: 14px !important;
      }
      
      [data-font-size="medium"] {
        font-size: 16px !important;
      }
      
      [data-font-size="large"] {
        font-size: 18px !important;
      }
      
      [data-font-size="extra-large"] {
        font-size: 20px !important;
      }
      
      /* Ensure minimum touch target size (44x44px) */
      button, .btn, input[type="button"], input[type="submit"], 
      input[type="reset"], .clickable, .touch-target {
        min-width: 44px;
        min-height: 44px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }
      
      /* Ensure sufficient color contrast */
      .low-contrast-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 14px;
        margin: 8px 0;
      }
      
      /* Enhanced table accessibility */
      table {
        border-collapse: collapse;
        width: 100%;
      }
      
      th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
      }
      
      th {
        background-color: #f5f5f5;
        font-weight: bold;
      }
      
      caption {
        font-weight: bold;
        margin-bottom: 8px;
        text-align: left;
      }
      
      /* Form accessibility enhancements */
      label {
        display: block;
        margin-bottom: 4px;
        font-weight: 500;
        color: #333;
      }
      
      input[required]::after,
      select[required]::after,
      textarea[required]::after {
        content: " *";
        color: #dc3545;
      }
      
      .form-error {
        color: #dc3545;
        font-size: 14px;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
      }
      
      input:invalid, select:invalid, textarea:invalid {
        border-color: #dc3545 !important;
        box-shadow: 0 0 0 2px rgba(220, 53, 69, 0.1) !important;
      }
      
      input:valid, select:valid, textarea:valid {
        border-color: #28a745 !important;
      }
      
      /* Loading states with better accessibility */
      .loading-content[aria-busy="true"] {
        position: relative;
      }
      
      .loading-content[aria-busy="true"]::after {
        content: "Loading...";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255, 255, 255, 0.9);
        padding: 8px 16px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        font-size: 14px;
        color: #666;
      }
      
      /* Enhanced error states */
      .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .warning-message {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .info-message {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      /* Responsive text sizing */
      @media (max-width: 768px) {
        body {
          font-size: 16px; /* Prevent zoom on iOS */
        }
        
        input, select, textarea {
          font-size: 16px; /* Prevent zoom on iOS */
        }
        
        .small-text {
          font-size: 14px;
        }
      }
      
      /* Enhanced keyboard navigation */
      .keyboard-nav-hint {
        position: absolute;
        top: -30px;
        left: 0;
        right: 0;
        text-align: center;
        background: #333;
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        opacity: 0;
        transition: opacity 0.2s;
        pointer-events: none;
      }
      
      *:focus + .keyboard-nav-hint,
      *:focus .keyboard-nav-hint {
        opacity: 1;
      }
      
      /* Print styles for accessibility */
      @media print {
        * {
          background: transparent !important;
          color: black !important;
          box-shadow: none !important;
          text-shadow: none !important;
        }
        
        a, a:visited {
          text-decoration: underline;
        }
        
        a[href]:after {
          content: " (" attr(href) ")";
        }
        
        .no-print, .tts-controls, .image-controls {
          display: none !important;
        }
        
        .print-only {
          display: block !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
          page-break-after: avoid;
        }
        
        blockquote, pre {
          page-break-inside: avoid;
        }
        
        img {
          max-width: 100% !important;
          page-break-inside: avoid;
        }
      }
      
      /* Dark mode accessibility */
      @media (prefers-color-scheme: dark) {
        :root {
          --text-primary: #f8f9fa;
          --text-secondary: #e9ecef;
          --background-primary: #212529;
          --background-secondary: #343a40;
          --border-color: #6c757d;
        }
      }
      
      [data-theme="dark"] {
        color-scheme: dark;
      }
      
      [data-theme="dark"] .loading-content[aria-busy="true"]::after {
        background: rgba(33, 37, 41, 0.9);
        color: #f8f9fa;
      }
      
      /* Animation preferences */
      .respect-motion-preference {
        animation-duration: 0.3s;
        transition-duration: 0.2s;
      }
      
      @media (prefers-reduced-motion: reduce) {
        .respect-motion-preference {
          animation-duration: 0.01ms;
          transition-duration: 0.01ms;
        }
      }
      
      /* Language support */
      [lang="ar"], [lang="he"], [lang="fa"] {
        direction: rtl;
        text-align: right;
      }
      
      [lang="ar"] .ltr, [lang="he"] .ltr, [lang="fa"] .ltr {
        direction: ltr;
        text-align: left;
      }
      
      /* Zoom compatibility */
      @media (min-resolution: 2dppx) {
        .high-dpi-image {
          image-rendering: -webkit-optimize-contrast;
          image-rendering: crisp-edges;
        }
      }
      
      /* Error prevention */
      .destructive-action {
        background: #dc3545 !important;
        color: white !important;
        border: 2px solid #dc3545 !important;
      }
      
      .destructive-action:hover {
        background: #c82333 !important;
        border-color: #bd2130 !important;
      }
      
      .destructive-action:focus {
        box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.3) !important;
      }
      
      /* Content spacing for readability */
      .content-spacing p {
        line-height: 1.6;
        margin-bottom: 1em;
      }
      
      .content-spacing h1, .content-spacing h2, 
      .content-spacing h3, .content-spacing h4, 
      .content-spacing h5, .content-spacing h6 {
        line-height: 1.3;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
      }
      
      .content-spacing ul, .content-spacing ol {
        margin: 1em 0;
        padding-left: 2em;
      }
      
      .content-spacing li {
        margin-bottom: 0.5em;
      }
      
      /* Custom scrollbar for better accessibility */
      .custom-scrollbar {
        scrollbar-width: thin;
        scrollbar-color: #6c757d transparent;
      }
      
      .custom-scrollbar::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      
      .custom-scrollbar::-webkit-scrollbar-track {
        background: transparent;
      }
      
      .custom-scrollbar::-webkit-scrollbar-thumb {
        background: #6c757d;
        border-radius: 4px;
      }
      
      .custom-scrollbar::-webkit-scrollbar-thumb:hover {
        background: #5a6268;
      }
    `}</style>
  );
}