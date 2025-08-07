"use client";

import React, { useState } from "react";
import { Sparkles, Star, Settings } from "lucide-react";
import ConfigurationPanel from "../components/ConfigurationPanel";

export default function Header() {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [uiSettings, setUiSettings] = useState({});

  const handleSettingsChange = (settings: any) => {
    setUiSettings(settings);
    // Apply settings globally here if needed
    console.log('Settings updated:', settings);
  };

  return (
    <>
      <div className="flex items-center justify-between p-2 px-4 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-blue-600" />
          <h1 className="font-semibold text-gray-900 dark:text-white">SOTA RAG Assistant</h1>
        </div>
        <div className="flex items-center justify-end gap-4">
          <button
            onClick={() => setIsConfigOpen(true)}
            className="flex items-center gap-2 rounded-md border border-gray-300 dark:border-gray-600 px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            title="Settings"
            aria-label="Open settings"
          >
            <Settings className="size-4" />
            <span className="hidden sm:inline">Settings</span>
          </button>
          
          <div className="flex items-center gap-2">
            <a
              href="https://www.llamaindex.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            >
              <span className="hidden md:inline">Built by LlamaIndex</span>
              <span className="md:hidden">LlamaIndex</span>
            </a>
            <img
              className="h-[24px] w-[24px] rounded-sm"
              src="https://ui.llamaindex.ai/llama.png"
              alt="Llama Logo"
            />
          </div>
          <a
            href="https://github.com/run-llama/LlamaIndexTS"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-md border border-gray-300 dark:border-gray-600 px-2 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <Star className="size-4" />
            <span className="hidden sm:inline">Star on GitHub</span>
            <span className="sm:hidden">★</span>
          </a>
        </div>
      </div>

      <ConfigurationPanel
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
        onSettingsChange={handleSettingsChange}
      />
    </>
  );
}
