"use client";

import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Volume2, 
  Image as ImageIcon, 
  Monitor,
  Moon,
  Sun,
  Accessibility,
  BarChart3,
  Shield,
  Zap,
  Eye,
  X,
  RotateCcw,
  Save,
  Download,
  Upload
} from 'lucide-react';

interface UISettings {
  // TTS Settings
  ttsEnabled: boolean;
  ttsEngine: 'browser' | 'pyttsx3' | 'gtts';
  voiceSpeed: number;
  voiceVolume: number;
  voicePitch: number;
  ttsLanguage: string;
  autoPlayResponses: boolean;
  
  // Multimodal Settings
  multimodalEnabled: boolean;
  imagePreviewEnabled: boolean;
  maxImagesPerResponse: number;
  imageQuality: 'low' | 'medium' | 'high';
  lazyLoadImages: boolean;
  
  // Display Settings
  theme: 'light' | 'dark' | 'system';
  fontSize: 'small' | 'medium' | 'large';
  compactMode: boolean;
  showConfidenceScores: boolean;
  showPerformanceMetrics: boolean;
  showCitations: boolean;
  animationsEnabled: boolean;
  
  // Accessibility Settings
  highContrast: boolean;
  reducedMotion: boolean;
  screenReaderOptimized: boolean;
  keyboardNavigationEnabled: boolean;
  
  // Performance Settings
  cacheEnabled: boolean;
  prefetchEnabled: boolean;
  compressionEnabled: boolean;
  maxCacheSize: number; // MB
  
  // Privacy Settings
  analyticsEnabled: boolean;
  crashReportingEnabled: boolean;
  dataCollectionEnabled: boolean;
}

interface ConfigurationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: UISettings) => void;
  className?: string;
}

const DEFAULT_SETTINGS: UISettings = {
  // TTS Settings
  ttsEnabled: false,
  ttsEngine: 'browser',
  voiceSpeed: 1.0,
  voiceVolume: 0.8,
  voicePitch: 1.0,
  ttsLanguage: 'en-US',
  autoPlayResponses: false,
  
  // Multimodal Settings
  multimodalEnabled: true,
  imagePreviewEnabled: true,
  maxImagesPerResponse: 10,
  imageQuality: 'medium',
  lazyLoadImages: true,
  
  // Display Settings
  theme: 'system',
  fontSize: 'medium',
  compactMode: false,
  showConfidenceScores: true,
  showPerformanceMetrics: false,
  showCitations: true,
  animationsEnabled: true,
  
  // Accessibility Settings
  highContrast: false,
  reducedMotion: false,
  screenReaderOptimized: false,
  keyboardNavigationEnabled: true,
  
  // Performance Settings
  cacheEnabled: true,
  prefetchEnabled: true,
  compressionEnabled: true,
  maxCacheSize: 100,
  
  // Privacy Settings
  analyticsEnabled: true,
  crashReportingEnabled: true,
  dataCollectionEnabled: true,
};

export default function ConfigurationPanel({ 
  isOpen, 
  onClose, 
  onSettingsChange,
  className = "" 
}: ConfigurationPanelProps) {
  const [settings, setSettings] = useState<UISettings>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState<string>('display');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('rag-ui-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    }
  }, []);

  // Apply theme changes immediately
  useEffect(() => {
    if (settings.theme !== 'system') {
      document.documentElement.setAttribute('data-theme', settings.theme);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    
    if (settings.highContrast) {
      document.documentElement.setAttribute('data-high-contrast', 'true');
    } else {
      document.documentElement.removeAttribute('data-high-contrast');
    }
  }, [settings.theme, settings.highContrast]);

  const updateSetting = <K extends keyof UISettings>(
    key: K, 
    value: UISettings[K]
  ) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    setHasUnsavedChanges(true);
    onSettingsChange(newSettings);
  };

  const saveSettings = () => {
    try {
      localStorage.setItem('rag-ui-settings', JSON.stringify(settings));
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    }
  };

  const resetSettings = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      setSettings(DEFAULT_SETTINGS);
      setHasUnsavedChanges(true);
      onSettingsChange(DEFAULT_SETTINGS);
    }
  };

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rag-ui-settings.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const importSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedSettings = JSON.parse(e.target?.result as string);
        const mergedSettings = { ...DEFAULT_SETTINGS, ...importedSettings };
        setSettings(mergedSettings);
        setHasUnsavedChanges(true);
        onSettingsChange(mergedSettings);
      } catch (error) {
        alert('Failed to import settings. Please check the file format.');
      }
    };
    reader.readAsText(file);
    
    // Reset the input
    event.target.value = '';
  };

  const tabs = [
    { id: 'display', label: 'Display', icon: <Monitor className="size-4" /> },
    { id: 'tts', label: 'Text-to-Speech', icon: <Volume2 className="size-4" /> },
    { id: 'multimodal', label: 'Images', icon: <ImageIcon className="size-4" /> },
    { id: 'accessibility', label: 'Accessibility', icon: <Accessibility className="size-4" /> },
    { id: 'performance', label: 'Performance', icon: <Zap className="size-4" /> },
    { id: 'privacy', label: 'Privacy', icon: <Shield className="size-4" /> },
  ];

  const renderDisplaySettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-label">Theme</label>
        <select
          value={settings.theme}
          onChange={(e) => updateSetting('theme', e.target.value as UISettings['theme'])}
          className="setting-select"
        >
          <option value="light">Light</option>
          <option value="dark">Dark</option>
          <option value="system">System</option>
        </select>
      </div>

      <div className="setting-group">
        <label className="setting-label">Font Size</label>
        <select
          value={settings.fontSize}
          onChange={(e) => updateSetting('fontSize', e.target.value as UISettings['fontSize'])}
          className="setting-select"
        >
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.compactMode}
            onChange={(e) => updateSetting('compactMode', e.target.checked)}
          />
          <span>Compact mode</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.showConfidenceScores}
            onChange={(e) => updateSetting('showConfidenceScores', e.target.checked)}
          />
          <span>Show confidence scores</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.showPerformanceMetrics}
            onChange={(e) => updateSetting('showPerformanceMetrics', e.target.checked)}
          />
          <span>Show performance metrics</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.showCitations}
            onChange={(e) => updateSetting('showCitations', e.target.checked)}
          />
          <span>Show citations</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.animationsEnabled}
            onChange={(e) => updateSetting('animationsEnabled', e.target.checked)}
          />
          <span>Enable animations</span>
        </label>
      </div>
    </div>
  );

  const renderTTSSettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.ttsEnabled}
            onChange={(e) => updateSetting('ttsEnabled', e.target.checked)}
          />
          <span>Enable Text-to-Speech</span>
        </label>
      </div>

      {settings.ttsEnabled && (
        <>
          <div className="setting-group">
            <label className="setting-label">Engine</label>
            <select
              value={settings.ttsEngine}
              onChange={(e) => updateSetting('ttsEngine', e.target.value as UISettings['ttsEngine'])}
              className="setting-select"
            >
              <option value="browser">Browser (Web Speech API)</option>
              <option value="pyttsx3">Offline (pyttsx3)</option>
              <option value="gtts">Google TTS</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-label">Speech Speed: {settings.voiceSpeed.toFixed(1)}x</label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={settings.voiceSpeed}
              onChange={(e) => updateSetting('voiceSpeed', parseFloat(e.target.value))}
              className="setting-range"
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">Volume: {Math.round(settings.voiceVolume * 100)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.voiceVolume}
              onChange={(e) => updateSetting('voiceVolume', parseFloat(e.target.value))}
              className="setting-range"
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">Pitch: {settings.voicePitch.toFixed(1)}</label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={settings.voicePitch}
              onChange={(e) => updateSetting('voicePitch', parseFloat(e.target.value))}
              className="setting-range"
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">Language</label>
            <select
              value={settings.ttsLanguage}
              onChange={(e) => updateSetting('ttsLanguage', e.target.value)}
              className="setting-select"
            >
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="es-ES">Spanish</option>
              <option value="fr-FR">French</option>
              <option value="de-DE">German</option>
              <option value="it-IT">Italian</option>
              <option value="pt-BR">Portuguese</option>
              <option value="ja-JP">Japanese</option>
              <option value="ko-KR">Korean</option>
              <option value="zh-CN">Chinese</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-checkbox">
              <input
                type="checkbox"
                checked={settings.autoPlayResponses}
                onChange={(e) => updateSetting('autoPlayResponses', e.target.checked)}
              />
              <span>Auto-play responses</span>
            </label>
          </div>
        </>
      )}
    </div>
  );

  const renderMultimodalSettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.multimodalEnabled}
            onChange={(e) => updateSetting('multimodalEnabled', e.target.checked)}
          />
          <span>Enable image display</span>
        </label>
      </div>

      {settings.multimodalEnabled && (
        <>
          <div className="setting-group">
            <label className="setting-checkbox">
              <input
                type="checkbox"
                checked={settings.imagePreviewEnabled}
                onChange={(e) => updateSetting('imagePreviewEnabled', e.target.checked)}
              />
              <span>Enable image previews</span>
            </label>
          </div>

          <div className="setting-group">
            <label className="setting-label">Max images per response: {settings.maxImagesPerResponse}</label>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={settings.maxImagesPerResponse}
              onChange={(e) => updateSetting('maxImagesPerResponse', parseInt(e.target.value))}
              className="setting-range"
            />
          </div>

          <div className="setting-group">
            <label className="setting-label">Image Quality</label>
            <select
              value={settings.imageQuality}
              onChange={(e) => updateSetting('imageQuality', e.target.value as UISettings['imageQuality'])}
              className="setting-select"
            >
              <option value="low">Low (faster loading)</option>
              <option value="medium">Medium (balanced)</option>
              <option value="high">High (best quality)</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-checkbox">
              <input
                type="checkbox"
                checked={settings.lazyLoadImages}
                onChange={(e) => updateSetting('lazyLoadImages', e.target.checked)}
              />
              <span>Lazy load images</span>
            </label>
          </div>
        </>
      )}
    </div>
  );

  const renderAccessibilitySettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.highContrast}
            onChange={(e) => updateSetting('highContrast', e.target.checked)}
          />
          <span>High contrast mode</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.reducedMotion}
            onChange={(e) => updateSetting('reducedMotion', e.target.checked)}
          />
          <span>Reduce motion</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.screenReaderOptimized}
            onChange={(e) => updateSetting('screenReaderOptimized', e.target.checked)}
          />
          <span>Screen reader optimized</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.keyboardNavigationEnabled}
            onChange={(e) => updateSetting('keyboardNavigationEnabled', e.target.checked)}
          />
          <span>Enhanced keyboard navigation</span>
        </label>
      </div>
    </div>
  );

  const renderPerformanceSettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.cacheEnabled}
            onChange={(e) => updateSetting('cacheEnabled', e.target.checked)}
          />
          <span>Enable caching</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.prefetchEnabled}
            onChange={(e) => updateSetting('prefetchEnabled', e.target.checked)}
          />
          <span>Enable prefetching</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.compressionEnabled}
            onChange={(e) => updateSetting('compressionEnabled', e.target.checked)}
          />
          <span>Enable compression</span>
        </label>
      </div>

      <div className="setting-group">
        <label className="setting-label">Max cache size: {settings.maxCacheSize} MB</label>
        <input
          type="range"
          min="10"
          max="1000"
          step="10"
          value={settings.maxCacheSize}
          onChange={(e) => updateSetting('maxCacheSize', parseInt(e.target.value))}
          className="setting-range"
        />
      </div>
    </div>
  );

  const renderPrivacySettings = () => (
    <div className="settings-section">
      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.analyticsEnabled}
            onChange={(e) => updateSetting('analyticsEnabled', e.target.checked)}
          />
          <span>Enable analytics</span>
        </label>
        <p className="setting-description">Help improve the application by sharing usage analytics</p>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.crashReportingEnabled}
            onChange={(e) => updateSetting('crashReportingEnabled', e.target.checked)}
          />
          <span>Enable crash reporting</span>
        </label>
        <p className="setting-description">Automatically send crash reports to help fix issues</p>
      </div>

      <div className="setting-group">
        <label className="setting-checkbox">
          <input
            type="checkbox"
            checked={settings.dataCollectionEnabled}
            onChange={(e) => updateSetting('dataCollectionEnabled', e.target.checked)}
          />
          <span>Enable data collection</span>
        </label>
        <p className="setting-description">Allow collection of anonymized usage data for research</p>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'display': return renderDisplaySettings();
      case 'tts': return renderTTSSettings();
      case 'multimodal': return renderMultimodalSettings();
      case 'accessibility': return renderAccessibilitySettings();
      case 'performance': return renderPerformanceSettings();
      case 'privacy': return renderPrivacySettings();
      default: return renderDisplaySettings();
    }
  };

  if (!isOpen) return null;

  return (
    <div className={`configuration-panel ${className}`}>
      <div className="panel-overlay" onClick={onClose} />
      <div className="panel-content">
        <div className="panel-header">
          <div className="header-title">
            <Settings className="size-5" />
            <h2>Settings</h2>
          </div>
          <button className="close-btn" onClick={onClose} aria-label="Close settings">
            <X className="size-5" />
          </button>
        </div>

        <div className="panel-body">
          <div className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          <div className="tab-content">
            {renderTabContent()}
          </div>
        </div>

        <div className="panel-footer">
          <div className="footer-actions">
            <button className="action-btn secondary" onClick={resetSettings}>
              <RotateCcw className="size-4" />
              Reset
            </button>
            
            <button className="action-btn secondary" onClick={exportSettings}>
              <Download className="size-4" />
              Export
            </button>
            
            <label className="action-btn secondary">
              <Upload className="size-4" />
              Import
              <input
                type="file"
                accept=".json"
                onChange={importSettings}
                style={{ display: 'none' }}
              />
            </label>
          </div>
          
          <div className="save-actions">
            <button 
              className={`action-btn primary ${hasUnsavedChanges ? 'has-changes' : ''}`}
              onClick={saveSettings}
              disabled={!hasUnsavedChanges}
            >
              <Save className="size-4" />
              {hasUnsavedChanges ? 'Save Changes' : 'Saved'}
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        .configuration-panel {
          position: fixed;
          inset: 0;
          z-index: 2000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .panel-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(2px);
        }
        
        .panel-content {
          position: relative;
          background: white;
          border-radius: 12px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
          width: 90vw;
          max-width: 800px;
          max-height: 90vh;
          display: flex;
          flex-direction: column;
        }
        
        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 24px;
          border-bottom: 1px solid #e1e5e9;
        }
        
        .header-title {
          display: flex;
          align-items: center;
          gap: 12px;
          color: #333;
        }
        
        .header-title h2 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
        }
        
        .close-btn {
          padding: 8px;
          border: none;
          background: none;
          cursor: pointer;
          border-radius: 6px;
          color: #6c757d;
        }
        
        .close-btn:hover {
          background: #f8f9fa;
          color: #495057;
        }
        
        .panel-body {
          flex: 1;
          display: flex;
          overflow: hidden;
        }
        
        .tabs {
          width: 200px;
          background: #f8f9fa;
          border-right: 1px solid #e1e5e9;
          display: flex;
          flex-direction: column;
          padding: 16px 0;
        }
        
        .tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 20px;
          border: none;
          background: none;
          cursor: pointer;
          text-align: left;
          color: #6c757d;
          font-size: 14px;
          transition: all 0.2s ease;
        }
        
        .tab:hover {
          background: rgba(0, 123, 255, 0.1);
          color: #007bff;
        }
        
        .tab.active {
          background: #007bff;
          color: white;
        }
        
        .tab-content {
          flex: 1;
          padding: 24px;
          overflow-y: auto;
        }
        
        .settings-section {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        
        .setting-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .setting-label {
          font-size: 14px;
          font-weight: 500;
          color: #333;
        }
        
        .setting-checkbox {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #495057;
          cursor: pointer;
        }
        
        .setting-checkbox input[type="checkbox"] {
          width: 16px;
          height: 16px;
          cursor: pointer;
        }
        
        .setting-select, .setting-range {
          border: 1px solid #ddd;
          border-radius: 6px;
          padding: 8px 12px;
          font-size: 14px;
          background: white;
        }
        
        .setting-select:focus, .setting-range:focus {
          border-color: #007bff;
          outline: none;
          box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.1);
        }
        
        .setting-range {
          padding: 4px 0;
          background: none;
          border: none;
        }
        
        .setting-description {
          font-size: 12px;
          color: #6c757d;
          margin: 0;
          line-height: 1.4;
        }
        
        .panel-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 24px;
          border-top: 1px solid #e1e5e9;
          background: #f8f9fa;
        }
        
        .footer-actions, .save-actions {
          display: flex;
          gap: 8px;
        }
        
        .action-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          border: 1px solid #ddd;
          border-radius: 6px;
          background: white;
          cursor: pointer;
          font-size: 14px;
          transition: all 0.2s ease;
        }
        
        .action-btn.primary {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }
        
        .action-btn.primary:hover:not(:disabled) {
          background: #0056b3;
        }
        
        .action-btn.primary.has-changes {
          background: #28a745;
          border-color: #28a745;
        }
        
        .action-btn.primary:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .action-btn.secondary:hover {
          background: #f8f9fa;
        }
        
        /* Dark theme support */
        [data-theme="dark"] .panel-content {
          background: #2d3748;
          color: #f7fafc;
        }
        
        [data-theme="dark"] .panel-header {
          border-color: #4a5568;
        }
        
        [data-theme="dark"] .tabs {
          background: #1a202c;
          border-color: #4a5568;
        }
        
        [data-theme="dark"] .tab {
          color: #a0aec0;
        }
        
        [data-theme="dark"] .tab:hover {
          background: rgba(66, 153, 225, 0.1);
          color: #63b3ed;
        }
        
        [data-theme="dark"] .tab.active {
          background: #3182ce;
        }
        
        [data-theme="dark"] .setting-select,
        [data-theme="dark"] .action-btn {
          background: #4a5568;
          border-color: #718096;
          color: #f7fafc;
        }
        
        [data-theme="dark"] .panel-footer {
          background: #1a202c;
          border-color: #4a5568;
        }
        
        /* High contrast mode */
        [data-high-contrast="true"] {
          --border-color: #000;
          --text-color: #000;
          --bg-color: #fff;
        }
        
        [data-high-contrast="true"] .panel-content {
          border: 3px solid var(--border-color);
        }
        
        [data-high-contrast="true"] .setting-select,
        [data-high-contrast="true"] .action-btn {
          border: 2px solid var(--border-color);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
          .panel-content {
            width: 95vw;
            max-height: 95vh;
          }
          
          .panel-body {
            flex-direction: column;
          }
          
          .tabs {
            width: 100%;
            flex-direction: row;
            overflow-x: auto;
            padding: 12px 16px;
          }
          
          .tab {
            padding: 8px 12px;
            white-space: nowrap;
          }
          
          .tab-content {
            padding: 16px;
          }
          
          .panel-footer {
            flex-direction: column;
            gap: 12px;
          }
          
          .footer-actions {
            order: 2;
          }
          
          .save-actions {
            order: 1;
            width: 100%;
          }
          
          .save-actions .action-btn {
            flex: 1;
            justify-content: center;
          }
        }
        
        @media (max-width: 480px) {
          .tabs {
            padding: 8px;
          }
          
          .tab {
            padding: 6px 8px;
            font-size: 12px;
          }
          
          .tab span {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}