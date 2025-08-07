"use client";

import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Volume2, 
  VolumeX, 
  Download, 
  Settings,
  SkipForward,
  SkipBack
} from 'lucide-react';

interface TTSControlsProps {
  text: string;
  enabled?: boolean;
  onToggleEnabled?: (enabled: boolean) => void;
  className?: string;
}

interface TTSSettings {
  engine: 'browser' | 'pyttsx3' | 'gtts';
  speed: number;
  volume: number;
  voice?: string;
  language: string;
  pitch: number;
}

export default function TTSControls({ 
  text, 
  enabled = false, 
  onToggleEnabled, 
  className = "" 
}: TTSControlsProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speechSynthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [settings, setSettings] = useState<TTSSettings>({
    engine: 'browser',
    speed: 1.0,
    volume: 0.8,
    language: 'en-US',
    pitch: 1.0
  });

  // Load available voices on component mount
  useEffect(() => {
    const loadVoices = () => {
      const voices = speechSynthesis.getVoices();
      setAvailableVoices(voices);
      
      // Set default voice if none selected
      if (!settings.voice && voices.length > 0) {
        const englishVoice = voices.find(voice => 
          voice.lang.startsWith('en') && voice.localService
        ) || voices[0];
        setSettings(prev => ({ ...prev, voice: englishVoice.name }));
      }
    };

    loadVoices();
    if (speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = loadVoices;
    }

    return () => {
      if (speechSynthesis.onvoiceschanged) {
        speechSynthesis.onvoiceschanged = null;
      }
    };
  }, [settings.voice]);

  // Clean text for TTS
  const cleanTextForTTS = (rawText: string): string => {
    if (!rawText) return "";
    
    // Remove citation markers
    let cleanText = rawText.replace(/\[citation:\d+\]/g, '');
    
    // Remove excessive whitespace
    cleanText = cleanText.replace(/\s+/g, ' ').trim();
    
    // Replace common abbreviations for better pronunciation
    const replacements: { [key: string]: string } = {
      'e.g.': 'for example',
      'i.e.': 'that is',
      'etc.': 'etcetera',
      'vs.': 'versus',
      'Dr.': 'Doctor',
      'Mr.': 'Mister',
      'Mrs.': 'Misses',
      'Ms.': 'Miss',
    };
    
    for (const [abbrev, replacement] of Object.entries(replacements)) {
      cleanText = cleanText.replace(new RegExp(abbrev, 'g'), replacement);
    }
    
    return cleanText;
  };

  const playWithBrowserTTS = async () => {
    const cleanText = cleanTextForTTS(text);
    if (!cleanText) return;

    // Stop any existing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    speechSynthesisRef.current = utterance;

    // Configure utterance
    utterance.rate = settings.speed;
    utterance.volume = isMuted ? 0 : settings.volume;
    utterance.pitch = settings.pitch;
    utterance.lang = settings.language;

    // Set voice
    if (settings.voice) {
      const selectedVoice = availableVoices.find(voice => voice.name === settings.voice);
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
    }

    // Event handlers
    utterance.onstart = () => {
      setIsPlaying(true);
      setIsPaused(false);
      startProgressTracking();
    };

    utterance.onend = () => {
      setIsPlaying(false);
      setIsPaused(false);
      setProgress(0);
      setCurrentTime(0);
      stopProgressTracking();
    };

    utterance.onerror = (event) => {
      console.error('TTS Error:', event.error);
      setIsPlaying(false);
      setIsPaused(false);
      stopProgressTracking();
    };

    utterance.onpause = () => {
      setIsPaused(true);
      stopProgressTracking();
    };

    utterance.onresume = () => {
      setIsPaused(false);
      startProgressTracking();
    };

    // Estimate duration (roughly 150-200 words per minute)
    const wordCount = cleanText.split(' ').length;
    const estimatedDuration = (wordCount / 180) * 60; // seconds
    setDuration(estimatedDuration);

    speechSynthesis.speak(utterance);
  };

  const startProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    progressIntervalRef.current = setInterval(() => {
      if (speechSynthesis.speaking && !speechSynthesis.paused) {
        setCurrentTime(prev => {
          const newTime = prev + 0.1;
          setProgress(duration > 0 ? (newTime / duration) * 100 : 0);
          return newTime;
        });
      }
    }, 100);
  };

  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const handlePlay = async () => {
    if (!enabled || !text) return;

    try {
      if (settings.engine === 'browser') {
        if (isPaused && speechSynthesis.paused) {
          speechSynthesis.resume();
        } else {
          await playWithBrowserTTS();
        }
      } else {
        // For server-side TTS engines, we would need to make API calls
        // This would require backend integration
        console.log(`Playing with ${settings.engine} engine (requires backend API)`);
      }
    } catch (error) {
      console.error('TTS playback failed:', error);
    }
  };

  const handlePause = () => {
    if (settings.engine === 'browser') {
      if (speechSynthesis.speaking) {
        speechSynthesis.pause();
      }
    } else if (audioRef.current) {
      audioRef.current.pause();
    }
  };

  const handleStop = () => {
    if (settings.engine === 'browser') {
      speechSynthesis.cancel();
    } else if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    
    setIsPlaying(false);
    setIsPaused(false);
    setProgress(0);
    setCurrentTime(0);
    stopProgressTracking();
  };

  const handleVolumeToggle = () => {
    setIsMuted(!isMuted);
    if (settings.engine === 'browser' && speechSynthesis.speaking) {
      // Restart speech with new volume
      const wasPlaying = isPlaying;
      handleStop();
      if (wasPlaying) {
        setTimeout(() => handlePlay(), 100);
      }
    }
  };

  const handleDownload = async () => {
    if (!text || settings.engine !== 'gtts') {
      alert('Download is only available with Google TTS engine');
      return;
    }

    try {
      // This would require a backend API call to generate and serve the audio file
      const response = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: cleanTextForTTS(text),
          engine: 'gtts',
          language: settings.language,
          slow: false
        })
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `speech_${Date.now()}.mp3`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download audio file');
    }
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      speechSynthesis.cancel();
      stopProgressTracking();
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  if (!text) {
    return null;
  }

  return (
    <div className={`tts-controls ${className}`}>
      <div className="tts-main-controls">
        <div className="control-buttons">
          <button
            onClick={handlePlay}
            disabled={!enabled || isPlaying}
            className="btn-primary"
            aria-label="Play text-to-speech"
            title="Play"
          >
            <Play className="size-4" />
          </button>
          
          <button
            onClick={handlePause}
            disabled={!enabled || !isPlaying || isPaused}
            className="btn-secondary"
            aria-label="Pause text-to-speech"
            title="Pause"
          >
            <Pause className="size-4" />
          </button>
          
          <button
            onClick={handleStop}
            disabled={!enabled || (!isPlaying && !isPaused)}
            className="btn-secondary"
            aria-label="Stop text-to-speech"
            title="Stop"
          >
            <Square className="size-4" />
          </button>
        </div>

        <div className="progress-section">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="time-display">
            <span>{formatTime(currentTime)}</span>
            <span>/</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <div className="volume-controls">
          <button
            onClick={handleVolumeToggle}
            className="btn-icon"
            aria-label={isMuted ? "Unmute" : "Mute"}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
          </button>
        </div>

        <div className="additional-controls">
          <button
            onClick={handleDownload}
            disabled={!enabled}
            className="btn-icon"
            aria-label="Download audio"
            title="Download audio file"
          >
            <Download className="size-4" />
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="btn-icon"
            aria-label="TTS settings"
            title="Settings"
          >
            <Settings className="size-4" />
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="tts-settings">
          <div className="settings-row">
            <label htmlFor="tts-engine">Engine:</label>
            <select
              id="tts-engine"
              value={settings.engine}
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                engine: e.target.value as TTSSettings['engine'] 
              }))}
            >
              <option value="browser">Browser</option>
              <option value="pyttsx3">Offline (pyttsx3)</option>
              <option value="gtts">Google TTS</option>
            </select>
          </div>

          <div className="settings-row">
            <label htmlFor="tts-speed">Speed:</label>
            <input
              id="tts-speed"
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={settings.speed}
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                speed: parseFloat(e.target.value) 
              }))}
            />
            <span>{settings.speed.toFixed(1)}x</span>
          </div>

          <div className="settings-row">
            <label htmlFor="tts-volume">Volume:</label>
            <input
              id="tts-volume"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.volume}
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                volume: parseFloat(e.target.value) 
              }))}
            />
            <span>{Math.round(settings.volume * 100)}%</span>
          </div>

          {settings.engine === 'browser' && availableVoices.length > 0 && (
            <div className="settings-row">
              <label htmlFor="tts-voice">Voice:</label>
              <select
                id="tts-voice"
                value={settings.voice || ''}
                onChange={(e) => setSettings(prev => ({ 
                  ...prev, 
                  voice: e.target.value 
                }))}
              >
                {availableVoices.map((voice) => (
                  <option key={voice.name} value={voice.name}>
                    {voice.name} ({voice.lang})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="settings-row">
            <label htmlFor="tts-language">Language:</label>
            <select
              id="tts-language"
              value={settings.language}
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                language: e.target.value 
              }))}
            >
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="es-ES">Spanish</option>
              <option value="fr-FR">French</option>
              <option value="de-DE">German</option>
              <option value="it-IT">Italian</option>
              <option value="pt-BR">Portuguese (Brazil)</option>
              <option value="ja-JP">Japanese</option>
              <option value="ko-KR">Korean</option>
              <option value="zh-CN">Chinese (Mandarin)</option>
            </select>
          </div>
        </div>
      )}

      <style jsx>{`
        .tts-controls {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 12px;
          margin: 8px 0;
          background: #f8f9fa;
        }
        
        .tts-main-controls {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        
        .control-buttons {
          display: flex;
          gap: 4px;
        }
        
        .btn-primary, .btn-secondary, .btn-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 6px 8px;
          border: 1px solid #ddd;
          border-radius: 6px;
          background: white;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .btn-primary {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }
        
        .btn-primary:hover:not(:disabled) {
          background: #0056b3;
        }
        
        .btn-secondary:hover:not(:disabled) {
          background: #f8f9fa;
        }
        
        .btn-icon:hover:not(:disabled) {
          background: #f8f9fa;
        }
        
        .btn-primary:disabled, .btn-secondary:disabled, .btn-icon:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .progress-section {
          flex: 1;
          min-width: 200px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .progress-bar {
          flex: 1;
          height: 4px;
          background: #e1e5e9;
          border-radius: 2px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background: #007bff;
          transition: width 0.1s ease;
        }
        
        .time-display {
          display: flex;
          gap: 4px;
          font-size: 12px;
          color: #666;
          min-width: 80px;
          justify-content: center;
        }
        
        .volume-controls, .additional-controls {
          display: flex;
          gap: 4px;
        }
        
        .tts-settings {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #e1e5e9;
        }
        
        .settings-row {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }
        
        .settings-row label {
          min-width: 60px;
          font-size: 12px;
          color: #666;
        }
        
        .settings-row select, .settings-row input[type="range"] {
          flex: 1;
        }
        
        .settings-row span {
          min-width: 40px;
          font-size: 12px;
          color: #666;
        }
        
        @media (max-width: 768px) {
          .tts-main-controls {
            flex-direction: column;
            align-items: stretch;
          }
          
          .control-buttons {
            justify-content: center;
          }
          
          .progress-section {
            min-width: auto;
          }
          
          .volume-controls, .additional-controls {
            justify-content: center;
          }
        }
      `}</style>
    </div>
  );
}