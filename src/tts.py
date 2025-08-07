"""
Text-to-Speech (TTS) integration module for voice output capabilities.

This module provides foundation for TTS functionality, enabling voice responses
from the RAG system for enhanced accessibility and user experience.
"""

import os
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available. Install for offline TTS support")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gTTS not available. Install for Google TTS support")

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    @abstractmethod
    def speak(self, text: str, **kwargs) -> bool:
        """Speak the given text."""
        pass
    
    @abstractmethod
    def save_to_file(self, text: str, file_path: str, **kwargs) -> bool:
        """Save speech to audio file."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if TTS engine is available."""
        pass


class PyttsxEngine(TTSEngine):
    """
    Offline TTS engine using pyttsx3.
    Provides immediate speech synthesis without internet connection.
    """
    
    def __init__(self):
        """Initialize pyttsx3 engine."""
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the pyttsx3 engine with configuration."""
        if not PYTTSX3_AVAILABLE:
            logger.error("pyttsx3 not available")
            return
        
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self._configure_voice()
            
            logger.info("pyttsx3 TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            self.engine = None
    
    def _configure_voice(self):
        """Configure voice properties from environment variables."""
        if not self.engine:
            return
        
        try:
            # Voice speed (words per minute)
            speed = int(os.getenv("VOICE_SPEED", "150"))
            self.engine.setProperty('rate', speed)
            
            # Voice volume (0.0 to 1.0)
            volume = float(os.getenv("VOICE_VOLUME", "0.8"))
            self.engine.setProperty('volume', volume)
            
            # Voice selection (optional)
            voice_id = os.getenv("VOICE_ID")
            if voice_id:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if voice_id.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            logger.debug(f"Voice configured: speed={speed}, volume={volume}")
            
        except Exception as e:
            logger.warning(f"Voice configuration failed: {e}")
    
    def speak(self, text: str, **kwargs) -> bool:
        """
        Speak the given text immediately.
        
        Args:
            text: Text to speak
            **kwargs: Additional parameters (async_mode, etc.)
            
        Returns:
            True if speech was successful
        """
        if not self.is_available():
            return False
        
        try:
            # Clean text for speech
            clean_text = self._clean_text_for_speech(text)
            
            if not clean_text.strip():
                logger.warning("No text to speak after cleaning")
                return False
            
            # Speak text
            async_mode = kwargs.get('async_mode', False)
            
            self.engine.say(clean_text)
            
            if async_mode:
                # Non-blocking speech
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()
            else:
                # Blocking speech
                self.engine.runAndWait()
            
            logger.debug(f"Successfully spoke {len(clean_text)} characters")
            return True
            
        except Exception as e:
            logger.error(f"Speech failed: {e}")
            return False
    
    def save_to_file(self, text: str, file_path: str, **kwargs) -> bool:
        """
        Save speech to audio file.
        
        Args:
            text: Text to convert to speech
            file_path: Path to save audio file
            **kwargs: Additional parameters
            
        Returns:
            True if file was saved successfully
        """
        if not self.is_available():
            return False
        
        try:
            clean_text = self._clean_text_for_speech(text)
            
            # Configure engine to save to file
            self.engine.save_to_file(clean_text, file_path)
            self.engine.runAndWait()
            
            # Verify file was created
            if Path(file_path).exists():
                logger.info(f"Speech saved to {file_path}")
                return True
            else:
                logger.error(f"Failed to create audio file: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to save speech to file: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if pyttsx3 engine is available."""
        return PYTTSX3_AVAILABLE and self.engine is not None
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean text for better speech synthesis.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text suitable for TTS
        """
        if not text:
            return ""
        
        # Remove citation markers
        import re
        text = re.sub(r'\[citation:\d+\]', '', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Replace common abbreviations for better pronunciation
        replacements = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
        }
        
        for abbrev, replacement in replacements.items():
            text = text.replace(abbrev, replacement)
        
        return text


class GoogleTTSEngine(TTSEngine):
    """
    Cloud-based TTS engine using Google Text-to-Speech.
    Provides high-quality speech synthesis but requires internet connection.
    """
    
    def __init__(self):
        """Initialize Google TTS engine."""
        self.language = os.getenv("TTS_LANGUAGE", "en")
        self.slow = os.getenv("TTS_SLOW_SPEECH", "false").lower() == "true"
    
    def speak(self, text: str, **kwargs) -> bool:
        """
        Speak text using Google TTS (saves to temp file and plays).
        
        Args:
            text: Text to speak
            **kwargs: Additional parameters
            
        Returns:
            True if speech was successful
        """
        if not self.is_available():
            return False
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save to temporary file
            if self.save_to_file(text, temp_path, **kwargs):
                # Play the audio file (requires system audio player)
                success = self._play_audio_file(temp_path)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Google TTS speech failed: {e}")
            return False
    
    def save_to_file(self, text: str, file_path: str, **kwargs) -> bool:
        """
        Save speech to audio file using Google TTS.
        
        Args:
            text: Text to convert to speech
            file_path: Path to save audio file
            **kwargs: Additional parameters (language, slow, etc.)
            
        Returns:
            True if file was saved successfully
        """
        if not self.is_available():
            return False
        
        try:
            # Get parameters
            language = kwargs.get('language', self.language)
            slow = kwargs.get('slow', self.slow)
            
            # Clean text
            clean_text = self._clean_text_for_speech(text)
            
            if not clean_text.strip():
                logger.warning("No text to convert to speech")
                return False
            
            # Create TTS object
            tts = gTTS(text=clean_text, lang=language, slow=slow)
            
            # Save to file
            tts.save(file_path)
            
            # Verify file was created
            if Path(file_path).exists():
                logger.info(f"Google TTS audio saved to {file_path}")
                return True
            else:
                logger.error(f"Failed to create audio file: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Google TTS save failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Google TTS is available."""
        return GTTS_AVAILABLE
    
    def _play_audio_file(self, file_path: str) -> bool:
        """
        Play audio file using system audio player.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if playback was successful
        """
        try:
            import platform
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                os.system(f"afplay '{file_path}'")
            elif system == "linux":
                os.system(f"mpg123 '{file_path}' || aplay '{file_path}' || paplay '{file_path}'")
            elif system == "windows":
                os.system(f"start '{file_path}'")
            else:
                logger.warning(f"Unsupported system for audio playback: {system}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for Google TTS."""
        if not text:
            return ""
        
        # Remove citation markers
        import re
        text = re.sub(r'\[citation:\d+\]', '', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit text length for Google TTS (max ~5000 characters)
        max_length = 4500
        if len(text) > max_length:
            text = text[:max_length].rsplit('.', 1)[0] + '.'
            logger.warning(f"Text truncated to {len(text)} characters for TTS")
        
        return text


class TTSManager:
    """
    Manager class for Text-to-Speech functionality.
    Handles engine selection and provides unified TTS interface.
    """
    
    def __init__(self):
        """Initialize TTS manager."""
        self.engines = {}
        self.default_engine = None
        self.enabled = os.getenv("TTS_INTEGRATION_ENABLED", "false").lower() == "true"
        
        if self.enabled:
            self._initialize_engines()
            self._select_default_engine()
        else:
            logger.info("TTS integration disabled via configuration")
    
    def _initialize_engines(self):
        """Initialize available TTS engines."""
        # Initialize pyttsx3 engine
        if PYTTSX3_AVAILABLE:
            try:
                pyttsx_engine = PyttsxEngine()
                if pyttsx_engine.is_available():
                    self.engines['pyttsx3'] = pyttsx_engine
                    logger.info("pyttsx3 TTS engine registered")
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3 engine: {e}")
        
        # Initialize Google TTS engine
        if GTTS_AVAILABLE:
            try:
                gtts_engine = GoogleTTSEngine()
                if gtts_engine.is_available():
                    self.engines['gtts'] = gtts_engine
                    logger.info("Google TTS engine registered")
            except Exception as e:
                logger.warning(f"Failed to initialize Google TTS engine: {e}")
    
    def _select_default_engine(self):
        """Select default TTS engine based on configuration."""
        preferred_engine = os.getenv("TTS_ENGINE", "pyttsx3")
        
        if preferred_engine in self.engines:
            self.default_engine = preferred_engine
            logger.info(f"Default TTS engine: {preferred_engine}")
        elif self.engines:
            self.default_engine = list(self.engines.keys())[0]
            logger.info(f"Fallback TTS engine: {self.default_engine}")
        else:
            logger.warning("No TTS engines available")
    
    def speak(self, text: str, engine: Optional[str] = None, **kwargs) -> bool:
        """
        Speak text using specified or default TTS engine.
        
        Args:
            text: Text to speak
            engine: Specific engine to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            True if speech was successful
        """
        if not self.is_enabled():
            logger.debug("TTS not enabled")
            return False
        
        engine_name = engine or self.default_engine
        
        if not engine_name or engine_name not in self.engines:
            logger.error(f"TTS engine not available: {engine_name}")
            return False
        
        try:
            start_time = time.time()
            success = self.engines[engine_name].speak(text, **kwargs)
            speech_time = time.time() - start_time
            
            if success:
                logger.info(f"TTS successful with {engine_name} in {speech_time:.2f}s")
            else:
                logger.warning(f"TTS failed with {engine_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"TTS error with {engine_name}: {e}")
            return False
    
    def save_audio(
        self, 
        text: str, 
        file_path: str, 
        engine: Optional[str] = None, 
        **kwargs
    ) -> bool:
        """
        Save text as audio file.
        
        Args:
            text: Text to convert
            file_path: Output file path
            engine: Specific engine to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            True if save was successful
        """
        if not self.is_enabled():
            return False
        
        engine_name = engine or self.default_engine
        
        if not engine_name or engine_name not in self.engines:
            logger.error(f"TTS engine not available: {engine_name}")
            return False
        
        try:
            return self.engines[engine_name].save_to_file(text, file_path, **kwargs)
        except Exception as e:
            logger.error(f"TTS save error with {engine_name}: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if TTS is enabled."""
        return self.enabled and bool(self.engines)
    
    def get_available_engines(self) -> list:
        """Get list of available TTS engines."""
        return list(self.engines.keys())
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about TTS configuration."""
        return {
            "enabled": self.is_enabled(),
            "default_engine": self.default_engine,
            "available_engines": self.get_available_engines(),
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "configuration": {
                "voice_speed": os.getenv("VOICE_SPEED", "150"),
                "voice_volume": os.getenv("VOICE_VOLUME", "0.8"),
                "tts_language": os.getenv("TTS_LANGUAGE", "en"),
                "slow_speech": os.getenv("TTS_SLOW_SPEECH", "false"),
            }
        }


# Global TTS manager instance
_tts_manager = None


def get_tts_manager() -> TTSManager:
    """Get the global TTS manager instance."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager()
    return _tts_manager


def speak_text(text: str, **kwargs) -> bool:
    """
    Convenience function to speak text.
    
    Args:
        text: Text to speak
        **kwargs: Additional parameters
        
    Returns:
        True if speech was successful
    """
    return get_tts_manager().speak(text, **kwargs)


def save_speech_audio(text: str, file_path: str, **kwargs) -> bool:
    """
    Convenience function to save text as audio.
    
    Args:
        text: Text to convert
        file_path: Output file path
        **kwargs: Additional parameters
        
    Returns:
        True if save was successful
    """
    return get_tts_manager().save_audio(text, file_path, **kwargs)


def is_tts_available() -> bool:
    """Check if TTS functionality is available."""
    return get_tts_manager().is_enabled()


def get_tts_info() -> Dict[str, Any]:
    """Get TTS system information."""
    return get_tts_manager().get_engine_info()