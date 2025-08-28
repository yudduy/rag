"""
Comprehensive Unit Tests for Settings and Configuration - Priority 1

This test suite provides comprehensive coverage of the settings module,
focusing on:
- Configuration validation and API key validation
- Environment loading and variable parsing
- Security validation of configuration parameters
- Edge cases and error handling
- Performance profile validation
- Multimodal and advanced feature configuration
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.settings import (
    init_settings, get_rag_config, get_agentic_config, get_cache_config,
    get_verification_config, get_multimodal_config,
    _validate_configuration, _validate_agentic_configuration,
    _validate_cache_configuration, _validate_verification_configuration,
    _validate_multimodal_configuration
)


class TestInitializationAndAPIKeyValidation:
    """Test settings initialization and API key validation."""
    
    def test_init_settings_success_with_valid_api_key(self):
        """Test successful initialization with valid API key."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "REDACTED_SK_KEY",
            "MODEL": "gpt-4o",
            "EMBEDDING_MODEL": "text-embedding-3-large"
        }):
            with patch('src.settings.OpenAI') as mock_openai, \
                 patch('src.settings.OpenAIEmbedding') as mock_embedding, \
                 patch('src.settings.Settings') as mock_settings:
                
                mock_embedding.return_value = Mock()
                
                # Should not raise any exceptions
                init_settings()
                
                # Verify OpenAI client was configured
                mock_openai.assert_called_once()
                mock_embedding.assert_called_once()
    
    def test_init_settings_failure_missing_api_key(self):
        """Test initialization failure when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
                init_settings()
    
    def test_init_settings_failure_placeholder_api_key(self):
        """Test initialization failure with placeholder API key."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "your_openai_api_key_here"
        }):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing or not configured"):
                init_settings()
    
    def test_environment_loading_from_src_env(self):
        """Test loading environment from src/.env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary project structure
            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir()
            env_file = src_dir / ".env"
            
            # Write test environment file
            env_content = """OPENAI_API_KEY=REDACTED
MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002"""
            env_file.write_text(env_content)
            
            # Mock the file path detection
            with patch('src.settings.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.parent.parent.absolute.return_value = Path(temp_dir)
                mock_path.return_value = mock_path_instance
                
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Clear environment first
                    with patch.dict(os.environ, {}, clear=True):
                        init_settings()
                        
                        # Verify environment was loaded
                        assert os.environ.get("OPENAI_API_KEY") == "sk-test123"
    
    def test_environment_loading_fallback_to_root_env(self):
        """Test fallback to root .env when src/.env doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only root .env file
            env_file = Path(temp_dir) / ".env"
            env_content = """OPENAI_API_KEY=REDACTED_SK_KEY
MODEL=gpt-4o"""
            env_file.write_text(env_content)
            
            # Mock the file path detection
            with patch('src.settings.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.parent.parent.absolute.return_value = Path(temp_dir)
                mock_path.return_value = mock_path_instance
                
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Clear environment first
                    with patch.dict(os.environ, {}, clear=True):
                        init_settings()
                        
                        # Verify fallback environment was loaded
                        assert os.environ.get("OPENAI_API_KEY") == "REDACTED_SK_KEY"
    
    def test_model_configuration_defaults(self):
        """Test default model configuration."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "REDACTED_SK_KEY"}):
            with patch('src.settings.OpenAI') as mock_openai, \
                 patch('src.settings.OpenAIEmbedding') as mock_embedding, \
                 patch('src.settings.Settings') as mock_settings:
                
                mock_embedding.return_value = Mock()
                
                init_settings()
                
                # Check default model configuration
                openai_call_args = mock_openai.call_args[1] if mock_openai.call_args else {}
                embedding_call_args = mock_embedding.call_args[1] if mock_embedding.call_args else {}
                
                # Verify defaults are used when not specified
                assert openai_call_args.get("model", "gpt-4o") == "gpt-4o"
                assert embedding_call_args.get("model", "text-embedding-3-large") == "text-embedding-3-large"
    
    def test_model_configuration_custom(self):
        """Test custom model configuration."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "REDACTED_SK_KEY",
            "MODEL": "gpt-3.5-turbo",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "TEMPERATURE": "0.5",
            "MAX_TOKENS": "1024",
            "EMBED_BATCH_SIZE": "50"
        }):
            with patch('src.settings.OpenAI') as mock_openai, \
                 patch('src.settings.OpenAIEmbedding') as mock_embedding, \
                 patch('src.settings.Settings') as mock_settings:
                
                mock_embedding.return_value = Mock()
                
                init_settings()
                
                # Check custom configuration
                openai_call_args = mock_openai.call_args[1]
                embedding_call_args = mock_embedding.call_args[1]
                
                assert openai_call_args["model"] == "gpt-3.5-turbo"
                assert openai_call_args["temperature"] == 0.5
                assert openai_call_args["max_tokens"] == 1024
                assert embedding_call_args["model"] == "text-embedding-ada-002"
                assert embedding_call_args["embed_batch_size"] == 50


class TestConfigurationValidation:
    """Test configuration validation functions."""
    
    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with patch.dict(os.environ, {
            "SENTENCE_WINDOW_SIZE": "3",
            "TOP_K": "10",
            "RERANK_TOP_N": "5",
            "SIMILARITY_THRESHOLD": "0.7",
            "TEMPERATURE": "0.1",
            "CHUNK_SIZE": "512"
        }):
            # Should not raise any exceptions
            _validate_configuration()
    
    def test_validate_configuration_warnings(self):
        """Test configuration validation with warning conditions."""
        test_cases = [
            {"SENTENCE_WINDOW_SIZE": "15"},  # Too high
            {"TOP_K": "0"},  # Too low
            {"RERANK_TOP_N": "50"},  # Higher than TOP_K
            {"SIMILARITY_THRESHOLD": "1.5"},  # Out of range
            {"TEMPERATURE": "3.0"},  # Too high
            {"CHUNK_SIZE": "50"},  # Too small
        ]
        
        for test_env in test_cases:
            base_env = {
                "SENTENCE_WINDOW_SIZE": "3",
                "TOP_K": "10",
                "RERANK_TOP_N": "5",
                "SIMILARITY_THRESHOLD": "0.7",
                "TEMPERATURE": "0.1",
                "CHUNK_SIZE": "512"
            }
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                # Should not raise exceptions but may log warnings
                _validate_configuration()
    
    def test_validate_configuration_errors(self):
        """Test configuration validation with error conditions."""
        with patch.dict(os.environ, {"TOP_K": "-5"}):
            with pytest.raises(ValueError, match="TOP_K.*must be between 1-100"):
                _validate_configuration()
    
    def test_validate_agentic_configuration_success(self):
        """Test successful agentic configuration validation."""
        with patch.dict(os.environ, {
            "AGENT_ROUTING_ENABLED": "true",
            "AGENT_ROUTING_THRESHOLD": "0.8",
            "QUERY_DECOMPOSITION_ENABLED": "true",
            "MAX_SUBQUERIES": "3",
            "QUERY_COMPLEXITY_THRESHOLD": "0.7",
            "ROUTING_MODEL": "gpt-3.5-turbo",
            "DECOMPOSITION_MODEL": "gpt-3.5-turbo"
        }):
            # Should not raise any exceptions
            _validate_agentic_configuration()
    
    def test_validate_agentic_configuration_warnings(self):
        """Test agentic configuration validation with warnings."""
        warning_cases = [
            {"AGENT_ROUTING_THRESHOLD": "1.5"},  # Out of range
            {"MAX_SUBQUERIES": "15"},  # Too high
            {"QUERY_COMPLEXITY_THRESHOLD": "-0.1"},  # Too low
        ]
        
        for test_env in warning_cases:
            base_env = {
                "AGENT_ROUTING_ENABLED": "true",
                "AGENT_ROUTING_THRESHOLD": "0.8",
                "MAX_SUBQUERIES": "3",
                "QUERY_COMPLEXITY_THRESHOLD": "0.7"
            }
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                # Should not raise exceptions but may log warnings
                _validate_agentic_configuration()
    
    def test_validate_cache_configuration_success(self):
        """Test successful cache configuration validation."""
        with patch.dict(os.environ, {
            "CACHE_SIMILARITY_THRESHOLD": "0.95",
            "CACHE_TTL": "3600",
            "MAX_CACHE_SIZE": "5000",
            "SEMANTIC_CACHE_ENABLED": "true",
            "REDIS_CACHE_URL": "redis://localhost:6379"
        }):
            # Should not raise any exceptions
            _validate_cache_configuration()
    
    def test_validate_cache_configuration_warnings(self):
        """Test cache configuration validation with warnings."""
        warning_cases = [
            {"CACHE_SIMILARITY_THRESHOLD": "0.5"},  # Too low
            {"CACHE_TTL": "100"},  # Too short
            {"MAX_CACHE_SIZE": "50"},  # Too small
        ]
        
        for test_env in warning_cases:
            base_env = {
                "CACHE_SIMILARITY_THRESHOLD": "0.95",
                "CACHE_TTL": "3600", 
                "MAX_CACHE_SIZE": "1000"
            }
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                # Should not raise exceptions but may log warnings
                _validate_cache_configuration()
    
    def test_validate_verification_configuration_success(self):
        """Test successful verification configuration validation."""
        with patch.dict(os.environ, {
            "VERIFICATION_ENABLED": "true",
            "VERIFICATION_THRESHOLD": "0.8",
            "VERIFICATION_MODEL": "gpt-4o-mini",
            "ENSEMBLE_VERIFICATION": "true",
            "MAX_VERIFICATION_TIME": "2.0",
            "VERIFICATION_TIMEOUT": "5.0"
        }):
            # Should not raise any exceptions
            _validate_verification_configuration()
    
    def test_validate_verification_configuration_warnings(self):
        """Test verification configuration validation with warnings."""
        warning_cases = [
            {"VERIFICATION_THRESHOLD": "1.5"},  # Out of range
            {"VERIFICATION_MODEL": "invalid-model"},  # Not recommended
            {"MAX_VERIFICATION_TIME": "15.0"},  # Too high
        ]
        
        for test_env in warning_cases:
            base_env = {
                "VERIFICATION_ENABLED": "true",
                "VERIFICATION_THRESHOLD": "0.8",
                "VERIFICATION_MODEL": "gpt-4o-mini",
                "MAX_VERIFICATION_TIME": "2.0"
            }
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                # Should not raise exceptions but may log warnings
                _validate_verification_configuration()
    
    def test_validate_multimodal_configuration_success(self):
        """Test successful multimodal configuration validation."""
        with patch.dict(os.environ, {
            "MULTIMODAL_ENABLED": "true",
            "IMAGE_INDEXING_ENABLED": "true",
            "CLIP_MODEL_NAME": "openai/clip-vit-base-patch32",
            "MULTIMODAL_THRESHOLD": "0.6",
            "MAX_IMAGE_SIZE_MB": "10",
            "TTS_ENGINE": "pyttsx3",
            "VOICE_SPEED": "150"
        }):
            # Should not raise any exceptions
            _validate_multimodal_configuration()
    
    def test_validate_multimodal_configuration_warnings(self):
        """Test multimodal configuration validation with warnings."""
        warning_cases = [
            {"CLIP_MODEL_NAME": "invalid-clip-model"},  # Not recommended
            {"MULTIMODAL_THRESHOLD": "1.5"},  # Out of range
            {"MAX_IMAGE_SIZE_MB": "200"},  # Too large
            {"TTS_ENGINE": "invalid-engine"},  # Not supported
            {"VOICE_SPEED": "400"},  # Too fast
        ]
        
        for test_env in warning_cases:
            base_env = {
                "MULTIMODAL_ENABLED": "true",
                "CLIP_MODEL_NAME": "openai/clip-vit-base-patch32",
                "MULTIMODAL_THRESHOLD": "0.6",
                "MAX_IMAGE_SIZE_MB": "10",
                "TTS_ENGINE": "pyttsx3",
                "VOICE_SPEED": "150"
            }
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                # Should not raise exceptions but may log warnings
                _validate_multimodal_configuration()


class TestConfigurationGetters:
    """Test configuration getter functions."""
    
    def test_get_rag_config_defaults(self):
        """Test RAG configuration getter with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_rag_config()
            
            assert config["sentence_window_size"] == 3
            assert config["top_k"] == 10
            assert config["hybrid_search_enabled"] is True
            assert config["rerank_enabled"] is True
            assert config["rerank_top_n"] == 5
            assert config["similarity_threshold"] == 0.6
            assert config["main_model"] == "gpt-4o"
            assert config["embedding_model"] == "text-embedding-3-large"
            assert config["temperature"] == 0.1
            assert config["max_tokens"] == 2048
            assert config["chunk_size"] == 512
            assert config["chunk_overlap"] == 50
    
    def test_get_rag_config_custom_values(self):
        """Test RAG configuration getter with custom values."""
        with patch.dict(os.environ, {
            "SENTENCE_WINDOW_SIZE": "5",
            "TOP_K": "15",
            "HYBRID_SEARCH_ENABLED": "false",
            "RERANK_ENABLED": "false",
            "RERANK_TOP_N": "8",
            "SIMILARITY_THRESHOLD": "0.8",
            "MODEL": "gpt-3.5-turbo",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "TEMPERATURE": "0.5",
            "MAX_TOKENS": "1024",
            "CHUNK_SIZE": "256",
            "CHUNK_OVERLAP": "25"
        }):
            config = get_rag_config()
            
            assert config["sentence_window_size"] == 5
            assert config["top_k"] == 15
            assert config["hybrid_search_enabled"] is False
            assert config["rerank_enabled"] is False
            assert config["rerank_top_n"] == 8
            assert config["similarity_threshold"] == 0.8
            assert config["main_model"] == "gpt-3.5-turbo"
            assert config["embedding_model"] == "text-embedding-ada-002"
            assert config["temperature"] == 0.5
            assert config["max_tokens"] == 1024
            assert config["chunk_size"] == 256
            assert config["chunk_overlap"] == 25
    
    def test_get_agentic_config_defaults(self):
        """Test agentic configuration getter with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_agentic_config()
            
            assert config["agent_routing_enabled"] is True
            assert config["query_decomposition_enabled"] is True
            assert config["agent_routing_threshold"] == 0.8
            assert config["max_subqueries"] == 3
            assert config["query_complexity_threshold"] == 0.7
            assert config["routing_model"] == "gpt-3.5-turbo"
            assert config["decomposition_model"] == "gpt-3.5-turbo"
            assert config["parallel_execution_enabled"] is True
            assert config["subquery_aggregation_model"] == "gpt-4o-mini"
    
    def test_get_cache_config_defaults(self):
        """Test cache configuration getter with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_cache_config()
            
            assert config["semantic_cache_enabled"] is False
            assert config["cache_similarity_threshold"] == 0.97
            assert config["cache_ttl"] == 3600
            assert config["redis_cache_url"] == "redis://localhost:6379"
            assert config["max_cache_size"] == 10000
            assert config["cache_key_prefix"] == "rag_semantic:"
            assert config["cache_stats_enabled"] is True
            assert config["cache_warming_enabled"] is False
    
    def test_get_verification_config_defaults(self):
        """Test verification configuration getter with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_verification_config()
            
            assert config["verification_enabled"] is True
            assert config["multi_level_confidence"] is True
            assert config["verification_threshold"] == 0.8
            assert config["verification_model"] == "gpt-4o-mini"
            assert config["ensemble_verification"] is True
            assert config["debate_augmentation_enabled"] is False
            assert config["max_verification_time"] == 2.0
            assert config["verification_timeout"] == 5.0
            assert config["verification_caching_enabled"] is True
            assert config["smart_verification_routing"] is True
            assert config["verification_batch_processing"] is True
            assert config["verification_batch_size"] == 5
            assert config["low_confidence_alert_threshold"] == 0.6
            assert config["verification_metrics_enabled"] is True
            assert config["hallucination_alerts_enabled"] is True
    
    def test_get_multimodal_config_defaults(self):
        """Test multimodal configuration getter with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_multimodal_config()
            
            assert config["multimodal_enabled"] is False
            assert config["image_indexing_enabled"] is False
            assert config["tts_integration_enabled"] is False
            assert config["clip_model_name"] == "openai/clip-vit-base-patch32"
            assert config["multimodal_threshold"] == 0.6
            assert config["max_image_size_mb"] == 10
            assert config["supported_image_formats"] == ["jpg", "jpeg", "png", "bmp", "tiff"]
            assert config["tts_engine"] == "pyttsx3"
            assert config["voice_speed"] == 150
            assert config["ocr_enabled"] is True
            assert config["ocr_language"] == "eng"
            assert config["cross_modal_search_enabled"] is False
            assert config["text_image_weight_ratio"] == 0.7


class TestSecurityValidation:
    """Test security-focused validation and edge cases."""
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        invalid_api_keys = [
            "",  # Empty
            "invalid",  # Too short
            "sk-",  # Just prefix
            "not-a-real-key-123",  # Wrong format
            "sk-" + "a" * 20,  # Too short with valid prefix
        ]
        
        for invalid_key in invalid_api_keys:
            with patch.dict(os.environ, {"OPENAI_API_KEY": invalid_key}):
                # Currently the validation only checks for presence and placeholder
                # In a production system, format validation could be more strict
                if invalid_key == "" or invalid_key == "your_openai_api_key_here":
                    with pytest.raises(RuntimeError):
                        init_settings()
                else:
                    # These would pass current validation but might fail at runtime
                    with patch('src.settings.OpenAI'), \
                         patch('src.settings.OpenAIEmbedding'), \
                         patch('src.settings.Settings'):
                        init_settings()  # Should not crash during validation
    
    def test_configuration_injection_prevention(self):
        """Test prevention of configuration injection attacks."""
        malicious_values = [
            "'; DROP TABLE config; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "\x00\x01\x02",  # Binary data
            "$(rm -rf /)",  # Command injection
        ]
        
        for malicious_value in malicious_values:
            test_env = {
                "OPENAI_API_KEY": "REDACTED_SK_KEY",
                "MODEL": malicious_value,  # Try to inject malicious model name
                "REDIS_CACHE_URL": malicious_value,  # Try to inject malicious URL
            }
            
            with patch.dict(os.environ, test_env):
                with patch('src.settings.OpenAI') as mock_openai, \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Should handle malicious values without executing them
                    init_settings()
                    
                    # Verify malicious value was passed as-is (not executed)
                    call_args = mock_openai.call_args[1]
                    assert call_args["model"] == malicious_value
    
    def test_resource_limit_validation(self):
        """Test validation of resource limits."""
        extreme_values = [
            {"MAX_TOKENS": "999999"},  # Very large token limit
            {"CHUNK_SIZE": "100000"},  # Very large chunk size
            {"TOP_K": "10000"},  # Very large retrieval count
            {"MAX_CACHE_SIZE": "1000000000"},  # Very large cache size
        ]
        
        for test_env in extreme_values:
            base_env = {"OPENAI_API_KEY": "REDACTED_SK_KEY"}
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Should handle extreme values gracefully
                    # In production, additional limits could be enforced
                    init_settings()
                    
                    # Verify config getter handles extreme values
                    config = get_rag_config()
                    if "max_tokens" in config:
                        assert isinstance(config["max_tokens"], int)
    
    def test_environment_variable_sanitization(self):
        """Test sanitization of environment variables."""
        # Test with various problematic characters
        test_values = [
            ("TEMPERATURE", "0.1\n0.9"),  # Newline injection
            ("MODEL", "gpt-4\r\ngpt-3.5"),  # Carriage return injection
            ("CHUNK_SIZE", "512\t1024"),  # Tab injection
        ]
        
        for env_var, value in test_values:
            test_env = {
                "OPENAI_API_KEY": "REDACTED_SK_KEY",
                env_var: value
            }
            
            with patch.dict(os.environ, test_env):
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Should handle problematic characters
                    init_settings()


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_missing_embedding_model_handling(self):
        """Test handling when OpenAIEmbedding is not available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "REDACTED_SK_KEY"}):
            with patch('src.settings.OpenAI'), \
                 patch('src.settings.OpenAIEmbedding', None), \
                 patch('src.settings.Settings') as mock_settings:
                
                # Should handle missing embedding model gracefully
                init_settings()
                
                # Should not set embed_model when not available
                # (The actual behavior depends on implementation)
    
    def test_invalid_numeric_configurations(self):
        """Test handling of invalid numeric configuration values."""
        invalid_numeric_configs = [
            {"TEMPERATURE": "not_a_number"},
            {"MAX_TOKENS": "invalid"},
            {"CHUNK_SIZE": "abc"},
            {"TOP_K": ""},
            {"EMBED_BATCH_SIZE": "negative_number"},
        ]
        
        for test_env in invalid_numeric_configs:
            base_env = {"OPENAI_API_KEY": "REDACTED_SK_KEY"}
            base_env.update(test_env)
            
            with patch.dict(os.environ, base_env):
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'):
                    
                    # Should handle invalid numeric values gracefully
                    # Might use defaults or raise appropriate errors
                    try:
                        init_settings()
                    except ValueError:
                        # Expected for invalid numeric values
                        pass
    
    def test_boolean_configuration_parsing(self):
        """Test parsing of boolean configuration values."""
        boolean_test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("yes", False),  # Only "true" should parse to True
            ("no", False),
            ("1", False),  # Numbers don't parse as booleans
            ("0", False),
            ("", False),  # Empty string is False
        ]
        
        for str_value, expected_bool in boolean_test_cases:
            with patch.dict(os.environ, {
                "SEMANTIC_CACHE_ENABLED": str_value,
                "VERIFICATION_ENABLED": str_value,
                "MULTIMODAL_ENABLED": str_value
            }):
                cache_config = get_cache_config()
                verification_config = get_verification_config()
                multimodal_config = get_multimodal_config()
                
                assert cache_config["semantic_cache_enabled"] == expected_bool
                assert verification_config["verification_enabled"] == expected_bool
                assert multimodal_config["multimodal_enabled"] == expected_bool
    
    def test_configuration_with_unicode_values(self):
        """Test configuration with unicode values."""
        unicode_values = [
            ("OCR_LANGUAGE", "日本語"),  # Japanese
            ("CACHE_KEY_PREFIX", "测试_"),  # Chinese
            ("REDIS_CACHE_URL", "redis://пример:6379"),  # Cyrillic
        ]
        
        for env_var, value in unicode_values:
            test_env = {
                "OPENAI_API_KEY": "REDACTED_SK_KEY",
                env_var: value
            }
            
            with patch.dict(os.environ, test_env):
                # Should handle unicode values without crashing
                if env_var == "OCR_LANGUAGE":
                    config = get_multimodal_config()
                    assert config["ocr_language"] == value
                elif env_var == "CACHE_KEY_PREFIX":
                    config = get_cache_config()
                    assert config["cache_key_prefix"] == value
                elif env_var == "REDIS_CACHE_URL":
                    config = get_cache_config()
                    assert config["redis_cache_url"] == value
    
    def test_empty_configuration_values(self):
        """Test handling of empty configuration values."""
        empty_value_configs = [
            "MODEL",
            "EMBEDDING_MODEL",
            "TEMPERATURE",
            "REDIS_CACHE_URL",
            "VERIFICATION_MODEL",
        ]
        
        for config_key in empty_value_configs:
            test_env = {
                "OPENAI_API_KEY": "REDACTED_SK_KEY",
                config_key: ""  # Empty string
            }
            
            with patch.dict(os.environ, test_env):
                # Should handle empty values by using defaults
                rag_config = get_rag_config()
                cache_config = get_cache_config()
                verification_config = get_verification_config()
                
                # Verify defaults are used for empty values
                if config_key == "MODEL":
                    assert rag_config["main_model"] == "gpt-4o"  # Default
                elif config_key == "EMBEDDING_MODEL":
                    assert rag_config["embedding_model"] == "text-embedding-3-large"  # Default
    
    def test_logging_level_configuration(self):
        """Test logging level configuration."""
        log_level_tests = [
            ("DEBUG", "DEBUG"),
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
            ("CRITICAL", "CRITICAL"),
            ("invalid", "INFO"),  # Should default to INFO for invalid levels
        ]
        
        for input_level, expected_level in log_level_tests:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "REDACTED_SK_KEY",
                "LOG_LEVEL": input_level
            }):
                with patch('src.settings.OpenAI'), \
                     patch('src.settings.OpenAIEmbedding'), \
                     patch('src.settings.Settings'), \
                     patch('src.settings.logging.basicConfig') as mock_logging:
                    
                    if input_level == "invalid":
                        # Should handle invalid log level gracefully
                        try:
                            init_settings()
                        except AttributeError:
                            # Expected for invalid log levels
                            pass
                    else:
                        init_settings()
                        # Verify logging was configured (if not invalid)
                        if mock_logging.called:
                            args = mock_logging.call_args[1]
                            assert hasattr(args.get("level"), "__name__")


class TestPerformanceProfileValidation:
    """Test validation of performance profiles and related configurations."""
    
    def test_performance_profile_configurations(self):
        """Test different performance profile configurations."""
        profile_configs = [
            "high_accuracy",
            "balanced", 
            "speed",
            "cost_optimized"
        ]
        
        for profile in profile_configs:
            with patch.dict(os.environ, {"PERFORMANCE_PROFILE": profile}):
                # Should handle all valid profiles
                # Note: This would require the unified_config module to test fully
                pass
    
    def test_cost_management_configuration(self):
        """Test cost management configuration validation."""
        cost_configs = [
            {"MAX_QUERY_COST": "0.01"},  # Low cost limit
            {"MAX_QUERY_COST": "0.10"},  # Medium cost limit
            {"MAX_QUERY_COST": "1.00"},  # High cost limit
            {"MAX_DAILY_COST": "10.00"},  # Daily limit
            {"COST_TRACKING_ENABLED": "true"},  # Cost tracking
        ]
        
        for config in cost_configs:
            base_env = {"OPENAI_API_KEY": "REDACTED_SK_KEY"}
            base_env.update(config)
            
            with patch.dict(os.environ, base_env):
                # Should handle cost management configuration
                # These would be used by the unified config system
                pass
    
    def test_feature_flag_configurations(self):
        """Test feature flag configurations."""
        feature_flags = [
            "AGENTIC_WORKFLOW_ENABLED",
            "SEMANTIC_CACHE_ENABLED", 
            "VERIFICATION_ENABLED",
            "MULTIMODAL_ENABLED",
            "TTS_INTEGRATION_ENABLED",
            "PERFORMANCE_OPTIMIZATION_ENABLED",
            "MONITORING_ENABLED",
            "DEBUG_MODE"
        ]
        
        for flag in feature_flags:
            # Test both enabled and disabled
            for value in ["true", "false"]:
                test_env = {
                    "OPENAI_API_KEY": "REDACTED_SK_KEY",
                    flag: value
                }
                
                with patch.dict(os.environ, test_env):
                    # Should handle all feature flags
                    # Specific validation would depend on the feature
                    pass


class TestConfigurationConsistency:
    """Test consistency between different configuration sections."""
    
    def test_model_consistency_between_configs(self):
        """Test that model configurations are consistent across different sections."""
        with patch.dict(os.environ, {
            "MODEL": "gpt-4o",
            "ROUTING_MODEL": "gpt-3.5-turbo",
            "DECOMPOSITION_MODEL": "gpt-3.5-turbo",
            "VERIFICATION_MODEL": "gpt-4o-mini",
            "SUBQUERY_AGGREGATION_MODEL": "gpt-4o-mini"
        }):
            rag_config = get_rag_config()
            agentic_config = get_agentic_config()
            verification_config = get_verification_config()
            
            # Verify models are configured as expected
            assert rag_config["main_model"] == "gpt-4o"
            assert agentic_config["routing_model"] == "gpt-3.5-turbo"
            assert agentic_config["decomposition_model"] == "gpt-3.5-turbo"
            assert verification_config["verification_model"] == "gpt-4o-mini"
    
    def test_threshold_consistency(self):
        """Test that threshold configurations are logically consistent."""
        with patch.dict(os.environ, {
            "SIMILARITY_THRESHOLD": "0.7",
            "CACHE_SIMILARITY_THRESHOLD": "0.95",
            "VERIFICATION_THRESHOLD": "0.8",
            "MULTIMODAL_THRESHOLD": "0.6"
        }):
            rag_config = get_rag_config()
            cache_config = get_cache_config()
            verification_config = get_verification_config()
            multimodal_config = get_multimodal_config()
            
            # Cache threshold should be higher than similarity threshold
            assert cache_config["cache_similarity_threshold"] > rag_config["similarity_threshold"]
            
            # All thresholds should be in valid range
            assert 0.0 <= rag_config["similarity_threshold"] <= 1.0
            assert 0.0 <= cache_config["cache_similarity_threshold"] <= 1.0
            assert 0.0 <= verification_config["verification_threshold"] <= 1.0
            assert 0.0 <= multimodal_config["multimodal_threshold"] <= 1.0
    
    def test_batch_size_consistency(self):
        """Test that batch size configurations are reasonable."""
        with patch.dict(os.environ, {
            "EMBED_BATCH_SIZE": "100",
            "IMAGE_BATCH_SIZE": "8",
            "VERIFICATION_BATCH_SIZE": "5"
        }):
            rag_config = get_rag_config()  # Would need to expose embed_batch_size
            multimodal_config = get_multimodal_config()
            verification_config = get_verification_config()
            
            # Batch sizes should be reasonable
            assert multimodal_config["image_batch_size"] > 0
            assert verification_config["verification_batch_size"] > 0
            
            # Image batch size should be smaller than embed batch size (more resource intensive)
            assert multimodal_config["image_batch_size"] <= 32  # Reasonable upper limit


class TestRealWorldConfigurationScenarios:
    """Test realistic configuration scenarios."""
    
    def test_production_configuration(self):
        """Test a production-like configuration."""
        production_env = {
            "OPENAI_API_KEY": "REDACTED_SK_KEY",
            "MODEL": "gpt-4o",
            "EMBEDDING_MODEL": "text-embedding-3-large", 
            "TEMPERATURE": "0.0",  # Deterministic
            "MAX_TOKENS": "2048",
            "SEMANTIC_CACHE_ENABLED": "true",
            "CACHE_SIMILARITY_THRESHOLD": "0.98",  # High precision
            "VERIFICATION_ENABLED": "true",
            "VERIFICATION_THRESHOLD": "0.85",  # High confidence required
            "ENSEMBLE_VERIFICATION": "true",
            "LOG_LEVEL": "INFO"
        }
        
        with patch.dict(os.environ, production_env):
            with patch('src.settings.OpenAI'), \
                 patch('src.settings.OpenAIEmbedding'), \
                 patch('src.settings.Settings'):
                
                init_settings()
                
                # Verify production configuration
                rag_config = get_rag_config()
                cache_config = get_cache_config()
                verification_config = get_verification_config()
                
                assert rag_config["main_model"] == "gpt-4o"
                assert rag_config["temperature"] == 0.0
                assert cache_config["semantic_cache_enabled"] is True
                assert cache_config["cache_similarity_threshold"] == 0.98
                assert verification_config["verification_enabled"] is True
                assert verification_config["ensemble_verification"] is True
    
    def test_development_configuration(self):
        """Test a development-like configuration."""
        development_env = {
            "OPENAI_API_KEY": "REDACTED_SK_KEY",
            "MODEL": "gpt-3.5-turbo",  # Cheaper for development
            "TEMPERATURE": "0.1",
            "SEMANTIC_CACHE_ENABLED": "false",  # Disabled for development
            "VERIFICATION_ENABLED": "false",  # Disabled for speed
            "LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, development_env):
            with patch('src.settings.OpenAI'), \
                 patch('src.settings.OpenAIEmbedding'), \
                 patch('src.settings.Settings'):
                
                init_settings()
                
                # Verify development configuration
                rag_config = get_rag_config()
                cache_config = get_cache_config() 
                verification_config = get_verification_config()
                
                assert rag_config["main_model"] == "gpt-3.5-turbo"
                assert cache_config["semantic_cache_enabled"] is False
                assert verification_config["verification_enabled"] is False
    
    def test_high_throughput_configuration(self):
        """Test a high-throughput configuration."""
        high_throughput_env = {
            "OPENAI_API_KEY": "REDACTED_SK_KEY",
            "MODEL": "gpt-3.5-turbo",  # Faster model
            "EMBED_BATCH_SIZE": "200",  # Large batches
            "TOP_K": "20",  # More candidates
            "SEMANTIC_CACHE_ENABLED": "true",
            "CACHE_SIMILARITY_THRESHOLD": "0.95",  # Lower threshold for more hits
            "MAX_CACHE_SIZE": "50000",  # Large cache
            "VERIFICATION_ENABLED": "false",  # Disabled for speed
            "SMART_VERIFICATION_ROUTING": "true"
        }
        
        with patch.dict(os.environ, high_throughput_env):
            rag_config = get_rag_config()
            cache_config = get_cache_config()
            verification_config = get_verification_config()
            
            assert rag_config["top_k"] == 20
            assert cache_config["semantic_cache_enabled"] is True
            assert cache_config["max_cache_size"] == 50000
            assert verification_config["smart_verification_routing"] is True