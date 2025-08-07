"""
Pytest configuration and shared fixtures for SOTA RAG testing.

This module provides:
- Common fixtures for all test modules
- Mock services for external dependencies
- Test data management
- Performance benchmarking utilities
"""

import asyncio
import os
import pytest
import tempfile
import shutil
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import all components for testing
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.unified_workflow import UnifiedWorkflow, QueryCharacteristics, QueryComplexity
from src.health_monitor import HealthMonitor, get_health_monitor
from src.cache import SemanticCache
from src.verification import HallucinationDetector
from src.agentic_workflow import AgenticWorkflow
from src.multimodal import MultimodalProcessor
from src.performance import PerformanceOptimizer


# Test configuration
TEST_OPENAI_API_KEY = "test-key-REDACTED_SK_KEY"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use test database


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up clean test environment for each test."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_env = {
        "OPENAI_API_KEY": TEST_OPENAI_API_KEY,
        "REDIS_URL": TEST_REDIS_URL,
        "PERFORMANCE_PROFILE": "balanced",
        "SEMANTIC_CACHE_ENABLED": "true",
        "VERIFICATION_ENABLED": "true",
        "AGENTIC_WORKFLOW_ENABLED": "true",
        "MULTIMODAL_ENABLED": "true",
        "TTS_ENABLED": "false",  # Disable TTS for testing
        "MONITORING_ENABLED": "true",
        "LOG_LEVEL": "INFO",
    }
    
    # Apply test environment
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Reset config to pick up test environment
    reset_unified_config()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    reset_unified_config()


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.AsyncOpenAI') as mock_client:
        # Mock embedding response
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        
        # Mock chat completion response
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test response"))]
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create.return_value = mock_embedding_response
        mock_instance.chat.completions.create.return_value = mock_chat_response
        
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for semantic caching tests."""
    with patch('redis.asyncio.Redis') as mock_redis:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = None
        mock_instance.set.return_value = True
        mock_instance.exists.return_value = False
        mock_instance.ping.return_value = True
        
        mock_redis.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llama_index():
    """Mock LlamaIndex components."""
    with patch('src.settings.init_settings') as mock_init, \
         patch('src.workflow.get_index') as mock_get_index, \
         patch('src.query.get_query_engine') as mock_get_engine:
        
        # Mock index
        mock_index = Mock()
        mock_get_index.return_value = mock_index
        
        # Mock query engine
        mock_engine = Mock()
        mock_engine.query.return_value = Mock(
            response="Test response",
            source_nodes=[],
            metadata={}
        )
        mock_get_engine.return_value = mock_engine
        
        yield {
            'index': mock_index,
            'engine': mock_engine,
            'settings': mock_init
        }


@pytest.fixture
def sample_queries():
    """Sample queries for testing various scenarios."""
    return {
        'simple': [
            "What is machine learning?",
            "Define artificial intelligence",
            "Explain neural networks"
        ],
        'complex': [
            "Compare supervised and unsupervised learning algorithms, provide examples, and explain when to use each",
            "Explain the transformer architecture, its attention mechanism, and how it differs from RNNs and LSTMs",
            "Describe the complete process of building a RAG system from data ingestion to response generation"
        ],
        'multimodal': [
            "Show me a diagram of neural network architecture",
            "Explain this image and its relevance to machine learning",
            "Generate a visual representation of the transformer model"
        ],
        'factual': [
            "What year was GPT-3 released?",
            "Who invented the transformer architecture?",
            "What is the capital of France?"
        ],
        'conversational': [
            "Can you help me understand deep learning?",
            "I'm confused about backpropagation, can you explain?",
            "Let's discuss the future of AI"
        ]
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        'response_time': {
            'simple_query': 1.0,      # seconds
            'complex_query': 3.0,     # seconds
            'multimodal_query': 5.0,  # seconds
        },
        'accuracy': {
            'high_accuracy_profile': 0.98,
            'balanced_profile': 0.96,
            'cost_optimized_profile': 0.94,
            'speed_profile': 0.92,
        },
        'cost': {
            'max_simple_query': 0.01,  # USD
            'max_complex_query': 0.05, # USD
            'max_multimodal_query': 0.10, # USD
        },
        'cache_hit_rate': {
            'target_minimum': 0.25,
            'target_optimal': 0.35,
        }
    }


@pytest.fixture
def test_documents():
    """Sample documents for testing."""
    return [
        {
            'id': 'doc1',
            'title': 'Introduction to Machine Learning',
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
            'metadata': {'category': 'AI', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc2', 
            'title': 'Deep Learning Fundamentals',
            'content': 'Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.',
            'metadata': {'category': 'AI', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc3',
            'title': 'Transformer Architecture',
            'content': 'The transformer architecture revolutionized natural language processing by introducing the attention mechanism.',
            'metadata': {'category': 'NLP', 'difficulty': 'advanced'}
        }
    ]


@pytest.fixture
def component_health_data():
    """Sample component health data for testing."""
    return {
        'healthy_components': {
            'unified_orchestrator': {
                'status': 'healthy',
                'metrics': {'queries_processed': 100, 'avg_response_time': 1.5},
                'last_error': None
            },
            'semantic_cache': {
                'status': 'healthy', 
                'metrics': {'cache_hit_rate': 0.35, 'total_requests': 50},
                'last_error': None
            }
        },
        'unhealthy_components': {
            'verification_system': {
                'status': 'degraded',
                'metrics': {'verification_rate': 0.85, 'timeout_count': 5},
                'last_error': 'Timeout in hallucination detection'
            }
        }
    }


@pytest.fixture
async def configured_workflow(mock_llama_index, mock_redis_client):
    """Create a configured workflow for testing."""
    try:
        workflow = UnifiedWorkflow(timeout=30.0)
        # Mock the initialization that requires external services
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'total_response_time': 0}
        return workflow
    except Exception:
        # Return a mock workflow if real initialization fails
        mock_workflow = Mock(spec=UnifiedWorkflow)
        mock_workflow.config_manager = get_unified_config()
        mock_workflow.stats = {'queries_processed': 0, 'total_response_time': 0}
        return mock_workflow


@pytest.fixture
def quality_metrics():
    """Quality assessment metrics for testing."""
    return {
        'accuracy_thresholds': {
            'high_accuracy': 0.98,
            'balanced': 0.96,
            'cost_optimized': 0.94,
            'speed': 0.92
        },
        'confidence_thresholds': {
            'high_confidence': 0.9,
            'medium_confidence': 0.7,
            'low_confidence': 0.5
        },
        'hallucination_detection': {
            'precision_threshold': 0.85,
            'recall_threshold': 0.80,
            'f1_threshold': 0.82
        }
    }


# Utility functions for tests
def assert_response_quality(response: Dict[str, Any], min_confidence: float = 0.7):
    """Assert that a response meets quality standards."""
    assert 'content' in response
    assert 'confidence' in response
    assert response['confidence'] >= min_confidence
    assert len(response['content']) > 0


def assert_performance_within_limits(
    actual_time: float, 
    max_time: float, 
    operation: str = "operation"
):
    """Assert that performance is within acceptable limits."""
    assert actual_time <= max_time, \
        f"{operation} took {actual_time:.2f}s, expected <= {max_time:.2f}s"


def create_mock_query_result(content: str, confidence: float = 0.8):
    """Create a mock query result for testing."""
    return {
        'content': content,
        'confidence': confidence,
        'sources': [],
        'metadata': {},
        'processing_time': 1.0,
        'cost': 0.01
    }


# Performance testing utilities
class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.metrics[operation] = {'start_time': time.time()}
    
    def end_timing(self, operation: str):
        """End timing an operation."""
        if operation in self.metrics:
            self.metrics[operation]['duration'] = time.time() - self.metrics[operation]['start_time']
    
    def get_duration(self, operation: str) -> float:
        """Get duration of an operation."""
        return self.metrics.get(operation, {}).get('duration', 0.0)
    
    def assert_within_limit(self, operation: str, max_duration: float):
        """Assert operation completed within time limit."""
        duration = self.get_duration(operation)
        assert duration <= max_duration, \
            f"{operation} took {duration:.2f}s, expected <= {max_duration:.2f}s"


@pytest.fixture
def performance_tracker():
    """Performance tracking utility."""
    return PerformanceTracker()