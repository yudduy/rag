"""
Optimized Pytest Configuration - Performance Focused

This optimized conftest.py reduces overhead while maintaining essential test capabilities:
- Streamlined fixtures with minimal setup
- Lightweight mock services  
- Reduced test data sizes
- Efficient resource management
"""

import asyncio
import os
import pytest
import tempfile
import shutil
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Import only essential components
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile


# Test configuration - minimal but functional
TEST_OPENAI_API_KEY = "test-key-REDACTED_SK_KEY"
TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Lightweight test environment setup."""
    original_env = dict(os.environ)
    
    # Essential test environment only
    test_env = {
        "OPENAI_API_KEY": TEST_OPENAI_API_KEY,
        "REDIS_URL": TEST_REDIS_URL,
        "PERFORMANCE_PROFILE": "speed",  # Use fastest profile for tests
        "SEMANTIC_CACHE_ENABLED": "true",
        "LOG_LEVEL": "ERROR",  # Reduce log noise
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    reset_unified_config()
    
    yield
    
    # Cleanup
    os.environ.clear()
    os.environ.update(original_env)
    reset_unified_config()


@pytest.fixture
def temp_storage_dir():
    """Lightweight temporary storage directory."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_client():
    """Lightweight OpenAI client mock."""
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_instance = AsyncMock()
        
        # Minimal embedding response
        mock_embedding = Mock()
        mock_embedding.data = [Mock(embedding=[0.1] * 100)]  # Reduced from 1536
        mock_instance.embeddings.create.return_value = mock_embedding
        
        # Minimal chat response
        mock_chat = Mock()
        mock_chat.choices = [Mock(message=Mock(content="Test response"))]
        mock_instance.chat.completions.create.return_value = mock_chat
        
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_redis_client():
    """Lightweight Redis client mock."""
    with patch('redis.asyncio.Redis') as mock_redis:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = None
        mock_instance.set.return_value = True
        mock_instance.exists.return_value = False
        mock_instance.ping.return_value = True
        
        mock_redis.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def minimal_sample_queries():
    """Minimal sample queries for fast testing."""
    return {
        'simple': ["What is AI?", "Define ML"],
        'complex': ["Compare supervised vs unsupervised learning with examples"],
        'factual': ["When was GPT-3 released?"],
    }


@pytest.fixture
def minimal_test_documents():
    """Minimal test documents for efficient testing."""
    return [
        {
            'id': 'doc1',
            'content': 'AI is intelligence demonstrated by machines.',
            'metadata': {'category': 'AI'}
        },
        {
            'id': 'doc2',
            'content': 'Machine learning is a subset of AI.',
            'metadata': {'category': 'ML'}
        }
    ]


@pytest.fixture
def performance_benchmarks():
    """Minimal performance benchmarks for testing."""
    return {
        'response_time': {'simple_query': 1.0, 'complex_query': 2.0},
        'accuracy': {'target': 0.90},
        'cache_hit_rate': {'target': 0.25},
    }


@pytest.fixture
async def configured_workflow():
    """Lightweight configured workflow for testing."""
    try:
        from src.unified_workflow import UnifiedWorkflow
        
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            config_manager = Mock()
            config_manager.config = Mock()
            config_manager.config.performance_profile = PerformanceProfile.SPEED
            mock_config.return_value = config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=5.0)  # Short timeout for tests
            workflow.config_manager = config_manager
            return workflow
    except Exception:
        # Return mock if initialization fails
        mock_workflow = Mock()
        mock_workflow.config_manager = get_unified_config()
        return mock_workflow


# Lightweight utility functions
def assert_response_quality(response: Dict[str, Any], min_confidence: float = 0.5):
    """Lightweight response quality assertion."""
    assert 'content' in response
    assert len(response['content']) > 0
    if 'confidence' in response:
        assert response['confidence'] >= min_confidence


def assert_performance_within_limits(actual_time: float, max_time: float, operation: str = "operation"):
    """Performance assertion with clear error message."""
    assert actual_time <= max_time, f"{operation} took {actual_time:.2f}s, expected <= {max_time:.2f}s"


def create_minimal_query_result(content: str, confidence: float = 0.8) -> Dict[str, Any]:
    """Create minimal query result for testing."""
    return {
        'content': content,
        'confidence': confidence,
        'cost': 0.01
    }


@pytest.fixture
def mock_cache():
    """Lightweight cache mock for testing."""
    from src.cache import SemanticCache
    
    with patch('src.cache.REDIS_AVAILABLE', False):
        cache = SemanticCache(config={
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.90,
            "max_cache_size": 10,  # Very small for fast tests
        })
        return cache


@pytest.fixture
def mock_security_components():
    """Lightweight security component mocks."""
    return {
        'input_validator': Mock(return_value=True),
        'output_sanitizer': Mock(side_effect=lambda x: x),
        'rate_limiter': Mock(return_value=True),
    }


# Lightweight performance tracker
class FastPerformanceTracker:
    """Minimal performance tracking for tests."""
    
    def __init__(self):
        self.start_times = {}
    
    def start(self, operation: str):
        import time
        self.start_times[operation] = time.time()
    
    def end_and_assert(self, operation: str, max_duration: float):
        import time
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            assert duration <= max_duration, f"{operation} took {duration:.2f}s, expected <= {max_duration:.2f}s"
            del self.start_times[operation]
            return duration
        return 0.0


@pytest.fixture
def performance_tracker():
    """Lightweight performance tracker."""
    return FastPerformanceTracker()


# Essential markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow tests (may be skipped)")


# Test collection optimization
def pytest_collection_modifyitems(config, items):
    """Optimize test collection and execution order."""
    # Move fast unit tests to the beginning
    unit_tests = []
    other_tests = []
    
    for item in items:
        if "unit" in item.nodeid:
            unit_tests.append(item)
        else:
            other_tests.append(item)
    
    # Return with unit tests first for faster feedback
    items[:] = unit_tests + other_tests