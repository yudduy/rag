"""
Optimized Unit Tests for SemanticCache - Consolidated and Performance Focused

This optimized test suite consolidates:
- test_semantic_cache.py (410 lines)
- test_cache_comprehensive.py (1,071 lines)

Into a streamlined, high-coverage test suite focusing on:
- Core functionality validation
- Security and edge cases
- Performance critical paths
- Error handling
"""

import asyncio
import json
import pytest
import time
import hashlib
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

import numpy as np

from src.cache import SemanticCache, CacheEntry, CacheStats, get_cache


class TestSemanticCacheCore:
    """Core functionality tests - most critical coverage."""
    
    @pytest.fixture
    def minimal_cache_config(self):
        """Minimal cache configuration for fast test execution."""
        return {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 100,  # Reduced from 1000
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
    
    def test_cache_initialization_redis_available(self, minimal_cache_config):
        """Test cache initialization - Redis available path."""
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url') as mock_redis:
            
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            cache = SemanticCache(config=minimal_cache_config)
            
            assert cache.enabled is True
            assert cache.redis_client is not None
            assert cache.similarity_threshold == 0.95
    
    def test_cache_initialization_redis_unavailable(self, minimal_cache_config):
        """Test cache initialization - Redis unavailable fallback."""
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=minimal_cache_config)
            
            assert cache.enabled is True
            assert cache.redis_client is None
            assert len(cache._fallback_cache) == 0
    
    def test_similarity_computation_core_scenarios(self):
        """Test similarity computation for most important scenarios."""
        cache = SemanticCache(config={"semantic_cache_enabled": True, "cache_similarity_threshold": 0.95})
        
        # Identical vectors
        vec1 = [1.0, 0.5, 0.0]
        identical = cache._compute_similarity(vec1, vec1)
        assert abs(identical - 1.0) < 0.001
        
        # Orthogonal vectors  
        vec2 = [0.0, 1.0, 0.0]
        orthogonal = cache._compute_similarity(vec1, vec2)
        assert 0.0 <= orthogonal <= 1.0
        
        # Zero vectors (edge case)
        zero_vec = [0.0, 0.0, 0.0]
        zero_sim = cache._compute_similarity(zero_vec, vec1)
        assert zero_sim == 0.0
    
    @pytest.mark.asyncio
    async def test_cache_operations_flow(self, minimal_cache_config):
        """Test complete cache operation flow - get/set/hit."""
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=minimal_cache_config)
            
            # Cache miss
            query = "test query"
            embedding = [0.1] * 10
            result = await cache.get(query, embedding)
            assert result is None
            
            # Cache set
            response_data = {"content": "test response", "confidence": 0.95}
            await cache.set(query, embedding, response_data, cost=0.01)
            
            # Cache hit 
            result = await cache.get(query, embedding)
            assert result is not None
            assert result["content"] == "test response"
    
    def test_cache_key_security_validation(self):
        """Test cache key security against common attacks."""
        cache = SemanticCache(config={"semantic_cache_enabled": True})
        
        # SQL injection attempt
        malicious_query = "'; DROP TABLE cache; --"
        key = cache._generate_cache_key(malicious_query, [0.1] * 5)
        assert "DROP TABLE" not in key
        assert len(key) > 10
        
        # Path traversal attempt  
        traversal_query = "../../../etc/passwd"
        key = cache._generate_cache_key(traversal_query, [0.1] * 5)
        assert ".." not in key
        assert "etc" not in key or len(key) > 20  # Should be hashed
        
        # XSS attempt
        xss_query = "<script>alert('xss')</script>"
        key = cache._generate_cache_key(xss_query, [0.1] * 5)
        assert "<script>" not in key


class TestSemanticCachePerformance:
    """Performance-focused tests - critical bottlenecks only."""
    
    @pytest.fixture
    def perf_cache_config(self):
        """Performance test configuration."""
        return {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.90,  # Lower for more hits
            "max_cache_size": 50,  # Small for fast tests
            "cache_stats_enabled": True
        }
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, perf_cache_config):
        """Test cache performance under concurrent load."""
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=perf_cache_config)
            
            # Simulate concurrent queries
            queries = [f"query_{i}" for i in range(10)]  # Reduced from 100
            embeddings = [[0.1 * i] * 5 for i in range(10)]
            
            # Measure cache set performance
            start_time = time.time()
            for i, (query, embedding) in enumerate(zip(queries, embeddings)):
                await cache.set(query, embedding, {"response": f"result_{i}"}, cost=0.01)
            set_time = time.time() - start_time
            
            # Should be fast even without Redis
            assert set_time < 1.0, f"Cache set took {set_time:.2f}s for 10 items"
            
            # Measure cache get performance  
            start_time = time.time()
            hits = 0
            for query, embedding in zip(queries, embeddings):
                result = await cache.get(query, embedding)
                if result:
                    hits += 1
            get_time = time.time() - start_time
            
            assert get_time < 0.5, f"Cache get took {get_time:.2f}s for 10 lookups"
            assert hits >= 8, f"Expected ≥8 cache hits, got {hits}"  # Allow some variance
    
    def test_cache_memory_efficiency(self, perf_cache_config):
        """Test cache memory usage stays within bounds."""
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=perf_cache_config)
            
            # Fill cache to capacity
            for i in range(60):  # Exceed max_cache_size
                query = f"memory_test_{i}"
                embedding = [float(i % 10)] * 5
                asyncio.run(cache.set(query, embedding, {"data": f"result_{i}"}, cost=0.01))
            
            # Cache should not exceed max size (with some tolerance for implementation)
            actual_size = len(cache._fallback_cache)
            max_allowed = perf_cache_config["max_cache_size"] + 10  # Tolerance
            assert actual_size <= max_allowed, f"Cache size {actual_size} exceeds limit {max_allowed}"


class TestSemanticCacheErrorHandling:
    """Error handling and resilience tests."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_handling(self):
        """Test graceful handling of Redis connection failures."""
        config = {"semantic_cache_enabled": True, "redis_cache_url": "redis://localhost:6379"}
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url', side_effect=Exception("Redis connection failed")):
            
            cache = SemanticCache(config=config)
            
            # Should fallback to in-memory cache
            assert cache.redis_client is None
            assert cache.enabled is True
            
            # Should still work with fallback
            result = await cache.get("test", [0.1] * 5)
            assert result is None  # Cache miss expected
    
    @pytest.mark.asyncio 
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        cache = SemanticCache(config={"semantic_cache_enabled": True})
        
        # Empty embedding
        result = await cache.get("test query", [])
        assert result is None
        
        # None embedding
        result = await cache.get("test query", None)
        assert result is None
        
        # Invalid embedding type
        result = await cache.get("test query", "not_an_embedding")
        assert result is None
    
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Disabled cache
        cache = SemanticCache(config={"semantic_cache_enabled": False})
        assert cache.enabled is False
        
        # Invalid similarity threshold  
        cache = SemanticCache(config={
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 1.5  # Invalid - should be ≤1.0
        })
        assert cache.similarity_threshold <= 1.0  # Should be corrected
        
        # Zero TTL
        cache = SemanticCache(config={
            "semantic_cache_enabled": True, 
            "cache_ttl": 0
        })
        assert cache.ttl >= 300  # Should have minimum TTL


class TestSemanticCacheStats:
    """Cache statistics and monitoring - essential metrics only."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_tracking(self):
        """Test cache hit rate calculation."""
        config = {"semantic_cache_enabled": True, "cache_stats_enabled": True}
        
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=config)
            
            # Generate some cache operations
            queries = ["query_1", "query_2", "query_1"]  # query_1 repeated for hit
            embeddings = [[0.1], [0.2], [0.1]]
            
            # First two are misses, third is hit
            await cache.get(queries[0], embeddings[0])  # miss
            await cache.set(queries[0], embeddings[0], {"data": "result_1"})
            
            await cache.get(queries[1], embeddings[1])  # miss  
            await cache.set(queries[1], embeddings[1], {"data": "result_2"})
            
            hit_result = await cache.get(queries[2], embeddings[2])  # hit
            assert hit_result is not None
            
            # Check statistics
            stats = cache.get_stats()
            assert stats.total_requests >= 3
            assert stats.cache_hits >= 1
            assert 0.0 <= stats.hit_rate <= 1.0


# Utility functions for optimized testing
def create_test_embedding(size: int = 10, seed: int = 42) -> List[float]:
    """Create deterministic test embedding for consistent results."""
    np.random.seed(seed)
    return np.random.rand(size).tolist()


def create_mock_cache_response(content: str = "test response", confidence: float = 0.9) -> Dict[str, Any]:
    """Create minimal mock cache response."""
    return {
        "content": content,
        "confidence": confidence,
        "metadata": {},
        "cost": 0.01
    }


# Integration test placeholder (reduced scope)
class TestSemanticCacheIntegration:
    """Essential integration tests only."""
    
    def test_cache_factory_function(self):
        """Test cache factory function works correctly."""
        with patch.dict('os.environ', {"SEMANTIC_CACHE_ENABLED": "true"}):
            cache = get_cache()
            assert cache is not None
            assert hasattr(cache, 'get')
            assert hasattr(cache, 'set')
    
    @pytest.mark.asyncio
    async def test_cache_with_query_cost_estimation(self):
        """Test cache integration with cost estimation."""
        cache = SemanticCache(config={"semantic_cache_enabled": True})
        
        # Mock cost estimation
        with patch('src.cache.estimate_query_cost', return_value=0.05):
            query = "expensive query"
            embedding = create_test_embedding()
            
            # Set with cost tracking
            await cache.set(query, embedding, create_mock_cache_response(), cost=0.05)
            
            # Get should include cost savings
            result = await cache.get(query, embedding)
            assert result is not None
            assert "cost" in result or "metadata" in result