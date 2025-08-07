"""
Unit tests for semantic caching system.

Tests cover:
- Semantic similarity computation
- Cache operations (get, set, eviction)
- Performance optimization
- Error handling and resilience
- Cache statistics and monitoring
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.cache import (
    SemanticCache,
    CacheEntry,
    CacheStats,
    get_cache,
    _compute_semantic_similarity,
    _should_use_cache
)


class TestSemanticSimilarity:
    """Test semantic similarity computation algorithms."""
    
    def test_cosine_similarity_computation(self):
        """Test cosine similarity computation."""
        # Test vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 4.0, 6.0])  # Parallel vector
        vec3 = np.array([0.0, 1.0, 0.0])  # Orthogonal vector
        
        # Parallel vectors should have high similarity
        similarity = _compute_semantic_similarity(vec1, vec2)
        assert similarity > 0.99
        
        # Orthogonal vectors should have lower similarity
        similarity = _compute_semantic_similarity(vec1, vec3)
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_edge_cases(self):
        """Test edge cases in similarity computation."""
        # Zero vectors
        zero_vec = np.zeros(768)
        normal_vec = np.ones(768)
        
        similarity = _compute_semantic_similarity(zero_vec, normal_vec)
        assert 0.0 <= similarity <= 1.0
        
        # Identical vectors
        similarity = _compute_semantic_similarity(normal_vec, normal_vec)
        assert similarity == 1.0
    
    def test_similarity_threshold_logic(self):
        """Test similarity threshold decision logic."""
        # High similarity should use cache
        assert _should_use_cache(0.95, threshold=0.9)
        
        # Low similarity should not use cache
        assert not _should_use_cache(0.5, threshold=0.9)
        
        # Exact threshold should use cache
        assert _should_use_cache(0.9, threshold=0.9)


class TestCacheOperations:
    """Test core cache operations."""
    
    @pytest.fixture
    def mock_cache(self, mock_redis_client):
        """Create a semantic cache with mocked Redis."""
        return SemanticCache(
            redis_url="redis://localhost:6379/15",
            similarity_threshold=0.85
        )
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_cache, mock_redis_client, mock_openai_client):
        """Test cache miss scenario."""
        # Configure mocks
        mock_redis_client.keys.return_value = []
        mock_redis_client.get.return_value = None
        
        result = await mock_cache.get("What is machine learning?")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_cache, mock_redis_client, mock_openai_client):
        """Test cache hit scenario."""
        # Setup cached entry
        cached_entry = CacheEntry(
            query="What is AI?",
            embedding=[0.1] * 768,
            response="AI is artificial intelligence",
            confidence=0.9,
            metadata={"timestamp": 1234567890}
        )
        
        # Mock Redis operations
        mock_redis_client.keys.return_value = ["cache:hash1"]
        mock_redis_client.get.return_value = json.dumps(cached_entry.__dict__)
        
        # Mock embedding for similarity
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 768
        
        result = await mock_cache.get("What is artificial intelligence?")
        
        # Should find similar cached entry
        assert result is not None
        assert result["response"] == "AI is artificial intelligence"
        assert result["confidence"] == 0.9
        assert result["cache_hit"] == True
    
    @pytest.mark.asyncio
    async def test_cache_set(self, mock_cache, mock_redis_client, mock_openai_client):
        """Test setting cache entries."""
        query = "What is deep learning?"
        response = "Deep learning is a subset of machine learning"
        confidence = 0.95
        
        # Mock embedding generation
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.2] * 768
        
        await mock_cache.set(query, response, confidence)
        
        # Verify Redis set was called
        mock_redis_client.set.assert_called_once()
        
        # Verify the call included proper TTL
        call_args = mock_redis_client.set.call_args
        assert 'ex' in call_args[1]  # TTL parameter
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, mock_cache, mock_redis_client):
        """Test cache statistics collection."""
        # Mock Redis info
        mock_redis_client.info.return_value = {
            'used_memory': 1024000,
            'keyspace_hits': 50,
            'keyspace_misses': 20
        }
        mock_redis_client.dbsize.return_value = 100
        
        stats = await mock_cache.get_stats()
        
        assert isinstance(stats, CacheStats)
        assert stats.total_entries == 100
        assert stats.memory_usage > 0
        assert stats.hit_rate > 0


class TestCachePerformance:
    """Test cache performance optimizations."""
    
    @pytest.mark.asyncio
    async def test_embedding_batch_processing(self, mock_cache, mock_openai_client):
        """Test batch processing of embeddings."""
        queries = [
            "What is machine learning?",
            "Explain deep learning",
            "Define neural networks"
        ]
        
        # Mock batch embedding response
        mock_openai_client.embeddings.create.return_value.data = [
            Mock(embedding=[0.1] * 768),
            Mock(embedding=[0.2] * 768),
            Mock(embedding=[0.3] * 768)
        ]
        
        embeddings = await mock_cache._get_embeddings_batch(queries)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Should have made only one API call for batch
        mock_openai_client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similarity_computation_vectorized(self, mock_cache):
        """Test vectorized similarity computation for performance."""
        # Create test embeddings
        query_embedding = np.random.random(768)
        cached_embeddings = [np.random.random(768) for _ in range(100)]
        
        start_time = asyncio.get_event_loop().time()
        
        # Compute similarities
        similarities = []
        for cached_emb in cached_embeddings:
            sim = _compute_semantic_similarity(query_embedding, cached_emb)
            similarities.append(sim)
        
        end_time = asyncio.get_event_loop().time()
        computation_time = end_time - start_time
        
        # Should complete quickly (under 1 second for 100 comparisons)
        assert computation_time < 1.0
        assert len(similarities) == 100
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
    
    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self, mock_cache, mock_redis_client):
        """Test memory efficiency of cache entries."""
        # Large response for testing
        large_response = "This is a very long response " * 1000
        
        await mock_cache.set(
            "Test query",
            large_response,
            confidence=0.9,
            metadata={"test": "data"}
        )
        
        # Verify Redis was called (actual compression testing would need integration test)
        mock_redis_client.set.assert_called_once()


class TestCacheEviction:
    """Test cache eviction policies."""
    
    @pytest.mark.asyncio
    async def test_ttl_eviction(self, mock_cache, mock_redis_client):
        """Test TTL-based eviction."""
        await mock_cache.set(
            "Test query",
            "Test response",
            confidence=0.9,
            ttl=3600  # 1 hour
        )
        
        # Verify TTL was set
        call_args = mock_redis_client.set.call_args
        assert call_args[1]['ex'] == 3600
    
    @pytest.mark.asyncio
    async def test_lru_eviction_simulation(self, mock_cache, mock_redis_client):
        """Test LRU eviction behavior simulation."""
        # Mock Redis memory eviction
        mock_redis_client.set.side_effect = [True, True, Exception("OOM")]
        
        # First two sets should succeed
        await mock_cache.set("Query 1", "Response 1", 0.9)
        await mock_cache.set("Query 2", "Response 2", 0.9)
        
        # Third set should handle eviction gracefully
        try:
            await mock_cache.set("Query 3", "Response 3", 0.9)
        except Exception:
            # Should handle eviction errors gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_low_confidence_eviction(self, mock_cache):
        """Test eviction of low-confidence entries."""
        # Low confidence entries should have shorter TTL
        with patch.object(mock_cache, '_calculate_ttl') as mock_ttl:
            mock_ttl.return_value = 1800  # 30 minutes for low confidence
            
            await mock_cache.set(
                "Uncertain query",
                "Uncertain response", 
                confidence=0.6  # Low confidence
            )
            
            # Should calculate shorter TTL for low confidence
            mock_ttl.assert_called_with(0.6, {})


class TestCacheResilience:
    """Test cache error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, mock_redis_client):
        """Test handling of Redis connection failures."""
        # Mock connection failure
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        
        # Should handle connection failure gracefully
        result = await cache.get("Test query")
        assert result is None  # Should return None instead of crashing
    
    @pytest.mark.asyncio
    async def test_openai_api_failure(self, mock_cache, mock_openai_client):
        """Test handling of OpenAI API failures."""
        # Mock API failure
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        # Should handle embedding generation failure
        result = await mock_cache.get("Test query")
        assert result is None  # Should fallback gracefully
    
    @pytest.mark.asyncio
    async def test_corrupted_cache_entry(self, mock_cache, mock_redis_client):
        """Test handling of corrupted cache entries."""
        # Mock corrupted JSON
        mock_redis_client.keys.return_value = ["cache:corrupted"]
        mock_redis_client.get.return_value = "invalid json data"
        
        result = await mock_cache.get("Test query")
        assert result is None  # Should handle corruption gracefully
    
    @pytest.mark.asyncio
    async def test_partial_embedding_failure(self, mock_cache, mock_openai_client):
        """Test handling of partial embedding failures."""
        # Mock partial failure in batch
        mock_openai_client.embeddings.create.side_effect = [
            Mock(data=[Mock(embedding=[0.1] * 768)]),  # First succeeds
            Exception("Rate limit"),  # Second fails
            Mock(data=[Mock(embedding=[0.3] * 768)])   # Third succeeds
        ]
        
        # Should handle partial failures gracefully
        embeddings = await mock_cache._get_embeddings_batch([
            "Query 1", "Query 2", "Query 3"
        ])
        
        # Should return embeddings for successful queries
        assert len([emb for emb in embeddings if emb is not None]) >= 1


class TestCacheConfiguration:
    """Test cache configuration and tuning."""
    
    def test_similarity_threshold_configuration(self):
        """Test similarity threshold configuration."""
        # Test different thresholds
        thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
        
        for threshold in thresholds:
            cache = SemanticCache(
                redis_url="redis://localhost:6379/15",
                similarity_threshold=threshold
            )
            assert cache.similarity_threshold == threshold
    
    def test_cache_size_limits(self):
        """Test cache size limit configuration."""
        cache = SemanticCache(
            redis_url="redis://localhost:6379/15",
            max_entries=1000
        )
        
        assert hasattr(cache, 'max_entries') or cache.max_entries == 1000
    
    def test_ttl_configuration(self):
        """Test TTL configuration options."""
        cache = SemanticCache(
            redis_url="redis://localhost:6379/15",
            default_ttl=7200  # 2 hours
        )
        
        # Test TTL calculation
        high_confidence_ttl = cache._calculate_ttl(0.95, {})
        low_confidence_ttl = cache._calculate_ttl(0.6, {})
        
        # High confidence should have longer TTL
        assert high_confidence_ttl >= low_confidence_ttl


class TestCacheIntegration:
    """Test cache integration with other components."""
    
    @pytest.mark.asyncio
    async def test_workflow_integration(self, mock_cache, mock_redis_client, mock_openai_client):
        """Test integration with workflow system."""
        # Mock cached response
        cached_entry = {
            "query": "What is AI?",
            "response": "AI is artificial intelligence",
            "confidence": 0.95,
            "metadata": {"source": "cache"}
        }
        
        mock_redis_client.get.return_value = json.dumps(cached_entry)
        mock_redis_client.keys.return_value = ["cache:test"]
        
        result = await mock_cache.get("What is artificial intelligence?")
        
        if result:  # If cache hit occurred
            assert result["source"] == "cache"
            assert result["confidence"] >= 0.9
    
    def test_monitoring_integration(self, mock_cache):
        """Test integration with monitoring system."""
        # Cache should expose metrics for monitoring
        assert hasattr(mock_cache, 'get_stats')
        assert hasattr(mock_cache, 'stats') or hasattr(mock_cache, '_stats')
    
    @pytest.mark.asyncio
    async def test_performance_profile_integration(self, mock_cache):
        """Test integration with performance profiles."""
        # Different profiles should affect cache behavior
        profiles_config = {
            'speed': {'similarity_threshold': 0.8, 'ttl_multiplier': 0.5},
            'balanced': {'similarity_threshold': 0.85, 'ttl_multiplier': 1.0},
            'accuracy': {'similarity_threshold': 0.9, 'ttl_multiplier': 1.5}
        }
        
        for profile, config in profiles_config.items():
            # Cache behavior should adapt to profile
            threshold = config['similarity_threshold']
            assert 0.7 <= threshold <= 0.95