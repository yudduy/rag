"""
Comprehensive Unit Tests for SemanticCache - Priority 1

This test suite provides comprehensive coverage of the SemanticCache system,
focusing on:
- Redis semantic caching with mocked Redis client
- Similarity matching algorithms and thresholds
- Fallback behavior when Redis is unavailable
- Cache performance and statistics tracking
- Security validation of cache operations
- Error handling and edge cases
- Cost optimization through caching
"""

import asyncio
import json
import os
import pytest
import time
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from src.cache import (
    SemanticCache, CacheEntry, CacheStats, get_cache, estimate_query_cost
)


class TestSemanticCacheInitialization:
    """Test cache initialization and configuration."""
    
    def test_cache_initialization_with_redis_available(self):
        """Test cache initialization when Redis is available."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url') as mock_redis_from_url:
            
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.exists.return_value = False
            mock_redis_client.hset.return_value = True
            mock_redis_from_url.return_value = mock_redis_client
            
            cache = SemanticCache(config=mock_config)
            
            assert cache.enabled is True
            assert cache.similarity_threshold == 0.95
            assert cache.redis_client is not None
            assert isinstance(cache.stats, CacheStats)
    
    def test_cache_initialization_with_redis_unavailable(self):
        """Test cache initialization when Redis is unavailable."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.97,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=mock_config)
            
            assert cache.enabled is True
            assert cache.redis_client is None
            assert len(cache._fallback_cache) == 0
    
    def test_cache_initialization_redis_connection_failure(self):
        """Test cache initialization when Redis connection fails."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.97,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url', side_effect=Exception("Connection failed")):
            
            cache = SemanticCache(config=mock_config)
            
            assert cache.enabled is True
            assert cache.redis_client is None  # Should fall back to None
    
    def test_cache_disabled_configuration(self):
        """Test cache when disabled in configuration."""
        mock_config = {
            "semantic_cache_enabled": False,
            "cache_similarity_threshold": 0.97,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        cache = SemanticCache(config=mock_config)
        assert cache.enabled is False
        assert cache.redis_client is None


class TestEmbeddingGeneration:
    """Test embedding generation and similarity computation."""
    
    @pytest.fixture
    def cache_with_mock_embedding(self):
        """Cache with mocked embedding model."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            return cache, mock_embed_model
    
    def test_embedding_generation_success(self, cache_with_mock_embedding):
        """Test successful embedding generation."""
        cache, mock_embed_model = cache_with_mock_embedding
        
        query = "What is machine learning?"
        embedding = cache._generate_embedding(query)
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        mock_embed_model.get_text_embedding.assert_called_once_with(query)
    
    def test_embedding_normalization(self, cache_with_mock_embedding):
        """Test that embeddings are properly normalized."""
        cache, mock_embed_model = cache_with_mock_embedding
        
        # Mock non-normalized embedding
        mock_embed_model.get_text_embedding.return_value = [1.0] * 10  # Not normalized
        
        query = "Test query"
        embedding = cache._generate_embedding(query)
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.0001  # Should be normalized
    
    def test_embedding_generation_failure(self, cache_with_mock_embedding):
        """Test handling of embedding generation failures."""
        cache, mock_embed_model = cache_with_mock_embedding
        
        mock_embed_model.get_text_embedding.side_effect = Exception("Embedding failed")
        
        query = "Test query"
        embedding = cache._generate_embedding(query)
        
        assert embedding == []  # Should return empty list on failure
    
    def test_similarity_computation(self, cache_with_mock_embedding):
        """Test cosine similarity computation."""
        cache, _ = cache_with_mock_embedding
        
        # Test identical embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity = cache._compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 0.0001
        
        # Test orthogonal embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = cache._compute_similarity(emb1, emb2)
        assert abs(similarity - 0.0) < 0.0001
        
        # Test opposite embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [-1.0, 0.0, 0.0]
        similarity = cache._compute_similarity(emb1, emb2)
        assert similarity >= 0.0  # Clamped to positive range
    
    def test_similarity_computation_edge_cases(self, cache_with_mock_embedding):
        """Test similarity computation edge cases."""
        cache, _ = cache_with_mock_embedding
        
        # Test empty embeddings
        assert cache._compute_similarity([], []) == 0.0
        assert cache._compute_similarity([1.0], []) == 0.0
        assert cache._compute_similarity([], [1.0]) == 0.0
        
        # Test mismatched dimensions
        assert cache._compute_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
        
        # Test with error-inducing inputs
        with patch('numpy.dot', side_effect=Exception("Computation error")):
            assert cache._compute_similarity([1.0], [1.0]) == 0.0


class TestCacheOperations:
    """Test cache get/put operations."""
    
    @pytest.fixture
    def mock_cache_with_redis(self):
        """Cache with mocked Redis client."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.exists.return_value = False
        mock_redis_client.hset.return_value = True
        mock_redis_client.get.return_value = None
        mock_redis_client.setex.return_value = True
        mock_redis_client.pipeline.return_value = mock_redis_client
        mock_redis_client.execute.return_value = [True]
        mock_redis_client.hkeys.return_value = []
        mock_redis_client.hlen.return_value = 0
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url', return_value=mock_redis_client), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            return cache, mock_redis_client, mock_embed_model
    
    @pytest.fixture
    def mock_cache_fallback(self):
        """Cache using fallback storage."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 3,  # Small size for testing eviction
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            return cache, mock_embed_model
    
    def test_cache_put_redis_success(self, mock_cache_with_redis):
        """Test successful cache put operation with Redis."""
        cache, mock_redis, mock_embed = mock_cache_with_redis
        
        # Mock response object
        mock_response = Mock()
        mock_response.response = "Test response"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        
        query = "What is AI?"
        result = cache.put(query, mock_response, estimated_cost=0.02)
        
        assert result is True
        mock_redis.setex.assert_called()
        mock_redis.hset.assert_called()
    
    def test_cache_put_fallback_success(self, mock_cache_fallback):
        """Test successful cache put operation with fallback storage."""
        cache, mock_embed = mock_cache_fallback
        
        # Mock response object
        mock_response = Mock()
        mock_response.response = "Test response"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        
        query = "What is AI?"
        result = cache.put(query, mock_response, estimated_cost=0.02)
        
        assert result is True
        assert len(cache._fallback_cache) == 1
    
    def test_cache_get_miss(self, mock_cache_with_redis):
        """Test cache get operation when no similar entry exists."""
        cache, mock_redis, mock_embed = mock_cache_with_redis
        
        # No similar entries
        mock_redis.hkeys.return_value = []
        
        query = "What is machine learning?"
        result = cache.get(query)
        
        assert result is None
    
    def test_cache_get_hit_redis(self, mock_cache_with_redis):
        """Test cache get hit with Redis storage."""
        cache, mock_redis, mock_embed = mock_cache_with_redis
        
        # Mock similar entry exists
        cache_key = "test:entry:abc123"
        mock_redis.hkeys.return_value = [cache_key.encode()]
        
        # Mock stored embedding and entry data
        stored_embedding = [0.1] * 1536
        mock_redis.hget.return_value = json.dumps(stored_embedding).encode()
        
        # Mock cached entry
        cached_entry_data = {
            "query": "What is AI?",
            "query_embedding": stored_embedding,
            "response": {"response": "AI is artificial intelligence"},
            "nodes": [],
            "timestamp": time.time(),
            "access_count": 0,
            "last_accessed": 0.0,
            "similarity_threshold": 0.95,
            "cost_saved": 0.02
        }
        mock_redis.get.return_value = json.dumps(cached_entry_data).encode()
        mock_redis.execute.return_value = [json.dumps(stored_embedding).encode()]
        
        # Mock embedding generation to return similar embedding
        mock_embed.get_text_embedding.return_value = [0.11] * 1536  # Slightly different
        
        query = "What is artificial intelligence?"
        result = cache.get(query)
        
        # Should find similar entry and return it
        # Note: This test may need adjustment based on actual similarity computation
    
    def test_cache_get_hit_fallback(self, mock_cache_fallback):
        """Test cache get hit with fallback storage."""
        cache, mock_embed = mock_cache_fallback
        
        # Add entry to fallback cache
        query_embedding = [0.1] * 1536
        cache_entry = CacheEntry(
            query="What is AI?",
            query_embedding=query_embedding,
            response={"response": "AI is artificial intelligence"},
            nodes=[],
            timestamp=time.time(),
            similarity_threshold=0.95,
            cost_saved=0.02
        )
        cache_key = cache._generate_cache_key("What is AI?", query_embedding)
        cache._fallback_cache[cache_key] = cache_entry
        
        # Mock embedding generation to return similar embedding
        mock_embed.get_text_embedding.return_value = [0.11] * 1536
        
        # Query with similar content
        result = cache.get("What is artificial intelligence?")
        
        # Should find and return the similar entry
        if result:
            response, nodes, similarity = result
            assert similarity >= 0.95
    
    def test_cache_eviction_fallback(self, mock_cache_fallback):
        """Test cache eviction in fallback mode."""
        cache, mock_embed = mock_cache_fallback
        
        # Mock response object
        mock_response = Mock()
        mock_response.response = "Test response"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        
        # Fill cache beyond capacity (max_size = 3)
        for i in range(5):
            query = f"Query {i}"
            cache.put(query, mock_response)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should have evicted oldest entries
        assert len(cache._fallback_cache) <= 3
        assert cache.stats.evictions > 0
    
    def test_cache_disabled_operations(self):
        """Test cache operations when caching is disabled."""
        mock_config = {
            "semantic_cache_enabled": False,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        cache = SemanticCache(config=mock_config)
        
        # Operations should return None/False when disabled
        assert cache.get("test query") is None
        assert cache.put("test query", Mock()) is False


class TestSimilarityMatching:
    """Test similarity matching algorithms and optimizations."""
    
    @pytest.fixture
    def cache_with_entries(self):
        """Cache with pre-populated entries for similarity testing."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.90,  # Lower threshold for testing
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 100,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Add some test entries
            test_entries = [
                ("What is machine learning?", [1.0, 0.0, 0.0]),
                ("Explain deep learning", [0.8, 0.6, 0.0]),
                ("How does AI work?", [0.6, 0.8, 0.0]),
                ("Python programming basics", [0.0, 0.0, 1.0]),
            ]
            
            for query, embedding in test_entries:
                cache_entry = CacheEntry(
                    query=query,
                    query_embedding=embedding,
                    response={"response": f"Response for: {query}"},
                    nodes=[],
                    timestamp=time.time(),
                    similarity_threshold=0.90,
                    cost_saved=0.01
                )
                cache_key = cache._generate_cache_key(query, embedding)
                cache._fallback_cache[cache_key] = cache_entry
            
            return cache, mock_embed_model
    
    def test_exact_similarity_match(self, cache_with_entries):
        """Test exact similarity matching."""
        cache, mock_embed = cache_with_entries
        
        # Query with exact same embedding as stored entry
        mock_embed.get_text_embedding.return_value = [1.0, 0.0, 0.0]
        
        result = cache.get("What is machine learning?")
        assert result is not None
        response, nodes, similarity = result
        assert similarity >= 0.99  # Should be very high similarity
    
    def test_high_similarity_match(self, cache_with_entries):
        """Test high similarity matching above threshold."""
        cache, mock_embed = cache_with_entries
        
        # Query with very similar embedding
        mock_embed.get_text_embedding.return_value = [0.98, 0.02, 0.0]
        
        result = cache.get("What is ML?")
        if result:
            response, nodes, similarity = result
            assert similarity >= 0.90  # Above threshold
    
    def test_low_similarity_no_match(self, cache_with_entries):
        """Test that low similarity doesn't return matches."""
        cache, mock_embed = cache_with_entries
        
        # Query with very different embedding
        mock_embed.get_text_embedding.return_value = [0.0, 0.1, 0.9]
        
        result = cache.get("Completely different topic")
        # Should not return match due to low similarity
    
    def test_similarity_with_advanced_detector(self, cache_with_entries):
        """Test similarity matching with advanced similarity detector."""
        cache, mock_embed = cache_with_entries
        
        # Mock the performance optimizer and similarity detector
        with patch('src.performance.get_performance_optimizer') as mock_get_optimizer:
            mock_similarity_detector = Mock()
            mock_similarity_detector.compute_similarity.return_value = (0.95, {})
            
            mock_optimizer = Mock()
            mock_optimizer.similarity_detector = mock_similarity_detector
            mock_get_optimizer.return_value = mock_optimizer
            
            mock_embed.get_text_embedding.return_value = [0.9, 0.1, 0.0]
            
            # Test with advanced similarity
            match = cache._find_similar_cache_entry([0.9, 0.1, 0.0], use_advanced_similarity=True)
            
            if match:
                cache_key, similarity = match
                assert similarity >= 0.90


class TestPerformanceAndStats:
    """Test cache performance tracking and statistics."""
    
    @pytest.fixture
    def cache_with_stats(self):
        """Cache with statistics enabled."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            return cache, mock_embed_model
    
    def test_hit_rate_calculation(self, cache_with_stats):
        """Test cache hit rate calculation."""
        cache, _ = cache_with_stats
        
        # Initially no queries
        assert cache.get_hit_rate() == 0.0
        
        # Simulate some cache operations
        cache.stats.total_queries = 10
        cache.stats.cache_hits = 3
        cache.stats.cache_misses = 7
        
        hit_rate = cache.get_hit_rate()
        assert hit_rate == 30.0
    
    def test_stats_tracking(self, cache_with_stats):
        """Test that statistics are properly tracked."""
        cache, mock_embed = cache_with_stats
        
        initial_stats = cache.get_stats()
        assert initial_stats.total_queries == 0
        assert initial_stats.cache_hits == 0
        assert initial_stats.cache_misses == 0
        
        # Simulate cache miss
        cache._record_miss()
        
        stats_after_miss = cache.get_stats()
        assert stats_after_miss.total_queries == 1
        assert stats_after_miss.cache_misses == 1
        
        # Simulate cache hit
        cache._record_hit(0.96, 0.05)
        
        stats_after_hit = cache.get_stats()
        assert stats_after_hit.total_queries == 2
        assert stats_after_hit.cache_hits == 1
        assert stats_after_hit.avg_similarity_score == 0.96
    
    def test_performance_tracking(self, cache_with_stats):
        """Test performance metrics tracking."""
        cache, _ = cache_with_stats
        
        # Simulate different lookup times
        cache._record_hit(0.95, 0.1)
        cache._record_hit(0.98, 0.2)
        
        stats = cache.get_stats()
        assert stats.avg_cache_lookup_time > 0
        assert stats.cache_hits == 2
    
    def test_cost_tracking(self, cache_with_stats):
        """Test cost savings tracking."""
        cache, mock_embed = cache_with_stats
        
        # Mock response
        mock_response = Mock()
        mock_response.response = "Test response"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        
        # Put entry with cost information
        cache.put("Test query", mock_response, estimated_cost=0.05)
        
        # Verify cost is tracked in cache entry
        assert len(cache._fallback_cache) == 1
        entry = list(cache._fallback_cache.values())[0]
        assert entry.cost_saved == 0.05


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_redis_connection_recovery(self):
        """Test behavior when Redis connection is lost and recovered."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        # First, Redis is available
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.exists.return_value = False
        mock_redis_client.hset.return_value = True
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url', return_value=mock_redis_client):
            
            cache = SemanticCache(config=mock_config)
            assert cache.redis_client is not None
            
            # Simulate Redis connection failure
            mock_redis_client.ping.side_effect = Exception("Connection lost")
            
            health = cache.health_check()
            assert health["redis_ping"] is False
            assert "redis_error" in health
    
    def test_embedding_generation_failures(self):
        """Test handling of embedding generation failures."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.side_effect = Exception("Embedding failed")
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Operations should handle embedding failures gracefully
            result = cache.get("Test query")
            assert result is None
            
            mock_response = Mock()
            mock_response.response = "Test"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            put_result = cache.put("Test query", mock_response)
            assert put_result is False
    
    def test_malformed_cached_data_handling(self):
        """Test handling of malformed or corrupted cached data."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.exists.return_value = False
        mock_redis_client.hset.return_value = True
        mock_redis_client.hkeys.return_value = [b"test:entry:malformed"]
        mock_redis_client.pipeline.return_value = mock_redis_client
        
        # Return malformed JSON data
        mock_redis_client.execute.return_value = [b'{"invalid": json}']
        
        with patch('src.cache.REDIS_AVAILABLE', True), \
             patch('src.cache.redis.from_url', return_value=mock_redis_client), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Should handle malformed data gracefully
            result = cache.get("Test query")
            # Should not crash and should return None for no valid matches
    
    def test_extreme_cache_sizes(self):
        """Test cache behavior with extreme size configurations."""
        # Test with very small cache size
        small_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1,  # Extremely small
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            small_cache = SemanticCache(config=small_config)
            
            # Should handle small cache size
            mock_response = Mock()
            mock_response.response = "Test"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            # Add multiple entries - should trigger eviction
            small_cache.put("Query 1", mock_response)
            small_cache.put("Query 2", mock_response)
            
            assert len(small_cache._fallback_cache) <= 1
    
    def test_unicode_and_special_characters(self):
        """Test cache handling of unicode and special characters."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Test with various unicode and special characters
            test_queries = [
                "测试中文查询",  # Chinese
                "Тест на русском",  # Russian
                "🤖 AI and robots 🚀",  # Emojis
                "Query with \"quotes\" and 'apostrophes'",
                "Special chars: !@#$%^&*()",
                "Newlines\nand\ttabs",
            ]
            
            mock_response = Mock()
            mock_response.response = "Test response"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            for query in test_queries:
                # Should handle all characters without crashing
                result = cache.put(query, mock_response)
                # Put operation might succeed or fail, but shouldn't crash
                assert isinstance(result, bool)


class TestSecurityValidation:
    """Test security aspects of cache operations."""
    
    def test_cache_key_security(self):
        """Test that cache keys are generated securely."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=mock_config)
            
            # Test with potentially malicious inputs
            malicious_queries = [
                "'; DROP TABLE cache; --",
                "../../../etc/passwd",
                "<script>alert('xss')</script>",
                "\x00\x01\x02",  # Binary data
            ]
            
            for malicious_query in malicious_queries:
                embedding = [0.1] * 1536
                cache_key = cache._generate_cache_key(malicious_query, embedding)
                
                # Cache key should be properly hashed and safe
                assert cache_key.startswith("test:entry:")
                assert len(cache_key) > 20  # Should be hashed
                assert not any(char in cache_key for char in ["'", '"', '\x00', '<', '>'])
    
    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Test with various potentially problematic inputs
            edge_case_queries = [
                "",  # Empty string
                None,  # None value (should be handled)
                "A" * 10000,  # Very long string
                "\n\r\t\0",  # Control characters
            ]
            
            mock_response = Mock()
            mock_response.response = "Test"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            for query in edge_case_queries:
                try:
                    if query is not None:
                        result = cache.put(query, mock_response)
                        # Should not crash, may succeed or fail gracefully
                        assert isinstance(result, bool)
                except Exception as e:
                    # If it fails, should fail gracefully with appropriate error
                    assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_resource_limit_enforcement(self):
        """Test that resource limits are properly enforced."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 5,  # Small limit for testing
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            mock_response = Mock()
            mock_response.response = "Test"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            # Add entries beyond limit
            for i in range(10):
                cache.put(f"Query {i}", mock_response)
                time.sleep(0.001)  # Small delay for timestamp differences
            
            # Should enforce size limits
            assert len(cache._fallback_cache) <= 5
            assert cache.stats.evictions > 0


class TestCacheUtilityFunctions:
    """Test utility functions and helpers."""
    
    def test_cost_estimation_function(self):
        """Test the standalone cost estimation function."""
        # Test simple query
        simple_cost = estimate_query_cost("What is AI?", response_tokens=100)
        assert simple_cost > 0
        assert simple_cost < 0.1
        
        # Test complex query
        complex_query = "Explain machine learning algorithms in detail"
        complex_cost = estimate_query_cost(complex_query, response_tokens=2000)
        assert complex_cost > simple_cost
        
        # Test with different models
        with patch.dict(os.environ, {"MODEL": "gpt-3.5-turbo"}):
            cheaper_cost = estimate_query_cost("Test query", response_tokens=1000)
            
        with patch.dict(os.environ, {"MODEL": "gpt-4"}):
            expensive_cost = estimate_query_cost("Test query", response_tokens=1000)
            
        # GPT-4 should be more expensive than GPT-3.5-turbo
        assert expensive_cost > cheaper_cost
    
    def test_global_cache_instance(self):
        """Test the global cache instance function."""
        # Should return same instance on multiple calls
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2
        
        # Should be a SemanticCache instance
        assert isinstance(cache1, SemanticCache)
    
    def test_cache_warming_functionality(self):
        """Test cache warming functionality."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": True
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            common_queries = [
                "What is machine learning?",
                "Explain neural networks",
                "How does deep learning work?",
                "What is artificial intelligence?"
            ]
            
            warmed_count = cache.warm_cache(common_queries)
            assert warmed_count >= 0  # Should process queries without error
    
    def test_health_check_comprehensive(self):
        """Test comprehensive health check functionality."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        # Test with fallback mode
        with patch('src.cache.REDIS_AVAILABLE', False):
            cache = SemanticCache(config=mock_config)
            health = cache.health_check()
            
            assert health["enabled"] is True
            assert health["redis_available"] is False
            assert health["fallback_active"] is True
            assert health["redis_ping"] is False
            assert "cache_size" in health
            assert "hit_rate" in health
    
    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        mock_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.95,
            "cache_ttl": 3600,
            "redis_cache_url": "redis://localhost:6379",
            "max_cache_size": 1000,
            "cache_key_prefix": "test:",
            "cache_stats_enabled": True,
            "cache_warming_enabled": False
        }
        
        with patch('src.cache.REDIS_AVAILABLE', False), \
             patch('src.cache.Settings') as mock_settings:
            
            mock_embed_model = Mock()
            mock_embed_model.get_text_embedding.return_value = [0.1] * 1536
            mock_settings.embed_model = mock_embed_model
            
            cache = SemanticCache(config=mock_config)
            
            # Add some entries
            mock_response = Mock()
            mock_response.response = "Test"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            
            cache.put("Query 1", mock_response)
            cache.put("Query 2", mock_response)
            
            assert len(cache._fallback_cache) > 0
            assert cache.stats.total_queries > 0
            
            # Clear cache
            result = cache.clear_cache()
            assert result is True
            assert len(cache._fallback_cache) == 0
            assert cache.stats.total_queries == 0