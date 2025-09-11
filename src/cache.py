import os
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle, NodeWithScore

from src.settings import get_cache_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry storing query, response, and similarity metadata.
    
    Used for semantic caching where similar queries can reuse cached responses
    based on embedding similarity rather than exact text matching.
    """
    query: str
    query_embedding: List[float]
    response: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    similarity_threshold: float = 0.97
    cost_saved: float = 0.0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_saved: float = 0.0
    avg_similarity_score: float = 0.0
    avg_cache_lookup_time: float = 0.0
    cache_size: int = 0
    evictions: int = 0


class SemanticCache:
    """Redis-based cache that matches queries by semantic similarity.
    
    Instead of exact string matching, this cache uses embedding similarity
    to find cached responses for semantically similar queries. Falls back
    to in-memory caching if Redis is unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic cache.
        
        Args:
            config: Optional configuration dictionary. If None, uses environment variables.
        """
        self.config = config or get_cache_config()
        self.enabled = self.config["semantic_cache_enabled"]
        self.similarity_threshold = self.config["cache_similarity_threshold"]
        self.ttl = self.config["cache_ttl"]
        self.max_size = self.config["max_cache_size"]
        self.key_prefix = self.config["cache_key_prefix"]
        self.stats_enabled = self.config["cache_stats_enabled"]
        
        self.redis_client: Optional[Redis] = None
        self.stats = CacheStats()
        self._fallback_cache: Dict[str, CacheEntry] = {}
        
        # Initialize Redis connection
        self._initialize_redis()
        
        logger.info(f"SemanticCache initialized: enabled={self.enabled}, threshold={self.similarity_threshold}")
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection with error handling."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, caching will use in-memory fallback")
            return
            
        if not self.enabled:
            logger.info("Semantic cache disabled")
            return
            
        try:
            redis_url = self.config["redis_url"]
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Set up cache metadata
            self._ensure_cache_metadata()
            
            logger.info(f"Redis connection established: {redis_url}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback")
            self.redis_client = None
    
    def _ensure_cache_metadata(self) -> None:
        """Ensure cache metadata structures exist in Redis."""
        if not self.redis_client:
            return
            
        try:
            # Initialize stats if they don't exist
            stats_key = f"{self.key_prefix}stats"
            if not self.redis_client.exists(stats_key):
                initial_stats = asdict(CacheStats())
                self.redis_client.hset(stats_key, mapping={
                    k: json.dumps(v) for k, v in initial_stats.items()
                })
            
            # Initialize embeddings index for similarity search
            embeddings_key = f"{self.key_prefix}embeddings"
            if not self.redis_client.exists(embeddings_key):
                self.redis_client.hset(embeddings_key, "initialized", "true")
                
        except Exception as e:
            logger.error(f"Failed to initialize cache metadata: {e}")
    
    def _generate_embedding(self, query: str) -> List[float]:
        """
        Generate normalized embedding for the query.
        
        Args:
            query: Query string to embed
            
        Returns:
            Normalized embedding vector
        """
        try:
            embed_model = Settings.embed_model
            if not embed_model:
                raise RuntimeError("No embedding model available")
                
            # Get raw embedding
            embedding = embed_model.get_text_embedding(query)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Compute cosine similarity between two normalized embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            if not emb1 or not emb2 or len(emb1) != len(emb2):
                return 0.0
                
            # Since embeddings are normalized, dot product = cosine similarity
            similarity = float(np.dot(emb1, emb2))
            
            # Ensure within valid range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _find_similar_cache_entry(self, query_embedding: List[float], use_advanced_similarity: bool = True) -> Optional[Tuple[str, float]]:
        """
        Find the most similar cache entry using embedding similarity.
        
        Args:
            query_embedding: Normalized embedding of the query
            use_advanced_similarity: Whether to use advanced multi-level similarity
            
        Returns:
            Tuple of (cache_key, similarity_score) if similar entry found, None otherwise
        """
        if not query_embedding:
            return None
            
        start_time = time.time()
        best_match = None
        best_similarity = 0.0
        
        try:
            # Import performance module for advanced similarity if enabled
            if use_advanced_similarity:
                try:
                    from src.performance import get_performance_optimizer
                    optimizer = get_performance_optimizer(semantic_cache=self)
                    similarity_detector = optimizer.similarity_detector
                except ImportError:
                    use_advanced_similarity = False
                    similarity_detector = None
            else:
                similarity_detector = None
            
            if self.redis_client:
                # Batch fetch for improved performance
                embeddings_key = f"{self.key_prefix}embeddings"
                cache_keys = self.redis_client.hkeys(embeddings_key)
                
                # Process in batches for better performance
                batch_size = 100
                for i in range(0, len(cache_keys), batch_size):
                    batch_keys = cache_keys[i:i+batch_size]
                    
                    # Use pipeline for batch fetching
                    pipe = self.redis_client.pipeline()
                    for key in batch_keys:
                        if key != b"initialized":
                            pipe.hget(embeddings_key, key)
                    
                    embeddings_batch = pipe.execute()
                    
                    for j, embedding_data in enumerate(embeddings_batch):
                        if embedding_data:
                            try:
                                stored_embedding = json.loads(embedding_data.decode('utf-8'))
                                
                                # Use advanced similarity if available
                                if use_advanced_similarity and similarity_detector:
                                    # Get cached query for advanced similarity
                                    cache_key = batch_keys[j].decode('utf-8')
                                    entry_data = self.redis_client.get(cache_key)
                                    if entry_data:
                                        entry = json.loads(entry_data.decode('utf-8'))
                                        original_query = entry.get('query', '')
                                        
                                        # Compute multi-level similarity
                                        similarity, _ = similarity_detector.compute_similarity(
                                            query_embedding,  # Current query
                                            original_query    # Cached query
                                        )
                                else:
                                    # Standard similarity computation
                                    similarity = self._compute_similarity(query_embedding, stored_embedding)
                                
                                if similarity > best_similarity and similarity >= self.similarity_threshold:
                                    best_similarity = similarity
                                    best_match = batch_keys[j].decode('utf-8')
                                    
                            except Exception as e:
                                logger.warning(f"Error processing cache entry: {e}")
                                continue
            else:
                # Search fallback cache
                for cache_key, entry in self._fallback_cache.items():
                    if use_advanced_similarity and similarity_detector:
                        similarity, _ = similarity_detector.compute_similarity(
                            query_embedding,
                            entry.query
                        )
                    else:
                        similarity = self._compute_similarity(query_embedding, entry.query_embedding)
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = cache_key
            
            search_time = time.time() - start_time
            if self.stats_enabled:
                self.stats.avg_cache_lookup_time = (
                    (self.stats.avg_cache_lookup_time * self.stats.total_queries + search_time) /
                    (self.stats.total_queries + 1)
                )
            
            if best_match:
                logger.debug(f"Found similar cache entry: {best_match} (similarity: {best_similarity:.3f})")
                return best_match, best_similarity
                
        except Exception as e:
            logger.error(f"Cache similarity search failed: {e}")
            
        return None
    
    def _generate_cache_key(self, query: str, query_embedding: List[float]) -> str:
        """
        Generate a unique cache key for the query.
        
        Args:
            query: Original query string
            query_embedding: Query embedding vector
            
        Returns:
            Unique cache key
        """
        # Use hash of query + embedding for uniqueness
        content = f"{query}:{json.dumps(query_embedding[:10])}"  # Use first 10 dims for key
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.key_prefix}entry:{hash_value}"
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """
        Serialize a Response object for caching.
        
        Args:
            response: LlamaIndex Response object
            
        Returns:
            Serializable dictionary
        """
        try:
            return {
                "response": str(response.response) if response.response else "",
                "source_nodes": [
                    {
                        "node_id": node.node.id_ if hasattr(node, 'node') and hasattr(node.node, 'id_') else str(i),
                        "text": node.node.text if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node),
                        "score": float(node.score) if hasattr(node, 'score') else 1.0,
                        "metadata": node.node.metadata if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {}
                    }
                    for i, node in enumerate(response.source_nodes or [])
                ],
                "metadata": response.metadata or {}
            }
        except Exception as e:
            logger.error(f"Response serialization failed: {e}")
            return {"response": str(response), "source_nodes": [], "metadata": {}}
    
    def _deserialize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize cached response data.
        
        Args:
            data: Cached response data
            
        Returns:
            Deserialized response data
        """
        return data  # Return as-is, conversion handled by caller
    
    def get(self, query: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], float]]:
        """
        Retrieve cached response for semantically similar query.
        
        Args:
            query: Query string to lookup
            
        Returns:
            Tuple of (response_data, nodes, similarity_score) if cache hit, None otherwise
        """
        if not self.enabled:
            return None
            
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return None
            
            # Find similar entry
            match_result = self._find_similar_cache_entry(query_embedding)
            if not match_result:
                self._record_miss()
                return None
                
            cache_key, similarity_score = match_result
            
            # Retrieve cache entry
            cache_entry = self._get_cache_entry(cache_key)
            if not cache_entry:
                self._record_miss()
                return None
            
            # Update access statistics
            self._update_access_stats(cache_key, cache_entry)
            
            # Record cache hit
            self._record_hit(similarity_score, time.time() - start_time)
            
            logger.info(f"Cache HIT: {cache_key} (similarity: {similarity_score:.3f})")
            
            return cache_entry.response, cache_entry.nodes, similarity_score
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self._record_miss()
            return None
    
    def put(
        self, 
        query: str, 
        response: Any, 
        estimated_cost: float = 0.0
    ) -> bool:
        """
        Cache a query response with semantic similarity indexing.
        
        Args:
            query: Original query string
            response: Response object to cache
            estimated_cost: Estimated cost of generating this response
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return False
            
            # Create cache entry
            cache_entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                response=self._serialize_response(response),
                nodes=[
                    {
                        "node_id": node.node.id_ if hasattr(node, 'node') and hasattr(node.node, 'id_') else str(i),
                        "text": node.node.text if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node),
                        "score": float(node.score) if hasattr(node, 'score') else 1.0,
                        "metadata": node.node.metadata if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {}
                    }
                    for i, node in enumerate(response.source_nodes or [])
                ],
                timestamp=time.time(),
                similarity_threshold=self.similarity_threshold,
                cost_saved=estimated_cost
            )
            
            # Generate cache key
            cache_key = self._generate_cache_key(query, query_embedding)
            
            # Store in cache
            success = self._store_cache_entry(cache_key, cache_entry)
            
            if success:
                # Ensure we don't exceed max cache size
                self._enforce_cache_limits()
                logger.debug(f"Cached response for query: {query[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache put failed: {e}")
            return False
    
    def _get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key."""
        try:
            if self.redis_client:
                # Get from Redis
                entry_data = self.redis_client.get(cache_key)
                if entry_data:
                    data = json.loads(entry_data.decode('utf-8'))
                    return CacheEntry(**data)
            else:
                # Get from fallback cache
                return self._fallback_cache.get(cache_key)
                
        except Exception as e:
            logger.error(f"Failed to retrieve cache entry {cache_key}: {e}")
            
        return None
    
    def _store_cache_entry(self, cache_key: str, entry: CacheEntry) -> bool:
        """Store cache entry."""
        try:
            if self.redis_client:
                # Store in Redis with TTL
                entry_data = json.dumps(asdict(entry), default=str)
                pipe = self.redis_client.pipeline()
                
                # Store entry
                pipe.setex(cache_key, self.ttl, entry_data)
                
                # Store embedding for similarity search
                embeddings_key = f"{self.key_prefix}embeddings"
                pipe.hset(embeddings_key, cache_key, json.dumps(entry.query_embedding))
                
                # Execute pipeline
                pipe.execute()
            else:
                # Store in fallback cache
                self._fallback_cache[cache_key] = entry
                
                # Simple LRU for fallback cache
                if len(self._fallback_cache) > self.max_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self._fallback_cache.keys(),
                        key=lambda k: self._fallback_cache[k].timestamp
                    )
                    del self._fallback_cache[oldest_key]
                    self.stats.evictions += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store cache entry {cache_key}: {e}")
            return False
    
    def _update_access_stats(self, cache_key: str, entry: CacheEntry) -> None:
        """Update access statistics for a cache entry."""
        try:
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Update in storage
            if self.redis_client:
                entry_data = json.dumps(asdict(entry), default=str)
                self.redis_client.setex(cache_key, self.ttl, entry_data)
            # Fallback cache entry is already updated by reference
            
        except Exception as e:
            logger.error(f"Failed to update access stats for {cache_key}: {e}")
    
    def _enforce_cache_limits(self) -> None:
        """Enforce cache size limits with LRU eviction."""
        try:
            if not self.redis_client:
                return  # Fallback cache handles this in _store_cache_entry
                
            # Get current cache size
            embeddings_key = f"{self.key_prefix}embeddings"
            current_size = self.redis_client.hlen(embeddings_key) - 1  # Exclude 'initialized' key
            
            if current_size <= self.max_size:
                return
            
            # Get all cache keys with their last access times
            cache_keys = []
            pipe = self.redis_client.pipeline()
            
            for key in self.redis_client.hkeys(embeddings_key):
                if key != b"initialized":
                    pipe.get(key.decode('utf-8'))
            
            entries = pipe.execute()
            
            # Parse entries and sort by last access time
            key_access_times = []
            for i, entry_data in enumerate(entries):
                if entry_data:
                    try:
                        data = json.loads(entry_data.decode('utf-8'))
                        key_access_times.append((
                            list(self.redis_client.hkeys(embeddings_key))[i].decode('utf-8'),
                            data.get('last_accessed', data.get('timestamp', 0))
                        ))
                    except:
                        continue
            
            # Sort by access time (oldest first)
            key_access_times.sort(key=lambda x: x[1])
            
            # Remove oldest entries
            eviction_count = current_size - self.max_size + 1
            for cache_key, _ in key_access_times[:eviction_count]:
                self._remove_cache_entry(cache_key)
                self.stats.evictions += 1
                
        except Exception as e:
            logger.error(f"Cache limit enforcement failed: {e}")
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry completely."""
        try:
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                pipe.delete(cache_key)
                pipe.hdel(f"{self.key_prefix}embeddings", cache_key)
                pipe.execute()
            else:
                self._fallback_cache.pop(cache_key, None)
                
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")
    
    def _record_hit(self, similarity_score: float, lookup_time: float) -> None:
        """Record cache hit statistics."""
        if not self.stats_enabled:
            return
            
        self.stats.cache_hits += 1
        self.stats.total_queries += 1
        
        # Update average similarity score
        total_scores = self.stats.avg_similarity_score * (self.stats.cache_hits - 1) + similarity_score
        self.stats.avg_similarity_score = total_scores / self.stats.cache_hits
        
        # Update average lookup time
        total_lookups = self.stats.total_queries - 1
        if total_lookups > 0:
            total_time = self.stats.avg_cache_lookup_time * total_lookups + lookup_time
            self.stats.avg_cache_lookup_time = total_time / self.stats.total_queries
        else:
            self.stats.avg_cache_lookup_time = lookup_time
        
        self._persist_stats()
    
    def _record_miss(self) -> None:
        """Record cache miss statistics."""
        if not self.stats_enabled:
            return
            
        self.stats.cache_misses += 1
        self.stats.total_queries += 1
        self._persist_stats()
    
    def _persist_stats(self) -> None:
        """Persist statistics to Redis."""
        try:
            if self.redis_client:
                stats_key = f"{self.key_prefix}stats"
                self.redis_client.hset(stats_key, mapping={
                    k: json.dumps(v) for k, v in asdict(self.stats).items()
                })
        except Exception as e:
            logger.error(f"Failed to persist cache stats: {e}")
    
    def get_stats(self) -> CacheStats:
        """
        Get current cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        # Update cache size
        try:
            if self.redis_client:
                embeddings_key = f"{self.key_prefix}embeddings"
                self.stats.cache_size = self.redis_client.hlen(embeddings_key) - 1  # Exclude 'initialized'
            else:
                self.stats.cache_size = len(self._fallback_cache)
        except:
            pass
            
        return self.stats
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate as percentage.
        
        Returns:
            Hit rate percentage (0-100)
        """
        if self.stats.total_queries == 0:
            return 0.0
        return (self.stats.cache_hits / self.stats.total_queries) * 100
    
    def warm_cache(self, common_queries: List[str]) -> int:
        """
        Warm the cache with common queries.
        
        Args:
            common_queries: List of common query strings to pre-cache
            
        Returns:
            Number of queries successfully warmed
        """
        if not self.enabled or not self.config["cache_warming_enabled"]:
            return 0
            
        warmed_count = 0
        
        for query in common_queries:
            try:
                # Generate embedding for query
                embedding = self._generate_embedding(query)
                if embedding:
                    # Check if similar query already cached
                    if not self._find_similar_cache_entry(embedding):
                        # Could implement actual query execution here
                        # For now, just log the warming intent
                        logger.info(f"Cache warming candidate: {query}")
                        warmed_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to warm cache for query '{query}': {e}")
                
        logger.info(f"Cache warming completed: {warmed_count} queries processed")
        return warmed_count
    
    def clear_cache(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            if self.redis_client:
                # Clear all cache keys
                pattern = f"{self.key_prefix}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                self._fallback_cache.clear()
            
            # Reset stats
            self.stats = CacheStats()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "enabled": self.enabled,
            "redis_available": self.redis_client is not None,
            "cache_size": self.get_stats().cache_size,
            "hit_rate": self.get_hit_rate(),
            "total_queries": self.stats.total_queries,
            "avg_lookup_time": self.stats.avg_cache_lookup_time,
            "fallback_active": self.redis_client is None and self.enabled
        }
        
        try:
            if self.redis_client:
                # Test Redis connectivity
                self.redis_client.ping()
                health["redis_ping"] = True
                health["redis_memory"] = self.redis_client.info("memory")
            else:
                health["redis_ping"] = False
                
        except Exception as e:
            health["redis_ping"] = False
            health["redis_error"] = str(e)
        
        return health

    def _get_secure_redis_config(self, redis_url: str) -> Dict[str, Any]:
        """Get secure Redis connection configuration."""
        config = {
            "decode_responses": False,  # We handle encoding ourselves
            "socket_connect_timeout": int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
            "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            "retry_on_timeout": True,
            "health_check_interval": int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
            "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
        }
        
        # Parse URL to check for security features
        parsed = urlparse(redis_url)
        
        # Enable SSL/TLS if specified
        if parsed.scheme == "rediss":
            config["ssl"] = True
            config["ssl_check_hostname"] = True
            
            # SSL certificate validation
            ssl_cert_reqs = os.getenv("REDIS_SSL_CERT_REQS", "required")
            if ssl_cert_reqs == "required":
                config["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            elif ssl_cert_reqs == "optional":
                config["ssl_cert_reqs"] = ssl.CERT_OPTIONAL
            else:
                config["ssl_cert_reqs"] = ssl.CERT_NONE
                
            # Custom SSL certificate paths if provided
            ssl_ca_certs = os.getenv("REDIS_SSL_CA_CERTS")
            if ssl_ca_certs:
                config["ssl_ca_certs"] = ssl_ca_certs
                
            ssl_cert_file = os.getenv("REDIS_SSL_CERT_FILE")
            ssl_key_file = os.getenv("REDIS_SSL_KEY_FILE")
            if ssl_cert_file and ssl_key_file:
                config["ssl_certfile"] = ssl_cert_file
                config["ssl_keyfile"] = ssl_key_file
        
        # Connection pooling and limits
        config["connection_pool_kwargs"] = {
            "max_connections": config["max_connections"],
            "retry_on_timeout": True,
        }
        
        return config
    
    def _sanitize_url_for_logging(self, url: str) -> str:
        """Remove sensitive information from URL for safe logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                # Replace password with asterisks
                sanitized = url.replace(parsed.password, "***")
                return sanitized
            return url
        except Exception:
            # If parsing fails, just show the scheme and host
            try:
                parsed = urlparse(url)
                return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 6379}"
            except Exception:
                return "redis://***"
    
    def _validate_redis_security(self) -> None:
        """Validate Redis security configuration."""
        if not self.redis_client:
            return
            
        try:
            # Check Redis server info for security features
            info = self.redis_client.info()
            
            # Warn about potential security issues
            redis_version = info.get("redis_version", "unknown")
            
            # Check for authentication
            try:
                # This will fail if auth is required but not provided
                self.redis_client.config_get("requirepass")
                logger.info("Redis authentication check passed")
            except Exception:
                logger.warning("Could not verify Redis authentication configuration")
            
            # Check for memory limits to prevent DoS
            try:
                memory_info = self.redis_client.info("memory")
                max_memory = memory_info.get("maxmemory", 0)
                if max_memory == 0:
                    logger.warning("Redis has no memory limit set - consider setting maxmemory for security")
            except Exception:
                logger.debug("Could not check Redis memory configuration")
            
        except Exception as e:
            logger.warning(f"Redis security validation failed: {e}")


# Global cache instance
_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    """
    Get the global semantic cache instance.
    
    Returns:
        SemanticCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


def estimate_query_cost(query: str, response_tokens: int = 2000) -> float:
    """
    Estimate the cost of a query based on tokens and model pricing.
    
    Args:
        query: Query string
        response_tokens: Estimated response tokens
        
    Returns:
        Estimated cost in USD
    """
    # Rough cost estimates for OpenAI models (as of 2024)
    # These should be updated based on actual pricing
    
    query_tokens = len(query.split()) * 1.3  # Rough token estimation
    embedding_tokens = query_tokens
    
    # Model costs per 1K tokens (approximate)
    model_costs = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "text-embedding-3-large": {"input": 0.00013, "output": 0}
    }
    
    # Get current model from settings
    main_model = os.getenv("MODEL", "gpt-4o")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Calculate costs
    main_model_cost = model_costs.get(main_model, {"input": 0.01, "output": 0.03})
    embedding_cost = model_costs.get(embedding_model, {"input": 0.0001, "output": 0})
    
    total_cost = (
        (query_tokens / 1000) * main_model_cost["input"] +
        (response_tokens / 1000) * main_model_cost["output"] +
        (embedding_tokens / 1000) * embedding_cost["input"]
    )
    
    return round(total_cost, 6)


def get_secure_cache_config() -> Dict[str, Any]:
    """Get secure cache configuration with validation."""
    config = get_cache_config()
    
    # Validate Redis URL security
    redis_url = config.get("redis_url", "")
    if redis_url:
        parsed = urlparse(redis_url)
        
        # Warn about insecure configurations
        if parsed.scheme == "redis" and parsed.hostname not in ("localhost", "127.0.0.1"):
            logger.warning("Redis connection is not using TLS - consider using rediss:// for production")
        
        if not parsed.password and parsed.hostname not in ("localhost", "127.0.0.1"):
            logger.warning("Redis connection may not include authentication for remote connections")
    
    # Set secure defaults
    config.setdefault("cache_ttl", min(config.get("cache_ttl", 3600), 86400))  # Max 24 hours
    config.setdefault("max_cache_size", min(config.get("max_cache_size", 10000), 100000))  # Max 100k entries
    
    return config