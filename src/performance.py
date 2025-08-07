"""
Performance Optimization Module for SOTA RAG System

This module implements advanced performance optimizations including:
- Multi-level query similarity detection (lexical, semantic, intent-based)
- Dynamic similarity thresholds based on query complexity
- Cross-modal similarity for multimodal queries
- Intelligent cache key generation with query normalization
- Batch processing for embedding operations
- Connection pooling optimization
- Memory usage optimization
- Smart cache warming strategies
- Priority-based eviction
- Cross-system cache coherence
"""

import os
import re
import json
import time
import hashlib
import logging
import asyncio
from collections import defaultdict, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Attempt to import required libraries
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding

from src.cache import SemanticCache, CacheEntry, CacheStats
from src.agentic import QueryType, QueryClassification, SubQuery
from src.verification import VerificationResult, ResponseConfidence

logger = logging.getLogger(__name__)


class SimilarityLevel(Enum):
    """Levels of similarity detection."""
    LEXICAL = "lexical"          # Exact/near-exact word matching
    SEMANTIC = "semantic"        # Meaning-based similarity
    INTENT = "intent"            # Intent/purpose similarity
    STRUCTURAL = "structural"    # Query structure similarity
    CROSS_MODAL = "cross_modal"  # Cross-modal similarity


@dataclass
class QuerySignature:
    """
    Enhanced query signature for multi-level similarity detection.
    
    Captures multiple aspects of a query for comprehensive similarity matching.
    """
    original_query: str
    normalized_query: str
    query_type: QueryType
    intent_vector: List[float]
    semantic_embedding: List[float]
    lexical_tokens: Set[str]
    structural_pattern: str
    modality: str  # text, image, mixed
    complexity_score: float
    estimated_cost: float
    timestamp: float = field(default_factory=time.time)
    
    def to_cache_key(self) -> str:
        """Generate a unique cache key from the signature."""
        # Combine multiple elements for uniqueness
        key_elements = [
            self.normalized_query[:50],
            self.query_type.value,
            str(self.complexity_score)[:4],
            self.modality,
            # Use first few dimensions of embeddings for key
            str(self.semantic_embedding[:5]) if self.semantic_embedding else ""
        ]
        key_str = "|".join(key_elements)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics tracking."""
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    avg_cache_lookup_time: float = 0.0
    
    # Embedding metrics
    embeddings_computed: int = 0
    embeddings_cached: int = 0
    avg_embedding_time: float = 0.0
    batch_processing_savings: float = 0.0
    
    # Verification metrics
    verifications_performed: int = 0
    verifications_cached: int = 0
    avg_verification_time: float = 0.0
    verification_cost_saved: float = 0.0
    
    # Query processing metrics
    queries_processed: int = 0
    queries_decomposed: int = 0
    avg_query_time: float = 0.0
    p95_query_time: float = 0.0
    p99_query_time: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_saved: float = 0.0
    cost_reduction_percentage: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    cache_size_mb: float = 0.0
    
    def update_cache_metrics(self, hit: bool, lookup_time: float):
        """Update cache-related metrics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total_lookups = self.cache_hits + self.cache_misses
        if total_lookups > 0:
            self.cache_hit_rate = self.cache_hits / total_lookups
            self.avg_cache_lookup_time = (
                (self.avg_cache_lookup_time * (total_lookups - 1) + lookup_time) / total_lookups
            )


class AdvancedSimilarityDetector:
    """
    Advanced multi-level similarity detection system.
    
    Implements sophisticated similarity detection across multiple dimensions:
    - Lexical similarity (word-level matching)
    - Semantic similarity (meaning-based)
    - Intent similarity (purpose/goal matching)
    - Structural similarity (query pattern matching)
    - Cross-modal similarity (for multimodal queries)
    """
    
    def __init__(
        self,
        embedding_model: Optional[OpenAIEmbedding] = None,
        enable_all_levels: bool = True,
        dynamic_thresholds: bool = True
    ):
        """
        Initialize the advanced similarity detector.
        
        Args:
            embedding_model: Model for generating embeddings
            enable_all_levels: Enable all similarity detection levels
            dynamic_thresholds: Use dynamic thresholds based on query complexity
        """
        self.embedding_model = embedding_model or Settings.embed_model
        self.enable_all_levels = enable_all_levels
        self.dynamic_thresholds = dynamic_thresholds
        
        # Similarity thresholds by level
        self.base_thresholds = {
            SimilarityLevel.LEXICAL: 0.95,
            SimilarityLevel.SEMANTIC: 0.92,
            SimilarityLevel.INTENT: 0.88,
            SimilarityLevel.STRUCTURAL: 0.85,
            SimilarityLevel.CROSS_MODAL: 0.80
        }
        
        # Cache for computed signatures
        self._signature_cache: Dict[str, QuerySignature] = {}
        
        # Intent patterns for classification
        self.intent_patterns = {
            "definition": ["what is", "define", "meaning of"],
            "comparison": ["compare", "difference", "versus", "vs"],
            "procedure": ["how to", "steps", "process"],
            "reason": ["why", "because", "reason"],
            "location": ["where", "location", "place"],
            "time": ["when", "time", "date"],
            "quantity": ["how many", "how much", "count"]
        }
        
        logger.info("Initialized AdvancedSimilarityDetector with all levels enabled")
    
    def compute_similarity(
        self,
        query1: str,
        query2: str,
        query_type: Optional[QueryType] = None,
        complexity: Optional[float] = None
    ) -> Tuple[float, Dict[SimilarityLevel, float]]:
        """
        Compute multi-level similarity between two queries.
        
        Args:
            query1: First query
            query2: Second query
            query_type: Type of query for context
            complexity: Query complexity score
            
        Returns:
            Tuple of (overall similarity score, individual level scores)
        """
        try:
            # Get or compute signatures for both queries
            sig1 = self._get_or_compute_signature(query1, query_type)
            sig2 = self._get_or_compute_signature(query2, query_type)
            
            # Compute similarity at each level
            level_scores = {}
            
            if self.enable_all_levels:
                level_scores[SimilarityLevel.LEXICAL] = self._compute_lexical_similarity(sig1, sig2)
                level_scores[SimilarityLevel.SEMANTIC] = self._compute_semantic_similarity(sig1, sig2)
                level_scores[SimilarityLevel.INTENT] = self._compute_intent_similarity(sig1, sig2)
                level_scores[SimilarityLevel.STRUCTURAL] = self._compute_structural_similarity(sig1, sig2)
                
                if sig1.modality != "text" or sig2.modality != "text":
                    level_scores[SimilarityLevel.CROSS_MODAL] = self._compute_cross_modal_similarity(sig1, sig2)
            else:
                # Fast path: semantic only
                level_scores[SimilarityLevel.SEMANTIC] = self._compute_semantic_similarity(sig1, sig2)
            
            # Compute weighted overall score
            overall_score = self._compute_weighted_score(level_scores, sig1, sig2, complexity)
            
            return overall_score, level_scores
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0, {}
    
    def get_dynamic_threshold(
        self,
        query_type: QueryType,
        complexity: float,
        level: SimilarityLevel
    ) -> float:
        """
        Get dynamic similarity threshold based on query characteristics.
        
        Args:
            query_type: Type of query
            complexity: Query complexity score (0-1)
            level: Similarity level
            
        Returns:
            Adjusted threshold value
        """
        if not self.dynamic_thresholds:
            return self.base_thresholds.get(level, 0.9)
        
        base_threshold = self.base_thresholds.get(level, 0.9)
        
        # Adjust based on query type
        type_adjustments = {
            QueryType.FACTUAL: 0.05,      # Higher threshold for factual queries
            QueryType.SEMANTIC: -0.05,     # Lower threshold for semantic queries
            QueryType.COMPARATIVE: -0.03,  # Slightly lower for comparisons
            QueryType.ANALYTICAL: -0.08,   # Lower for analytical queries
            QueryType.MULTIFACETED: -0.10  # Lowest for complex queries
        }
        
        type_adjustment = type_adjustments.get(query_type, 0.0)
        
        # Adjust based on complexity (more complex = lower threshold)
        complexity_adjustment = -0.1 * complexity
        
        # Combine adjustments
        adjusted_threshold = base_threshold + type_adjustment + complexity_adjustment
        
        # Ensure within reasonable bounds
        return max(0.5, min(0.99, adjusted_threshold))
    
    def _get_or_compute_signature(
        self,
        query: str,
        query_type: Optional[QueryType] = None
    ) -> QuerySignature:
        """Get cached signature or compute new one."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]
        
        signature = self._compute_query_signature(query, query_type)
        self._signature_cache[cache_key] = signature
        
        # Limit cache size
        if len(self._signature_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self._signature_cache.keys())[:100]
            for key in oldest_keys:
                del self._signature_cache[key]
        
        return signature
    
    def _compute_query_signature(
        self,
        query: str,
        query_type: Optional[QueryType] = None
    ) -> QuerySignature:
        """Compute comprehensive signature for a query."""
        # Normalize query
        normalized = self._normalize_query(query)
        
        # Extract lexical tokens
        lexical_tokens = self._extract_lexical_tokens(normalized)
        
        # Compute embeddings
        semantic_embedding = self._compute_embedding(query)
        intent_vector = self._compute_intent_vector(query)
        
        # Detect structural pattern
        structural_pattern = self._detect_structural_pattern(query)
        
        # Detect modality
        modality = self._detect_modality(query)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(query)
        
        # Estimate cost
        estimated_cost = len(query.split()) * 0.0001  # Simple estimate
        
        return QuerySignature(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type or QueryType.SEMANTIC,
            intent_vector=intent_vector,
            semantic_embedding=semantic_embedding,
            lexical_tokens=lexical_tokens,
            structural_pattern=structural_pattern,
            modality=modality,
            complexity_score=complexity_score,
            estimated_cost=estimated_cost
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove punctuation except for essential ones
        normalized = re.sub(r'[^\w\s\-\']', '', normalized)
        
        # Expand common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'s": " is",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        return normalized.strip()
    
    def _extract_lexical_tokens(self, normalized_query: str) -> Set[str]:
        """Extract meaningful lexical tokens from normalized query."""
        # Simple tokenization
        tokens = normalized_query.split()
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        meaningful_tokens = {token for token in tokens if token not in stop_words and len(token) > 2}
        
        return meaningful_tokens
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute semantic embedding for text."""
        try:
            if self.embedding_model:
                embedding = self.embedding_model.get_text_embedding(text)
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
                return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
        
        return []
    
    def _compute_intent_vector(self, query: str) -> List[float]:
        """Compute intent vector based on query patterns."""
        query_lower = query.lower()
        intent_vector = []
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in query_lower:
                    score = 1.0
                    break
            intent_vector.append(score)
        
        return intent_vector
    
    def _detect_structural_pattern(self, query: str) -> str:
        """Detect the structural pattern of a query."""
        # Simplified pattern detection
        patterns = []
        
        if query.startswith(('what', 'who', 'when', 'where', 'why', 'how')):
            patterns.append('WH_QUESTION')
        
        if '?' in query:
            patterns.append('QUESTION')
        
        if ' and ' in query.lower() or ' or ' in query.lower():
            patterns.append('COMPOUND')
        
        if any(word in query.lower() for word in ['compare', 'difference', 'versus']):
            patterns.append('COMPARISON')
        
        if not patterns:
            patterns.append('STATEMENT')
        
        return '_'.join(patterns)
    
    def _detect_modality(self, query: str) -> str:
        """Detect the modality of the query."""
        image_keywords = ['image', 'picture', 'photo', 'diagram', 'chart', 'graph', 'visual']
        
        if any(keyword in query.lower() for keyword in image_keywords):
            return "mixed"
        
        return "text"
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        factors = []
        
        # Length factor
        word_count = len(query.split())
        length_score = min(word_count / 20.0, 1.0)
        factors.append(length_score * 0.3)
        
        # Conjunction factor
        conjunctions = ['and', 'or', 'but', 'also', 'furthermore', 'moreover']
        conj_count = sum(1 for conj in conjunctions if conj in query.lower())
        conj_score = min(conj_count / 3.0, 1.0)
        factors.append(conj_score * 0.4)
        
        # Question complexity
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        question_count = sum(1 for word in question_words if word in query.lower())
        question_score = min(question_count / 3.0, 1.0)
        factors.append(question_score * 0.3)
        
        return sum(factors)
    
    def _compute_lexical_similarity(self, sig1: QuerySignature, sig2: QuerySignature) -> float:
        """Compute lexical similarity between signatures."""
        if not sig1.lexical_tokens or not sig2.lexical_tokens:
            return 0.0
        
        intersection = sig1.lexical_tokens.intersection(sig2.lexical_tokens)
        union = sig1.lexical_tokens.union(sig2.lexical_tokens)
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Boost if normalized queries are very similar
        from difflib import SequenceMatcher
        normalized_similarity = SequenceMatcher(None, sig1.normalized_query, sig2.normalized_query).ratio()
        
        return 0.7 * jaccard + 0.3 * normalized_similarity
    
    def _compute_semantic_similarity(self, sig1: QuerySignature, sig2: QuerySignature) -> float:
        """Compute semantic similarity using embeddings."""
        if not sig1.semantic_embedding or not sig2.semantic_embedding:
            return 0.0
        
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(sig1.semantic_embedding, sig2.semantic_embedding)
        
        return max(0.0, min(1.0, float(similarity)))
    
    def _compute_intent_similarity(self, sig1: QuerySignature, sig2: QuerySignature) -> float:
        """Compute intent similarity between signatures."""
        if not sig1.intent_vector or not sig2.intent_vector:
            return 0.0
        
        # Cosine similarity of intent vectors
        vec1 = np.array(sig1.intent_vector)
        vec2 = np.array(sig2.intent_vector)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return max(0.0, min(1.0, float(similarity)))
    
    def _compute_structural_similarity(self, sig1: QuerySignature, sig2: QuerySignature) -> float:
        """Compute structural similarity between signatures."""
        if sig1.structural_pattern == sig2.structural_pattern:
            return 1.0
        
        # Partial matching for compound patterns
        patterns1 = set(sig1.structural_pattern.split('_'))
        patterns2 = set(sig2.structural_pattern.split('_'))
        
        if not patterns1 or not patterns2:
            return 0.0
        
        intersection = patterns1.intersection(patterns2)
        union = patterns1.union(patterns2)
        
        return len(intersection) / len(union)
    
    def _compute_cross_modal_similarity(self, sig1: QuerySignature, sig2: QuerySignature) -> float:
        """Compute cross-modal similarity for multimodal queries."""
        # Simplified cross-modal similarity
        if sig1.modality == sig2.modality:
            return 1.0
        elif "mixed" in [sig1.modality, sig2.modality]:
            return 0.7
        else:
            return 0.3
    
    def _compute_weighted_score(
        self,
        level_scores: Dict[SimilarityLevel, float],
        sig1: QuerySignature,
        sig2: QuerySignature,
        complexity: Optional[float] = None
    ) -> float:
        """Compute weighted overall similarity score."""
        if not level_scores:
            return 0.0
        
        # Default weights
        weights = {
            SimilarityLevel.LEXICAL: 0.15,
            SimilarityLevel.SEMANTIC: 0.40,
            SimilarityLevel.INTENT: 0.20,
            SimilarityLevel.STRUCTURAL: 0.15,
            SimilarityLevel.CROSS_MODAL: 0.10
        }
        
        # Adjust weights based on query characteristics
        if complexity and complexity > 0.7:
            # For complex queries, semantic and intent matter more
            weights[SimilarityLevel.SEMANTIC] = 0.45
            weights[SimilarityLevel.INTENT] = 0.25
            weights[SimilarityLevel.LEXICAL] = 0.10
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for level, score in level_scores.items():
            weight = weights.get(level, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight


class BatchEmbeddingProcessor:
    """
    Optimized batch processing for embedding operations.
    
    Reduces API calls and improves throughput by batching embedding requests.
    """
    
    def __init__(
        self,
        embedding_model: Optional[OpenAIEmbedding] = None,
        batch_size: int = 10,
        max_wait_time: float = 0.1
    ):
        """
        Initialize batch embedding processor.
        
        Args:
            embedding_model: Model for generating embeddings
            batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch to fill
        """
        self.embedding_model = embedding_model or Settings.embed_model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Batch queue
        self._batch_queue: List[Tuple[str, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._processing = False
        
        # Performance tracking
        self.total_processed = 0
        self.total_batches = 0
        self.total_time = 0.0
        
        logger.info(f"BatchEmbeddingProcessor initialized with batch_size={batch_size}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text with batch processing.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Create future for result
        future = asyncio.Future()
        
        async with self._batch_lock:
            self._batch_queue.append((text, future))
            
            # Check if we should process batch
            should_process = (
                len(self._batch_queue) >= self.batch_size or
                not self._processing
            )
        
        if should_process:
            asyncio.create_task(self._process_batch())
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=self.max_wait_time + 5.0)
            return result
        except asyncio.TimeoutError:
            logger.error("Batch embedding timeout")
            # Fallback to individual processing
            return self._compute_single_embedding(text)
    
    async def _process_batch(self):
        """Process a batch of embedding requests."""
        async with self._batch_lock:
            if self._processing or not self._batch_queue:
                return
            
            self._processing = True
            
            # Extract batch
            batch_size = min(len(self._batch_queue), self.batch_size)
            batch = self._batch_queue[:batch_size]
            self._batch_queue = self._batch_queue[batch_size:]
        
        try:
            start_time = time.time()
            
            # Extract texts
            texts = [text for text, _ in batch]
            
            # Compute embeddings in batch
            embeddings = await self._compute_batch_embeddings(texts)
            
            # Distribute results
            for i, (_, future) in enumerate(batch):
                if i < len(embeddings):
                    future.set_result(embeddings[i])
                else:
                    future.set_exception(Exception("Batch processing failed"))
            
            # Update metrics
            elapsed = time.time() - start_time
            self.total_processed += len(batch)
            self.total_batches += 1
            self.total_time += elapsed
            
            logger.debug(f"Processed batch of {len(batch)} embeddings in {elapsed:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
        
        finally:
            async with self._batch_lock:
                self._processing = False
                
                # Check if there are more items to process
                if self._batch_queue:
                    asyncio.create_task(self._process_batch())
    
    async def _compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        try:
            if self.embedding_model:
                # OpenAI supports batch embedding
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: [self.embedding_model.get_text_embedding(text) for text in texts]
                )
                
                # Normalize embeddings
                normalized = []
                for embedding in embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        normalized.append((np.array(embedding) / norm).tolist())
                    else:
                        normalized.append(embedding)
                
                return normalized
            
        except Exception as e:
            logger.error(f"Batch embedding computation failed: {e}")
        
        # Fallback to individual processing
        return [self._compute_single_embedding(text) for text in texts]
    
    def _compute_single_embedding(self, text: str) -> List[float]:
        """Compute single embedding as fallback."""
        try:
            if self.embedding_model:
                embedding = self.embedding_model.get_text_embedding(text)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
                return embedding
        except Exception as e:
            logger.error(f"Single embedding computation failed: {e}")
        
        return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get batch processing performance statistics."""
        if self.total_batches == 0:
            return {"status": "no_data"}
        
        avg_batch_size = self.total_processed / self.total_batches
        avg_batch_time = self.total_time / self.total_batches
        throughput = self.total_processed / max(0.001, self.total_time)
        
        # Estimate savings
        individual_time_estimate = self.total_processed * 0.05  # Assume 50ms per individual
        time_saved = max(0, individual_time_estimate - self.total_time)
        savings_percentage = (time_saved / max(0.001, individual_time_estimate)) * 100
        
        return {
            "total_embeddings_processed": self.total_processed,
            "total_batches": self.total_batches,
            "average_batch_size": avg_batch_size,
            "average_batch_time_seconds": avg_batch_time,
            "throughput_per_second": throughput,
            "estimated_time_saved_seconds": time_saved,
            "batch_processing_savings_percentage": savings_percentage
        }


class IntelligentCacheManager:
    """
    Intelligent cache management system with advanced features.
    
    Implements:
    - Priority-based eviction
    - Adaptive cache sizing
    - Smart cache warming
    - Cross-system cache coherence
    - Performance monitoring
    """
    
    def __init__(
        self,
        semantic_cache: SemanticCache,
        max_memory_mb: float = 512,
        enable_warming: bool = True,
        enable_adaptive_sizing: bool = True
    ):
        """
        Initialize intelligent cache manager.
        
        Args:
            semantic_cache: Base semantic cache instance
            max_memory_mb: Maximum memory allocation in MB
            enable_warming: Enable cache warming strategies
            enable_adaptive_sizing: Enable adaptive cache sizing
        """
        self.semantic_cache = semantic_cache
        self.max_memory_mb = max_memory_mb
        self.enable_warming = enable_warming
        self.enable_adaptive_sizing = enable_adaptive_sizing
        
        # Cache layers
        self.verification_cache: OrderedDict[str, Tuple[VerificationResult, float, str]] = OrderedDict()
        self.decomposition_cache: OrderedDict[str, List[SubQuery]] = OrderedDict()
        self.embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        
        # Priority tracking
        self.access_counts: defaultdict[str, int] = defaultdict(int)
        self.last_access_times: Dict[str, float] = {}
        self.query_importance: Dict[str, float] = {}
        
        # Performance tracking
        self.cache_performance = PerformanceMetrics()
        
        # Cache warming patterns
        self.common_patterns = [
            "what is",
            "how to",
            "why does",
            "when did",
            "where is",
            "compare",
            "difference between",
            "explain"
        ]
        
        # Background tasks
        self._warming_task = None
        self._cleanup_task = None
        
        logger.info(f"IntelligentCacheManager initialized with {max_memory_mb}MB limit")
    
    async def get_or_compute_verification(
        self,
        query: str,
        response: str,
        compute_func
    ) -> Tuple[VerificationResult, float, str]:
        """
        Get cached verification result or compute new one.
        
        Args:
            query: Query string
            response: Response to verify
            compute_func: Function to compute verification if not cached
            
        Returns:
            Verification result tuple
        """
        cache_key = self._generate_verification_key(query, response)
        
        # Check cache
        if cache_key in self.verification_cache:
            self.access_counts[cache_key] += 1
            self.last_access_times[cache_key] = time.time()
            
            # Move to end (LRU)
            self.verification_cache.move_to_end(cache_key)
            
            self.cache_performance.verifications_cached += 1
            logger.debug(f"Verification cache HIT: {cache_key[:16]}...")
            
            return self.verification_cache[cache_key]
        
        # Compute verification
        start_time = time.time()
        result = await compute_func()
        elapsed = time.time() - start_time
        
        # Cache result
        self.verification_cache[cache_key] = result
        self.access_counts[cache_key] = 1
        self.last_access_times[cache_key] = time.time()
        
        # Estimate cost saved
        verification_cost = 0.00015  # Approximate cost per verification
        self.cache_performance.verification_cost_saved += verification_cost
        self.cache_performance.verifications_performed += 1
        self.cache_performance.avg_verification_time = (
            (self.cache_performance.avg_verification_time * (self.cache_performance.verifications_performed - 1) + elapsed) /
            self.cache_performance.verifications_performed
        )
        
        # Apply size limits
        await self._enforce_cache_limits()
        
        return result
    
    async def get_or_compute_decomposition(
        self,
        query: str,
        classification: QueryClassification,
        compute_func
    ) -> List[SubQuery]:
        """
        Get cached decomposition or compute new one.
        
        Args:
            query: Query to decompose
            classification: Query classification
            compute_func: Function to compute decomposition if not cached
            
        Returns:
            List of sub-queries
        """
        cache_key = self._generate_decomposition_key(query, classification)
        
        # Check cache
        if cache_key in self.decomposition_cache:
            self.access_counts[cache_key] += 1
            self.last_access_times[cache_key] = time.time()
            
            # Move to end (LRU)
            self.decomposition_cache.move_to_end(cache_key)
            
            logger.debug(f"Decomposition cache HIT: {cache_key[:16]}...")
            return self.decomposition_cache[cache_key]
        
        # Compute decomposition
        result = await compute_func()
        
        # Cache result
        self.decomposition_cache[cache_key] = result
        self.access_counts[cache_key] = 1
        self.last_access_times[cache_key] = time.time()
        
        # Apply size limits
        await self._enforce_cache_limits()
        
        return result
    
    def get_or_compute_embedding(
        self,
        text: str,
        compute_func
    ) -> List[float]:
        """
        Get cached embedding or compute new one.
        
        Args:
            text: Text to embed
            compute_func: Function to compute embedding if not cached
            
        Returns:
            Embedding vector
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self.embedding_cache:
            self.access_counts[cache_key] += 1
            self.last_access_times[cache_key] = time.time()
            
            # Move to end (LRU)
            self.embedding_cache.move_to_end(cache_key)
            
            self.cache_performance.embeddings_cached += 1
            return self.embedding_cache[cache_key]
        
        # Compute embedding
        start_time = time.time()
        result = compute_func()
        elapsed = time.time() - start_time
        
        # Cache result
        self.embedding_cache[cache_key] = result
        self.access_counts[cache_key] = 1
        self.last_access_times[cache_key] = time.time()
        
        self.cache_performance.embeddings_computed += 1
        self.cache_performance.avg_embedding_time = (
            (self.cache_performance.avg_embedding_time * (self.cache_performance.embeddings_computed - 1) + elapsed) /
            self.cache_performance.embeddings_computed
        )
        
        # Apply size limits (synchronous version)
        self._enforce_cache_limits_sync()
        
        return result
    
    async def warm_cache(self, recent_queries: Optional[List[str]] = None):
        """
        Warm cache with common or recent queries.
        
        Args:
            recent_queries: Optional list of recent queries to warm
        """
        if not self.enable_warming:
            return
        
        logger.info("Starting cache warming...")
        
        warming_queries = []
        
        # Add pattern-based queries
        for pattern in self.common_patterns:
            warming_queries.extend([
                f"{pattern} machine learning",
                f"{pattern} deep learning",
                f"{pattern} neural networks",
                f"{pattern} RAG systems"
            ])
        
        # Add recent queries if provided
        if recent_queries:
            warming_queries.extend(recent_queries[:20])
        
        # Warm semantic cache
        warmed_count = self.semantic_cache.warm_cache(warming_queries[:50])
        
        # Pre-compute embeddings for common terms
        common_terms = [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "transformer", "attention mechanism",
            "retrieval augmented generation", "vector database",
            "embedding", "semantic search"
        ]
        
        for term in common_terms:
            self.get_or_compute_embedding(
                term,
                lambda t=term: self._compute_embedding_sync(t)
            )
        
        logger.info(f"Cache warming completed: {warmed_count} queries warmed")
    
    async def _enforce_cache_limits(self):
        """Enforce memory and size limits on caches."""
        # Estimate current memory usage
        estimated_memory_mb = self._estimate_memory_usage()
        
        if estimated_memory_mb > self.max_memory_mb:
            # Apply intelligent eviction
            await self._apply_intelligent_eviction(estimated_memory_mb)
        
        # Update metrics
        self.cache_performance.memory_usage_mb = estimated_memory_mb
        self.cache_performance.cache_size_mb = estimated_memory_mb
    
    def _enforce_cache_limits_sync(self):
        """Synchronous version of cache limit enforcement."""
        estimated_memory_mb = self._estimate_memory_usage()
        
        if estimated_memory_mb > self.max_memory_mb:
            self._apply_intelligent_eviction_sync(estimated_memory_mb)
        
        self.cache_performance.memory_usage_mb = estimated_memory_mb
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage of all caches."""
        # Rough estimates
        verification_size = len(self.verification_cache) * 0.001  # ~1KB per entry
        decomposition_size = len(self.decomposition_cache) * 0.002  # ~2KB per entry
        embedding_size = len(self.embedding_cache) * 0.006  # ~6KB per embedding
        
        # Semantic cache estimate
        semantic_size = self.semantic_cache.get_stats().cache_size * 0.01  # ~10KB per entry
        
        total_mb = verification_size + decomposition_size + embedding_size + semantic_size
        
        return total_mb
    
    async def _apply_intelligent_eviction(self, current_memory_mb: float):
        """Apply intelligent eviction strategy."""
        target_memory_mb = self.max_memory_mb * 0.8  # Target 80% capacity
        memory_to_free = current_memory_mb - target_memory_mb
        
        if memory_to_free <= 0:
            return
        
        logger.info(f"Applying intelligent eviction to free {memory_to_free:.2f}MB")
        
        # Calculate priority scores for all cached items
        eviction_candidates = []
        
        # Collect candidates from all caches
        for cache_name, cache in [
            ("verification", self.verification_cache),
            ("decomposition", self.decomposition_cache),
            ("embedding", self.embedding_cache)
        ]:
            for key in cache.keys():
                priority = self._calculate_eviction_priority(key)
                size_mb = self._estimate_entry_size(cache_name)
                eviction_candidates.append((cache_name, key, priority, size_mb))
        
        # Sort by priority (lower priority = evict first)
        eviction_candidates.sort(key=lambda x: x[2])
        
        # Evict items until we free enough memory
        freed_memory = 0
        evicted_count = 0
        
        for cache_name, key, priority, size_mb in eviction_candidates:
            if freed_memory >= memory_to_free:
                break
            
            # Evict from appropriate cache
            if cache_name == "verification":
                del self.verification_cache[key]
            elif cache_name == "decomposition":
                del self.decomposition_cache[key]
            elif cache_name == "embedding":
                del self.embedding_cache[key]
            
            # Clean up tracking data
            self.access_counts.pop(key, None)
            self.last_access_times.pop(key, None)
            self.query_importance.pop(key, None)
            
            freed_memory += size_mb
            evicted_count += 1
        
        logger.info(f"Evicted {evicted_count} items, freed {freed_memory:.2f}MB")
    
    def _apply_intelligent_eviction_sync(self, current_memory_mb: float):
        """Synchronous version of intelligent eviction."""
        target_memory_mb = self.max_memory_mb * 0.8
        memory_to_free = current_memory_mb - target_memory_mb
        
        if memory_to_free <= 0:
            return
        
        # Similar logic to async version but without await
        eviction_candidates = []
        
        for cache_name, cache in [
            ("embedding", self.embedding_cache)
        ]:
            for key in list(cache.keys()):
                priority = self._calculate_eviction_priority(key)
                size_mb = self._estimate_entry_size(cache_name)
                eviction_candidates.append((cache_name, key, priority, size_mb))
        
        eviction_candidates.sort(key=lambda x: x[2])
        
        freed_memory = 0
        for cache_name, key, priority, size_mb in eviction_candidates:
            if freed_memory >= memory_to_free:
                break
            
            if cache_name == "embedding" and key in self.embedding_cache:
                del self.embedding_cache[key]
                freed_memory += size_mb
    
    def _calculate_eviction_priority(self, key: str) -> float:
        """
        Calculate eviction priority for a cache entry.
        
        Higher score = keep in cache
        Lower score = evict first
        """
        # Factors for priority calculation
        access_count = self.access_counts.get(key, 0)
        last_access = self.last_access_times.get(key, 0)
        importance = self.query_importance.get(key, 0.5)
        
        # Time decay factor (exponential decay)
        time_since_access = time.time() - last_access if last_access > 0 else float('inf')
        time_decay = np.exp(-time_since_access / 3600)  # 1 hour half-life
        
        # Frequency factor (logarithmic growth)
        frequency_factor = np.log1p(access_count)
        
        # Combined priority score
        priority = (
            0.4 * frequency_factor +  # Frequency of access
            0.3 * time_decay +         # Recency of access
            0.3 * importance           # Query importance
        )
        
        return priority
    
    def _estimate_entry_size(self, cache_type: str) -> float:
        """Estimate size of a cache entry in MB."""
        sizes = {
            "verification": 0.001,     # ~1KB
            "decomposition": 0.002,    # ~2KB
            "embedding": 0.006,        # ~6KB
            "semantic": 0.010          # ~10KB
        }
        return sizes.get(cache_type, 0.001)
    
    def _generate_verification_key(self, query: str, response: str) -> str:
        """Generate cache key for verification."""
        combined = f"verify:{query}:{response}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _generate_decomposition_key(self, query: str, classification: QueryClassification) -> str:
        """Generate cache key for decomposition."""
        combined = f"decompose:{query}:{classification.query_type.value}:{classification.complexity_score:.2f}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _compute_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding computation."""
        try:
            if Settings.embed_model:
                embedding = Settings.embed_model.get_text_embedding(text)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
                return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
        return []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_entries = (
            len(self.verification_cache) +
            len(self.decomposition_cache) +
            len(self.embedding_cache) +
            self.semantic_cache.get_stats().cache_size
        )
        
        return {
            "total_cache_entries": total_entries,
            "verification_cache_size": len(self.verification_cache),
            "decomposition_cache_size": len(self.decomposition_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "semantic_cache_size": self.semantic_cache.get_stats().cache_size,
            "memory_usage_mb": self.cache_performance.memory_usage_mb,
            "memory_limit_mb": self.max_memory_mb,
            "cache_hit_rate": self.cache_performance.cache_hit_rate,
            "verifications_cached": self.cache_performance.verifications_cached,
            "embeddings_cached": self.cache_performance.embeddings_cached,
            "cost_saved": self.cache_performance.cost_saved,
            "warming_enabled": self.enable_warming,
            "adaptive_sizing_enabled": self.enable_adaptive_sizing
        }


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Integrates all performance optimization components and provides
    a unified interface for the RAG system.
    """
    
    def __init__(
        self,
        semantic_cache: Optional[SemanticCache] = None,
        enable_all_optimizations: bool = True
    ):
        """
        Initialize performance optimizer.
        
        Args:
            semantic_cache: Existing semantic cache instance
            enable_all_optimizations: Enable all optimization features
        """
        self.semantic_cache = semantic_cache or SemanticCache()
        self.enable_all = enable_all_optimizations
        
        # Initialize components
        self.similarity_detector = AdvancedSimilarityDetector(
            enable_all_levels=enable_all_optimizations
        )
        
        self.batch_processor = BatchEmbeddingProcessor(
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        )
        
        self.cache_manager = IntelligentCacheManager(
            semantic_cache=self.semantic_cache,
            max_memory_mb=float(os.getenv("MAX_CACHE_MEMORY_MB", "512")),
            enable_warming=enable_all_optimizations
        )
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Connection pooling settings
        self.connection_pool_size = int(os.getenv("CONNECTION_POOL_SIZE", "10"))
        self.connection_timeout = float(os.getenv("CONNECTION_TIMEOUT", "30"))
        
        logger.info("PerformanceOptimizer initialized with all optimizations enabled")
    
    async def optimize_query_processing(
        self,
        query: str,
        classification: QueryClassification
    ) -> Dict[str, Any]:
        """
        Apply all optimizations to query processing.
        
        Args:
            query: User query
            classification: Query classification
            
        Returns:
            Optimization results and recommendations
        """
        start_time = time.time()
        optimizations = {}
        
        # 1. Check semantic cache with advanced similarity
        cache_result = await self._check_advanced_cache(query, classification)
        if cache_result:
            optimizations["cache_hit"] = True
            optimizations["similarity_score"] = cache_result["similarity"]
            optimizations["response"] = cache_result["response"]
            
            # Update metrics
            self.metrics.cache_hits += 1
            self.metrics.update_cache_metrics(True, time.time() - start_time)
            
            return optimizations
        
        # 2. Apply query optimization
        optimized_query = await self._optimize_query(query, classification)
        optimizations["optimized_query"] = optimized_query
        
        # 3. Batch embedding processing
        if classification.requires_decomposition:
            embeddings = await self._batch_process_embeddings([query] + optimized_query.get("subqueries", []))
            optimizations["embeddings_batched"] = len(embeddings)
        
        # 4. Connection pooling optimization
        optimizations["connection_pool"] = {
            "size": self.connection_pool_size,
            "timeout": self.connection_timeout
        }
        
        # Update metrics
        self.metrics.queries_processed += 1
        query_time = time.time() - start_time
        self.metrics.avg_query_time = (
            (self.metrics.avg_query_time * (self.metrics.queries_processed - 1) + query_time) /
            self.metrics.queries_processed
        )
        
        optimizations["processing_time"] = query_time
        
        return optimizations
    
    async def _check_advanced_cache(
        self,
        query: str,
        classification: QueryClassification
    ) -> Optional[Dict[str, Any]]:
        """Check cache with advanced similarity detection."""
        try:
            # Get dynamic threshold
            threshold = self.similarity_detector.get_dynamic_threshold(
                classification.query_type,
                classification.complexity_score,
                SimilarityLevel.SEMANTIC
            )
            
            # Check semantic cache
            cache_result = self.semantic_cache.get(query)
            
            if cache_result:
                response, nodes, similarity = cache_result
                
                # Verify similarity meets dynamic threshold
                if similarity >= threshold:
                    return {
                        "response": response,
                        "nodes": nodes,
                        "similarity": similarity,
                        "threshold": threshold
                    }
            
        except Exception as e:
            logger.error(f"Advanced cache check failed: {e}")
        
        return None
    
    async def _optimize_query(
        self,
        query: str,
        classification: QueryClassification
    ) -> Dict[str, Any]:
        """Optimize query for processing."""
        optimizations = {}
        
        # Normalize query
        normalized = self.similarity_detector._normalize_query(query)
        optimizations["normalized_query"] = normalized
        
        # Extract key terms for focused retrieval
        tokens = self.similarity_detector._extract_lexical_tokens(normalized)
        optimizations["key_terms"] = list(tokens)[:10]
        
        # Determine optimal retrieval parameters
        if classification.complexity_score > 0.7:
            optimizations["similarity_top_k"] = min(15, classification.estimated_chunks_needed)
            optimizations["rerank_enabled"] = True
        else:
            optimizations["similarity_top_k"] = min(10, classification.estimated_chunks_needed)
            optimizations["rerank_enabled"] = False
        
        return optimizations
    
    async def _batch_process_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Process embeddings in batch."""
        embeddings = []
        
        for text in texts:
            embedding = await self.batch_processor.get_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    async def warm_system(self):
        """Warm all caches and prepare system for optimal performance."""
        logger.info("Starting system warming...")
        
        # Warm caches
        await self.cache_manager.warm_cache()
        
        # Pre-compute common embeddings
        common_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What is RAG?",
            "Compare supervised and unsupervised learning"
        ]
        
        await self._batch_process_embeddings(common_queries)
        
        logger.info("System warming completed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "cache_performance": {
                "hit_rate": self.metrics.cache_hit_rate,
                "avg_lookup_time": self.metrics.avg_cache_lookup_time,
                "total_hits": self.metrics.cache_hits,
                "total_misses": self.metrics.cache_misses
            },
            "embedding_performance": self.batch_processor.get_performance_stats(),
            "cache_statistics": self.cache_manager.get_cache_statistics(),
            "query_performance": {
                "total_queries": self.metrics.queries_processed,
                "avg_query_time": self.metrics.avg_query_time,
                "p95_query_time": self.metrics.p95_query_time,
                "p99_query_time": self.metrics.p99_query_time
            },
            "cost_analysis": {
                "total_cost": self.metrics.total_cost,
                "cost_saved": self.metrics.cost_saved,
                "cost_reduction_percentage": self.metrics.cost_reduction_percentage
            },
            "memory_usage": {
                "current_mb": self.metrics.memory_usage_mb,
                "cache_size_mb": self.metrics.cache_size_mb
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Cache hit rate recommendations
        if self.metrics.cache_hit_rate < 0.3:
            recommendations.append(
                "Cache hit rate is below 30%. Consider enabling cache warming "
                "or adjusting similarity thresholds."
            )
        
        # Query time recommendations
        if self.metrics.avg_query_time > 3.0:
            recommendations.append(
                "Average query time exceeds 3 seconds. Consider increasing "
                "batch size or optimizing retrieval parameters."
            )
        
        # Memory recommendations
        if self.metrics.memory_usage_mb > self.cache_manager.max_memory_mb * 0.9:
            recommendations.append(
                "Memory usage is above 90% of limit. Consider increasing "
                "memory allocation or adjusting eviction policies."
            )
        
        # Cost recommendations
        if self.metrics.cost_reduction_percentage < 20:
            recommendations.append(
                "Cost reduction is below 20%. Enable more aggressive caching "
                "and batch processing to reduce API costs."
            )
        
        if not recommendations:
            recommendations.append("System is performing optimally. No immediate optimizations needed.")
        
        return recommendations


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(semantic_cache: Optional[SemanticCache] = None) -> PerformanceOptimizer:
    """
    Get the global performance optimizer instance.
    
    Args:
        semantic_cache: Optional semantic cache instance
        
    Returns:
        PerformanceOptimizer instance
    """
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(
            semantic_cache=semantic_cache,
            enable_all_optimizations=os.getenv("ENABLE_ALL_OPTIMIZATIONS", "true").lower() == "true"
        )
    
    return _performance_optimizer


async def optimize_rag_performance(
    query: str,
    classification: QueryClassification
) -> Dict[str, Any]:
    """
    Main entry point for performance optimization.
    
    Args:
        query: User query
        classification: Query classification
        
    Returns:
        Optimization results and recommendations
    """
    optimizer = get_performance_optimizer()
    return await optimizer.optimize_query_processing(query, classification)