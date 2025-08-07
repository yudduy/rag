import os
import time
import logging
from typing import Any, Optional, List, Tuple, Union

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.postprocessor.metadata_replacement import MetadataReplacementPostProcessor
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import QueryBundle

from src.agentic import QueryType, QueryClassification
from src.cache import get_cache, estimate_query_cost
from src.verification_integration import create_verified_query_tool
from src.multimodal import (
    get_multimodal_embedding_model,
    is_multimodal_enabled,
    is_cross_modal_search_enabled
)

logger = logging.getLogger(__name__)


class CrossModalRetriever:
    """
    Cross-modal retriever that can search across text and image modalities.
    
    Enables queries like:
    - Text queries that retrieve relevant images
    - Mixed-modal results with proper weighting
    - Cross-modal similarity scoring
    """
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        multimodal_model: Optional[Any] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        multimodal_threshold: float = 0.6
    ):
        """
        Initialize cross-modal retriever.
        
        Args:
            vector_retriever: Base vector retriever
            multimodal_model: Multimodal embedding model
            text_weight: Weight for text similarity
            image_weight: Weight for image similarity
            multimodal_threshold: Minimum similarity for cross-modal results
        """
        self.vector_retriever = vector_retriever
        self.multimodal_model = multimodal_model
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.multimodal_threshold = multimodal_threshold
        
        logger.info(f"CrossModalRetriever initialized with weights T:{text_weight}/I:{image_weight}")
    
    def retrieve(self, query_bundle: QueryBundle) -> List:
        """
        Perform cross-modal retrieval.
        
        Args:
            query_bundle: Query information
            
        Returns:
            List of retrieved nodes with cross-modal scoring
        """
        try:
            query_str = query_bundle.query_str
            
            # Get base retrieval results
            base_results = self.vector_retriever.retrieve(query_bundle)
            
            if not is_cross_modal_search_enabled() or not self.multimodal_model:
                return base_results
            
            # Enhance with cross-modal scoring
            enhanced_results = self._apply_cross_modal_scoring(query_str, base_results)
            
            # Filter by multimodal threshold
            filtered_results = [
                result for result in enhanced_results
                if result.score >= self.multimodal_threshold
            ]
            
            logger.debug(f"Cross-modal retrieval: {len(base_results)} -> {len(filtered_results)} nodes")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Cross-modal retrieval failed: {e}")
            return self.vector_retriever.retrieve(query_bundle)
    
    def _apply_cross_modal_scoring(self, query: str, results: List) -> List:
        """
        Apply cross-modal similarity scoring to results.
        
        Args:
            query: Query string
            results: Base retrieval results
            
        Returns:
            Results with enhanced cross-modal scores
        """
        try:
            # Get query embedding in multimodal space
            query_embedding = self.multimodal_model.get_text_embedding(query)
            
            enhanced_results = []
            
            for result in results:
                node = result.node
                original_score = result.score
                
                # Check if node has multimodal metadata
                modality = node.metadata.get("modality", "text")
                
                if modality == "text":
                    # For text nodes, use text weight
                    cross_modal_score = original_score * self.text_weight
                    
                elif modality == "image":
                    # For image nodes, compute cross-modal similarity
                    if hasattr(node, 'embedding') and node.embedding:
                        # Use existing CLIP embedding
                        image_embedding = node.embedding
                    else:
                        # Compute CLIP embedding for image
                        image_path = node.metadata.get("image_path")
                        if image_path:
                            image_embedding = self.multimodal_model.get_image_embedding(image_path)
                        else:
                            image_embedding = None
                    
                    if image_embedding:
                        # Compute cross-modal similarity
                        similarity = self.multimodal_model.compute_similarity(
                            query_embedding, image_embedding
                        )
                        cross_modal_score = similarity * self.image_weight
                    else:
                        cross_modal_score = original_score * self.image_weight
                
                else:
                    # Unknown modality, use original score
                    cross_modal_score = original_score
                
                # Create enhanced result
                from llama_index.core.schema import NodeWithScore
                enhanced_result = NodeWithScore(
                    node=node,
                    score=cross_modal_score
                )
                
                enhanced_results.append(enhanced_result)
            
            # Sort by enhanced scores
            enhanced_results.sort(key=lambda x: x.score, reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Cross-modal scoring failed: {e}")
            return results


class MultimodalQueryEngine:
    """
    Query engine with multimodal capabilities and cross-modal search.
    """
    
    def __init__(
        self,
        base_query_engine: BaseQueryEngine,
        enable_cross_modal: bool = True
    ):
        """
        Initialize multimodal query engine.
        
        Args:
            base_query_engine: Base query engine
            enable_cross_modal: Whether to enable cross-modal search
        """
        self.base_query_engine = base_query_engine
        self.enable_cross_modal = enable_cross_modal
        self.multimodal_model = None
        
        # Initialize multimodal model if enabled
        if enable_cross_modal and is_multimodal_enabled():
            self.multimodal_model = get_multimodal_embedding_model()
            if self.multimodal_model:
                logger.info("Multimodal query engine initialized with cross-modal search")
            else:
                logger.warning("Multimodal model unavailable, falling back to standard search")
                self.enable_cross_modal = False
        else:
            self.enable_cross_modal = False
            logger.info("Multimodal query engine initialized in text-only mode")
    
    def query(self, query: Union[str, QueryBundle], **kwargs) -> Any:
        """
        Perform multimodal query.
        
        Args:
            query: Query string or bundle
            **kwargs: Additional query parameters
            
        Returns:
            Query response with multimodal results
        """
        try:
            # Convert to QueryBundle if needed
            if isinstance(query, str):
                query_bundle = QueryBundle(query_str=query)
            else:
                query_bundle = query
            
            # Add multimodal context to query if enabled
            if self.enable_cross_modal and self.multimodal_model:
                enhanced_query = self._enhance_query_for_multimodal(query_bundle)
                response = self.base_query_engine.query(enhanced_query, **kwargs)
                
                # Post-process response for multimodal presentation
                response = self._enhance_response_with_multimodal_info(response)
            else:
                response = self.base_query_engine.query(query_bundle, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"Multimodal query failed: {e}")
            # Fallback to base query engine
            return self.base_query_engine.query(query, **kwargs)
    
    def _enhance_query_for_multimodal(self, query_bundle: QueryBundle) -> QueryBundle:
        """
        Enhance query with multimodal context.
        
        Args:
            query_bundle: Original query bundle
            
        Returns:
            Enhanced query bundle
        """
        try:
            query_str = query_bundle.query_str
            
            # Add multimodal search hints to query
            multimodal_hints = [
                "Consider both text and image content in the search.",
                "Include relevant visual information if available.",
                "Cross-reference text descriptions with image content."
            ]
            
            # Check if query suggests image content
            image_keywords = ["image", "picture", "photo", "visual", "diagram", "chart", "graph", "figure"]
            has_image_intent = any(keyword in query_str.lower() for keyword in image_keywords)
            
            if has_image_intent:
                enhanced_query = f"{query_str}\n\nNote: This query may benefit from visual content analysis."
            else:
                enhanced_query = query_str
            
            return QueryBundle(
                query_str=enhanced_query,
                custom_embedding_strs=query_bundle.custom_embedding_strs,
                embedding=query_bundle.embedding
            )
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query_bundle
    
    def _enhance_response_with_multimodal_info(self, response: Any) -> Any:
        """
        Enhance response with multimodal information.
        
        Args:
            response: Original response
            
        Returns:
            Enhanced response with multimodal context
        """
        try:
            if not hasattr(response, 'source_nodes') or not response.source_nodes:
                return response
            
            # Analyze source nodes for multimodal content
            text_nodes = []
            image_nodes = []
            
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                modality = node.metadata.get("modality", "text")
                
                if modality == "image":
                    image_nodes.append(node_with_score)
                else:
                    text_nodes.append(node_with_score)
            
            # Add multimodal context to response
            if image_nodes:
                multimodal_context = f"\n\nThis response includes information from {len(image_nodes)} image(s) and {len(text_nodes)} text source(s)."
                
                # Add image references
                image_refs = []
                for i, img_node in enumerate(image_nodes[:3], 1):  # Limit to 3 images
                    img_name = img_node.node.metadata.get("image_name", f"Image {i}")
                    image_refs.append(f"[Image {i}: {img_name}]")
                
                if image_refs:
                    multimodal_context += f"\n\nReferenced images: {', '.join(image_refs)}"
                
                # Enhance response text
                response.response = response.response + multimodal_context
            
            return response
            
        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            return response
    
    # Delegate other methods to base query engine
    def __getattr__(self, name):
        return getattr(self.base_query_engine, name)


class CachedQueryEngine:
    """
    A wrapper around BaseQueryEngine that provides semantic caching capabilities.
    
    This class intercepts queries and checks for semantically similar cached responses
    before falling back to the underlying query engine. It provides significant
    performance improvements and cost savings for repeated or similar queries.
    """
    
    def __init__(self, query_engine: BaseQueryEngine, enable_cache: bool = True):
        """
        Initialize the cached query engine.
        
        Args:
            query_engine: The underlying query engine to wrap
            enable_cache: Whether to enable caching for this instance
        """
        self.query_engine = query_engine
        self.enable_cache = enable_cache
        self.cache = get_cache() if enable_cache else None
        
        logger.info(f"CachedQueryEngine initialized with caching {'enabled' if enable_cache else 'disabled'}")
    
    def query(self, query: str) -> Any:
        """
        Query with semantic caching.
        
        Args:
            query: Query string
            
        Returns:
            Response object from cache or query engine
        """
        start_time = time.time()
        
        # Try cache first if enabled
        if self.enable_cache and self.cache and self.cache.enabled:
            try:
                cache_result = self.cache.get(query)
                if cache_result:
                    response_data, nodes, similarity_score = cache_result
                    
                    # Reconstruct Response object from cached data
                    cached_response = self._reconstruct_response(response_data, nodes)
                    
                    cache_time = time.time() - start_time
                    logger.info(
                        f"Cache HIT: Query served from cache in {cache_time:.3f}s "
                        f"(similarity: {similarity_score:.3f})"
                    )
                    
                    return cached_response
                    
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}. Falling back to query engine")
        
        # Cache miss or cache disabled - use underlying query engine
        logger.debug("Cache MISS: Executing query on underlying engine")
        
        try:
            response = self.query_engine.query(query)
            query_time = time.time() - start_time
            
            # Cache the response if caching is enabled
            if self.enable_cache and self.cache and self.cache.enabled:
                try:
                    # Estimate cost for statistics
                    estimated_cost = estimate_query_cost(query, len(str(response.response).split()) * 1.3)
                    
                    # Cache the response
                    self.cache.put(query, response, estimated_cost)
                    
                    logger.debug(f"Query response cached (estimated cost: ${estimated_cost:.4f})")
                    
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            logger.info(f"Query executed in {query_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _reconstruct_response(self, response_data: dict, nodes: List[dict]) -> Any:
        """
        Reconstruct a Response object from cached data.
        
        Args:
            response_data: Cached response data
            nodes: Cached node data
            
        Returns:
            Reconstructed Response object
        """
        from llama_index.core.schema import NodeWithScore, TextNode
        from llama_index.core.base.response.schema import Response
        
        try:
            # Reconstruct source nodes
            source_nodes = []
            for node_data in nodes:
                # Create TextNode from cached data
                text_node = TextNode(
                    id_=node_data.get("node_id", "unknown"),
                    text=node_data.get("text", ""),
                    metadata=node_data.get("metadata", {})
                )
                
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=text_node,
                    score=node_data.get("score", 1.0)
                )
                source_nodes.append(node_with_score)
            
            # Create Response object
            response = Response(
                response=response_data.get("response", ""),
                source_nodes=source_nodes,
                metadata=response_data.get("metadata", {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to reconstruct response: {e}")
            # Return minimal response if reconstruction fails
            return Response(response=response_data.get("response", ""), source_nodes=[])
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache:
            stats = self.cache.get_stats()
            return {
                "hit_rate": self.cache.get_hit_rate(),
                "total_queries": stats.total_queries,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "cache_size": stats.cache_size,
                "avg_lookup_time": stats.avg_cache_lookup_time,
                "total_cost_saved": stats.total_cost_saved,
                "evictions": stats.evictions
            }
        return {}
    
    def clear_cache(self) -> bool:
        """
        Clear the cache.
        
        Returns:
            True if cache was cleared successfully
        """
        if self.cache:
            return self.cache.clear_cache()
        return False
    
    def warm_cache(self, queries: List[str]) -> int:
        """
        Warm the cache with common queries.
        
        Args:
            queries: List of queries to warm the cache with
            
        Returns:
            Number of queries successfully warmed
        """
        if not self.cache:
            return 0
            
        return self.cache.warm_cache(queries)
    
    # Delegate other methods to the underlying query engine
    def __getattr__(self, name):
        return getattr(self.query_engine, name)


def create_query_engine(
    index: BaseIndex, 
    classification: Optional[QueryClassification] = None,
    enable_cache: bool = True,
    **kwargs: Any
) -> BaseQueryEngine:
    """
    Create a sophisticated query engine with multi-stage retrieval cascade and semantic caching.
    
    Implements:
    - Hybrid search combining vector similarity and BM25 keyword search
    - Query fusion with reciprocal reranking
    - LLM-based reranking for improved relevance scoring
    - Metadata replacement for sentence window context expansion
    - Dynamic routing based on query type and complexity
    - Semantic caching with embedding-based similarity matching
    
    Args:
        index: The index to create a query engine for.
        classification: Optional query classification for routing decisions
        enable_cache: Whether to enable semantic caching (default: True)
        **kwargs: Additional parameters for query engine configuration
        
    Returns:
        BaseQueryEngine: Configured query engine with multi-stage retrieval and caching
        
    Raises:
        ValueError: If index is invalid or configuration is incorrect
        RuntimeError: If query engine creation fails
    """
    try:
        # Configuration from environment with validation
        top_k = int(os.getenv("TOP_K", "10"))
        hybrid_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
        rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"
        rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))
        rerank_model = os.getenv("RERANK_MODEL", "gpt-3.5-turbo")
        
        # Apply intelligent routing based on query classification
        if classification:
            top_k, hybrid_enabled, rerank_enabled, rerank_top_n = _apply_routing_strategy(
                classification, top_k, hybrid_enabled, rerank_enabled, rerank_top_n
            )
        
        # Override with passed parameters
        top_k = kwargs.get("similarity_top_k", top_k)
        
        # Validation
        if top_k <= 0:
            raise ValueError(f"TOP_K must be positive, got {top_k}")
        if rerank_top_n <= 0:
            raise ValueError(f"RERANK_TOP_N must be positive, got {rerank_top_n}")
        if rerank_top_n > top_k:
            logger.warning(f"RERANK_TOP_N ({rerank_top_n}) > TOP_K ({top_k}), adjusting to {top_k}")
            rerank_top_n = top_k
        
        logger.info(f"Creating query engine: Hybrid={hybrid_enabled}, Rerank={rerank_enabled}, TOP_K={top_k}")
        
        start_time = time.time()
        
        # Create base vector retriever
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )
        
        # Wrap with cross-modal retriever if enabled
        if is_cross_modal_search_enabled():
            multimodal_model = get_multimodal_embedding_model()
            if multimodal_model:
                text_weight = float(os.getenv("TEXT_IMAGE_WEIGHT_RATIO", "0.7"))
                image_weight = 1.0 - text_weight
                multimodal_threshold = float(os.getenv("MULTIMODAL_THRESHOLD", "0.6"))
                
                cross_modal_retriever = CrossModalRetriever(
                    vector_retriever=vector_retriever,
                    multimodal_model=multimodal_model,
                    text_weight=text_weight,
                    image_weight=image_weight,
                    multimodal_threshold=multimodal_threshold
                )
                
                # Create a wrapper that implements the retriever interface
                class CrossModalRetrieverWrapper:
                    def __init__(self, cross_modal_retriever):
                        self.cross_modal_retriever = cross_modal_retriever
                    
                    def retrieve(self, query_bundle):
                        return self.cross_modal_retriever.retrieve(query_bundle)
                    
                    def __getattr__(self, name):
                        return getattr(self.cross_modal_retriever.vector_retriever, name)
                
                retriever = CrossModalRetrieverWrapper(cross_modal_retriever)
                logger.info("Cross-modal retriever enabled")
            else:
                retriever = vector_retriever
                logger.warning("Cross-modal search requested but multimodal model unavailable")
        else:
            retriever = vector_retriever
        
        # Add BM25 hybrid search if enabled
        if hybrid_enabled:
            try:
                # Get nodes for BM25 retriever
                nodes = list(index.docstore.docs.values())
                if not nodes:
                    logger.warning("No nodes found in index docstore, falling back to vector-only search")
                    hybrid_enabled = False
                else:
                    bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=top_k,
                    )
                    
                    # Combine retrievers with reciprocal rerank fusion
                    retriever = QueryFusionRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        retriever_weights=[0.7, 0.3],  # Favor vector search slightly
                        similarity_top_k=top_k,
                        num_queries=1,
                        mode=FUSION_MODES.RECIPROCAL_RANK,
                        use_async=False,
                    )
                    logger.info("Hybrid search enabled with vector + BM25 retrievers")
            except Exception as e:
                logger.warning(f"Failed to enable hybrid search: {e}. Falling back to vector-only search")
                hybrid_enabled = False
        
        # Configure post-processors
        postprocessors = []
        
        # Metadata replacement for sentence window expansion
        metadata_replacement = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        postprocessors.append(metadata_replacement)
        
        # LLM reranking with cost-efficient model
        if rerank_enabled:
            try:
                rerank_llm = OpenAI(
                    model=rerank_model,
                    temperature=0,
                    max_tokens=1024,  # Limit tokens for cost efficiency
                )
                
                reranker = LLMRerank(
                    top_n=rerank_top_n,
                    choice_batch_size=5,
                    llm=rerank_llm,
                )
                postprocessors.append(reranker)
                logger.info(f"LLM reranking enabled with {rerank_model}, top_n={rerank_top_n}")
            except Exception as e:
                logger.warning(f"Failed to enable LLM reranking: {e}. Continuing without reranking")
                rerank_enabled = False
        
        # Similarity threshold filtering
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
        if 0.0 <= similarity_threshold <= 1.0:
            similarity_processor = SimilarityPostprocessor(
                similarity_cutoff=similarity_threshold
            )
            postprocessors.append(similarity_processor)
            logger.info(f"Similarity filtering enabled with threshold: {similarity_threshold}")
        else:
            logger.warning(f"Invalid similarity threshold: {similarity_threshold}. Skipping similarity filtering")
        
        # Create base query engine
        base_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            **kwargs
        )
        
        setup_time = time.time() - start_time
        logger.info(f"Base query engine setup completed in {setup_time:.2f}s")
        
        # Wrap with multimodal capabilities if enabled
        if is_multimodal_enabled():
            multimodal_query_engine = MultimodalQueryEngine(
                base_query_engine=base_query_engine,
                enable_cross_modal=is_cross_modal_search_enabled()
            )
            
            # Wrap with semantic cache if enabled
            if enable_cache:
                cached_query_engine = CachedQueryEngine(multimodal_query_engine, enable_cache=True)
                logger.info("Query engine wrapped with multimodal capabilities and semantic caching")
                return cached_query_engine
            else:
                logger.info("Query engine wrapped with multimodal capabilities")
                return multimodal_query_engine
        else:
            # Standard text-only engine
            if enable_cache:
                cached_query_engine = CachedQueryEngine(base_query_engine, enable_cache=True)
                logger.info("Query engine wrapped with semantic caching")
                return cached_query_engine
            else:
                logger.info("Standard text-only query engine created")
                return base_query_engine
        
    except Exception as e:
        logger.error(f"Failed to create query engine: {str(e)}")
        raise RuntimeError(f"Query engine creation failed: {str(e)}") from e


def get_query_engine_tool(
    index: BaseIndex,
    name: Optional[str] = None,
    description: Optional[str] = None,
    enable_cache: bool = True,
    **kwargs: Any,
) -> QueryEngineTool:
    """
    Get a query engine tool for the given index with enhanced retrieval capabilities and semantic caching.

    Args:
        index: The index to create a query engine for
        name: Optional name for the tool (defaults to "query_index")
        description: Optional description for the tool
        enable_cache: Whether to enable semantic caching (default: True)
        **kwargs: Additional parameters passed to create_query_engine
        
    Returns:
        QueryEngineTool: Configured tool with multi-stage retrieval and caching
        
    Raises:
        ValueError: If index is None or invalid
        RuntimeError: If tool creation fails
    """
    try:
        if index is None:
            raise ValueError("Index cannot be None")
            
        if name is None:
            name = "query_index"
            
        if description is None:
            cache_desc = " with intelligent semantic caching" if enable_cache else ""
            description = (
                f"Use this tool to retrieve information from a knowledge base using advanced "
                f"hybrid search combining semantic similarity and keyword matching{cache_desc}. "
                f"Provide a specific query and the system will return relevant information "
                f"with proper citations."
            )
            
        logger.info(f"Creating query engine tool: {name} (caching {'enabled' if enable_cache else 'disabled'})")
        
        query_engine = create_query_engine(index, enable_cache=enable_cache, **kwargs)
        
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=name,
            description=description,
        )
        
        # Add verification capabilities if enabled
        verified_tool = create_verified_query_tool(tool)
        
        logger.info(f"Successfully created query engine tool: {name}")
        return verified_tool
        
    except Exception as e:
        logger.error(f"Failed to create query engine tool: {str(e)}")
        raise RuntimeError(f"Query engine tool creation failed: {str(e)}") from e


def _apply_routing_strategy(
    classification: QueryClassification,
    default_top_k: int,
    default_hybrid: bool,
    default_rerank: bool,
    default_rerank_top_n: int
) -> Tuple[int, bool, bool, int]:
    """
    Apply intelligent routing strategy based on query classification.
    
    This function optimizes retrieval parameters based on query type and complexity
    to improve performance and accuracy while managing costs.
    
    Args:
        classification: Query classification results
        default_top_k: Default top_k value
        default_hybrid: Default hybrid search setting
        default_rerank: Default reranking setting  
        default_rerank_top_n: Default rerank top_n value
        
    Returns:
        Tuple of optimized (top_k, hybrid_enabled, rerank_enabled, rerank_top_n)
    """
    try:
        # Start with defaults
        top_k = default_top_k
        hybrid_enabled = default_hybrid
        rerank_enabled = default_rerank
        rerank_top_n = default_rerank_top_n
        
        # Routing strategy based on query type
        if classification.query_type == QueryType.FACTUAL:
            # Factual queries: Lower top_k, prefer hybrid search for precision
            top_k = min(classification.estimated_chunks_needed, 8)
            hybrid_enabled = True
            rerank_enabled = True
            rerank_top_n = min(top_k, 3)  # Strong reranking for precision
            
        elif classification.query_type == QueryType.SEMANTIC:
            # Semantic queries: Higher top_k, vector search focus
            top_k = min(classification.estimated_chunks_needed + 2, 12)
            hybrid_enabled = classification.complexity_score > 0.6
            rerank_enabled = True
            rerank_top_n = min(top_k, 5)
            
        elif classification.query_type == QueryType.COMPARATIVE:
            # Comparative queries: High top_k, strong reranking
            top_k = min(classification.estimated_chunks_needed + 3, 15)
            hybrid_enabled = True
            rerank_enabled = True
            rerank_top_n = min(top_k, 7)  # More candidates for comparison
            
        elif classification.query_type == QueryType.PROCEDURAL:
            # Procedural queries: High top_k, hybrid for step coverage
            top_k = min(classification.estimated_chunks_needed + 2, 12)
            hybrid_enabled = True
            rerank_enabled = True
            rerank_top_n = min(top_k, 6)
            
        elif classification.query_type == QueryType.ANALYTICAL:
            # Analytical queries: Very high top_k, comprehensive search
            top_k = min(classification.estimated_chunks_needed + 4, 18)
            hybrid_enabled = True
            rerank_enabled = True
            rerank_top_n = min(top_k, 8)
            
        elif classification.query_type == QueryType.MULTIFACETED:
            # Multifaceted queries: Maximum coverage
            top_k = min(classification.estimated_chunks_needed + 5, 20)
            hybrid_enabled = True
            rerank_enabled = True
            rerank_top_n = min(top_k, 10)
        
        # Adjust based on complexity score
        complexity_multiplier = 1.0 + (classification.complexity_score * 0.3)
        top_k = int(top_k * complexity_multiplier)
        
        # Adjust based on confidence - lower confidence = more retrieval
        if classification.confidence < 0.7:
            top_k = int(top_k * 1.2)
            rerank_top_n = min(int(rerank_top_n * 1.2), top_k)
        
        # Apply limits to prevent excessive costs
        top_k = min(max(top_k, 3), 25)  # Ensure reasonable bounds
        rerank_top_n = min(max(rerank_top_n, 2), top_k, 12)  # Limit reranking cost
        
        # Cost optimization: Disable expensive features for simple queries
        if (classification.query_type == QueryType.FACTUAL and 
            classification.complexity_score < 0.5 and 
            classification.confidence > 0.8):
            rerank_enabled = False  # Skip reranking for simple factual queries
        
        logger.info(
            f"Applied routing strategy for {classification.query_type.value}: "
            f"top_k={top_k}, hybrid={hybrid_enabled}, rerank={rerank_enabled}, "
            f"rerank_top_n={rerank_top_n}"
        )
        
        return top_k, hybrid_enabled, rerank_enabled, rerank_top_n
        
    except Exception as e:
        logger.error(f"Routing strategy failed: {e}, using defaults")
        return default_top_k, default_hybrid, default_rerank, default_rerank_top_n


def get_cache_statistics() -> dict:
    """
    Get comprehensive cache statistics across all query engines.
    
    Returns:
        Dictionary with cache performance metrics
    """
    try:
        cache = get_cache()
        stats = cache.get_stats()
        
        return {
            "cache_enabled": cache.enabled,
            "hit_rate_percentage": cache.get_hit_rate(),
            "total_queries": stats.total_queries,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "cache_size": stats.cache_size,
            "max_cache_size": cache.max_size,
            "average_lookup_time_ms": stats.avg_cache_lookup_time * 1000,
            "average_similarity_score": stats.avg_similarity_score,
            "total_cost_saved_usd": stats.total_cost_saved,
            "cache_evictions": stats.evictions,
            "similarity_threshold": cache.similarity_threshold,
            "ttl_seconds": cache.ttl,
            "health": cache.health_check()
        }
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        return {"error": str(e), "cache_enabled": False}


def clear_global_cache() -> bool:
    """
    Clear the global semantic cache.
    
    Returns:
        True if cache was cleared successfully
    """
    try:
        cache = get_cache()
        return cache.clear_cache()
    except Exception as e:
        logger.error(f"Failed to clear global cache: {e}")
        return False


def warm_global_cache(queries: List[str]) -> int:
    """
    Warm the global cache with common queries.
    
    Args:
        queries: List of queries to warm the cache with
        
    Returns:
        Number of queries successfully warmed
    """
    try:
        cache = get_cache()
        return cache.warm_cache(queries)
    except Exception as e:
        logger.error(f"Failed to warm global cache: {e}")
        return 0


def estimate_cache_benefits(query_volume_per_hour: int, duplicate_rate: float = 0.31) -> dict:
    """
    Estimate potential cache benefits based on query patterns.
    
    Args:
        query_volume_per_hour: Expected queries per hour
        duplicate_rate: Expected rate of duplicate/similar queries (0-1)
        
    Returns:
        Dictionary with estimated benefits
    """
    try:
        # Estimate cost per query (rough approximation)
        avg_cost_per_query = 0.02  # $0.02 per query average
        
        # Calculate potential savings
        duplicate_queries = query_volume_per_hour * duplicate_rate
        cost_savings_per_hour = duplicate_queries * avg_cost_per_query
        
        # Calculate performance improvements
        avg_query_time_no_cache = 2.5  # 2.5 seconds average
        avg_cache_lookup_time = 0.05   # 50ms cache lookup
        time_saved_per_query = avg_query_time_no_cache - avg_cache_lookup_time
        total_time_saved_per_hour = duplicate_queries * time_saved_per_query
        
        return {
            "queries_per_hour": query_volume_per_hour,
            "duplicate_rate": duplicate_rate,
            "potential_cache_hits_per_hour": duplicate_queries,
            "cost_savings_per_hour_usd": round(cost_savings_per_hour, 4),
            "cost_savings_per_day_usd": round(cost_savings_per_hour * 24, 2),
            "cost_savings_per_month_usd": round(cost_savings_per_hour * 24 * 30, 2),
            "time_saved_per_hour_seconds": round(total_time_saved_per_hour, 1),
            "average_speedup_for_cached_queries": f"{avg_query_time_no_cache / avg_cache_lookup_time:.1f}x",
            "recommended_cache_size": min(max(query_volume_per_hour * 24, 1000), 50000),  # 1 day worth, max 50k
            "recommended_ttl_seconds": 3600 * 8,  # 8 hours for dynamic content
        }
    except Exception as e:
        logger.error(f"Failed to estimate cache benefits: {e}")
        return {"error": str(e)}
