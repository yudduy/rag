"""
Hallucination Detection and Multi-Level Confidence System for RAG

This module implements comprehensive hallucination detection infrastructure including:
- Multi-level confidence calculation (graph-level, node-level, response-level)
- Post-generation verification framework
- Consistency checking mechanisms
- Confidence-based response filtering
- Ensemble verification for critical queries
- GPT-4o-mini verification integration
- Monitoring and alerting for low-confidence responses
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from llama_index.core import QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for response reliability."""
    VERY_LOW = "very_low"      # < 0.4
    LOW = "low"                # 0.4 - 0.6
    MEDIUM = "medium"          # 0.6 - 0.8
    HIGH = "high"              # 0.8 - 0.9
    VERY_HIGH = "very_high"    # > 0.9


class VerificationResult(Enum):
    """Result of verification process."""
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    ERROR = "error"


@dataclass
class NodeConfidence:
    """Confidence metrics for individual retrieved nodes."""
    node_id: str
    similarity_score: float
    semantic_coherence: float
    factual_consistency: float
    source_reliability: float
    overall_confidence: float = field(init=False)
    
    def __post_init__(self):
        """Calculate overall confidence from component scores."""
        weights = {
            'similarity': 0.3,
            'coherence': 0.25,
            'consistency': 0.25,
            'reliability': 0.2
        }
        self.overall_confidence = (
            weights['similarity'] * self.similarity_score +
            weights['coherence'] * self.semantic_coherence +
            weights['consistency'] * self.factual_consistency +
            weights['reliability'] * self.source_reliability
        )


@dataclass
class GraphConfidence:
    """Graph-level confidence for the entire retrieval result."""
    query_id: str
    node_confidences: List[NodeConfidence]
    cross_validation_score: float
    consensus_score: float
    coverage_score: float
    redundancy_penalty: float
    graph_confidence: float = field(init=False)
    
    def __post_init__(self):
        """Calculate graph-level confidence."""
        if not self.node_confidences:
            self.graph_confidence = 0.0
            return
            
        # Average node confidence weighted by position
        node_scores = []
        for i, node_conf in enumerate(self.node_confidences):
            # Higher weight for top-ranked nodes
            weight = 1.0 / (i + 1) ** 0.5
            node_scores.append(node_conf.overall_confidence * weight)
        
        avg_node_confidence = sum(node_scores) / sum(1.0 / (i + 1) ** 0.5 for i in range(len(node_scores)))
        
        # Combine with graph-level metrics
        self.graph_confidence = (
            0.4 * avg_node_confidence +
            0.2 * self.cross_validation_score +
            0.2 * self.consensus_score +
            0.15 * self.coverage_score +
            0.05 * (1.0 - self.redundancy_penalty)  # Penalty reduces confidence
        )


@dataclass
class ResponseConfidence:
    """Response-level confidence for generated answers."""
    response_id: str
    graph_confidence: GraphConfidence
    generation_consistency: float
    citation_accuracy: float
    hallucination_risk: float
    verification_score: float
    response_confidence: float = field(init=False)
    confidence_level: ConfidenceLevel = field(init=False)
    
    def __post_init__(self):
        """Calculate final response confidence and level."""
        # Calculate response confidence
        self.response_confidence = (
            0.3 * self.graph_confidence.graph_confidence +
            0.25 * self.generation_consistency +
            0.2 * self.citation_accuracy +
            0.15 * (1.0 - self.hallucination_risk) +
            0.1 * self.verification_score
        )
        
        # Determine confidence level
        if self.response_confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.response_confidence >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.response_confidence >= 0.6:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.response_confidence >= 0.4:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


@dataclass
class VerificationMetrics:
    """Metrics for monitoring verification system performance."""
    total_queries: int = 0
    verified_responses: int = 0
    rejected_responses: int = 0
    uncertain_responses: int = 0
    error_responses: int = 0
    average_confidence: float = 0.0
    average_verification_time: float = 0.0
    low_confidence_alerts: int = 0
    hallucination_detections: int = 0
    
    def update_metrics(self, confidence: ResponseConfidence, verification_time: float, result: VerificationResult):
        """Update metrics with new verification result."""
        self.total_queries += 1
        self.average_confidence = ((self.average_confidence * (self.total_queries - 1)) + 
                                  confidence.response_confidence) / self.total_queries
        self.average_verification_time = ((self.average_verification_time * (self.total_queries - 1)) + 
                                         verification_time) / self.total_queries
        
        if result == VerificationResult.VERIFIED:
            self.verified_responses += 1
        elif result == VerificationResult.REJECTED:
            self.rejected_responses += 1
            self.hallucination_detections += 1
        elif result == VerificationResult.UNCERTAIN:
            self.uncertain_responses += 1
        else:
            self.error_responses += 1
        
        if confidence.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            self.low_confidence_alerts += 1


class HallucinationDetector:
    """
    Comprehensive hallucination detection system with multi-level confidence calculation.
    
    This system provides:
    1. Node-level confidence scoring
    2. Graph-level confidence aggregation
    3. Response-level confidence calculation
    4. Post-generation verification with GPT-4o-mini
    5. Debate-augmented verification for critical queries
    6. Ensemble verification capabilities
    7. Citation accuracy verification
    8. Smart verification routing for cost optimization
    9. Verification result caching
    10. Confidence-based filtering
    """
    
    def __init__(
        self,
        verification_model: str = "gpt-4o-mini",
        verification_threshold: float = 0.8,
        ensemble_verification: bool = False,
        embedding_model: Optional[OpenAIEmbedding] = None,
        enable_debate_verification: bool = False,
        enable_verification_caching: bool = True,
        smart_routing_enabled: bool = True
    ):
        """Initialize the hallucination detector.
        
        Args:
            verification_model: Model for post-generation verification
            verification_threshold: Minimum confidence threshold
            ensemble_verification: Whether to use ensemble verification
            embedding_model: Embedding model for semantic similarity
            enable_debate_verification: Enable debate-augmented verification
            enable_verification_caching: Enable caching of verification results
            smart_routing_enabled: Enable smart routing to optimize costs
        """
        self.verification_model = verification_model
        self.verification_threshold = verification_threshold
        self.ensemble_verification = ensemble_verification
        self.enable_debate_verification = enable_debate_verification
        self.enable_verification_caching = enable_verification_caching
        self.smart_routing_enabled = smart_routing_enabled
        
        # Initialize verification LLMs
        self.verification_llm = OpenAI(
            model=verification_model,
            temperature=0.0,  # Deterministic for verification
            max_tokens=512
        )
        
        # Secondary LLM for ensemble/debate verification
        if ensemble_verification or enable_debate_verification:
            self.secondary_verification_llm = OpenAI(
                model=verification_model,
                temperature=0.1,  # Slight temperature for diversity
                max_tokens=512
            )
        else:
            self.secondary_verification_llm = None
        
        # Initialize embedding model for semantic analysis
        self.embedding_model = embedding_model or OpenAIEmbedding(
            model="text-embedding-3-small"  # Smaller model for efficiency
        )
        
        # Metrics tracking
        self.metrics = VerificationMetrics()
        
        # Caches for performance optimization
        self._embedding_cache: Dict[str, List[float]] = {}
        self._verification_cache: Dict[str, Tuple[VerificationResult, float, str]] = {}
        
        # Performance tracking
        self._verification_times: List[float] = []
        self._cost_tracker: Dict[str, float] = defaultdict(float)
        
        # Performance optimization settings
        self.max_verification_time = float(os.getenv("MAX_VERIFICATION_TIME", "2.0"))
        self.verification_timeout = float(os.getenv("VERIFICATION_TIMEOUT", "5.0"))
        
        # Batch processing for efficiency
        self._enable_batch_processing = os.getenv("VERIFICATION_BATCH_PROCESSING", "true").lower() == "true"
        self._batch_size = int(os.getenv("VERIFICATION_BATCH_SIZE", "5"))
        
        logger.info(f"Initialized HallucinationDetector with model: {verification_model}")
        logger.info(f"Features: Ensemble={ensemble_verification}, Debate={enable_debate_verification}, ")
        logger.info(f"Caching={enable_verification_caching}, Smart routing={smart_routing_enabled}")
    
    def calculate_node_confidence(
        self, 
        node: NodeWithScore, 
        query: QueryBundle,
        context_nodes: List[NodeWithScore]
    ) -> NodeConfidence:
        """Calculate confidence metrics for a single retrieved node."""
        node_id = node.node.node_id
        
        # 1. Similarity score (from retrieval)
        similarity_score = float(node.score) if node.score is not None else 0.0
        
        # 2. Semantic coherence with query
        semantic_coherence = self._calculate_semantic_coherence(node.node.text, query.query_str)
        
        # 3. Factual consistency with other nodes
        factual_consistency = self._calculate_factual_consistency(node, context_nodes)
        
        # 4. Source reliability (based on metadata)
        source_reliability = self._calculate_source_reliability(node.node.metadata)
        
        return NodeConfidence(
            node_id=node_id,
            similarity_score=min(1.0, max(0.0, similarity_score)),
            semantic_coherence=semantic_coherence,
            factual_consistency=factual_consistency,
            source_reliability=source_reliability
        )
    
    def calculate_graph_confidence(
        self,
        query: QueryBundle,
        nodes: List[NodeWithScore]
    ) -> GraphConfidence:
        """Calculate graph-level confidence for the entire retrieval result."""
        query_id = self._generate_query_id(query.query_str)
        
        # Calculate node confidences
        node_confidences = [
            self.calculate_node_confidence(node, query, nodes) 
            for node in nodes
        ]
        
        # Cross-validation score
        cross_validation_score = self._calculate_cross_validation_score(nodes)
        
        # Consensus score
        consensus_score = self._calculate_consensus_score(nodes)
        
        # Coverage score
        coverage_score = self._calculate_coverage_score(query.query_str, nodes)
        
        # Redundancy penalty
        redundancy_penalty = self._calculate_redundancy_penalty(nodes)
        
        return GraphConfidence(
            query_id=query_id,
            node_confidences=node_confidences,
            cross_validation_score=cross_validation_score,
            consensus_score=consensus_score,
            coverage_score=coverage_score,
            redundancy_penalty=redundancy_penalty
        )
    
    def calculate_response_confidence(
        self,
        response: str,
        graph_confidence: GraphConfidence,
        citations: List[str],
        query: QueryBundle
    ) -> ResponseConfidence:
        """Calculate response-level confidence for generated answers."""
        response_id = self._generate_response_id(response)
        
        # Generation consistency
        generation_consistency = self._calculate_generation_consistency(
            response, [nc.node_id for nc in graph_confidence.node_confidences]
        )
        
        # Citation accuracy
        citation_accuracy = self._calculate_citation_accuracy(response, citations)
        
        # Hallucination risk assessment
        hallucination_risk = self._assess_hallucination_risk(response, graph_confidence)
        
        # Initial verification score (will be updated after post-generation verification)
        verification_score = 0.8  # Placeholder
        
        return ResponseConfidence(
            response_id=response_id,
            graph_confidence=graph_confidence,
            generation_consistency=generation_consistency,
            citation_accuracy=citation_accuracy,
            hallucination_risk=hallucination_risk,
            verification_score=verification_score
        )
    
    async def verify_response(
        self,
        response: str,
        confidence: ResponseConfidence,
        query: QueryBundle,
        retrieved_nodes: List[NodeWithScore]
    ) -> Tuple[VerificationResult, float, Optional[str]]:
        """
        Advanced post-generation verification using GPT-4o-mini with multiple strategies.
        
        Returns:
            Tuple of (verification result, updated confidence score, explanation)
        """
        start_time = time.time()
        
        try:
            # Check if verification should be skipped based on smart routing
            if self._should_skip_verification(confidence, query.query_str, response):
                logger.debug("Skipping verification based on smart routing")
                return VerificationResult.VERIFIED, confidence.response_confidence, "Skipped - high confidence simple query"
            
            # Try to use IntelligentCacheManager for verification caching
            try:
                from src.performance import get_performance_optimizer
                optimizer = get_performance_optimizer()
                cache_manager = optimizer.cache_manager
                
                # Use intelligent cache manager for verification
                async def compute_verification():
                    return await self._perform_verification(
                        query, response, confidence, retrieved_nodes
                    )
                
                result, updated_confidence, explanation = await cache_manager.get_or_compute_verification(
                    query.query_str,
                    response,
                    compute_verification
                )
                
                verification_time = time.time() - start_time
                self._verification_times.append(verification_time)
                self.metrics.update_metrics(confidence, verification_time, result)
                self._update_cost_tracking(verification_time)
                
                return result, updated_confidence, explanation
                
            except ImportError:
                # Fallback to standard caching if performance module not available
                pass
            
            # Check verification cache first (fallback)
            cache_key = self._generate_verification_cache_key(query.query_str, response)
            if self.enable_verification_caching and cache_key in self._verification_cache:
                cached_result, cached_confidence, cached_explanation = self._verification_cache[cache_key]
                logger.debug(f"Verification cache HIT for key: {cache_key[:16]}...")
                return cached_result, cached_confidence, cached_explanation
            
            # Perform the actual verification
            result, updated_confidence, explanation = await self._perform_verification(
                query, response, confidence, retrieved_nodes
            )
            
            # Update confidence with verification score
            confidence.verification_score = updated_confidence
            confidence.response_confidence = (
                0.85 * confidence.response_confidence + 0.15 * updated_confidence
            )
            
            verification_time = time.time() - start_time
            self._verification_times.append(verification_time)
            
            # Cache the result if caching is enabled
            if self.enable_verification_caching:
                self._verification_cache[cache_key] = (result, updated_confidence, explanation)
                # Limit cache size
                if len(self._verification_cache) > 1000:
                    # Remove oldest entry
                    oldest_key = next(iter(self._verification_cache))
                    del self._verification_cache[oldest_key]
            
            # Update metrics and cost tracking
            self.metrics.update_metrics(confidence, verification_time, result)
            self._update_cost_tracking(verification_time)
            
            logger.debug(f"Verification completed in {verification_time:.3f}s: {result.value}")
            
            return result, confidence.response_confidence, explanation
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            verification_time = time.time() - start_time
            self.metrics.update_metrics(confidence, verification_time, VerificationResult.ERROR)
            return VerificationResult.ERROR, confidence.response_confidence, f"Error: {str(e)}"
    
    def should_filter_response(self, confidence: ResponseConfidence) -> bool:
        """Determine if response should be filtered based on confidence."""
        return (
            confidence.response_confidence < self.verification_threshold or
            confidence.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]
        )
    
    def get_confidence_explanation(self, confidence: ResponseConfidence) -> str:
        """Generate human-readable explanation of confidence scores."""
        explanation = []
        
        explanation.append(f"Overall Confidence: {confidence.response_confidence:.2f} ({confidence.confidence_level.value})")
        explanation.append(f"Graph Confidence: {confidence.graph_confidence.graph_confidence:.2f}")
        explanation.append(f"Generation Consistency: {confidence.generation_consistency:.2f}")
        explanation.append(f"Citation Accuracy: {confidence.citation_accuracy:.2f}")
        explanation.append(f"Hallucination Risk: {confidence.hallucination_risk:.2f}")
        explanation.append(f"Verification Score: {confidence.verification_score:.2f}")
        
        if confidence.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            explanation.append("⚠️  Low confidence - response may contain inaccuracies")
        elif confidence.confidence_level == ConfidenceLevel.VERY_HIGH:
            explanation.append("✅ High confidence - response is highly reliable")
        
        return "\n".join(explanation)
    
    # Private helper methods
    
    def _calculate_semantic_coherence(self, text: str, query: str) -> float:
        """Calculate semantic coherence between text and query."""
        try:
            text_embedding = self._get_embedding(text)
            query_embedding = self._get_embedding(query)
            
            # Cosine similarity
            similarity = np.dot(text_embedding, query_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(query_embedding)
            )
            
            return max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"Failed to calculate semantic coherence: {e}")
            return 0.5  # Default moderate score
    
    def _calculate_factual_consistency(self, node: NodeWithScore, context_nodes: List[NodeWithScore]) -> float:
        """Calculate factual consistency with other retrieved nodes."""
        if len(context_nodes) <= 1:
            return 0.7  # Default score for single node
        
        # Compare with other nodes
        similarities = []
        for other_node in context_nodes:
            if other_node.node.node_id != node.node.node_id:
                similarity = self._calculate_semantic_coherence(
                    node.node.text, other_node.node.text
                )
                similarities.append(similarity)
        
        if not similarities:
            return 0.7
        
        # Return average similarity with penalty for very low similarities
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        
        # Penalty if any similarity is very low (indicates contradiction)
        penalty = 1.0 if min_similarity > 0.3 else (min_similarity / 0.3)
        
        return avg_similarity * penalty
    
    def _calculate_source_reliability(self, metadata: Dict[str, Any]) -> float:
        """Calculate source reliability based on metadata."""
        reliability_score = 0.8  # Base score
        
        # Boost for certain file types
        if 'file_type' in metadata:
            file_type = metadata['file_type'].lower()
            if file_type in ['pdf', 'doc', 'docx']:
                reliability_score += 0.1
        
        # Boost for recent documents
        if 'last_modified' in metadata:
            # Implementation would check recency
            pass
        
        # Boost for authoritative sources
        if 'source' in metadata:
            source = metadata['source'].lower()
            if any(domain in source for domain in ['edu', 'gov', 'org']):
                reliability_score += 0.1
        
        return min(1.0, reliability_score)
    
    def _calculate_cross_validation_score(self, nodes: List[NodeWithScore]) -> float:
        """Calculate cross-validation score between nodes."""
        if len(nodes) <= 1:
            return 0.5
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self._calculate_semantic_coherence(
                    nodes[i].node.text, nodes[j].node.text
                )
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _calculate_consensus_score(self, nodes: List[NodeWithScore]) -> float:
        """Calculate consensus score based on agreement between nodes."""
        # Simplified implementation - would use more sophisticated NLP techniques
        return self._calculate_cross_validation_score(nodes)
    
    def _calculate_coverage_score(self, query: str, nodes: List[NodeWithScore]) -> float:
        """Calculate how well the retrieved nodes cover the query."""
        if not nodes:
            return 0.0
        
        # Simple implementation based on semantic similarity
        max_similarity = 0.0
        for node in nodes:
            similarity = self._calculate_semantic_coherence(node.node.text, query)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_redundancy_penalty(self, nodes: List[NodeWithScore]) -> float:
        """Calculate redundancy penalty for duplicate information."""
        if len(nodes) <= 1:
            return 0.0
        
        # Calculate similarity between consecutive nodes
        redundancy_scores = []
        for i in range(len(nodes) - 1):
            similarity = self._calculate_semantic_coherence(
                nodes[i].node.text, nodes[i + 1].node.text
            )
            redundancy_scores.append(similarity)
        
        # High similarity indicates redundancy
        avg_redundancy = sum(redundancy_scores) / len(redundancy_scores)
        return max(0.0, avg_redundancy - 0.7) if avg_redundancy > 0.7 else 0.0
    
    def _calculate_generation_consistency(self, response: str, node_ids: List[str]) -> float:
        """Calculate consistency between generated response and source nodes."""
        # Simplified implementation - would analyze factual consistency
        return 0.8  # Placeholder
    
    def _calculate_citation_accuracy(self, response: str, citations: List[str]) -> float:
        """Calculate accuracy of citations in the response."""
        # Count citation patterns in response
        citation_count = response.count('[citation:')
        
        if citation_count == 0 and not citations:
            return 1.0  # No citations needed or provided
        
        if citation_count == 0 and citations:
            return 0.0  # Citations available but not used
        
        if citation_count > 0 and not citations:
            return 0.5  # Citations used but none available
        
        # Calculate accuracy based on valid citations
        valid_citations = sum(1 for cit in citations if f'[citation:{cit}]' in response)
        return valid_citations / max(citation_count, len(citations))
    
    def _assess_hallucination_risk(self, response: str, graph_confidence: GraphConfidence) -> float:
        """Assess risk of hallucination in the response."""
        risk_factors = []
        
        # Low graph confidence increases risk
        risk_factors.append(1.0 - graph_confidence.graph_confidence)
        
        # Very long responses have higher risk
        response_length = len(response.split())
        if response_length > 500:
            risk_factors.append(0.3)
        elif response_length < 50:
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.0)
        
        # Specific phrases that indicate uncertainty
        uncertainty_phrases = ['i think', 'might be', 'possibly', 'not sure']
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response.lower())
        risk_factors.append(min(0.5, uncertainty_count * 0.1))
        
        return min(1.0, sum(risk_factors) / len(risk_factors))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        embedding = self.embedding_model.get_text_embedding(text)
        self._embedding_cache[text_hash] = embedding
        
        return embedding
    
    def _generate_query_id(self, query: str) -> str:
        """Generate unique ID for query."""
        return hashlib.md5(f"query_{query}".encode()).hexdigest()[:16]
    
    def _generate_response_id(self, response: str) -> str:
        """Generate unique ID for response."""
        return hashlib.md5(f"response_{response}".encode()).hexdigest()[:16]
    
    def _prepare_verification_context(self, nodes: List[NodeWithScore]) -> str:
        """Prepare context for verification prompt."""
        context_parts = []
        for i, node in enumerate(nodes[:3]):  # Use top 3 nodes
            context_parts.append(f"Source {i+1}: {node.node.text[:500]}")
        
        return "\n\n".join(context_parts)
    
    def _create_verification_prompt(
        self,
        query: str,
        response: str,
        context: str,
        confidence: ResponseConfidence,
        query_type: str = "general",
        citation_score: float = 0.8
    ) -> str:
        """Create optimized prompt for post-generation verification."""
        
        # Determine prompt strategy based on query type and confidence
        if query_type == "factual" or "fact" in query.lower():
            verification_focus = (
                "Focus heavily on factual accuracy. Every claim must be supported by the context. "
                "Pay special attention to numbers, dates, names, and specific details."
            )
        elif query_type == "comparative" or any(word in query.lower() for word in ["compare", "difference", "versus"]):
            verification_focus = (
                "Focus on the accuracy of comparisons and contrasts. Ensure all compared elements "
                "are properly represented and the relationships between them are correctly stated."
            )
        elif query_type == "procedural" or any(word in query.lower() for word in ["how to", "steps", "process"]):
            verification_focus = (
                "Focus on the accuracy and completeness of procedural information. Ensure steps "
                "are in the correct order and no critical steps are missing or incorrect."
            )
        else:
            verification_focus = (
                "Focus on overall accuracy and consistency with the provided context. "
                "Identify any claims that lack proper support from the sources."
            )
        
        confidence_context = ""
        if confidence.response_confidence < 0.6:
            confidence_context = (
                "\n**ALERT**: This response has LOW CONFIDENCE. Be extra vigilant for:"
                "\n- Fabricated information not present in context"
                "\n- Overgeneralization beyond what sources support"
                "\n- Missing or incorrect citations"
            )
        
        return f"""You are an expert fact-checker specializing in RAG system verification. Your task is to verify if a generated response is accurate and well-supported by the provided context sources.

**Query:** {query}

**Generated Response:** {response}

**Context Sources:**
{context}

**Current Assessment:**
- Overall Confidence: {confidence.response_confidence:.2f}
- Hallucination Risk: {confidence.hallucination_risk:.2f}
- Citation Accuracy: {citation_score:.2f}{confidence_context}

**Verification Instructions:**
{verification_focus}

**Evaluation Criteria:**
1. **Factual Accuracy**: Are all factual claims supported by the context?
2. **Citation Alignment**: Do citations properly correspond to their claims?
3. **Completeness**: Are any important details missing or misrepresented?
4. **Consistency**: Is the response internally consistent and aligned with sources?
5. **Fabrication Risk**: Are there any unsupported claims or potential hallucinations?

**Response Format:**
VERIFICATION_RESULT: [VERIFIED|REJECTED|UNCERTAIN]
CONFIDENCE_SCORE: [0.0-1.0]
EXPLANATION: [Specific explanation with examples of issues found or confirmation of accuracy]

**Guidelines:**
- VERIFIED: All claims are well-supported by context, citations are accurate
- REJECTED: Contains fabricated information, incorrect facts, or misleading claims
- UNCERTAIN: Some concerns but not definitively incorrect, needs human review

Be precise and provide specific examples in your explanation."""
    
    async def _get_verification_response(self, prompt: str, use_secondary: bool = False) -> str:
        """Get response from verification model with retry logic and timeout handling."""
        llm = self.secondary_verification_llm if use_secondary and self.secondary_verification_llm else self.verification_llm
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                response = await asyncio.wait_for(
                    llm.acomplete(prompt), 
                    timeout=self.verification_timeout
                )
                return response.text
            except asyncio.TimeoutError:
                logger.warning(f"Verification timeout (attempt {attempt + 1}) after {self.verification_timeout}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    return "VERIFICATION_RESULT: ERROR\nCONFIDENCE_SCORE: 0.5\nEXPLANATION: Verification timeout after multiple attempts"
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Verification API call failed (attempt {attempt + 1}): {e}. Retrying...")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Verification API call failed after {max_retries} attempts: {e}")
                    return "VERIFICATION_RESULT: ERROR\nCONFIDENCE_SCORE: 0.5\nEXPLANATION: Verification failed due to API error after multiple retries"
    
    def _parse_verification_response(
        self,
        verification_response: str,
        current_confidence: float
    ) -> Tuple[VerificationResult, float]:
        """Parse verification response from the model."""
        try:
            lines = verification_response.strip().split('\n')
            result_line = next((line for line in lines if line.startswith('VERIFICATION_RESULT:')), None)
            confidence_line = next((line for line in lines if line.startswith('CONFIDENCE_SCORE:')), None)
            
            if result_line:
                result_str = result_line.split(':', 1)[1].strip().upper()
                if result_str == 'VERIFIED':
                    result = VerificationResult.VERIFIED
                elif result_str == 'REJECTED':
                    result = VerificationResult.REJECTED
                elif result_str == 'UNCERTAIN':
                    result = VerificationResult.UNCERTAIN
                else:
                    result = VerificationResult.ERROR
            else:
                result = VerificationResult.ERROR
            
            if confidence_line:
                try:
                    confidence_score = float(confidence_line.split(':', 1)[1].strip())
                    confidence_score = max(0.0, min(1.0, confidence_score))
                except ValueError:
                    confidence_score = current_confidence
            else:
                confidence_score = current_confidence
            
            return result, confidence_score
            
        except Exception as e:
            logger.error(f"Failed to parse verification response: {e}")
            return VerificationResult.ERROR, current_confidence
    
    async def _standard_verification(
        self,
        query: str,
        response: str,
        context: str,
        confidence: ResponseConfidence,
        citation_score: float
    ) -> Tuple[VerificationResult, float, str]:
        """Standard single-pass verification using GPT-4o-mini."""
        query_type = self._classify_query_type(query)
        
        verification_prompt = self._create_verification_prompt(
            query, response, context, confidence, query_type, citation_score
        )
        
        verification_response = await self._get_verification_response(verification_prompt)
        result, updated_confidence = self._parse_verification_response(
            verification_response, confidence.response_confidence
        )
        
        explanation = self._extract_explanation_from_response(verification_response)
        return result, updated_confidence, explanation
    
    async def _ensemble_verification(
        self,
        query: str,
        response: str,
        context: str,
        confidence: ResponseConfidence,
        citation_score: float
    ) -> Tuple[VerificationResult, float, str]:
        """Ensemble verification using multiple verification approaches."""
        verifications = []
        explanations = []
        
        # Primary verification
        primary_result, primary_confidence, primary_explanation = await self._standard_verification(
            query, response, context, confidence, citation_score
        )
        verifications.append((primary_result, primary_confidence))
        explanations.append(f"Primary: {primary_explanation}")
        
        # Secondary verification with different prompt
        if self.secondary_verification_llm:
            secondary_prompt = self._create_secondary_verification_prompt(
                query, response, context, confidence
            )
            secondary_response = await self._get_verification_response(secondary_prompt, use_secondary=True)
            secondary_result, secondary_confidence = self._parse_verification_response(
                secondary_response, confidence.response_confidence
            )
            verifications.append((secondary_result, secondary_confidence))
            explanations.append(f"Secondary: {self._extract_explanation_from_response(secondary_response)}")
        
        # Citation-focused verification
        citation_result, citation_confidence = await self._citation_focused_verification(
            response, context, citation_score
        )
        verifications.append((citation_result, citation_confidence))
        explanations.append(f"Citation: Score={citation_score:.2f}")
        
        # Ensemble decision making
        final_result, final_confidence = self._make_ensemble_decision(verifications)
        final_explanation = " | ".join(explanations[:2])  # Limit explanation length
        
        return final_result, final_confidence, final_explanation
    
    async def _debate_augmented_verification(
        self,
        query: str,
        response: str,
        context: str,
        confidence: ResponseConfidence,
        citation_score: float
    ) -> Tuple[VerificationResult, float, str]:
        """Debate-augmented verification for critical/uncertain queries."""
        
        # First verification: Find potential issues
        critic_prompt = self._create_critic_verification_prompt(
            query, response, context, confidence
        )
        critic_response = await self._get_verification_response(critic_prompt)
        
        # Second verification: Defend against criticism
        defender_prompt = self._create_defender_verification_prompt(
            query, response, context, critic_response
        )
        defender_response = await self._get_verification_response(defender_prompt, use_secondary=True)
        
        # Final arbitration
        arbitration_prompt = self._create_arbitration_prompt(
            query, response, context, critic_response, defender_response
        )
        final_response = await self._get_verification_response(arbitration_prompt)
        
        result, updated_confidence = self._parse_verification_response(
            final_response, confidence.response_confidence
        )
        
        explanation = f"Debate: {self._extract_explanation_from_response(final_response)}"
        
        return result, updated_confidence, explanation
    
    async def _perform_verification(
        self,
        query: QueryBundle,
        response: str,
        confidence: ResponseConfidence,
        retrieved_nodes: List[NodeWithScore]
    ) -> Tuple[VerificationResult, float, str]:
        """
        Perform the actual verification logic.
        
        Args:
            query: Query bundle
            response: Response to verify
            confidence: Response confidence
            retrieved_nodes: Retrieved nodes for context
            
        Returns:
            Tuple of (result, confidence, explanation)
        """
        # Prepare context from retrieved nodes
        context = self._prepare_verification_context(retrieved_nodes)
        
        # Perform citation verification
        citation_verification_score = await self._verify_citations(response, retrieved_nodes)
        
        # Choose verification strategy based on confidence and query complexity
        if (self.enable_debate_verification and 
            (confidence.response_confidence < 0.7 or 
             confidence.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW])):
            
            result, updated_confidence, explanation = await self._debate_augmented_verification(
                query.query_str, response, context, confidence, citation_verification_score
            )
        elif self.ensemble_verification:
            result, updated_confidence, explanation = await self._ensemble_verification(
                query.query_str, response, context, confidence, citation_verification_score
            )
        else:
            result, updated_confidence, explanation = await self._standard_verification(
                query.query_str, response, context, confidence, citation_verification_score
            )
        
        return result, updated_confidence, explanation
    
    async def _verify_citations(self, response: str, retrieved_nodes: List[NodeWithScore]) -> float:
        """Verify citation accuracy against retrieved contexts."""
        if not retrieved_nodes:
            return 1.0  # No nodes to verify against
        
        # Extract citations from response
        import re
        citation_pattern = r'\[citation:([^\]]+)\]'
        cited_ids = re.findall(citation_pattern, response)
        
        if not cited_ids:
            # No citations in response
            return 0.8 if len(retrieved_nodes) == 0 else 0.3
        
        # Check if cited IDs match actual node IDs
        available_ids = {node.node.node_id for node in retrieved_nodes}
        valid_citations = sum(1 for cited_id in cited_ids if cited_id in available_ids)
        
        # Calculate accuracy
        citation_accuracy = valid_citations / len(cited_ids)
        
        # Bonus for using all available high-scoring nodes
        if citation_accuracy > 0.8:
            high_score_nodes = [node for node in retrieved_nodes[:3] if node.score and node.score > 0.8]
            if len(high_score_nodes) > 0:
                used_high_score = sum(1 for node in high_score_nodes if node.node.node_id in cited_ids)
                bonus = (used_high_score / len(high_score_nodes)) * 0.1
                citation_accuracy = min(1.0, citation_accuracy + bonus)
        
        return citation_accuracy
    
    async def _citation_focused_verification(
        self, response: str, context: str, citation_score: float
    ) -> Tuple[VerificationResult, float]:
        """Focus specifically on citation accuracy and alignment."""
        if citation_score >= 0.9:
            return VerificationResult.VERIFIED, 0.95
        elif citation_score >= 0.7:
            return VerificationResult.VERIFIED, citation_score
        elif citation_score >= 0.5:
            return VerificationResult.UNCERTAIN, citation_score
        else:
            return VerificationResult.REJECTED, citation_score
    
    def _make_ensemble_decision(self, verifications: List[Tuple[VerificationResult, float]]) -> Tuple[VerificationResult, float]:
        """Make final decision based on ensemble of verifications."""
        if not verifications:
            return VerificationResult.ERROR, 0.5
        
        # Count results
        result_counts = defaultdict(int)
        confidence_sum = 0.0
        
        for result, conf in verifications:
            result_counts[result] += 1
            confidence_sum += conf
        
        avg_confidence = confidence_sum / len(verifications)
        
        # Decision logic
        if result_counts[VerificationResult.VERIFIED] >= len(verifications) / 2:
            return VerificationResult.VERIFIED, min(0.95, avg_confidence)
        elif result_counts[VerificationResult.REJECTED] >= len(verifications) / 3:
            return VerificationResult.REJECTED, max(0.2, avg_confidence * 0.7)
        else:
            return VerificationResult.UNCERTAIN, avg_confidence
    
    def _should_skip_verification(self, confidence: ResponseConfidence, query: str, response: str) -> bool:
        """Determine if verification can be skipped based on smart routing."""
        if not self.smart_routing_enabled:
            return False
        
        # Skip for very high confidence simple queries
        if (confidence.response_confidence >= 0.9 and 
            confidence.confidence_level == ConfidenceLevel.VERY_HIGH and
            len(response.split()) < 100 and
            len(query.split()) < 10):
            return True
        
        # Skip for straightforward factual queries with high confidence
        simple_patterns = ['what is', 'who is', 'when was', 'where is']
        if (confidence.response_confidence >= 0.85 and
            any(pattern in query.lower() for pattern in simple_patterns) and
            '[citation:' in response):  # Has citations
            return True
        
        return False
    
    def _generate_verification_cache_key(self, query: str, response: str) -> str:
        """Generate cache key for verification results."""
        combined = f"{query}|||{response}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for optimized verification."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'who is', 'when', 'where', 'define']):
            return 'factual'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparative'
        elif any(word in query_lower for word in ['how to', 'steps', 'process', 'procedure']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'explain', 'analyze']):
            return 'analytical'
        else:
            return 'general'
    
    def _create_secondary_verification_prompt(self, query: str, response: str, context: str, confidence: ResponseConfidence) -> str:
        """Create alternative verification prompt for ensemble approach."""
        return f"""As a secondary fact-checker, independently evaluate this response for accuracy.

Query: {query}
Response: {response}
Context: {context}

Focus on:
1. Alternative interpretation of sources
2. Potential ambiguities or unclear statements
3. Completeness of the response

VERIFICATION_RESULT: [VERIFIED|REJECTED|UNCERTAIN]
CONFIDENCE_SCORE: [0.0-1.0]
EXPLANATION: [Your independent assessment]"""
    
    def _create_critic_verification_prompt(self, query: str, response: str, context: str, confidence: ResponseConfidence) -> str:
        """Create critic prompt for debate verification."""
        return f"""You are a critical fact-checker. Your job is to find potential problems with this response.

Query: {query}
Response: {response}
Context: {context}

Look for:
1. Unsupported claims
2. Misrepresented information
3. Missing context
4. Potential inaccuracies

CRITIQUE: [List specific concerns or state 'NO MAJOR ISSUES FOUND']
CONFIDENCE_SCORE: [0.0-1.0]"""
    
    def _create_defender_verification_prompt(self, query: str, response: str, context: str, critic_response: str) -> str:
        """Create defender prompt for debate verification."""
        return f"""You are defending this response against criticism. Address the concerns raised.

Query: {query}
Response: {response}
Context: {context}
Criticism: {critic_response}

For each concern raised:
1. Is the criticism valid?
2. Is the response actually supported by context?
3. Provide counter-arguments where appropriate

DEFENSE: [Address each criticism]
CONFIDENCE_SCORE: [0.0-1.0]"""
    
    def _create_arbitration_prompt(self, query: str, response: str, context: str, critic_response: str, defender_response: str) -> str:
        """Create final arbitration prompt for debate verification."""
        return f"""As a neutral arbitrator, make the final decision on this response's accuracy.

Query: {query}
Response: {response}
Context: {context}
Critic's concerns: {critic_response}
Defender's response: {defender_response}

Make your final judgment:
VERIFICATION_RESULT: [VERIFIED|REJECTED|UNCERTAIN]
CONFIDENCE_SCORE: [0.0-1.0]
EXPLANATION: [Final reasoned decision]"""
    
    def _extract_explanation_from_response(self, verification_response: str) -> str:
        """Extract explanation from verification response."""
        lines = verification_response.strip().split('\n')
        explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), None)
        if explanation_line:
            return explanation_line.split(':', 1)[1].strip()
        return "No explanation provided"
    
    def _update_cost_tracking(self, verification_time: float):
        """Update cost tracking metrics."""
        # Estimate cost based on model and tokens (rough approximation)
        estimated_tokens = 800  # Average tokens per verification
        cost_per_1k_tokens = 0.00015 if "gpt-4o-mini" in self.verification_model else 0.002
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens
        
        self._cost_tracker["total_cost"] += estimated_cost
        self._cost_tracker["total_verifications"] += 1
        
        if len(self._verification_times) > 100:
            # Keep only recent times for performance tracking
            self._verification_times = self._verification_times[-50:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self._verification_times:
            return {"error": "No verification data available"}
        
        return {
            "average_verification_time": sum(self._verification_times) / len(self._verification_times),
            "median_verification_time": sorted(self._verification_times)[len(self._verification_times) // 2],
            "max_verification_time": max(self._verification_times),
            "min_verification_time": min(self._verification_times),
            "total_verifications": len(self._verification_times),
            "estimated_total_cost": self._cost_tracker.get("total_cost", 0.0),
            "average_cost_per_verification": self._cost_tracker.get("total_cost", 0.0) / max(1, self._cost_tracker.get("total_verifications", 1)),
            "cache_hit_rate": len(self._verification_cache) / max(1, self._cost_tracker.get("total_verifications", 1)),
            "cache_size": len(self._verification_cache)
        }


def create_hallucination_detector() -> Optional[HallucinationDetector]:
    """Create a hallucination detector with configuration from environment variables."""
    
    verification_enabled = os.getenv("VERIFICATION_ENABLED", "true").lower() == "true"
    if not verification_enabled:
        logger.info("Verification disabled in configuration")
        return None
    
    verification_threshold = float(os.getenv("VERIFICATION_THRESHOLD", "0.8"))
    ensemble_verification = os.getenv("ENSEMBLE_VERIFICATION", "true").lower() == "true"
    verification_model = os.getenv("VERIFICATION_MODEL", "gpt-4o-mini")
    enable_debate_verification = os.getenv("DEBATE_AUGMENTATION_ENABLED", "false").lower() == "true"
    enable_verification_caching = os.getenv("VERIFICATION_CACHING_ENABLED", "true").lower() == "true"
    smart_routing_enabled = os.getenv("SMART_VERIFICATION_ROUTING", "true").lower() == "true"
    
    try:
        detector = HallucinationDetector(
            verification_model=verification_model,
            verification_threshold=verification_threshold,
            ensemble_verification=ensemble_verification,
            enable_debate_verification=enable_debate_verification,
            enable_verification_caching=enable_verification_caching,
            smart_routing_enabled=smart_routing_enabled
        )
        
        logger.info(f"Created HallucinationDetector with threshold: {verification_threshold}")
        logger.info(f"Features enabled: Ensemble={ensemble_verification}, Debate={enable_debate_verification}")
        return detector
        
    except Exception as e:
        logger.error(f"Failed to create HallucinationDetector: {e}")
        return None