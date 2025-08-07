"""
Integration layer for hallucination detection system with the RAG workflow.

This module provides seamless integration of the verification system with:
- Query engine tools
- Citation system
- Agentic workflow
- Response processing
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools.query_engine import QueryEngineTool

from src.verification import (
    HallucinationDetector,
    ResponseConfidence,
    VerificationResult,
    create_hallucination_detector
)
from src.settings import get_verification_config

logger = logging.getLogger(__name__)


class VerifiedQueryEngineTool:
    """
    Wrapper for QueryEngineTool that adds hallucination detection and verification.
    
    This class integrates the verification system with the existing query engine
    to provide confidence scoring and response filtering.
    """
    
    def __init__(
        self,
        query_engine_tool: QueryEngineTool,
        detector: Optional[HallucinationDetector] = None
    ):
        """Initialize verified query engine tool.
        
        Args:
            query_engine_tool: Original query engine tool to wrap
            detector: Hallucination detector instance
        """
        self.query_engine_tool = query_engine_tool
        self.detector = detector or create_hallucination_detector()
        self.verification_config = get_verification_config()
        
        # Store original metadata and update with verification info
        self.metadata = query_engine_tool.metadata
        if self.detector:
            self.metadata.description += (
                "\n\nThis tool includes advanced hallucination detection and "
                "multi-level confidence scoring. Responses are verified for "
                "accuracy and reliability before being returned."
            )
        
        logger.info("Initialized VerifiedQueryEngineTool with hallucination detection")
    
    async def acall(self, query: str, **kwargs) -> str:
        """
        Asynchronously call the query engine with verification.
        
        Args:
            query: Query string
            **kwargs: Additional arguments
            
        Returns:
            Verified response string with confidence information
        """
        if not self.detector:
            # Fallback to original tool if verification is disabled
            return await self.query_engine_tool.acall(query, **kwargs)
        
        start_time = time.time()
        
        try:
            # Step 1: Execute original query
            logger.debug(f"Executing query with verification: {query}")
            response = await self.query_engine_tool.acall(query, **kwargs)
            
            # Step 2: Get retrieval information for verification
            query_bundle = QueryBundle(query_str=query)
            retrieved_nodes = await self._get_retrieved_nodes(query_bundle)
            
            # Step 3: Calculate multi-level confidence
            confidence = await self._calculate_confidence(
                query_bundle, retrieved_nodes, response
            )
            
            # Step 4: Post-generation verification with explanation
            verification_result, updated_confidence, verification_explanation = await self.detector.verify_response(
                response, confidence, query_bundle, retrieved_nodes
            )
            
            # Step 5: Apply confidence-based filtering
            if self.detector.should_filter_response(confidence):
                filtered_response = await self._handle_low_confidence_response(
                    response, confidence, verification_result, verification_explanation
                )
                return filtered_response
            
            # Step 6: Add confidence information to response
            verified_response = self._add_confidence_info(
                response, confidence, verification_result, verification_explanation
            )
            
            total_time = time.time() - start_time
            logger.info(f"Query verified in {total_time:.2f}s with confidence: {confidence.response_confidence:.2f}")
            
            return verified_response
            
        except Exception as e:
            logger.error(f"Verification failed for query: {query}. Error: {str(e)}")
            # Fallback to original response without verification
            return await self.query_engine_tool.acall(query, **kwargs)
    
    def call(self, query: str, **kwargs) -> str:
        """
        Synchronous call wrapper.
        
        Args:
            query: Query string
            **kwargs: Additional arguments
            
        Returns:
            Verified response string
        """
        # For synchronous calls, we'll use asyncio.run
        return asyncio.run(self.acall(query, **kwargs))
    
    async def _get_retrieved_nodes(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Extract retrieved nodes from the query engine."""
        try:
            # Access the underlying query engine
            query_engine = self.query_engine_tool.query_engine
            
            # Get retriever if available
            if hasattr(query_engine, '_retriever'):
                retriever = query_engine._retriever
                nodes = await retriever.aretrieve(query_bundle)
                return nodes
            elif hasattr(query_engine, 'retriever'):
                retriever = query_engine.retriever
                nodes = await retriever.aretrieve(query_bundle)
                return nodes
            else:
                logger.warning("Could not access retriever for confidence calculation")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get retrieved nodes: {str(e)}")
            return []
    
    async def _calculate_confidence(
        self,
        query_bundle: QueryBundle,
        retrieved_nodes: List[NodeWithScore],
        response: str
    ) -> ResponseConfidence:
        """Calculate multi-level confidence for the response."""
        if not retrieved_nodes:
            # Create minimal confidence for responses without retrieval info
            from src.verification import GraphConfidence, NodeConfidence
            
            graph_confidence = GraphConfidence(
                query_id=self.detector._generate_query_id(query_bundle.query_str),
                node_confidences=[],
                cross_validation_score=0.5,
                consensus_score=0.5,
                coverage_score=0.5,
                redundancy_penalty=0.0
            )
        else:
            # Calculate full graph confidence
            graph_confidence = self.detector.calculate_graph_confidence(
                query_bundle, retrieved_nodes
            )
        
        # Extract citations from response
        citations = self._extract_citations(response)
        
        # Calculate response confidence
        response_confidence = self.detector.calculate_response_confidence(
            response, graph_confidence, citations, query_bundle
        )
        
        return response_confidence
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract citation IDs from response text."""
        citation_pattern = r'\[citation:([^\]]+)\]'
        citations = re.findall(citation_pattern, response)
        return citations
    
    async def _handle_low_confidence_response(
        self,
        response: str,
        confidence: ResponseConfidence,
        verification_result: VerificationResult,
        verification_explanation: Optional[str] = None
    ) -> str:
        """Handle responses with low confidence."""
        confidence_explanation = self.detector.get_confidence_explanation(confidence)
        
        if verification_result == VerificationResult.REJECTED:
            # Response was rejected by verification
            filtered_response = (
                "I apologize, but I cannot provide a reliable answer to your question "
                "based on the available information. The generated response failed "
                "verification checks for accuracy and reliability.\n\n"
                "Please try rephrasing your question or asking for more specific information.\n\n"
                f"**Confidence Analysis:**\n{confidence_explanation}"
            )
            if verification_explanation:
                filtered_response += f"\n\n**Verification Details:**\n{verification_explanation}"
        elif confidence.confidence_level.value in ['very_low', 'low']:
            # Low confidence warning
            filtered_response = (
                f"{response}\n\n"
                "**⚠️ Confidence Warning:**\n"
                "This response has low confidence and may contain inaccuracies. "
                "Please verify the information independently.\n\n"
                f"**Confidence Analysis:**\n{confidence_explanation}"
            )
            if verification_explanation:
                filtered_response += f"\n\n**Verification Notes:**\n{verification_explanation}"
        else:
            # Medium confidence with explanation
            filtered_response = (
                f"{response}\n\n"
                f"**Confidence Analysis:**\n{confidence_explanation}"
            )
        
        return filtered_response
    
    def _add_confidence_info(
        self,
        response: str,
        confidence: ResponseConfidence,
        verification_result: VerificationResult,
        verification_explanation: Optional[str] = None
    ) -> str:
        """Add confidence information to high-confidence responses."""
        
        # Only add detailed confidence info in debug mode or for medium confidence
        show_confidence = (
            logger.isEnabledFor(logging.DEBUG) or
            confidence.confidence_level.value == 'medium'
        )
        
        if verification_result == VerificationResult.VERIFIED and show_confidence:
            confidence_info = (
                f"\n\n**Response Verification:**\n"
                f"✅ Verified (Confidence: {confidence.response_confidence:.2f})"
            )
            if verification_explanation and logger.isEnabledFor(logging.DEBUG):
                confidence_info += f"\n**Verification Details:** {verification_explanation}"
            return response + confidence_info
        
        return response


def create_verified_query_tool(query_engine_tool: QueryEngineTool) -> QueryEngineTool:
    """
    Create a verified query engine tool with hallucination detection.
    
    Args:
        query_engine_tool: Original query engine tool
        
    Returns:
        Enhanced query engine tool with verification capabilities
    """
    verification_config = get_verification_config()
    
    if not verification_config.get('verification_enabled', True):
        logger.info("Verification disabled, returning original tool")
        return query_engine_tool
    
    # Create hallucination detector
    detector = create_hallucination_detector()
    
    if detector is None:
        logger.warning("Failed to create hallucination detector, returning original tool")
        return query_engine_tool
    
    # Wrap the original tool
    verified_tool = VerifiedQueryEngineTool(query_engine_tool, detector)
    
    # Create a new QueryEngineTool that uses the verified wrapper
    class VerifiedTool:
        def __init__(self, verified_wrapper):
            self.verified_wrapper = verified_wrapper
            self.metadata = verified_wrapper.metadata
            self.query_engine = verified_wrapper.query_engine_tool.query_engine
        
        async def acall(self, query: str, **kwargs) -> str:
            return await self.verified_wrapper.acall(query, **kwargs)
        
        def call(self, query: str, **kwargs) -> str:
            return self.verified_wrapper.call(query, **kwargs)
    
    return VerifiedTool(verified_tool)


class VerificationReporter:
    """Enhanced reporter for verification metrics and monitoring."""
    
    def __init__(self, detector: HallucinationDetector):
        """Initialize reporter with detector instance."""
        self.detector = detector
        self.config = get_verification_config()
        self.start_time = time.time()
        
        # Additional metrics tracking
        self._query_type_stats = defaultdict(int)
        self._verification_method_stats = defaultdict(int)
        self._hourly_stats = defaultdict(lambda: defaultdict(int))
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report with enhanced details."""
        metrics = self.detector.metrics
        performance_stats = self.detector.get_performance_stats()
        
        # Calculate uptime
        uptime_hours = (time.time() - self.start_time) / 3600
        
        report = {
            "summary": {
                "total_queries": metrics.total_queries,
                "verification_rate": metrics.verified_responses / max(1, metrics.total_queries),
                "rejection_rate": metrics.rejected_responses / max(1, metrics.total_queries),
                "uncertain_rate": metrics.uncertain_responses / max(1, metrics.total_queries),
                "error_rate": metrics.error_responses / max(1, metrics.total_queries),
                "average_confidence": metrics.average_confidence,
                "average_verification_time": metrics.average_verification_time,
                "uptime_hours": uptime_hours,
                "queries_per_hour": metrics.total_queries / max(0.1, uptime_hours),
            },
            "quality_metrics": {
                "verified_responses": metrics.verified_responses,
                "rejected_responses": metrics.rejected_responses,
                "uncertain_responses": metrics.uncertain_responses,
                "error_responses": metrics.error_responses,
                "hallucination_detections": metrics.hallucination_detections,
                "low_confidence_alerts": metrics.low_confidence_alerts,
                "quality_score": self._calculate_quality_score(metrics),
            },
            "performance_metrics": {
                "average_verification_time_ms": metrics.average_verification_time * 1000,
                "median_verification_time_ms": performance_stats.get("median_verification_time", 0) * 1000,
                "max_verification_time_ms": performance_stats.get("max_verification_time", 0) * 1000,
                "min_verification_time_ms": performance_stats.get("min_verification_time", 0) * 1000,
                "cache_hit_rate": performance_stats.get("cache_hit_rate", 0),
                "cache_size": performance_stats.get("cache_size", 0),
                "verification_enabled": self.config.get('verification_enabled', True),
                "verification_threshold": self.config.get('verification_threshold', 0.8),
            },
            "cost_metrics": {
                "estimated_total_cost_usd": performance_stats.get("estimated_total_cost", 0.0),
                "average_cost_per_verification_usd": performance_stats.get("average_cost_per_verification", 0.0),
                "estimated_monthly_cost_usd": self._estimate_monthly_cost(performance_stats, uptime_hours),
                "cost_savings_from_caching_usd": self._estimate_cache_savings(performance_stats),
            },
            "feature_usage": {
                "ensemble_verification_enabled": self.config.get('ensemble_verification', False),
                "debate_verification_enabled": self.config.get('debate_augmentation_enabled', False),
                "smart_routing_enabled": self.detector.smart_routing_enabled,
                "verification_caching_enabled": self.detector.enable_verification_caching,
            },
            "configuration": self.config
        }
        
        return report
    
    def _calculate_quality_score(self, metrics: Any) -> float:
        """Calculate overall quality score based on verification metrics."""
        if metrics.total_queries == 0:
            return 0.0
        
        # Base score from verification success rate
        verification_rate = metrics.verified_responses / metrics.total_queries
        quality_score = verification_rate * 0.7
        
        # Penalty for high rejection rate (indicates poor base quality)
        rejection_rate = metrics.rejected_responses / metrics.total_queries
        quality_score -= rejection_rate * 0.3
        
        # Bonus for high average confidence
        quality_score += (metrics.average_confidence - 0.5) * 0.3
        
        # Penalty for errors
        error_rate = metrics.error_responses / metrics.total_queries
        quality_score -= error_rate * 0.5
        
        return max(0.0, min(1.0, quality_score))
    
    def _estimate_monthly_cost(self, performance_stats: Dict[str, Any], uptime_hours: float) -> float:
        """Estimate monthly cost based on current usage patterns."""
        if uptime_hours < 0.1:
            return 0.0
        
        avg_cost_per_verification = performance_stats.get("average_cost_per_verification", 0.0)
        total_verifications = performance_stats.get("total_verifications", 0)
        
        verifications_per_hour = total_verifications / uptime_hours
        monthly_verifications = verifications_per_hour * 24 * 30
        
        return monthly_verifications * avg_cost_per_verification
    
    def _estimate_cache_savings(self, performance_stats: Dict[str, Any]) -> float:
        """Estimate cost savings from verification caching."""
        cache_hit_rate = performance_stats.get("cache_hit_rate", 0)
        avg_cost_per_verification = performance_stats.get("average_cost_per_verification", 0.0)
        total_verifications = performance_stats.get("total_verifications", 0)
        
        cache_hits = total_verifications * cache_hit_rate
        return cache_hits * avg_cost_per_verification
    
    def check_alerts(self) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        metrics = self.detector.metrics
        
        if metrics.total_queries > 0:
            rejection_rate = metrics.rejected_responses / metrics.total_queries
            if rejection_rate > 0.1:  # More than 10% rejection rate
                alerts.append(f"High rejection rate: {rejection_rate:.1%}")
            
            if metrics.average_confidence < 0.6:
                alerts.append(f"Low average confidence: {metrics.average_confidence:.2f}")
            
            if metrics.average_verification_time > 2.0:
                alerts.append(f"Slow verification: {metrics.average_verification_time:.2f}s average")
        
        return alerts
    
    def log_metrics_summary(self):
        """Log a summary of verification metrics."""
        report = self.generate_metrics_report()
        summary = report['summary']
        
        logger.info("Verification Metrics Summary:")
        logger.info(f"  Total queries: {summary['total_queries']}")
        logger.info(f"  Verification rate: {summary['verification_rate']:.1%}")
        logger.info(f"  Average confidence: {summary['average_confidence']:.2f}")
        logger.info(f"  Average verification time: {summary['average_verification_time']:.2f}s")
        
        alerts = self.check_alerts()
        if alerts:
            logger.warning(f"Verification alerts: {', '.join(alerts)}")


# Global reporter instance for monitoring
_global_reporter: Optional[VerificationReporter] = None


def get_verification_reporter() -> Optional[VerificationReporter]:
    """Get global verification reporter instance."""
    global _global_reporter
    return _global_reporter


def initialize_verification_monitoring(detector: HallucinationDetector):
    """Initialize global verification monitoring."""
    global _global_reporter
    _global_reporter = VerificationReporter(detector)
    logger.info("Initialized verification monitoring")


def log_verification_summary():
    """Log verification summary if monitoring is enabled."""
    reporter = get_verification_reporter()
    if reporter:
        reporter.log_metrics_summary()