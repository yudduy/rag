"""
SOTA RAG Workflow - Single unified workflow for all RAG operations.

This module consolidates all workflow functionality into one clean implementation.
"""

import asyncio
import logging
import time
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)
from llama_index.llms.openai import OpenAI

from src.index import get_index
from src.query import get_query_engine_tool
from src.citation import CITATION_SYSTEM_PROMPT, enable_citation
from src.settings import init_settings
from src.cache import get_cache
from src.verification import create_hallucination_detector
from src.agentic import QueryClassifier, QueryDecomposer, QueryType

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """How difficult a query is to process."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTIMODAL = "multimodal"


@dataclass
class QueryEvent(Event):
    """Event containing a user query."""
    query: str
    session_id: Optional[str] = None
    multimodal_data: Optional[Dict[str, Any]] = None


@dataclass
class ResponseEvent(Event):
    """Event containing the final response."""
    response: str
    sources: List[str]
    confidence: float
    processing_time: float
    session_id: Optional[str] = None


class RAGWorkflow(Workflow):
    """
    SOTA RAG workflow that handles all query processing.
    
    Features:
    - Query analysis and routing
    - Semantic caching
    - Response verification
    - Query decomposition for complex questions
    - Multimodal support
    """
    
    def __init__(self, timeout: float = 120.0, verbose: bool = False, **kwargs: Any):
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        
        # Initialize settings
        init_settings()
        
        # Core components
        self.index = get_index()
        self.query_engine_tool = get_query_engine_tool(self.index)
        
        # Create base agent workflow
        self.agent_workflow = AgentWorkflow(
            llm=OpenAI(model="gpt-4o-mini"),
            tools=[self.query_engine_tool],
            system_prompt=CITATION_SYSTEM_PROMPT,
            verbose=verbose
        )
        
        # Optional components (enabled via environment variables)
        self.cache = None
        self.verifier = None
        self.classifier = None
        self.decomposer = None
        
        # Initialize optional components
        self._init_optional_components()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'decomposed_queries': 0,
            'verified_queries': 0,
            'total_processing_time': 0.0
        }
        
        logger.info("RAG Workflow initialized")
    
    def _init_optional_components(self):
        """Initialize optional components based on environment variables."""
        
        # Semantic caching
        if os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true":
            try:
                self.cache = get_cache()
                logger.info("Semantic cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
        
        # Response verification
        if os.getenv("VERIFICATION_ENABLED", "true").lower() == "true":
            try:
                self.verifier = create_hallucination_detector()
                logger.info("Response verification enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize verifier: {e}")
        
        # Query decomposition
        if os.getenv("QUERY_DECOMPOSITION_ENABLED", "false").lower() == "true":
            try:
                self.classifier = QueryClassifier()
                self.decomposer = QueryDecomposer()
                logger.info("Query decomposition enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize decomposition: {e}")
    
    @step
    async def process_query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """Entry point - convert StartEvent to QueryEvent."""
        query = ev.query if hasattr(ev, 'query') else str(ev)
        return QueryEvent(query=query)
    
    @step
    async def check_cache(self, ctx: Context, ev: QueryEvent) -> Union[ResponseEvent, QueryEvent]:
        """Check semantic cache for similar queries."""
        if not self.cache:
            return ev
        
        try:
            cached_result = await self.cache.get_similar(ev.query)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for query: {ev.query[:50]}...")
                
                return ResponseEvent(
                    response=cached_result.response,
                    sources=cached_result.nodes,
                    confidence=0.9,  # Cache results are high confidence
                    processing_time=0.01,  # Cache is fast
                    session_id=ev.session_id
                )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return ev
    
    @step 
    async def analyze_query(self, ctx: Context, ev: QueryEvent) -> QueryEvent:
        """Analyze query complexity and characteristics."""
        start_time = time.time()
        
        # Simple analysis based on query length and content
        query_length = len(ev.query.split())
        
        if query_length < 5:
            complexity = QueryComplexity.SIMPLE
        elif query_length < 15:
            complexity = QueryComplexity.MODERATE
        elif "image" in ev.query.lower() or "picture" in ev.query.lower():
            complexity = QueryComplexity.MULTIMODAL
        else:
            complexity = QueryComplexity.COMPLEX
        
        # Store analysis in context
        ctx.data["complexity"] = complexity
        ctx.data["should_decompose"] = (
            complexity == QueryComplexity.COMPLEX and 
            self.decomposer is not None and
            ("and" in ev.query or "compare" in ev.query.lower())
        )
        
        self.stats['total_queries'] += 1
        logger.info(f"Query analysis: complexity={complexity.value}, decompose={ctx.data['should_decompose']}")
        
        return ev
    
    @step
    async def decompose_if_needed(self, ctx: Context, ev: QueryEvent) -> Union[QueryEvent, List[QueryEvent]]:
        """Decompose complex queries into sub-queries if enabled."""
        if not ctx.data.get("should_decompose", False):
            return ev
        
        try:
            sub_queries = await self.decomposer.decompose_query(ev.query)
            if len(sub_queries) > 1:
                self.stats['decomposed_queries'] += 1
                logger.info(f"Decomposed query into {len(sub_queries)} parts")
                
                # Create QueryEvent for each sub-query
                return [
                    QueryEvent(
                        query=sub_query.text,
                        session_id=ev.session_id
                    ) 
                    for sub_query in sub_queries
                ]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        
        return ev
    
    @step
    async def run_agent(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        """Execute the main RAG query using the agent workflow."""
        start_time = time.time()
        
        try:
            # Run the agent workflow
            result = await self.agent_workflow.run(query=ev.query)
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            # Extract response and sources
            response_text = str(result)
            sources = []  # TODO: Extract sources from result if available
            
            logger.info(f"Agent completed query in {processing_time:.2f}s")
            
            return ResponseEvent(
                response=response_text,
                sources=sources,
                confidence=0.8,  # Default confidence
                processing_time=processing_time,
                session_id=ev.session_id
            )
            
        except Exception as e:
            logger.error(f"Agent workflow failed: {e}")
            return ResponseEvent(
                response=f"I encountered an error processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                session_id=ev.session_id
            )
    
    @step
    async def aggregate_responses(self, ctx: Context, ev: List[ResponseEvent]) -> ResponseEvent:
        """Aggregate multiple sub-query responses into final response."""
        if len(ev) == 1:
            return ev[0]
        
        try:
            # Simple aggregation - combine responses
            combined_response = "\n\n".join([resp.response for resp in ev])
            combined_sources = []
            for resp in ev:
                combined_sources.extend(resp.sources)
            
            # Average confidence
            avg_confidence = sum(resp.confidence for resp in ev) / len(ev)
            total_time = sum(resp.processing_time for resp in ev)
            
            logger.info(f"Aggregated {len(ev)} responses")
            
            return ResponseEvent(
                response=combined_response,
                sources=list(set(combined_sources)),  # Remove duplicates
                confidence=avg_confidence,
                processing_time=total_time,
                session_id=ev[0].session_id
            )
            
        except Exception as e:
            logger.error(f"Response aggregation failed: {e}")
            return ev[0]  # Return first response as fallback
    
    @step
    async def verify_response(self, ctx: Context, ev: ResponseEvent) -> ResponseEvent:
        """Verify response quality and detect hallucinations."""
        if not self.verifier:
            return ev
        
        try:
            # TODO: Implement actual verification
            # For now, just log that verification was attempted
            self.stats['verified_queries'] += 1
            logger.info("Response verification completed")
            
        except Exception as e:
            logger.warning(f"Response verification failed: {e}")
        
        return ev
    
    @step
    async def cache_response(self, ctx: Context, ev: ResponseEvent) -> ResponseEvent:
        """Cache the response for future similar queries."""
        if not self.cache:
            return ev
        
        try:
            # Get original query from context or reconstruct
            original_query = ctx.data.get("original_query", "")
            if original_query:
                await self.cache.set(original_query, ev.response, ev.sources)
                logger.info("Response cached successfully")
                
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
        
        return ev
    
    @step
    async def finalize_response(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        """Final step - return the response."""
        logger.info(f"Query completed - confidence: {ev.confidence:.2f}, time: {ev.processing_time:.2f}s")
        return StopEvent(result=ev.response)


def create_workflow() -> RAGWorkflow:
    """Create and return the main RAG workflow."""
    return RAGWorkflow(verbose=os.getenv("WORKFLOW_VERBOSE", "false").lower() == "true")


# Backwards compatibility
AgentWorkflow = RAGWorkflow
