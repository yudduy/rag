"""
Enhanced agentic workflow extending the existing AgentWorkflow with SOTA RAG capabilities.

This module provides an intelligent workflow that uses query routing, decomposition,
and parallel execution to optimize retrieval performance and accuracy.
"""

import os
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)
from llama_index.core.tools import ToolSelection
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

from src.agentic import (
    QueryClassifier,
    QueryDecomposer,
    QueryClassification,
    SubQuery,
    QueryType
)
from src.settings import get_agentic_config
from src.cache import estimate_query_cost, get_cache
from src.verification import create_hallucination_detector

logger = logging.getLogger(__name__)


@dataclass
class QueryEvent(Event):
    """Event containing a user query to process."""
    query: str
    session_id: Optional[str] = None


@dataclass
class ClassificationEvent(Event):
    """Event containing query classification results."""
    query: str
    classification: QueryClassification
    session_id: Optional[str] = None


@dataclass
class DecompositionEvent(Event):
    """Event containing query decomposition results."""
    query: str
    classification: QueryClassification
    subqueries: List[SubQuery]
    session_id: Optional[str] = None


@dataclass
class ExecutionEvent(Event):
    """Event containing execution results."""
    original_query: str
    subquery_results: Dict[int, str]  # priority -> result
    subqueries: Optional[List[SubQuery]] = None  # Original subqueries for context
    session_id: Optional[str] = None


@dataclass
class AggregationEvent(Event):
    """Event for final result aggregation."""
    original_query: str
    subquery_results: Dict[int, str]
    final_response: str
    session_id: Optional[str] = None


class AgenticWorkflow(Workflow):
    """
    Enhanced agentic workflow with intelligent query processing.
    
    This workflow provides:
    1. Dynamic query routing based on query type and complexity
    2. Intelligent query decomposition for complex multi-part questions  
    3. Parallel execution of sub-queries with dependency management
    4. Smart result aggregation with citation preservation
    5. Event-driven async architecture for optimal performance
    """
    
    def __init__(
        self,
        agent_workflow: AgentWorkflow,
        timeout: float = 120.0,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        
        self.agent_workflow = agent_workflow
        self.config = get_agentic_config()
        
        # Initialize agentic components
        self.classifier = QueryClassifier(self.config.get("routing_model"))
        self.decomposer = QueryDecomposer(self.config.get("decomposition_model"))
        
        # Aggregation LLM for combining results
        self.aggregation_llm = OpenAI(
            model=self.config.get("subquery_aggregation_model", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=2048,
        )
        
        # Cost monitoring and controls
        self.cost_metrics = {
            'total_queries': 0,
            'total_estimated_cost': 0.0,
            'decomposed_queries': 0,
            'decomposed_query_cost': 0.0,
            'aggregation_cost': 0.0,
            'cost_savings_from_optimization': 0.0
        }
        self.max_query_cost = float(os.getenv("MAX_QUERY_COST", "2.00"))
        self.cost_monitoring_enabled = os.getenv("COST_MONITORING_ENABLED", "true").lower() == "true"
        
        # Performance monitoring
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'classification_time': 0.0,
            'decomposition_time': 0.0,
            'execution_time': 0.0,
            'aggregation_time': 0.0,
            'total_processing_time': 0.0,
            'decomposition_success_rate': 0.0,
            'parallel_execution_count': 0,
            'sequential_execution_count': 0,
            'cache_hits': 0,
            'fallback_activations': 0,
            'error_recovery_attempts': 0,
            'error_recovery_successes': 0
        }
        self.performance_monitoring_enabled = os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true"
        
        # Integration with verification and caching systems
        self.hallucination_detector = create_hallucination_detector()
        self.global_cache = get_cache() if os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true" else None
        self.verification_enabled = os.getenv("AGENTIC_VERIFICATION_ENABLED", "true").lower() == "true"
        self.cache_enabled = self.global_cache is not None
        
        logger.info(f"Initialized AgenticWorkflow with enhanced capabilities, cost monitoring, verification: {self.verification_enabled}, caching: {self.cache_enabled}")
    
    @step
    async def classify_query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        Entry point: Extract query and emit QueryEvent.
        """
        start_time = time.time()
        try:
            if self.performance_monitoring_enabled:
                self.performance_metrics['total_queries'] += 1
            # Extract query from various possible input formats
            query = None
            session_id = None
            
            if hasattr(ev, 'query'):
                query = ev.query
            elif hasattr(ev, 'input'):
                query = ev.input
            elif hasattr(ev, 'message'):
                query = ev.message
            else:
                # Try to get query from the first argument
                if hasattr(ev, 'dict') and ev.dict():
                    query = list(ev.dict().values())[0]
            
            if not query or not isinstance(query, str):
                raise ValueError(f"Could not extract query from StartEvent: {ev}")
            
            # Extract session ID if available
            if hasattr(ev, 'session_id'):
                session_id = ev.session_id
            
            logger.info(f"Processing query: {query[:100]}...")
            
            if self.performance_monitoring_enabled:
                classification_time = time.time() - start_time
                self.performance_metrics['classification_time'] += classification_time
            
            return QueryEvent(query=query, session_id=session_id)
            
        except Exception as e:
            logger.error(f"Failed to extract query: {e}")
            # Return a fallback query event
            return QueryEvent(query="", session_id=None)
    
    @step
    async def analyze_query(self, ctx: Context, ev: QueryEvent) -> ClassificationEvent:
        """
        Analyze and classify the query for routing decisions.
        """
        start_time = time.time()
        try:
            if not ev.query.strip():
                raise ValueError("Empty query received")
            
            # Check if agentic routing is enabled
            if not self.config.get("agent_routing_enabled", True):
                # Skip classification, use default
                classification = QueryClassification(
                    query_type=QueryType.SEMANTIC,
                    confidence=1.0,
                    complexity_score=0.5,
                    keywords=ev.query.split()[:5],
                    requires_decomposition=False,
                    estimated_chunks_needed=5
                )
            else:
                # Perform intelligent classification
                classification = await self.classifier.classify_query(ev.query)
            
            logger.info(
                f"Query classified as {classification.query_type.value} "
                f"(confidence: {classification.confidence:.2f}, "
                f"complexity: {classification.complexity_score:.2f})"
            )
            
            if self.performance_monitoring_enabled:
                analysis_time = time.time() - start_time
                self.performance_metrics['classification_time'] += analysis_time
            
            return ClassificationEvent(
                query=ev.query,
                classification=classification,
                session_id=ev.session_id
            )
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback classification
            fallback_classification = QueryClassification(
                query_type=QueryType.SEMANTIC,
                confidence=0.5,
                complexity_score=0.5,
                keywords=ev.query.split()[:3] if ev.query else [],
                requires_decomposition=False,
                estimated_chunks_needed=5
            )
            return ClassificationEvent(
                query=ev.query,
                classification=fallback_classification,
                session_id=ev.session_id
            )
    
    @step
    async def decompose_query(self, ctx: Context, ev: ClassificationEvent) -> DecompositionEvent:
        """
        Decompose complex queries into manageable sub-queries.
        """
        start_time = time.time()
        try:
            # Check if decomposition is enabled and needed
            if (not self.config.get("query_decomposition_enabled", True) or 
                not ev.classification.requires_decomposition):
                
                # No decomposition needed, create single sub-query
                subqueries = [SubQuery(
                    text=ev.query,
                    priority=1,
                    depends_on=[],
                    query_type=ev.classification.query_type
                )]
            else:
                # Perform intelligent decomposition
                subqueries = await self.decomposer.decompose_query(ev.query, ev.classification)
            
            logger.info(f"Query decomposed into {len(subqueries)} sub-queries")
            
            if self.performance_monitoring_enabled:
                decomposition_time = time.time() - start_time
                self.performance_metrics['decomposition_time'] += decomposition_time
                if len(subqueries) > 1:
                    self.performance_metrics['decomposition_success_rate'] = (
                        (self.performance_metrics['decomposition_success_rate'] * (self.performance_metrics['decomposed_queries'] - 1) + 1.0) /
                        self.performance_metrics['decomposed_queries'] if self.performance_metrics['decomposed_queries'] > 0 else 1.0
                    )
            
            return DecompositionEvent(
                query=ev.query,
                classification=ev.classification,
                subqueries=subqueries,
                session_id=ev.session_id
            )
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            # Fallback: single query
            fallback_subqueries = [SubQuery(
                text=ev.query,
                priority=1,
                depends_on=[],
                query_type=ev.classification.query_type
            )]
            return DecompositionEvent(
                query=ev.query,
                classification=ev.classification,
                subqueries=fallback_subqueries,
                session_id=ev.session_id
            )
    
    @step
    async def execute_subqueries(self, ctx: Context, ev: DecompositionEvent) -> ExecutionEvent:
        """
        Execute sub-queries with dependency management, parallel processing, and cost monitoring.
        """
        start_time = time.time()
        try:
            # Cost monitoring and pre-execution checks
            if self.cost_monitoring_enabled:
                estimated_total_cost = sum(sq.estimated_cost for sq in ev.subqueries)
                self.cost_metrics['total_queries'] += 1
                
                # Check if total cost exceeds limits
                if estimated_total_cost > self.max_query_cost:
                    logger.warning(f"Query cost ${estimated_total_cost:.4f} exceeds limit ${self.max_query_cost:.2f}")
                    # Apply cost optimization
                    optimized_subqueries = self._apply_query_cost_limits(ev.subqueries, self.max_query_cost)
                    cost_saved = estimated_total_cost - sum(sq.estimated_cost for sq in optimized_subqueries)
                    self.cost_metrics['cost_savings_from_optimization'] += cost_saved
                    logger.info(f"Applied cost optimization, saved ${cost_saved:.4f}")
                    ev.subqueries = optimized_subqueries
                
                self.cost_metrics['total_estimated_cost'] += sum(sq.estimated_cost for sq in ev.subqueries)
                if len(ev.subqueries) > 1:
                    self.cost_metrics['decomposed_queries'] += 1
                    self.cost_metrics['decomposed_query_cost'] += sum(sq.estimated_cost for sq in ev.subqueries)
            
            subquery_results = {}
            
            if len(ev.subqueries) == 1:
                # Single query - direct execution
                result = await self._execute_single_query(ev.subqueries[0].text)
                subquery_results[ev.subqueries[0].priority] = result
            else:
                # Multiple sub-queries - execute with dependency management
                if self.config.get("parallel_execution_enabled", True):
                    if self.performance_monitoring_enabled:
                        self.performance_metrics['parallel_execution_count'] += 1
                    subquery_results = await self._execute_parallel_with_dependencies(ev.subqueries)
                else:
                    if self.performance_monitoring_enabled:
                        self.performance_metrics['sequential_execution_count'] += 1
                    subquery_results = await self._execute_sequential(ev.subqueries)
            
            execution_time = time.time() - start_time
            logger.info(f"Executed {len(subquery_results)} sub-queries successfully in {execution_time:.2f}s")
            
            if self.performance_monitoring_enabled:
                self.performance_metrics['execution_time'] += execution_time
                self.performance_metrics['successful_queries'] += 1
            
            return ExecutionEvent(
                original_query=ev.query,
                subquery_results=subquery_results,
                subqueries=ev.subqueries,
                session_id=ev.session_id
            )
            
        except Exception as e:
            logger.error(f"Sub-query execution failed: {e}")
            
            if self.performance_monitoring_enabled:
                execution_time = time.time() - start_time
                self.performance_metrics['execution_time'] += execution_time
                self.performance_metrics['failed_queries'] += 1
                self.performance_metrics['fallback_activations'] += 1
            
            # Enhanced fallback with progressive error handling
            return await self._handle_execution_failure(ev, e)
    
    @step
    async def aggregate_results(self, ctx: Context, ev: ExecutionEvent) -> StopEvent:
        """
        Aggregate sub-query results into a coherent final response with cost monitoring.
        """
        start_time = time.time()
        try:
            if len(ev.subquery_results) == 1:
                # Single result - return directly
                final_response = list(ev.subquery_results.values())[0]
            else:
                # Multiple results - aggregate intelligently
                if self.cost_monitoring_enabled:
                    # Estimate aggregation cost
                    total_input_length = sum(len(result) for result in ev.subquery_results.values())
                    aggregation_cost = estimate_query_cost(ev.original_query + str(total_input_length), 500)  # Estimated output tokens
                    self.cost_metrics['aggregation_cost'] += aggregation_cost
                    
                    logger.info(f"Estimated aggregation cost: ${aggregation_cost:.4f}")
                
                final_response = await self._aggregate_multiple_results(
                    ev.original_query,
                    ev.subquery_results,
                    ev.subqueries
                )
                
                # Apply final verification to aggregated response if enabled
                if self.verification_enabled and self.hallucination_detector:
                    final_response = await self._verify_aggregated_result(
                        ev.original_query, final_response, ev.subquery_results
                    )
            
            aggregation_time = time.time() - start_time
            logger.info(f"Results aggregated successfully in {aggregation_time:.2f}s")
            
            if self.performance_monitoring_enabled:
                self.performance_metrics['aggregation_time'] += aggregation_time
                total_time = (self.performance_metrics['classification_time'] + 
                            self.performance_metrics['decomposition_time'] + 
                            self.performance_metrics['execution_time'] + 
                            self.performance_metrics['aggregation_time'])
                self.performance_metrics['total_processing_time'] = total_time
            
            # Log cost metrics if monitoring is enabled
            if self.cost_monitoring_enabled and len(ev.subquery_results) > 1:
                self._log_cost_metrics()
                
            # Log performance metrics if monitoring is enabled
            if self.performance_monitoring_enabled:
                self._log_performance_metrics()
            
            return StopEvent(result=final_response)
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            # Enhanced fallback with error context
            fallback_response = await self._handle_aggregation_failure(ev, e)
            return StopEvent(result=fallback_response)
    
    async def _execute_single_query(self, query: str) -> str:
        """Execute a single query using the underlying agent workflow."""
        try:
            # Create a simple event-like object for the agent workflow
            class SimpleEvent:
                def __init__(self, query: str):
                    self.query = query
                    self.input = query
                    self.message = query
                
                def dict(self):
                    return {"query": self.query, "input": self.query, "message": self.query}
            
            # Use the agent workflow to process the query
            result = await self.agent_workflow.arun(query)
            return str(result)
            
        except Exception as e:
            logger.error(f"Single query execution failed: {e}")
            raise
    
    async def _execute_parallel_with_dependencies(self, subqueries: List[SubQuery]) -> Dict[int, str]:
        """Execute sub-queries in parallel with optimized dependency resolution and context sharing."""
        results = {}
        completed = set()
        context_cache = {}  # Shared context between queries
        
        # Analyze dependency graph for optimization opportunities
        dependency_graph = self._build_dependency_graph(subqueries)
        execution_plan = self._create_optimized_execution_plan(dependency_graph, subqueries)
        
        logger.info(f"Created optimized execution plan with {len(execution_plan)} waves")
        
        # Execute in optimized waves
        for wave_num, wave_queries in enumerate(execution_plan):
            logger.info(f"Executing wave {wave_num + 1} with {len(wave_queries)} queries")
            
            # Prepare enhanced queries with shared context
            enhanced_queries = []
            query_to_subquery = {}
            
            for sq in wave_queries:
                # Build context from dependencies and shared context cache
                context = self._build_enhanced_context(sq, results, context_cache)
                enhanced_query = self._enhance_query_with_context(sq.text, context, sq)
                
                enhanced_queries.append(enhanced_query)
                query_to_subquery[enhanced_query] = sq
            
            # Execute wave in parallel with timeout and resource management
            tasks = []
            for enhanced_query in enhanced_queries:
                task = asyncio.create_task(self._execute_single_query_with_timeout(enhanced_query))
                tasks.append(task)
            
            # Wait for all tasks with timeout
            wave_timeout = min(30.0, 10.0 * len(wave_queries))  # Adaptive timeout
            try:
                task_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=wave_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Wave {wave_num + 1} timed out after {wave_timeout}s")
                # Handle partial results and timeouts
                task_results = await self._handle_wave_timeout(tasks, enhanced_queries)
            
            # Process results and update context cache
            for i, enhanced_query in enumerate(enhanced_queries):
                sq = query_to_subquery[enhanced_query]
                
                if isinstance(task_results[i], Exception):
                    logger.error(f"Sub-query {sq.priority} failed: {task_results[i]}")
                    # Attempt recovery with simpler query
                    recovery_result = await self._attempt_query_recovery(sq, task_results[i])
                    results[sq.priority] = recovery_result
                    
                    if self.performance_monitoring_enabled:
                        self.performance_metrics['error_recovery_attempts'] += 1
                        if "simplified" in recovery_result.lower() or "key terms" in recovery_result.lower():
                            self.performance_metrics['error_recovery_successes'] += 1
                else:
                    result = task_results[i]
                    
                    # Apply verification if enabled
                    if self.verification_enabled and self.hallucination_detector:
                        result = await self._verify_subquery_result(sq, result)
                    
                    results[sq.priority] = result
                    
                    # Extract key information for context sharing
                    self._update_context_cache(context_cache, sq, result)
                    
                    # Cache individual subquery results if caching is enabled
                    if self.cache_enabled and self.global_cache:
                        await self._cache_subquery_result(sq, result)
                
                completed.add(sq.priority)
            
            logger.info(f"Wave {wave_num + 1} completed, {len(completed)} total queries done")
        
        # Handle any remaining incomplete queries (shouldn't happen with good planning)
        for sq in subqueries:
            if sq.priority not in results:
                logger.warning(f"Sub-query {sq.priority} not completed, adding fallback")
                results[sq.priority] = f"Unable to process: {sq.text}"
        
        return results
    
    async def _execute_sequential(self, subqueries: List[SubQuery]) -> Dict[int, str]:
        """Execute sub-queries sequentially in priority order."""
        results = {}
        
        # Sort by priority
        sorted_subqueries = sorted(subqueries, key=lambda x: x.priority)
        
        for sq in sorted_subqueries:
            try:
                # Include dependency results in the query context
                enhanced_query = self._enhance_query_with_context(
                    sq.text,
                    {dep: results.get(dep, "") for dep in sq.depends_on},
                    sq
                )
                
                result = await self._execute_single_query(enhanced_query)
                
                # Apply verification if enabled
                if self.verification_enabled and self.hallucination_detector:
                    result = await self._verify_subquery_result(sq, result)
                
                results[sq.priority] = result
                
                # Cache individual subquery results if caching is enabled
                if self.cache_enabled and self.global_cache:
                    await self._cache_subquery_result(sq, result)
                
            except Exception as e:
                logger.error(f"Sub-query {sq.priority} failed: {e}")
                # Attempt recovery with simpler query
                recovery_result = await self._attempt_query_recovery(sq, e)
                results[sq.priority] = recovery_result
        
        return results
    
    def _enhance_query_with_context(self, query: str, dependency_results: Dict[int, str], subquery: Optional[SubQuery] = None) -> str:
        """Enhance a query with context from dependency results with citation preservation."""
        if not dependency_results or not any(dependency_results.values()):
            return query
        
        context_parts = []
        for dep_id, result in dependency_results.items():
            if result.strip():
                # Extract citations from dependency results
                citations = self._extract_citations(result)
                
                # Create context summary preserving key information
                context_summary = self._create_context_summary(result, subquery, dep_id)
                context_parts.append(f"Context from step {dep_id}: {context_summary}")
                
                # If there are citations, include them
                if citations:
                    context_parts.append(f"Citations from step {dep_id}: {', '.join(citations)}")
        
        if context_parts:
            context = "\n".join(context_parts)
            enhanced_query = f"""Context from previous sub-queries:
{context}

Based on the above context, please answer: {query}

Important: 
- Use the provided context to inform your answer
- Preserve any citations from the context in your response
- Maintain factual accuracy based on the context provided"""
            return enhanced_query
        
        return query
    
    async def _aggregate_multiple_results(self, original_query: str, results: Dict[int, str], subqueries: Optional[List[SubQuery]] = None) -> str:
        """Aggregate multiple sub-query results with enhanced citation preservation."""
        try:
            # Extract and organize citations from all results
            all_citations = self._extract_all_citations(results)
            
            # Create enhanced result structure with context relationships
            structured_results = self._structure_results_with_context(results, subqueries or [])
            
            # Use LLM to aggregate results intelligently with citation guidance
            aggregation_prompt = f"""
You are tasked with combining multiple sub-query answers into a comprehensive response to the original question.

Original Question: "{original_query}"

Sub-query Results:
{structured_results}

Available Citations: {', '.join(all_citations) if all_citations else 'None'}

Instructions for creating the final answer:
1. Directly address the original question with a complete, coherent response
2. Integrate information from all sub-answers logically and smoothly
3. PRESERVE ALL CITATIONS: When using information from sub-answers, maintain their original citations in the format [citation:id]
4. Remove redundant information but ensure completeness
5. Structure the response clearly with proper flow between ideas
6. If sub-answers contradict, acknowledge differences and provide balanced perspective
7. Ensure the final answer is self-contained and comprehensive

Citation Requirements:
- Keep ALL citations from sub-answers in your final response
- Do not modify citation IDs
- Place citations immediately after the relevant information
- If combining information from multiple sources, include all relevant citations

Final Comprehensive Answer:
"""
            
            response = await self.aggregation_llm.acomplete(aggregation_prompt)
            aggregated_response = response.text.strip()
            
            # Validate citation preservation
            final_citations = self._extract_citations(aggregated_response)
            missing_citations = set(all_citations) - set(final_citations)
            
            if missing_citations:
                logger.warning(f"Citations lost during aggregation: {missing_citations}")
                # Attempt to recover missing citations
                aggregated_response = self._recover_missing_citations(aggregated_response, results, missing_citations)
            
            logger.info(f"Aggregated response with {len(final_citations)} citations preserved")
            return aggregated_response
            
        except Exception as e:
            logger.error(f"Enhanced LLM aggregation failed: {e}")
            # Fallback to citation-aware concatenation
            return self._citation_aware_concatenation(results)
    
    def _simple_concatenation(self, results: Dict[int, str]) -> str:
        """Simple fallback aggregation by concatenating results."""
        if not results:
            return "I apologize, but I couldn't generate a response to your query."
        
        if len(results) == 1:
            return list(results.values())[0]
        
        # Concatenate with clear separators
        parts = []
        for priority in sorted(results.keys()):
            parts.append(results[priority])
        
        return "\n\n".join(parts)
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation IDs from text."""
        import re
        citation_pattern = r'\[citation:([^\]]+)\]'
        citations = re.findall(citation_pattern, text)
        return list(set(citations))  # Remove duplicates
    
    def _extract_all_citations(self, results: Dict[int, str]) -> List[str]:
        """Extract all unique citations from multiple results."""
        all_citations = []
        for result in results.values():
            all_citations.extend(self._extract_citations(result))
        return list(set(all_citations))  # Remove duplicates
    
    def _create_context_summary(self, result: str, subquery: Optional[SubQuery], dep_id: int) -> str:
        """Create a context summary focusing on information relevant to the current subquery."""
        # If subquery has specific context requirements, focus on those
        if subquery and subquery.context_requirements:
            # Extract relevant parts based on context requirements
            context_req = subquery.context_requirements[0].lower()
            
            # Simple keyword-based extraction
            sentences = result.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in context_req.split()):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return '. '.join(relevant_sentences[:2])  # Limit to 2 most relevant sentences
        
        # Default: return first part of the result
        return result[:300] + ("..." if len(result) > 300 else "")
    
    def _structure_results_with_context(self, results: Dict[int, str], subqueries: List[SubQuery]) -> str:
        """Structure results showing context relationships between sub-queries."""
        structured_parts = []
        
        for priority in sorted(results.keys()):
            result = results[priority]
            
            # Find corresponding subquery
            subquery = next((sq for sq in subqueries if sq.priority == priority), None)
            
            if subquery:
                dependencies = subquery.depends_on
                if dependencies:
                    dep_info = f" (depends on: {', '.join(map(str, dependencies))})"
                else:
                    dep_info = " (independent)"
                    
                structured_parts.append(f"Sub-query {priority}{dep_info}:")
                structured_parts.append(f"Query: {subquery.text}")
            else:
                structured_parts.append(f"Sub-answer {priority}:")
            
            structured_parts.append(f"Answer: {result}")
            structured_parts.append("")  # Empty line for separation
        
        return "\n".join(structured_parts)
    
    def _recover_missing_citations(self, aggregated_response: str, original_results: Dict[int, str], missing_citations: set) -> str:
        """Attempt to recover missing citations by finding where they should be placed."""
        # For each missing citation, find the content it was associated with
        for citation_id in missing_citations:
            # Find which result contained this citation
            for priority, result in original_results.items():
                if f'[citation:{citation_id}]' in result:
                    # Extract the sentence containing the citation
                    sentences = result.split('. ')
                    for sentence in sentences:
                        if f'[citation:{citation_id}]' in sentence:
                            # Find similar content in the aggregated response
                            clean_sentence = sentence.replace(f'[citation:{citation_id}]', '').strip()
                            if clean_sentence and len(clean_sentence) > 10:
                                # Try to find similar content and add citation
                                words = clean_sentence.split()[:5]  # First 5 words
                                search_phrase = ' '.join(words)
                                
                                if search_phrase.lower() in aggregated_response.lower():
                                    # Insert citation after this phrase
                                    import re
                                    pattern = re.escape(search_phrase)
                                    replacement = search_phrase + f' [citation:{citation_id}]'
                                    aggregated_response = re.sub(
                                        pattern, replacement, aggregated_response, count=1, flags=re.IGNORECASE
                                    )
                                    logger.info(f"Recovered citation {citation_id}")
                                    break
                            break
                    break
        
        return aggregated_response
    
    def _citation_aware_concatenation(self, results: Dict[int, str]) -> str:
        """Citation-aware fallback aggregation by concatenating results."""
        if not results:
            return "I apologize, but I couldn't generate a response to your query."
        
        if len(results) == 1:
            return list(results.values())[0]
        
        # Concatenate with clear separators while preserving all citations
        parts = []
        for priority in sorted(results.keys()):
            result = results[priority]
            # Add section headers for clarity
            if len(results) > 2:  # Only add headers for 3+ results
                parts.append(f"**Regarding part {priority} of your question:**")
            parts.append(result)
        
        concatenated = "\n\n".join(parts)
        
        # Ensure all citations are preserved
        all_citations = self._extract_all_citations(results)
        final_citations = self._extract_citations(concatenated)
        
        if len(final_citations) < len(all_citations):
            logger.warning("Some citations may have been lost in concatenation")
        
        return concatenated
    
    def _apply_query_cost_limits(self, subqueries: List[SubQuery], max_cost: float) -> List[SubQuery]:
        """Apply cost limits to sub-queries by prioritizing and optimizing."""
        if not subqueries:
            return subqueries
        
        # Sort by priority (lower number = higher priority)
        sorted_queries = sorted(subqueries, key=lambda x: x.priority)
        optimized_queries = []
        total_cost = 0.0
        
        for sq in sorted_queries:
            if total_cost + sq.estimated_cost <= max_cost:
                # Query fits within budget
                optimized_queries.append(sq)
                total_cost += sq.estimated_cost
            else:
                # Try to fit by reducing query complexity
                remaining_budget = max_cost - total_cost
                if remaining_budget > 0:
                    # Simplify the query to fit
                    simplified_query = self._simplify_query_for_budget(sq, remaining_budget)
                    if simplified_query:
                        optimized_queries.append(simplified_query)
                        total_cost += simplified_query.estimated_cost
                        logger.info(f"Simplified sub-query {sq.priority} to fit budget")
                    else:
                        logger.warning(f"Dropped sub-query {sq.priority} due to cost constraints")
                else:
                    logger.warning(f"Dropped sub-query {sq.priority} - no remaining budget")
        
        return optimized_queries
    
    def _simplify_query_for_budget(self, subquery: SubQuery, available_budget: float) -> Optional[SubQuery]:
        """Simplify a sub-query to fit within the available budget."""
        if available_budget <= 0:
            return None
        
        # Calculate how much we need to reduce the query
        reduction_ratio = available_budget / subquery.estimated_cost
        
        if reduction_ratio >= 0.5:  # Only simplify if we can keep at least 50% of the query
            # Simple approach: truncate query text
            target_length = int(len(subquery.text) * reduction_ratio * 0.8)  # Be conservative
            if target_length < 20:  # Don't make queries too short
                return None
                
            simplified_text = subquery.text[:target_length].strip()
            if simplified_text.endswith(' '):
                simplified_text = simplified_text.rstrip() + "..."
            
            # Estimate new cost
            new_cost = available_budget * 0.9  # Use most of available budget
            
            return SubQuery(
                text=simplified_text,
                priority=subquery.priority,
                depends_on=subquery.depends_on,
                query_type=subquery.query_type,
                context_requirements=subquery.context_requirements,
                estimated_cost=new_cost,
                confidence_threshold=max(0.6, subquery.confidence_threshold - 0.1)  # Lower threshold
            )
        
        return None
    
    def _log_cost_metrics(self) -> None:
        """Log current cost metrics for monitoring."""
        metrics = self.cost_metrics
        total_queries = metrics['total_queries']
        
        if total_queries > 0:
            avg_cost_per_query = metrics['total_estimated_cost'] / total_queries
            decomposition_rate = metrics['decomposed_queries'] / total_queries
            
            logger.info(f"""Cost Metrics Summary:
- Total Queries: {total_queries}
- Average Cost per Query: ${avg_cost_per_query:.4f}
- Decomposition Rate: {decomposition_rate:.2%}
- Total Estimated Cost: ${metrics['total_estimated_cost']:.4f}
- Decomposed Query Cost: ${metrics['decomposed_query_cost']:.4f}
- Aggregation Cost: ${metrics['aggregation_cost']:.4f}
- Cost Savings from Optimization: ${metrics['cost_savings_from_optimization']:.4f}""")
    
    def get_cost_metrics(self) -> Dict[str, Any]:
        """Get current cost metrics for external monitoring."""
        metrics = self.cost_metrics.copy()
        
        if metrics['total_queries'] > 0:
            metrics['average_cost_per_query'] = metrics['total_estimated_cost'] / metrics['total_queries']
            metrics['decomposition_rate'] = metrics['decomposed_queries'] / metrics['total_queries']
            
            if metrics['decomposed_queries'] > 0:
                metrics['average_decomposed_query_cost'] = metrics['decomposed_query_cost'] / metrics['decomposed_queries']
            else:
                metrics['average_decomposed_query_cost'] = 0.0
        else:
            metrics['average_cost_per_query'] = 0.0
            metrics['decomposition_rate'] = 0.0
            metrics['average_decomposed_query_cost'] = 0.0
        
        return metrics
    
    def reset_cost_metrics(self) -> None:
        """Reset cost metrics (useful for testing or periodic resets)."""
        self.cost_metrics = {
            'total_queries': 0,
            'total_estimated_cost': 0.0,
            'decomposed_queries': 0,
            'decomposed_query_cost': 0.0,
            'aggregation_cost': 0.0,
            'cost_savings_from_optimization': 0.0
        }
        logger.info("Cost metrics reset")
    
    async def _handle_execution_failure(self, ev: DecompositionEvent, error: Exception) -> ExecutionEvent:
        """Handle execution failure with progressive fallback strategies."""
        logger.error(f"Handling execution failure: {error}")
        
        # Strategy 1: Try simplified sub-queries if we have multiple\n        if len(ev.subqueries) > 1:\n            logger.info("Attempting simplified sub-query execution")\n            try:\n                simplified_subqueries = self._create_simplified_fallback_queries(ev.subqueries)\n                if simplified_subqueries:\n                    # Try executing simplified queries\n                    subquery_results = {}\n                    for sq in simplified_subqueries:\n                        try:\n                            result = await self._execute_single_query(sq.text)\n                            subquery_results[sq.priority] = result\n                        except Exception as sq_error:\n                            logger.warning(f"Simplified sub-query {sq.priority} also failed: {sq_error}")\n                            subquery_results[sq.priority] = f"Unable to process this part of your query: {sq.text[:50]}..."\n                    \n                    if subquery_results:\n                        logger.info("Simplified sub-query execution partially successful")\n                        return ExecutionEvent(\n                            original_query=ev.query,\n                            subquery_results=subquery_results,\n                            subqueries=simplified_subqueries,\n                            session_id=ev.session_id\n                        )\n            except Exception as simplification_error:\n                logger.warning(f"Simplified sub-query execution failed: {simplification_error}")\n        \n        # Strategy 2: Try original query as fallback\n        logger.info("Attempting original query execution as fallback")\n        try:\n            fallback_result = await self._execute_single_query(ev.query)\n            subquery_results = {1: fallback_result}\n            logger.info("Original query fallback successful")\n        except Exception as fallback_error:\n            logger.error(f"Original query fallback also failed: {fallback_error}")\n            \n            # Strategy 3: Create informative error response\n            error_context = self._create_error_context(error, fallback_error)\n            subquery_results = {1: f"""I apologize, but I encountered difficulties processing your query. Here's what I attempted:\n\n1. Query decomposition and parallel execution\n2. Simplified query execution\n3. Direct query execution\n\nError details: {error_context}\n\nPlease try rephrasing your question or breaking it into simpler parts."""}\n        \n        return ExecutionEvent(\n            original_query=ev.query,\n            subquery_results=subquery_results,\n            subqueries=ev.subqueries,\n            session_id=ev.session_id\n        )\n    \n    async def _handle_aggregation_failure(self, ev: ExecutionEvent, error: Exception) -> str:\n        """Handle aggregation failure with fallback strategies."""        logger.error(f"Handling aggregation failure: {error}")\n        \n        # Strategy 1: Try simple concatenation with error context\n        try:\n            fallback_response = self._citation_aware_concatenation(ev.subquery_results)\n            \n            # Add error context if concatenation worked\n            error_notice = "\\n\\n*Note: There was an issue with response synthesis, so I've provided the information in sections above.*"\n            return fallback_response + error_notice\n            \n        except Exception as concat_error:\n            logger.error(f"Concatenation fallback also failed: {concat_error}")\n            \n            # Strategy 2: Manual result combination\n            try:\n                manual_response_parts = []\n                for priority in sorted(ev.subquery_results.keys()):\n                    result = ev.subquery_results[priority]\n                    if result and result.strip():\n                        manual_response_parts.append(f"**Part {priority}:** {result}")\n                \n                if manual_response_parts:\n                    manual_response = "\\n\\n".join(manual_response_parts)\n                    error_notice = "\\n\\n*Note: I encountered an issue processing your query but was able to provide partial information above.*"\n                    return manual_response + error_notice\n                    \n            except Exception as manual_error:\n                logger.error(f"Manual aggregation also failed: {manual_error}")\n            \n            # Strategy 3: Return error message with context\n            return f"""I apologize, but I encountered a significant error while processing and combining the results for your query: "{ev.original_query}"\n\nError details: {str(error)}\n\nPlease try rephrasing your question or ask about specific aspects separately."""\n    \n    async def _attempt_query_recovery(self, subquery: SubQuery, error: Exception) -> str:\n        """Attempt to recover from a sub-query execution failure."""        logger.info(f"Attempting recovery for sub-query {subquery.priority}: {subquery.text[:50]}...")\n        \n        # Strategy 1: Try a simplified version of the query\n        try:\n            simplified_text = self._create_simplified_query_text(subquery.text)\n            if simplified_text and simplified_text != subquery.text:\n                logger.info(f"Trying simplified version: {simplified_text[:50]}...")\n                result = await self._execute_single_query(simplified_text)\n                logger.info(f"Simplified query recovery successful for sub-query {subquery.priority}")\n                return result + " *(Note: This response was generated using a simplified version of your question due to processing issues.)*"\n        except Exception as simplify_error:\n            logger.warning(f"Simplified query recovery failed: {simplify_error}")\n        \n        # Strategy 2: Try extracting key terms and creating a basic query\n        try:\n            key_terms = self._extract_key_terms(subquery.text)\n            if key_terms:\n                basic_query = f"Please provide information about: {', '.join(key_terms[:3])}"\n                logger.info(f"Trying basic query: {basic_query}")\n                result = await self._execute_single_query(basic_query)\n                logger.info(f"Basic query recovery successful for sub-query {subquery.priority}")\n                return result + f" *(Note: This response addresses key terms from your question: {', '.join(key_terms[:3])})*"\n        except Exception as basic_error:\n            logger.warning(f"Basic query recovery failed: {basic_error}")\n        \n        # Strategy 3: Return informative error message\n        return f"""I was unable to process this part of your question: "{subquery.text}"\n\nError: {str(error)[:100]}\n\nPlease try asking about this topic in a different way or break it down into simpler questions."""\n    \n    def _create_simplified_fallback_queries(self, subqueries: List[SubQuery]) -> List[SubQuery]:\n        """Create simplified versions of sub-queries for fallback execution."""        simplified = []\n        \n        for sq in subqueries[:2]:  # Limit to first 2 queries for fallback\n            simplified_text = self._create_simplified_query_text(sq.text)\n            if simplified_text:\n                simplified_sq = SubQuery(\n                    text=simplified_text,\n                    priority=sq.priority,\n                    depends_on=[],  # Remove dependencies for simpler execution\n                    query_type=sq.query_type,\n                    context_requirements=[],\n                    estimated_cost=sq.estimated_cost * 0.7,  # Assume 30% cost reduction\n                    confidence_threshold=max(0.5, sq.confidence_threshold - 0.2)\n                )\n                simplified.append(simplified_sq)\n        \n        return simplified\n    \n    def _create_simplified_query_text(self, query_text: str) -> str:\n        """Create a simplified version of a query text."""        if not query_text or len(query_text) < 20:\n            return query_text\n        \n        # Remove complex phrases and simplify\n        simplified = query_text\n        \n        # Remove parenthetical information\n        import re\n        simplified = re.sub(r'\\([^)]*\\)', '', simplified)\n        \n        # Remove complex conjunctions and convert to simpler form\n        complex_patterns = [\n            (r'\\bhowever\\b', 'but'),\n            (r'\\bmoreover\\b', 'also'),\n            (r'\\bfurthermore\\b', 'also'),\n            (r'\\bin addition to\\b', 'and'),\n            (r'\\bconsequently\\b', 'so'),\n        ]\n        \n        for pattern, replacement in complex_patterns:\n            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)\n        \n        # Limit length and clean up\n        if len(simplified) > 100:\n            sentences = simplified.split('. ')\n            if len(sentences) > 1:\n                simplified = sentences[0] + '.'\n            else:\n                simplified = simplified[:100] + '...'\n        \n        return simplified.strip()\n    \n    def _extract_key_terms(self, text: str) -> List[str]:\n        """Extract key terms from query text."""        import re\n        \n        # Remove stop words and extract meaningful terms\n        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', \n                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',\n                     'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', \n                     'where', 'who', 'which', 'this', 'that', 'these', 'those'}\n        \n        # Extract words\n        words = re.findall(r'\\b[a-zA-Z]+\\b', text.lower())\n        \n        # Filter and get important terms\n        key_terms = [word for word in words if word not in stop_words and len(word) > 3]\n        \n        # Return top 5 most meaningful terms (could be enhanced with TF-IDF or other methods)\n        return key_terms[:5]\n    \n    def _create_error_context(self, primary_error: Exception, secondary_error: Exception) -> str:\n        """Create a concise error context for user feedback."""        error_types = []\n        \n        # Categorize errors for better user understanding\n        if 'timeout' in str(primary_error).lower() or 'timeout' in str(secondary_error).lower():\n            error_types.append('processing timeout')\n        \n        if 'rate limit' in str(primary_error).lower() or 'rate limit' in str(secondary_error).lower():\n            error_types.append('service rate limiting')\n        \n        if 'connection' in str(primary_error).lower() or 'connection' in str(secondary_error).lower():\n            error_types.append('connection issues')\n        \n        if 'model' in str(primary_error).lower() or 'model' in str(secondary_error).lower():\n            error_types.append('model processing issues')\n        \n        if not error_types:\n            error_types.append('processing complexity')\n        \n        return ', '.join(error_types)
    
    def _log_performance_metrics(self) -> None:
        """Log current performance metrics for monitoring."""
        metrics = self.performance_metrics
        total_queries = metrics['total_queries']
        
        if total_queries > 0:
            success_rate = metrics['successful_queries'] / total_queries
            failure_rate = metrics['failed_queries'] / total_queries
            avg_total_time = metrics['total_processing_time'] / total_queries
            
            decomposed_queries = self.cost_metrics.get('decomposed_queries', 0)
            
            recovery_success_rate = 0.0
            if metrics['error_recovery_attempts'] > 0:
                recovery_success_rate = metrics['error_recovery_successes'] / metrics['error_recovery_attempts']
            
            logger.info(f"""Performance Metrics Summary:
- Total Queries Processed: {total_queries}
- Success Rate: {success_rate:.2%}
- Failure Rate: {failure_rate:.2%}
- Average Processing Time: {avg_total_time:.3f}s
- Decomposed Queries: {decomposed_queries}
- Parallel Executions: {metrics['parallel_execution_count']}
- Sequential Executions: {metrics['sequential_execution_count']}
- Fallback Activations: {metrics['fallback_activations']}
- Error Recovery Attempts: {metrics['error_recovery_attempts']}
- Error Recovery Success Rate: {recovery_success_rate:.2%}
- Average Stage Times:
  - Classification: {metrics['classification_time'] / total_queries:.3f}s
  - Decomposition: {metrics['decomposition_time'] / total_queries:.3f}s
  - Execution: {metrics['execution_time'] / total_queries:.3f}s
  - Aggregation: {metrics['aggregation_time'] / total_queries:.3f}s""")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for external monitoring."""
        metrics = self.performance_metrics.copy()
        total_queries = metrics['total_queries']
        
        if total_queries > 0:
            metrics['success_rate'] = metrics['successful_queries'] / total_queries
            metrics['failure_rate'] = metrics['failed_queries'] / total_queries
            metrics['average_processing_time'] = metrics['total_processing_time'] / total_queries
            metrics['average_classification_time'] = metrics['classification_time'] / total_queries
            metrics['average_decomposition_time'] = metrics['decomposition_time'] / total_queries
            metrics['average_execution_time'] = metrics['execution_time'] / total_queries
            metrics['average_aggregation_time'] = metrics['aggregation_time'] / total_queries
            
            if metrics['error_recovery_attempts'] > 0:
                metrics['error_recovery_success_rate'] = metrics['error_recovery_successes'] / metrics['error_recovery_attempts']
            else:
                metrics['error_recovery_success_rate'] = 0.0
        else:
            metrics.update({
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'average_processing_time': 0.0,
                'average_classification_time': 0.0,
                'average_decomposition_time': 0.0,
                'average_execution_time': 0.0,
                'average_aggregation_time': 0.0,
                'error_recovery_success_rate': 0.0
            })
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics (useful for testing or periodic resets)."""
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'classification_time': 0.0,
            'decomposition_time': 0.0,
            'execution_time': 0.0,
            'aggregation_time': 0.0,
            'total_processing_time': 0.0,
            'decomposition_success_rate': 0.0,
            'parallel_execution_count': 0,
            'sequential_execution_count': 0,
            'cache_hits': 0,
            'fallback_activations': 0,
            'error_recovery_attempts': 0,
            'error_recovery_successes': 0
        }
        logger.info("Performance metrics reset")
    
    def _build_dependency_graph(self, subqueries: List[SubQuery]) -> Dict[int, List[int]]:
        """Build a dependency graph for optimization analysis."""
        dependency_graph = {}
        for sq in subqueries:
            dependency_graph[sq.priority] = sq.depends_on.copy()
        return dependency_graph
    
    def _create_optimized_execution_plan(self, dependency_graph: Dict[int, List[int]], subqueries: List[SubQuery]) -> List[List[SubQuery]]:
        """Create an optimized execution plan that maximizes parallelism."""
        execution_plan = []
        remaining_queries = {sq.priority: sq for sq in subqueries}
        completed = set()
        
        max_iterations = len(subqueries) + 1  # Prevent infinite loops
        iteration = 0
        
        while remaining_queries and iteration < max_iterations:
            # Find queries ready to execute (all dependencies satisfied)
            ready_queries = []
            
            for priority, sq in remaining_queries.items():
                if all(dep in completed for dep in sq.depends_on):
                    ready_queries.append(sq)
            
            if not ready_queries:
                # Handle circular dependencies by breaking them
                logger.warning("Detected circular dependencies, applying resolution strategy")
                ready_queries = self._resolve_circular_dependencies(remaining_queries, completed)
            
            if ready_queries:
                execution_plan.append(ready_queries)
                for sq in ready_queries:
                    completed.add(sq.priority)
                    del remaining_queries[sq.priority]
            
            iteration += 1
        
        # Add any remaining queries (shouldn't happen with proper resolution)
        if remaining_queries:
            logger.warning(f"Adding {len(remaining_queries)} unresolved queries to final wave")
            execution_plan.append(list(remaining_queries.values()))
        
        return execution_plan
    
    def _resolve_circular_dependencies(self, remaining_queries: Dict[int, SubQuery], completed: set) -> List[SubQuery]:
        """Resolve circular dependencies by breaking them strategically."""
        # Strategy: Select the query with the fewest unmet dependencies
        best_query = None
        min_unmet_deps = float('inf')
        
        for sq in remaining_queries.values():
            unmet_deps = sum(1 for dep in sq.depends_on if dep not in completed)
            if unmet_deps < min_unmet_deps:
                min_unmet_deps = unmet_deps
                best_query = sq
        
        if best_query:
            logger.info(f"Breaking circular dependency by executing query {best_query.priority} with {min_unmet_deps} unmet dependencies")
            return [best_query]
        
        # Fallback: return the first query
        return [list(remaining_queries.values())[0]] if remaining_queries else []
    
    def _build_enhanced_context(self, subquery: SubQuery, results: Dict[int, str], context_cache: Dict[str, str]) -> Dict[int, str]:
        """Build enhanced context combining dependency results and shared context cache."""
        context = {}
        
        # Add dependency results
        for dep in subquery.depends_on:
            if dep in results:
                context[dep] = results[dep]
        
        # Add relevant shared context based on subquery requirements
        if subquery.context_requirements:
            for req in subquery.context_requirements:
                # Look for relevant cached context
                for cache_key, cache_value in context_cache.items():
                    if any(keyword in cache_key.lower() for keyword in req.lower().split()):
                        # Create a pseudo-dependency ID for cached context
                        cache_dep_id = 0  # Special ID for cached context
                        context[cache_dep_id] = f"Relevant context: {cache_value}"
                        break
        
        return context
    
    async def _execute_single_query_with_timeout(self, query: str, timeout: float = 15.0) -> str:
        """Execute a single query with timeout for better resource management."""
        try:
            return await asyncio.wait_for(self._execute_single_query(query), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Query execution timed out after {timeout}s: {query[:50]}...")
            raise asyncio.TimeoutError(f"Query execution timeout: {query[:50]}...")
    
    async def _handle_wave_timeout(self, tasks: List[asyncio.Task], queries: List[str]) -> List:
        """Handle wave timeout by collecting partial results."""
        results = []
        
        for i, task in enumerate(tasks):
            if task.done():
                try:
                    result = task.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
            else:
                # Cancel incomplete task and create timeout exception
                task.cancel()
                timeout_error = asyncio.TimeoutError(f"Query timed out: {queries[i][:50]}...")
                results.append(timeout_error)
        
        return results
    
    def _update_context_cache(self, context_cache: Dict[str, str], subquery: SubQuery, result: str):
        """Update the shared context cache with key information from successful queries."""
        try:
            # Extract key concepts and information for future context sharing
            key_terms = self._extract_key_terms(subquery.text)
            
            if key_terms and result and len(result.strip()) > 50:
                # Create cache entries for key terms
                for term in key_terms[:3]:  # Limit to top 3 terms
                    cache_key = f"{term}_{subquery.query_type.value}"
                    
                    # Extract relevant sentences containing the term
                    sentences = result.split('. ')
                    relevant_sentences = [
                        sent for sent in sentences 
                        if term.lower() in sent.lower() and len(sent.strip()) > 20
                    ]
                    
                    if relevant_sentences:
                        # Take the first relevant sentence as context
                        context_value = relevant_sentences[0][:200]
                        context_cache[cache_key] = context_value
                        
                        # Limit cache size to prevent memory issues
                        if len(context_cache) > 50:
                            # Remove oldest entries (simple FIFO)
                            oldest_key = next(iter(context_cache))
                            del context_cache[oldest_key]
                            
        except Exception as e:
            logger.debug(f"Failed to update context cache: {e}")
    
    async def _verify_subquery_result(self, subquery: SubQuery, result: str) -> str:
        """Verify a sub-query result using the hallucination detection system."""
        try:
            if not self.hallucination_detector or not result.strip():
                return result
            
            # Create a simple QueryBundle for verification
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=subquery.text)
            
            # Create mock retrieved nodes (in real implementation, these would come from the retrieval)
            from llama_index.core.schema import TextNode, NodeWithScore
            mock_nodes = [NodeWithScore(
                node=TextNode(text=result[:500], id_=f"subquery_{subquery.priority}"),
                score=0.8
            )]
            
            # Calculate confidence scores
            graph_confidence = self.hallucination_detector.calculate_graph_confidence(query_bundle, mock_nodes)
            response_confidence = self.hallucination_detector.calculate_response_confidence(
                result, graph_confidence, [], query_bundle
            )
            
            # Check if response should be filtered
            if self.hallucination_detector.should_filter_response(response_confidence):
                logger.warning(f"Sub-query {subquery.priority} failed verification (confidence: {response_confidence.response_confidence:.3f})")
                
                # Add verification notice to result
                verification_notice = f" *(Note: This response has low confidence - {response_confidence.confidence_level.value})*"
                return result + verification_notice
            else:
                logger.debug(f"Sub-query {subquery.priority} passed verification (confidence: {response_confidence.response_confidence:.3f})")
                return result
                
        except Exception as e:
            logger.warning(f"Verification failed for sub-query {subquery.priority}: {e}")
            return result  # Return original result if verification fails
    
    async def _verify_aggregated_result(self, original_query: str, aggregated_result: str, subquery_results: Dict[int, str]) -> str:
        """Verify the final aggregated result using the hallucination detection system."""
        try:
            if not self.hallucination_detector or not aggregated_result.strip():
                return aggregated_result
            
            # Create QueryBundle for original query
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=original_query)
            
            # Create mock retrieved nodes from all sub-query results
            from llama_index.core.schema import TextNode, NodeWithScore
            mock_nodes = []
            for priority, result in subquery_results.items():
                if result.strip():
                    node = NodeWithScore(
                        node=TextNode(
                            text=result[:500], 
                            id_=f"aggregated_subquery_{priority}",
                            metadata={"subquery_priority": priority}
                        ),
                        score=0.8
                    )
                    mock_nodes.append(node)
            
            # Calculate confidence scores for the aggregated result
            graph_confidence = self.hallucination_detector.calculate_graph_confidence(query_bundle, mock_nodes)
            
            # Extract citations from the aggregated result
            citations = self._extract_citations(aggregated_result)
            
            response_confidence = self.hallucination_detector.calculate_response_confidence(
                aggregated_result, graph_confidence, citations, query_bundle
            )
            
            # Perform post-generation verification
            verification_result, updated_confidence = await self.hallucination_detector.verify_response(
                aggregated_result, response_confidence, query_bundle, mock_nodes
            )
            
            # Update performance metrics
            if self.performance_monitoring_enabled:
                if verification_result.name in ['VERIFIED', 'UNCERTAIN']:
                    verification_success = True
                else:
                    verification_success = False
                    
                # Update verification metrics in performance tracking
                if 'verification_attempts' not in self.performance_metrics:
                    self.performance_metrics['verification_attempts'] = 0
                    self.performance_metrics['verification_successes'] = 0
                
                self.performance_metrics['verification_attempts'] += 1
                if verification_success:
                    self.performance_metrics['verification_successes'] += 1
            
            # Handle verification results
            if verification_result.name == 'REJECTED':
                logger.warning(f"Aggregated result failed verification (confidence: {updated_confidence:.3f})")
                verification_notice = f"\\n\\n*Note: This response has been flagged as potentially unreliable (confidence: {response_confidence.confidence_level.value}). Please verify the information independently.*"
                return aggregated_result + verification_notice
            elif verification_result.name == 'UNCERTAIN':
                logger.info(f"Aggregated result has uncertain verification (confidence: {updated_confidence:.3f})")
                verification_notice = f"\\n\\n*Note: The confidence in this response is moderate ({response_confidence.confidence_level.value}). Some information should be verified.*"
                return aggregated_result + verification_notice
            else:
                logger.debug(f"Aggregated result passed verification (confidence: {updated_confidence:.3f})")
                return aggregated_result
                
        except Exception as e:
            logger.warning(f"Verification failed for aggregated result: {e}")
            return aggregated_result  # Return original result if verification fails
    
    async def _cache_subquery_result(self, subquery: SubQuery, result: str):
        """Cache individual sub-query results for future use."""
        try:
            if not self.global_cache or not result.strip():
                return
            
            # Create a cache key based on sub-query text and type
            cache_key = f"subquery_{subquery.query_type.value}_{hash(subquery.text) % 10000}"
            
            # Estimate cost for caching decision
            estimated_cost = subquery.estimated_cost
            
            # Create a simple response object for caching
            class SimpleResponse:
                def __init__(self, response_text):
                    self.response = response_text
                    
                def __str__(self):
                    return self.response
            
            response_obj = SimpleResponse(result)
            
            # Cache the result
            self.global_cache.put(cache_key, response_obj, estimated_cost)
            
            if self.performance_monitoring_enabled:
                if 'subquery_cache_stores' not in self.performance_metrics:
                    self.performance_metrics['subquery_cache_stores'] = 0
                self.performance_metrics['subquery_cache_stores'] += 1
            
            logger.debug(f"Cached sub-query {subquery.priority} result")\
            
        except Exception as e:
            logger.debug(f"Failed to cache sub-query result: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of integration with verification and caching systems."""
        status = {
            "verification_enabled": self.verification_enabled,
            "hallucination_detector_available": self.hallucination_detector is not None,
            "cache_enabled": self.cache_enabled,
            "global_cache_available": self.global_cache is not None,
        }
        
        # Add verification metrics if available
        if self.verification_enabled and 'verification_attempts' in self.performance_metrics:
            status['verification_attempts'] = self.performance_metrics['verification_attempts']
            status['verification_successes'] = self.performance_metrics['verification_successes']
            if self.performance_metrics['verification_attempts'] > 0:
                status['verification_success_rate'] = (
                    self.performance_metrics['verification_successes'] / 
                    self.performance_metrics['verification_attempts']
                )
            else:
                status['verification_success_rate'] = 0.0
        
        # Add cache metrics if available
        if self.cache_enabled and self.global_cache:
            try:
                cache_stats = self.global_cache.get_stats()
                status['cache_hit_rate'] = self.global_cache.get_hit_rate()
                status['cache_size'] = cache_stats.cache_size
                status['total_cache_cost_saved'] = cache_stats.total_cost_saved
            except Exception as e:
                logger.debug(f"Failed to get cache stats: {e}")
                status['cache_error'] = str(e)
        
        # Add subquery caching metrics
        if 'subquery_cache_stores' in self.performance_metrics:
            status['subquery_cache_stores'] = self.performance_metrics['subquery_cache_stores']
        
        return status