"""
Unified Workflow Orchestrator for SOTA RAG System

This module provides the master workflow orchestrator that intelligently coordinates
all SOTA components based on query characteristics, configuration, and system health.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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

from src.unified_config import get_unified_config, PerformanceProfile, FeatureStatus
from src.agentic_workflow import AgenticWorkflow
from src.cache import get_cache
from src.verification import create_hallucination_detector
from src.settings import init_settings

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_MODAL = "multi_modal"


@dataclass
class QueryCharacteristics:
    """Analysis of query characteristics for intelligent routing."""
    original_query: str
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    complexity_score: float = 0.0
    estimated_tokens: int = 0
    requires_decomposition: bool = False
    has_images: bool = False
    requires_verification: bool = True
    estimated_cost: float = 0.0
    modality: str = "text"
    intent: str = "informational"
    domain: str = "general"


@dataclass
class ProcessingPlan:
    """Plan for processing a query with selected components."""
    use_agentic_workflow: bool = False
    use_semantic_cache: bool = False
    use_hallucination_detection: bool = True
    use_multimodal_support: bool = False
    use_performance_optimization: bool = True
    use_tts_output: bool = False
    
    # Component-specific settings
    agentic_settings: Dict[str, Any] = None
    cache_settings: Dict[str, Any] = None
    verification_settings: Dict[str, Any] = None
    
    # Processing metadata
    estimated_processing_time: float = 0.0
    estimated_cost: float = 0.0
    confidence_threshold: float = 0.8


class UnifiedWorkflow(Workflow):
    """
    Master workflow orchestrator that intelligently coordinates all SOTA components.
    
    This workflow:
    1. Analyzes incoming queries to determine characteristics and requirements
    2. Creates an optimal processing plan based on configuration and system health
    3. Orchestrates the execution through selected components
    4. Handles errors gracefully with fallback strategies
    5. Monitors performance and adjusts behavior dynamically
    """
    
    def __init__(self, timeout: float = 300.0, verbose: bool = False, **kwargs: Any):
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        
        # Initialize configuration and components
        self.config_manager = get_unified_config()
        self.config = self.config_manager.config
        
        # Initialize core components with error handling
        self._initialize_components()
        
        # Processing statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'agentic_queries': 0,
            'verified_queries': 0,
            'multimodal_queries': 0,
            'fallback_activations': 0,
            'total_processing_time': 0.0,
            'total_cost': 0.0
        }
        
        logger.info(f"UnifiedWorkflow initialized with profile: {self.config.performance_profile.value}")
    
    def _initialize_components(self):
        """Initialize all SOTA components with error handling."""
        # Initialize base agent workflow
        try:
            # Import here to avoid circular dependencies
            from src.workflow import create_workflow
            
            # Create base workflow without agentic features
            import os
            os.environ["AGENT_ROUTING_ENABLED"] = "false"
            os.environ["QUERY_DECOMPOSITION_ENABLED"] = "false"
            
            self.base_workflow = create_workflow()
            logger.info("Base workflow initialized successfully")
            self.config_manager.update_component_health("base_workflow", "healthy")
            
        except Exception as e:
            logger.error(f"Failed to initialize base workflow: {e}")
            self.base_workflow = None
            self.config_manager.update_component_health(
                "base_workflow", "error", error_message=str(e)
            )
        
        # Initialize agentic workflow
        try:
            if self.config.agentic_workflow.enabled and self.base_workflow:
                self.agentic_workflow = AgenticWorkflow(
                    agent_workflow=self.base_workflow,
                    timeout=120.0,
                    verbose=self.config.debug_mode
                )
                logger.info("Agentic workflow initialized successfully")
                self.config_manager.update_component_health("agentic_workflow", "healthy")
            else:
                self.agentic_workflow = None
                logger.info("Agentic workflow disabled")
        except Exception as e:
            logger.error(f"Failed to initialize agentic workflow: {e}")
            self.agentic_workflow = None
            self.config_manager.update_component_health(
                "agentic_workflow", "error", error_message=str(e)
            )
        
        # Initialize semantic cache
        try:
            if self.config.semantic_cache.enabled:
                self.semantic_cache = get_cache()
                cache_health = self.semantic_cache.health_check()
                
                if cache_health.get("redis_available", False):
                    logger.info("Semantic cache initialized successfully")
                    self.config_manager.update_component_health("semantic_cache", "healthy")
                else:
                    logger.warning("Semantic cache using fallback mode")
                    self.config_manager.update_component_health(
                        "semantic_cache", "degraded", 
                        error_message="Redis not available, using fallback"
                    )
            else:
                self.semantic_cache = None
                logger.info("Semantic cache disabled")
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            self.semantic_cache = None
            self.config_manager.update_component_health(
                "semantic_cache", "error", error_message=str(e)
            )
        
        # Initialize hallucination detector
        try:
            if self.config.hallucination_detection.enabled:
                self.hallucination_detector = create_hallucination_detector()
                logger.info("Hallucination detector initialized successfully")
                self.config_manager.update_component_health("hallucination_detection", "healthy")
            else:
                self.hallucination_detector = None
                logger.info("Hallucination detection disabled")
        except Exception as e:
            logger.error(f"Failed to initialize hallucination detector: {e}")
            self.hallucination_detector = None
            self.config_manager.update_component_health(
                "hallucination_detection", "error", error_message=str(e)
            )
        
        # Initialize multimodal support
        try:
            if self.config.multimodal_support.enabled:
                from src.multimodal import MultimodalEmbedding
                self.multimodal_embedding = MultimodalEmbedding(
                    model_name=self.config.multimodal_support.settings["clip_model_name"]
                )
                logger.info("Multimodal support initialized successfully")
                self.config_manager.update_component_health("multimodal_support", "healthy")
            else:
                self.multimodal_embedding = None
                logger.info("Multimodal support disabled")
        except Exception as e:
            logger.error(f"Failed to initialize multimodal support: {e}")
            self.multimodal_embedding = None
            self.config_manager.update_component_health(
                "multimodal_support", "error", error_message=str(e)
            )
        
        # Initialize performance optimizer
        try:
            if self.config.performance_optimization.enabled:
                from src.performance import get_performance_optimizer
                self.performance_optimizer = get_performance_optimizer(
                    semantic_cache=self.semantic_cache
                )
                logger.info("Performance optimizer initialized successfully")
                self.config_manager.update_component_health("performance_optimization", "healthy")
            else:
                self.performance_optimizer = None
                logger.info("Performance optimization disabled")
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            self.performance_optimizer = None
            self.config_manager.update_component_health(
                "performance_optimization", "error", error_message=str(e)
            )
        
        # Initialize TTS integration
        try:
            if self.config.tts_integration.enabled:
                from src.tts import create_tts_engine
                self.tts_engine = create_tts_engine(
                    engine=self.config.tts_integration.settings["engine"]
                )
                logger.info("TTS integration initialized successfully")
                self.config_manager.update_component_health("tts_integration", "healthy")
            else:
                self.tts_engine = None
                logger.info("TTS integration disabled")
        except Exception as e:
            logger.error(f"Failed to initialize TTS integration: {e}")
            self.tts_engine = None
            self.config_manager.update_component_health(
                "tts_integration", "error", error_message=str(e)
            )
    
    @step
    async def analyze_query(self, ctx: Context, ev: StartEvent) -> Event:
        """Analyze the incoming query to determine characteristics and processing requirements."""
        start_time = time.time()
        
        try:
            # Extract query from various input formats
            query = self._extract_query(ev)
            if not query:
                return StopEvent(result="Error: Could not extract query from input")
            
            logger.info(f"Analyzing query: {query[:100]}...")
            
            # Analyze query characteristics
            characteristics = await self._analyze_query_characteristics(query)
            
            # Check cache first if enabled
            cache_result = None
            if self.semantic_cache and self.config_manager.is_feature_enabled("semantic_cache"):
                cache_result = self.semantic_cache.get(query)
                if cache_result:
                    self.stats['cache_hits'] += 1
                    logger.info("Query found in semantic cache")
                    
                    # Apply verification if enabled for cached results
                    if (self.hallucination_detector and 
                        self.config_manager.is_feature_enabled("hallucination_detection")):
                        cached_response, cached_nodes, similarity_score = cache_result
                        verified_result = await self._verify_cached_result(
                            query, cached_response, cached_nodes
                        )
                        return StopEvent(result=verified_result)
                    else:
                        return StopEvent(result=cache_result[0]["response"])
            
            # Create processing plan based on characteristics
            processing_plan = await self._create_processing_plan(characteristics)
            
            # Store context for next steps
            ctx.data["query"] = query
            ctx.data["characteristics"] = characteristics
            ctx.data["processing_plan"] = processing_plan
            ctx.data["start_time"] = start_time
            
            # Route to appropriate processing step
            return Event()
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            self.stats['failed_queries'] += 1
            return StopEvent(result=f"Error analyzing query: {str(e)}")
    
    @step
    async def execute_processing(self, ctx: Context, ev: Event) -> StopEvent:
        """Execute the processing plan with the selected components."""
        query = ctx.data["query"]
        characteristics = ctx.data["characteristics"]
        processing_plan = ctx.data["processing_plan"]
        start_time = ctx.data["start_time"]
        
        try:
            logger.info(f"Executing processing plan: {processing_plan}")
            
            # Execute query through selected components
            result = await self._execute_with_plan(query, characteristics, processing_plan)
            
            # Cache the result if caching is enabled
            if (self.semantic_cache and 
                self.config_manager.is_feature_enabled("semantic_cache") and
                result and len(str(result)) > 10):  # Only cache substantial results
                try:
                    # Create mock response for caching
                    class MockResponse:
                        def __init__(self, text):
                            self.response = text
                            self.source_nodes = []
                            self.metadata = {}
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.semantic_cache.put,
                        query, 
                        MockResponse(result), 
                        processing_plan.estimated_cost
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to cache result: {cache_error}")
            
            # Add TTS output if enabled
            if (processing_plan.use_tts_output and self.tts_engine and
                isinstance(result, str) and len(result.strip()) > 0):
                try:
                    audio_result = await self._add_tts_output(result)
                    if audio_result:
                        result = {"text": result, "audio": audio_result}
                except Exception as tts_error:
                    logger.warning(f"TTS output failed: {tts_error}")
                    # Continue with text-only result
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_queries'] += 1
            self.stats['successful_queries'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['total_cost'] += processing_plan.estimated_cost
            
            if processing_plan.use_agentic_workflow:
                self.stats['agentic_queries'] += 1
            if processing_plan.use_hallucination_detection:
                self.stats['verified_queries'] += 1
            if processing_plan.use_multimodal_support:
                self.stats['multimodal_queries'] += 1
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return StopEvent(result=result)
            
        except Exception as e:
            logger.error(f"Processing execution failed: {e}")
            
            # Attempt fallback processing
            fallback_result = await self._attempt_fallback_processing(
                query, characteristics, e
            )
            
            processing_time = time.time() - start_time
            self.stats['total_queries'] += 1
            self.stats['failed_queries'] += 1
            self.stats['fallback_activations'] += 1
            self.stats['total_processing_time'] += processing_time
            
            return StopEvent(result=fallback_result)
    
    def _extract_query(self, event: StartEvent) -> Optional[str]:
        """Extract query string from various event formats."""
        if hasattr(event, 'query'):
            return event.query
        elif hasattr(event, 'input'):
            return event.input
        elif hasattr(event, 'message'):
            return event.message
        else:
            # Try to get query from the first argument
            if hasattr(event, 'dict') and event.dict():
                values = list(event.dict().values())
                if values and isinstance(values[0], str):
                    return values[0]
        return None
    
    async def _analyze_query_characteristics(self, query: str) -> QueryCharacteristics:
        """Analyze query characteristics for intelligent routing decisions."""
        characteristics = QueryCharacteristics(original_query=query)
        
        # Basic analysis
        characteristics.estimated_tokens = len(query.split()) * 1.3  # Rough estimation
        
        # Complexity analysis
        complexity_indicators = [
            len(query.split()) > 20,  # Long queries
            '?' in query and query.count('?') > 1,  # Multiple questions
            any(word in query.lower() for word in ['compare', 'analyze', 'explain', 'why', 'how']),
            any(word in query.lower() for word in ['and', 'but', 'however', 'moreover']),  # Complex conjunctions
            len([c for c in query if c in '.,;:']) > 2  # Complex punctuation
        ]
        
        complexity_score = sum(complexity_indicators) / len(complexity_indicators)
        characteristics.complexity_score = complexity_score
        
        if complexity_score < 0.3:
            characteristics.complexity = QueryComplexity.SIMPLE
        elif complexity_score < 0.6:
            characteristics.complexity = QueryComplexity.MODERATE
        else:
            characteristics.complexity = QueryComplexity.COMPLEX
        
        # Determine if decomposition is needed
        characteristics.requires_decomposition = (
            characteristics.complexity in [QueryComplexity.COMPLEX] or
            complexity_score > self.config.agentic_workflow.settings.get("complexity_threshold", 0.7)
        )
        
        # Check for multimodal requirements
        characteristics.has_images = any(word in query.lower() for word in ['image', 'picture', 'photo', 'visual'])
        if characteristics.has_images:
            characteristics.complexity = QueryComplexity.MULTI_MODAL
            characteristics.modality = "mixed"
        
        # Intent analysis (simple heuristics)
        if any(word in query.lower() for word in ['what', 'who', 'where', 'when']):
            characteristics.intent = "informational"
        elif any(word in query.lower() for word in ['how', 'explain', 'describe']):
            characteristics.intent = "explanatory"
        elif any(word in query.lower() for word in ['compare', 'contrast', 'difference']):
            characteristics.intent = "comparative"
        else:
            characteristics.intent = "general"
        
        # Estimate cost
        characteristics.estimated_cost = self._estimate_query_cost(characteristics)
        
        logger.debug(f"Query characteristics: complexity={characteristics.complexity.value}, "
                    f"score={complexity_score:.2f}, decomposition={characteristics.requires_decomposition}")
        
        return characteristics
    
    async def _create_processing_plan(self, characteristics: QueryCharacteristics) -> ProcessingPlan:
        """Create an optimal processing plan based on query characteristics and system state."""
        plan = ProcessingPlan()
        
        # Get current system health
        system_health = self.config_manager.get_system_health()
        
        # Determine component usage based on characteristics and configuration
        
        # Agentic workflow decision
        plan.use_agentic_workflow = (
            self.config_manager.should_auto_enable_feature("agentic_workflow", {
                "complexity_score": characteristics.complexity_score,
                "query_type": characteristics.complexity.value
            }) and
            self.agentic_workflow is not None and
            system_health["features_enabled"]["agentic_workflow"]
        )
        
        # Semantic cache decision (already checked in analyze_query, but plan it for write)
        plan.use_semantic_cache = (
            self.config_manager.is_feature_enabled("semantic_cache") and
            self.semantic_cache is not None
        )
        
        # Hallucination detection decision
        plan.use_hallucination_detection = (
            self.config_manager.should_auto_enable_feature("hallucination_detection", {
                "complexity_score": characteristics.complexity_score,
                "query_type": characteristics.complexity.value
            }) and
            self.hallucination_detector is not None and
            system_health["features_enabled"]["hallucination_detection"]
        )
        
        # Multimodal support decision
        plan.use_multimodal_support = (
            characteristics.has_images and
            self.config_manager.is_feature_enabled("multimodal_support") and
            self.multimodal_embedding is not None
        )
        
        # Performance optimization decision
        plan.use_performance_optimization = (
            self.config_manager.is_feature_enabled("performance_optimization") and
            self.performance_optimizer is not None
        )
        
        # TTS output decision
        plan.use_tts_output = (
            self.config_manager.is_feature_enabled("tts_integration") and
            self.tts_engine is not None and
            characteristics.estimated_tokens > 20  # Only for substantial responses
        )
        
        # Configure component-specific settings based on performance profile
        profile = self.config.performance_profile
        
        if profile == PerformanceProfile.HIGH_ACCURACY:
            plan.confidence_threshold = 0.9
            plan.agentic_settings = {"enable_ensemble": True, "max_iterations": 3}
            plan.verification_settings = {"strict_mode": True, "ensemble_verification": True}
        elif profile == PerformanceProfile.SPEED:
            plan.confidence_threshold = 0.7
            plan.agentic_settings = {"enable_ensemble": False, "max_iterations": 1}
            plan.verification_settings = {"strict_mode": False, "timeout": 2.0}
        elif profile == PerformanceProfile.COST_OPTIMIZED:
            plan.confidence_threshold = 0.75
            plan.agentic_settings = {"use_cheaper_models": True}
            plan.verification_settings = {"use_cheaper_model": True, "batch_processing": True}
        else:  # BALANCED
            plan.confidence_threshold = 0.8
            plan.agentic_settings = {}
            plan.verification_settings = {}
        
        # Estimate processing cost and time
        plan.estimated_cost = self._estimate_processing_cost(characteristics, plan)
        plan.estimated_processing_time = self._estimate_processing_time(characteristics, plan)
        
        # Check cost constraints
        if plan.estimated_cost > self.config.cost_management["max_query_cost"]:
            logger.warning(f"Query cost ${plan.estimated_cost:.4f} exceeds limit, optimizing plan")
            plan = await self._optimize_plan_for_cost(plan, characteristics)
        
        return plan
    
    async def _execute_with_plan(self, query: str, characteristics: QueryCharacteristics, 
                                plan: ProcessingPlan) -> str:
        """Execute query processing with the specified plan."""
        
        # Choose the appropriate workflow
        if plan.use_agentic_workflow and self.agentic_workflow:
            logger.info("Using agentic workflow for processing")
            
            # Create event for agentic workflow
            class QueryEvent:
                def __init__(self, query_text):
                    self.query = query_text
                    self.input = query_text
                    self.message = query_text
                
                def dict(self):
                    return {"query": self.query}
            
            result = await self.agentic_workflow.arun(QueryEvent(query))
            
        elif self.base_workflow:
            logger.info("Using base workflow for processing")
            result = await self.base_workflow.arun(query)
            
        else:
            raise RuntimeError("No workflow available for processing")
        
        # Apply verification if enabled
        if plan.use_hallucination_detection and self.hallucination_detector:
            try:
                logger.info("Applying hallucination detection")
                verified_result = await self._verify_result(query, result, plan.verification_settings or {})
                result = verified_result
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                # Continue with unverified result
        
        return str(result)
    
    async def _verify_result(self, query: str, result: str, 
                           verification_settings: Dict[str, Any]) -> str:
        """Apply hallucination detection to the result."""
        try:
            from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore
            
            # Create QueryBundle
            query_bundle = QueryBundle(query_str=query)
            
            # Create mock nodes from result
            mock_nodes = [NodeWithScore(
                node=TextNode(text=result[:500], id_="generated_result"),
                score=0.9
            )]
            
            # Calculate confidence scores
            graph_confidence = self.hallucination_detector.calculate_graph_confidence(
                query_bundle, mock_nodes
            )
            response_confidence = self.hallucination_detector.calculate_response_confidence(
                result, graph_confidence, [], query_bundle
            )
            
            # Perform verification
            verification_result, updated_confidence = await self.hallucination_detector.verify_response(
                result, response_confidence, query_bundle, mock_nodes
            )
            
            # Handle verification results
            if verification_result.name == 'REJECTED':
                logger.warning(f"Result rejected by verification (confidence: {updated_confidence:.3f})")
                result += f"\n\n*Note: This response has been flagged as potentially unreliable. Please verify independently.*"
            elif verification_result.name == 'UNCERTAIN':
                logger.info(f"Result has uncertain verification (confidence: {updated_confidence:.3f})")
                result += f"\n\n*Note: The confidence in this response is moderate. Some information should be verified.*"
            
            return result
            
        except Exception as e:
            logger.error(f"Verification process failed: {e}")
            return result  # Return original result on verification failure
    
    async def _verify_cached_result(self, query: str, cached_response: Dict[str, Any], 
                                   cached_nodes: List[Dict[str, Any]]) -> str:
        """Verify a cached result."""
        try:
            response_text = cached_response.get("response", "")
            if not response_text:
                return "Cached result is empty"
            
            # Apply lightweight verification for cached results
            verified_result = await self._verify_result(query, response_text, {})
            return verified_result
            
        except Exception as e:
            logger.warning(f"Cached result verification failed: {e}")
            return cached_response.get("response", "Error retrieving cached result")
    
    async def _add_tts_output(self, text_result: str) -> Optional[str]:
        """Add TTS audio output to the result."""
        try:
            if not self.tts_engine:
                return None
            
            # Generate audio from text (implementation depends on TTS engine)
            audio_path = await self.tts_engine.text_to_speech(text_result)
            return audio_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    async def _attempt_fallback_processing(self, query: str, characteristics: QueryCharacteristics, 
                                         original_error: Exception) -> str:
        """Attempt fallback processing when main processing fails."""
        logger.warning(f"Attempting fallback processing due to: {original_error}")
        
        try:
            # Try simple base workflow without advanced features
            if self.base_workflow:
                logger.info("Trying base workflow fallback")
                result = await self.base_workflow.arun(query)
                return f"{str(result)}\n\n*Note: Response generated using simplified processing due to system issues.*"
            
        except Exception as fallback_error:
            logger.error(f"Fallback processing also failed: {fallback_error}")
        
        # Final fallback - return informative error message
        return f"""I apologize, but I encountered difficulties processing your query: "{query[:100]}..."

The system attempted multiple processing strategies but encountered errors:
1. Main processing: {str(original_error)[:100]}
2. Fallback processing: System unavailable

Please try:
- Rephrasing your question in simpler terms
- Breaking complex questions into smaller parts
- Checking back in a few moments if this is a temporary issue

If the problem persists, please contact support."""
    
    def _estimate_query_cost(self, characteristics: QueryCharacteristics) -> float:
        """Estimate the cost of processing a query."""
        base_cost = 0.01  # Base cost for simple queries
        
        # Adjust based on complexity
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 1.5,
            QueryComplexity.COMPLEX: 2.5,
            QueryComplexity.MULTI_MODAL: 3.0
        }
        
        estimated_cost = base_cost * complexity_multiplier[characteristics.complexity]
        
        # Adjust based on token count
        estimated_cost += (characteristics.estimated_tokens / 1000) * 0.002
        
        return round(estimated_cost, 6)
    
    def _estimate_processing_cost(self, characteristics: QueryCharacteristics, 
                                plan: ProcessingPlan) -> float:
        """Estimate the total cost of processing with the given plan."""
        base_cost = self._estimate_query_cost(characteristics)
        
        # Add component costs
        if plan.use_agentic_workflow:
            base_cost *= 2.0  # Agentic workflow uses more tokens
        
        if plan.use_hallucination_detection:
            base_cost += 0.005  # Verification cost
        
        if plan.use_multimodal_support:
            base_cost += 0.01  # CLIP processing cost
        
        return round(base_cost, 6)
    
    def _estimate_processing_time(self, characteristics: QueryCharacteristics, 
                                plan: ProcessingPlan) -> float:
        """Estimate the processing time for the given plan."""
        base_time = 1.0  # Base processing time in seconds
        
        # Adjust based on complexity
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 1.5,
            QueryComplexity.COMPLEX: 2.5,
            QueryComplexity.MULTI_MODAL: 3.0
        }
        
        estimated_time = base_time * complexity_multiplier[characteristics.complexity]
        
        # Add component processing times
        if plan.use_agentic_workflow:
            estimated_time += 2.0  # Additional time for decomposition and aggregation
        
        if plan.use_hallucination_detection:
            estimated_time += 0.5  # Verification time
        
        if plan.use_multimodal_support:
            estimated_time += 1.0  # CLIP processing time
        
        return estimated_time
    
    async def _optimize_plan_for_cost(self, plan: ProcessingPlan, 
                                    characteristics: QueryCharacteristics) -> ProcessingPlan:
        """Optimize the processing plan to meet cost constraints."""
        logger.info("Optimizing processing plan for cost constraints")
        
        # Disable expensive features if cost is too high
        if plan.estimated_cost > self.config.cost_management["max_query_cost"]:
            
            # First, try disabling multimodal support
            if plan.use_multimodal_support:
                plan.use_multimodal_support = False
                logger.info("Disabled multimodal support for cost optimization")
            
            # Then try using simpler verification
            if plan.use_hallucination_detection and plan.verification_settings:
                plan.verification_settings = {"use_cheaper_model": True, "timeout": 2.0}
                logger.info("Simplified verification for cost optimization")
            
            # Finally, try disabling agentic workflow for very high costs
            if plan.estimated_cost > self.config.cost_management["max_query_cost"] * 1.5:
                if plan.use_agentic_workflow:
                    plan.use_agentic_workflow = False
                    logger.info("Disabled agentic workflow for cost optimization")
        
        # Recalculate cost
        plan.estimated_cost = self._estimate_processing_cost(characteristics, plan)
        plan.estimated_processing_time = self._estimate_processing_time(characteristics, plan)
        
        return plan
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_queries']
            stats['avg_cost_per_query'] = stats['total_cost'] / stats['total_queries']
        else:
            stats.update({
                'success_rate': 0.0,
                'cache_hit_rate': 0.0,
                'avg_processing_time': 0.0,
                'avg_cost_per_query': 0.0
            })
        
        # Add system health
        stats['system_health'] = self.config_manager.get_system_health()
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'agentic_queries': 0,
            'verified_queries': 0,
            'multimodal_queries': 0,
            'fallback_activations': 0,
            'total_processing_time': 0.0,
            'total_cost': 0.0
        }
        logger.info("Processing statistics reset")


def create_unified_workflow() -> UnifiedWorkflow:
    """
    Create and initialize the unified workflow orchestrator.
    
    Returns:
        UnifiedWorkflow instance ready for deployment
    """
    try:
        # Initialize settings first
        init_settings()
        
        # Create unified workflow
        workflow = UnifiedWorkflow(timeout=300.0, verbose=False)
        
        logger.info("Unified workflow created successfully")
        return workflow
        
    except Exception as e:
        logger.error(f"Failed to create unified workflow: {e}")
        raise RuntimeError(f"Unified workflow creation failed: {str(e)}") from e