"""
Comprehensive Integration Tests for Enhanced RAG System Workflow Orchestration

This module provides comprehensive integration tests for the UnifiedWorkflow orchestrator,
focusing on:
1. End-to-End workflow orchestration from query to response
2. Component selection and coordination logic
3. Performance profile switching and validation
4. Error handling and graceful degradation
5. Real-world scenario testing

Tests validate both happy path and error scenarios with appropriate mocking for external services.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from src.unified_workflow import UnifiedWorkflow, QueryCharacteristics, QueryComplexity, ProcessingPlan
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.health_monitor import get_health_monitor


class TestUnifiedWorkflowOrchestration:
    """Test the unified workflow orchestrator with comprehensive scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_orchestration_simple_query(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test complete orchestration for a simple query from analysis to response."""
        
        # Setup mock responses
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 1536
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Python is a high-level programming language."
        
        # No cache hit initially
        mock_redis_client.get.return_value = None
        mock_redis_client.keys.return_value = []
        
        # Mock query engine
        mock_llama_index['engine'].query.return_value = Mock(
            response="Python is a high-level programming language created by Guido van Rossum.",
            source_nodes=[Mock(text="Python programming info", score=0.9, id_="node1")],
            metadata={'processing_time': 1.2}
        )
        
        # Create workflow
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock workflow methods to track orchestration
        with patch.object(workflow, '_analyze_query_characteristics') as mock_analyze, \
             patch.object(workflow, '_create_processing_plan') as mock_plan, \
             patch.object(workflow, '_execute_with_plan') as mock_execute:
            
            # Setup mock returns
            test_characteristics = QueryCharacteristics(
                original_query="What is Python?",
                complexity=QueryComplexity.SIMPLE,
                complexity_score=0.2,
                estimated_tokens=10,
                estimated_cost=0.01
            )
            mock_analyze.return_value = test_characteristics
            
            test_plan = ProcessingPlan(
                use_agentic_workflow=False,
                use_semantic_cache=True,
                use_hallucination_detection=True,
                estimated_cost=0.01,
                estimated_processing_time=2.0
            )
            mock_plan.return_value = test_plan
            
            mock_execute.return_value = "Python is a high-level programming language created by Guido van Rossum."
            
            # Execute workflow
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            result = await workflow.arun(MockStartEvent("What is Python?"))
            
            # Validate orchestration flow
            mock_analyze.assert_called_once()
            mock_plan.assert_called_once_with(test_characteristics)
            mock_execute.assert_called_once()
            
            assert result is not None
            assert "Python" in str(result)
    
    @pytest.mark.asyncio
    async def test_component_selection_logic_complex_query(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test component selection logic for complex queries requiring agentic processing."""
        
        workflow = UnifiedWorkflow(timeout=60.0)
        
        # Complex query that should trigger agentic workflow
        complex_query = "Compare supervised and unsupervised learning algorithms, provide examples of each, and explain their use cases in different industries"
        
        # Test query analysis
        characteristics = await workflow._analyze_query_characteristics(complex_query)
        
        assert characteristics.complexity in [QueryComplexity.COMPLEX]
        assert characteristics.requires_decomposition == True
        assert characteristics.complexity_score > 0.6
        
        # Test processing plan creation
        plan = await workflow._create_processing_plan(characteristics)
        
        # Complex query should enable appropriate components
        assert plan.use_hallucination_detection == True  # Always verify complex responses
        
        # Plan should have reasonable cost and time estimates
        assert plan.estimated_cost > 0.01  # More expensive than simple queries
        assert plan.estimated_processing_time > 1.0  # Takes more time
    
    @pytest.mark.asyncio
    async def test_performance_profile_switching(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test dynamic performance profile switching and component adaptation."""
        
        query = "Explain machine learning algorithms"
        
        profiles_to_test = [
            PerformanceProfile.HIGH_ACCURACY,
            PerformanceProfile.SPEED,
            PerformanceProfile.COST_OPTIMIZED,
            PerformanceProfile.BALANCED
        ]
        
        for profile in profiles_to_test:
            # Reset and configure
            reset_unified_config()
            config_manager = get_unified_config()
            config_manager.config.performance_profile = profile
            
            workflow = UnifiedWorkflow(timeout=30.0)
            
            # Analyze query with different profiles
            characteristics = await workflow._analyze_query_characteristics(query)
            plan = await workflow._create_processing_plan(characteristics)
            
            # Validate profile-specific behavior
            if profile == PerformanceProfile.HIGH_ACCURACY:
                assert plan.confidence_threshold >= 0.9
                assert plan.verification_settings.get("strict_mode", False) == True
            
            elif profile == PerformanceProfile.SPEED:
                assert plan.confidence_threshold <= 0.75
                assert plan.verification_settings.get("timeout", 10) <= 3.0
            
            elif profile == PerformanceProfile.COST_OPTIMIZED:
                assert plan.verification_settings.get("use_cheaper_model", False) == True
                # Should prefer cache usage
                assert plan.use_semantic_cache == True
    
    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test error handling and graceful degradation scenarios."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        query = "What is artificial intelligence?"
        
        # Test case 1: Cache failure but workflow continues
        mock_redis_client.get.side_effect = Exception("Redis connection failed")
        
        with patch.object(workflow, '_execute_with_plan') as mock_execute:
            mock_execute.return_value = "AI is a field of computer science focused on creating intelligent machines."
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            result = await workflow.arun(MockStartEvent(query))
            
            # Should still get a response despite cache failure
            assert result is not None
            assert "AI" in str(result)
            mock_execute.assert_called_once()
        
        # Reset mock
        mock_redis_client.get.side_effect = None
        
        # Test case 2: Main processing failure triggers fallback
        with patch.object(workflow, '_execute_with_plan') as mock_execute, \
             patch.object(workflow, '_attempt_fallback_processing') as mock_fallback:
            
            mock_execute.side_effect = Exception("Main processing failed")
            mock_fallback.return_value = "Fallback response about AI concepts."
            
            result = await workflow.arun(MockStartEvent(query))
            
            # Should use fallback processing
            mock_fallback.assert_called_once()
            assert "Fallback" in str(result)
    
    @pytest.mark.asyncio
    async def test_cost_optimization_logic(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test cost optimization logic when queries exceed budget limits."""
        
        # Configure strict cost limits
        reset_unified_config()
        config_manager = get_unified_config()
        config_manager.config.cost_management["max_query_cost"] = 0.02  # Very low limit
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # High-cost query
        expensive_query = "Provide a comprehensive analysis of all machine learning algorithms with detailed examples, comparisons, and implementation guidelines"
        
        characteristics = await workflow._analyze_query_characteristics(expensive_query)
        original_plan = await workflow._create_processing_plan(characteristics)
        
        # Plan should be optimized for cost
        optimized_plan = await workflow._optimize_plan_for_cost(original_plan, characteristics)
        
        # Cost should be reduced
        assert optimized_plan.estimated_cost <= original_plan.estimated_cost
        assert optimized_plan.estimated_cost <= config_manager.config.cost_management["max_query_cost"]
        
        # Expensive features should be disabled
        if original_plan.use_multimodal_support and optimized_plan.estimated_cost < original_plan.estimated_cost:
            assert optimized_plan.use_multimodal_support == False
    
    @pytest.mark.asyncio
    async def test_statistics_tracking_throughout_orchestration(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test that statistics are properly tracked throughout orchestration."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Initial stats should be zero
        initial_stats = workflow.get_stats()
        assert initial_stats['total_queries'] == 0
        assert initial_stats['successful_queries'] == 0
        
        # Mock successful processing
        mock_redis_client.get.return_value = None  # Cache miss
        
        with patch.object(workflow, '_execute_with_plan') as mock_execute:
            mock_execute.return_value = "Test response"
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            # Process multiple queries
            queries = ["Query 1", "Query 2", "Query 3"]
            
            for query in queries:
                await workflow.arun(MockStartEvent(query))
            
            # Check updated stats
            updated_stats = workflow.get_stats()
            
            assert updated_stats['total_queries'] == len(queries)
            assert updated_stats['successful_queries'] == len(queries)
            assert updated_stats['success_rate'] == 1.0
            assert updated_stats['total_processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_multimodal_query_orchestration(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test orchestration of multimodal queries with visual content."""
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        multimodal_query = "Show me a diagram of neural network architecture and explain how backpropagation works"
        
        # Test query analysis for multimodal content
        characteristics = await workflow._analyze_query_characteristics(multimodal_query)
        
        assert characteristics.has_images == True
        assert characteristics.complexity == QueryComplexity.MULTI_MODAL
        assert characteristics.modality == "mixed"
        
        # Test processing plan for multimodal
        plan = await workflow._create_processing_plan(characteristics)
        
        # Should enable multimodal support if available
        if workflow.multimodal_embedding:
            assert plan.use_multimodal_support == True
        
        # Multimodal should have higher cost estimates
        assert plan.estimated_cost > 0.02
        assert plan.estimated_processing_time > 2.0
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of concurrent queries through the orchestrator."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock processing with delays to simulate real processing
        async def mock_execute_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay
            return f"Response for query"
        
        with patch.object(workflow, '_execute_with_plan', side_effect=mock_execute_with_delay):
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            # Process multiple queries concurrently
            queries = [f"Query {i}" for i in range(5)]
            start_time = time.time()
            
            tasks = [workflow.arun(MockStartEvent(q)) for q in queries]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # All queries should complete successfully
            assert len(results) == len(queries)
            assert all("Response" in str(r) for r in results)
            
            # Should be faster than sequential processing
            assert end_time - start_time < len(queries) * 0.1 * 1.5  # Some concurrency benefit
            
            # Stats should reflect all queries
            stats = workflow.get_stats()
            assert stats['total_queries'] == len(queries)


class TestWorkflowStateManagement:
    """Test workflow state management and context handling."""
    
    @pytest.mark.asyncio
    async def test_context_data_flow_through_steps(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test that context data flows correctly through workflow steps."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock the step methods to inspect context
        original_execute = workflow.execute_processing
        
        context_data_captured = {}
        
        async def mock_execute_processing(ctx, ev):
            # Capture context data
            context_data_captured.update(ctx.data)
            return await original_execute(ctx, ev)
        
        with patch.object(workflow, 'execute_processing', side_effect=mock_execute_processing), \
             patch.object(workflow, '_execute_with_plan') as mock_execute:
            
            mock_execute.return_value = "Test response"
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            await workflow.arun(MockStartEvent("Test query"))
            
            # Validate context data was properly passed
            assert 'query' in context_data_captured
            assert 'characteristics' in context_data_captured
            assert 'processing_plan' in context_data_captured
            assert 'start_time' in context_data_captured
            
            # Validate data types
            assert isinstance(context_data_captured['characteristics'], QueryCharacteristics)
            assert isinstance(context_data_captured['processing_plan'], ProcessingPlan)
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test workflow timeout handling and cleanup."""
        
        # Create workflow with short timeout
        workflow = UnifiedWorkflow(timeout=1.0)
        
        # Mock long-running processing
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return "This should timeout"
        
        with patch.object(workflow, '_execute_with_plan', side_effect=slow_execute):
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            # Should handle timeout gracefully
            start_time = time.time()
            
            try:
                result = await workflow.arun(MockStartEvent("Test query"))
                # If it completes, check it did so quickly (fallback path)
                elapsed = time.time() - start_time
                assert elapsed < 5.0  # Should fail fast, not hang
                
            except asyncio.TimeoutError:
                # Timeout is acceptable behavior
                elapsed = time.time() - start_time
                assert elapsed <= 2.0  # Should timeout around the limit


class TestWorkflowComponentCoordination:
    """Test coordination between different workflow components."""
    
    @pytest.mark.asyncio
    async def test_agentic_workflow_coordination(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test coordination with agentic workflow component."""
        
        workflow = UnifiedWorkflow(timeout=60.0)
        
        # Mock agentic workflow
        mock_agentic = Mock()
        mock_agentic.arun = AsyncMock(return_value="Agentic response")
        workflow.agentic_workflow = mock_agentic
        
        complex_query = "Analyze the impact of artificial intelligence on various industries and provide detailed examples"
        
        characteristics = QueryCharacteristics(
            original_query=complex_query,
            complexity=QueryComplexity.COMPLEX,
            requires_decomposition=True
        )
        
        plan = ProcessingPlan(
            use_agentic_workflow=True,
            use_hallucination_detection=True
        )
        
        # Test execution with agentic workflow
        result = await workflow._execute_with_plan(complex_query, characteristics, plan)
        
        # Should have used agentic workflow
        mock_agentic.arun.assert_called_once()
        assert result == "Agentic response"
    
    @pytest.mark.asyncio
    async def test_verification_system_coordination(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test coordination with verification system."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock verification system
        mock_detector = Mock()
        workflow.hallucination_detector = mock_detector
        
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        characteristics = QueryCharacteristics(original_query=query)
        plan = ProcessingPlan(use_hallucination_detection=True)
        
        # Mock base workflow response
        mock_base = Mock()
        mock_base.arun = AsyncMock(return_value=response)
        workflow.base_workflow = mock_base
        
        with patch.object(workflow, '_verify_result') as mock_verify:
            mock_verify.return_value = f"{response}\n\n*Note: This response has been verified.*"
            
            result = await workflow._execute_with_plan(query, characteristics, plan)
            
            # Should have verified the response
            mock_verify.assert_called_once_with(query, response, {})
            assert "verified" in result.lower()
    
    @pytest.mark.asyncio
    async def test_cache_system_coordination(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test coordination with semantic cache system."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock cache with hit
        cached_response = ("Cached answer about AI", [], 0.9)
        
        mock_cache = Mock()
        mock_cache.get.return_value = cached_response
        workflow.semantic_cache = mock_cache
        
        query = "What is AI?"
        
        class MockStartEvent:
            def __init__(self, query):
                self.query = query
                
            def dict(self):
                return {"query": self.query}
        
        # Test with verification enabled for cached results
        mock_detector = Mock()
        workflow.hallucination_detector = mock_detector
        
        with patch.object(workflow, '_verify_cached_result') as mock_verify_cached:
            mock_verify_cached.return_value = "Verified cached answer"
            
            result = await workflow.arun(MockStartEvent(query))
            
            # Should use cached result with verification
            assert workflow.stats['cache_hits'] > 0
            mock_verify_cached.assert_called_once()


class TestWorkflowRealWorldScenarios:
    """Test workflow with real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_research_query_scenario(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of research-style queries with multiple information needs."""
        
        research_query = "What are the latest developments in quantum computing, how do they compare to classical computing, and what are the potential applications in cryptography?"
        
        workflow = UnifiedWorkflow(timeout=90.0)
        
        # This should trigger complex processing
        characteristics = await workflow._analyze_query_characteristics(research_query)
        plan = await workflow._create_processing_plan(characteristics)
        
        # Should be classified as complex
        assert characteristics.complexity in [QueryComplexity.COMPLEX]
        assert characteristics.requires_decomposition == True
        
        # Should enable comprehensive processing
        assert plan.use_hallucination_detection == True
        assert plan.estimated_cost > 0.02  # Research queries are expensive
    
    @pytest.mark.asyncio
    async def test_educational_query_scenario(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of educational/explanatory queries."""
        
        educational_query = "Can you explain how photosynthesis works in simple terms?"
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        characteristics = await workflow._analyze_query_characteristics(educational_query)
        plan = await workflow._create_processing_plan(characteristics)
        
        # Should be moderate complexity
        assert characteristics.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        assert characteristics.intent == "explanatory"
        
        # Should prioritize clarity over advanced features
        assert plan.confidence_threshold <= 0.85  # Don't need ultra-high confidence
    
    @pytest.mark.asyncio
    async def test_troubleshooting_query_scenario(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of troubleshooting/problem-solving queries."""
        
        troubleshooting_query = "My Python code is throwing a 'KeyError: missing key' exception. How can I debug and fix this?"
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        characteristics = await workflow._analyze_query_characteristics(troubleshooting_query)
        plan = await workflow._create_processing_plan(characteristics)
        
        # Should enable verification for technical accuracy
        assert plan.use_hallucination_detection == True
        
        # Should be reasonably fast for user experience
        assert plan.estimated_processing_time < 10.0
    
    @pytest.mark.asyncio
    async def test_conversational_followup_scenario(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of conversational follow-up queries."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Simulate conversation context
        initial_query = "What is machine learning?"
        followup_query = "Can you give me some examples?"
        
        # Process initial query
        with patch.object(workflow, '_execute_with_plan') as mock_execute:
            mock_execute.return_value = "Machine learning is a subset of AI..."
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            await workflow.arun(MockStartEvent(initial_query))
            
            # Process follow-up
            followup_characteristics = await workflow._analyze_query_characteristics(followup_query)
            
            # Follow-up should be classified appropriately
            assert followup_characteristics.intent in ["informational", "general"]
            
            # Should be processed efficiently (could use cache or simplified processing)
            followup_plan = await workflow._create_processing_plan(followup_characteristics)
            assert followup_plan.estimated_processing_time < 5.0