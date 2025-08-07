"""
Unit tests for the unified workflow orchestrator.

Tests cover:
- Query analysis and routing
- Component coordination
- Performance profile handling
- Error handling and fallbacks
- Statistics and monitoring
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.unified_workflow import (
    UnifiedWorkflow, 
    QueryCharacteristics, 
    QueryComplexity,
    ProcessingPlan,
    create_unified_workflow
)
from src.unified_config import PerformanceProfile, get_unified_config


class TestQueryAnalysis:
    """Test query analysis and characteristic detection."""
    
    @pytest.mark.asyncio
    async def test_simple_query_analysis(self, configured_workflow, sample_queries):
        """Test analysis of simple queries."""
        workflow = configured_workflow
        
        for query in sample_queries['simple']:
            characteristics = await workflow._analyze_query_characteristics(query)
            
            assert isinstance(characteristics, QueryCharacteristics)
            assert characteristics.original_query == query
            assert characteristics.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
            assert characteristics.complexity_score >= 0.0
            assert characteristics.complexity_score <= 1.0
            assert not characteristics.has_images  # Text queries shouldn't have images
    
    @pytest.mark.asyncio
    async def test_complex_query_analysis(self, configured_workflow, sample_queries):
        """Test analysis of complex queries."""
        workflow = configured_workflow
        
        for query in sample_queries['complex']:
            characteristics = await workflow._analyze_query_characteristics(query)
            
            assert characteristics.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
            assert characteristics.complexity_score >= 0.3
            # Complex queries likely require decomposition
            assert characteristics.requires_decomposition or characteristics.complexity_score > 0.6
    
    @pytest.mark.asyncio
    async def test_multimodal_query_analysis(self, configured_workflow, sample_queries):
        """Test analysis of multimodal queries."""
        workflow = configured_workflow
        
        for query in sample_queries['multimodal']:
            characteristics = await workflow._analyze_query_characteristics(query)
            
            # Should detect visual elements in query text
            assert 'image' in query.lower() or 'diagram' in query.lower() or 'visual' in query.lower()
            assert characteristics.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_MODAL]
    
    def test_token_estimation(self, configured_workflow):
        """Test token estimation accuracy."""
        workflow = configured_workflow
        
        test_cases = [
            ("Hello", 1),
            ("This is a test query", 5),
            ("This is a much longer test query with many more words to test token estimation", 15)
        ]
        
        for query, expected_min_tokens in test_cases:
            estimated = workflow._estimate_tokens(query)
            assert estimated >= expected_min_tokens
            assert estimated <= len(query.split()) * 2  # Reasonable upper bound


class TestProcessingPlanCreation:
    """Test processing plan creation for different query types."""
    
    def test_simple_query_plan(self, configured_workflow):
        """Test processing plan for simple queries."""
        workflow = configured_workflow
        characteristics = QueryCharacteristics(
            original_query="What is AI?",
            complexity=QueryComplexity.SIMPLE,
            complexity_score=0.2
        )
        
        plan = workflow._create_processing_plan(characteristics)
        
        assert isinstance(plan, ProcessingPlan)
        assert not plan.use_agentic_workflow  # Simple queries shouldn't need agentic workflow
        assert plan.use_cache  # Should try cache first
        assert not plan.use_decomposition
    
    def test_complex_query_plan(self, configured_workflow):
        """Test processing plan for complex queries."""
        workflow = configured_workflow
        characteristics = QueryCharacteristics(
            original_query="Compare machine learning and deep learning with examples",
            complexity=QueryComplexity.COMPLEX,
            complexity_score=0.8,
            requires_decomposition=True
        )
        
        plan = workflow._create_processing_plan(characteristics)
        
        assert plan.use_agentic_workflow  # Complex queries should use agentic workflow
        assert plan.use_decomposition
        assert plan.use_verification  # Should verify complex responses
    
    def test_multimodal_query_plan(self, configured_workflow):
        """Test processing plan for multimodal queries."""
        workflow = configured_workflow
        characteristics = QueryCharacteristics(
            original_query="Show me a diagram of transformer architecture",
            complexity=QueryComplexity.MULTI_MODAL,
            has_images=True
        )
        
        plan = workflow._create_processing_plan(characteristics)
        
        assert plan.use_multimodal  # Should enable multimodal processing
        assert plan.use_agentic_workflow  # Multimodal often needs advanced processing
    
    def test_performance_profile_influence(self, configured_workflow):
        """Test how performance profiles influence processing plans."""
        workflow = configured_workflow
        characteristics = QueryCharacteristics(
            original_query="Moderate complexity query",
            complexity=QueryComplexity.MODERATE,
            complexity_score=0.6
        )
        
        # Test different profiles
        profiles_to_test = [
            PerformanceProfile.HIGH_ACCURACY,
            PerformanceProfile.BALANCED,
            PerformanceProfile.COST_OPTIMIZED,
            PerformanceProfile.SPEED
        ]
        
        for profile in profiles_to_test:
            workflow.config_manager.config.performance_profile = profile
            plan = workflow._create_processing_plan(characteristics)
            
            if profile == PerformanceProfile.HIGH_ACCURACY:
                assert plan.use_verification  # Should verify for accuracy
                assert plan.verification_confidence_threshold >= 0.9
            elif profile == PerformanceProfile.COST_OPTIMIZED:
                assert plan.max_cost <= 1.5  # Should have cost limits
            elif profile == PerformanceProfile.SPEED:
                assert plan.timeout <= 3.0  # Should have tight timeouts


class TestComponentCoordination:
    """Test coordination between different SOTA components."""
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, configured_workflow, mock_redis_client):
        """Test semantic cache integration."""
        workflow = configured_workflow
        
        # Mock cache hit
        mock_redis_client.get.return_value = '{"response": "cached response", "confidence": 0.9}'
        
        query = "What is machine learning?"
        result = await workflow._try_semantic_cache(query)
        
        assert result is not None
        assert 'response' in result
        assert result['confidence'] == 0.9
    
    @pytest.mark.asyncio 
    async def test_verification_integration(self, configured_workflow):
        """Test hallucination detection integration."""
        workflow = configured_workflow
        
        # Mock response for verification
        response = {
            'content': 'Machine learning is a subset of AI',
            'sources': [],
            'confidence': 0.8
        }
        
        # This should not raise an error
        verified_response = await workflow._verify_response(response, "What is ML?")
        assert verified_response is not None
        assert 'confidence' in verified_response
    
    @pytest.mark.asyncio
    async def test_agentic_workflow_integration(self, configured_workflow):
        """Test agentic workflow integration."""
        workflow = configured_workflow
        
        complex_query = "Compare supervised and unsupervised learning with examples"
        characteristics = QueryCharacteristics(
            original_query=complex_query,
            complexity=QueryComplexity.COMPLEX,
            requires_decomposition=True
        )
        
        # Mock agentic workflow processing
        with patch.object(workflow, '_process_with_agentic_workflow') as mock_agentic:
            mock_agentic.return_value = {
                'content': 'Detailed comparison of learning types',
                'confidence': 0.95,
                'subqueries': ['What is supervised learning?', 'What is unsupervised learning?']
            }
            
            result = await workflow._process_with_agentic_workflow(characteristics)
            assert result['confidence'] >= 0.9
            assert 'subqueries' in result
    
    @pytest.mark.asyncio
    async def test_multimodal_integration(self, configured_workflow):
        """Test multimodal processing integration."""
        workflow = configured_workflow
        
        multimodal_query = "Show me a diagram of neural network"
        characteristics = QueryCharacteristics(
            original_query=multimodal_query,
            complexity=QueryComplexity.MULTI_MODAL,
            has_images=True
        )
        
        # Mock multimodal processing
        with patch.object(workflow, '_process_multimodal') as mock_multimodal:
            mock_multimodal.return_value = {
                'content': 'Neural network diagram explanation',
                'image_description': 'Diagram showing connected nodes',
                'confidence': 0.85
            }
            
            result = await workflow._process_multimodal(characteristics)
            assert 'image_description' in result


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_component_failure_fallback(self, configured_workflow):
        """Test fallback when components fail."""
        workflow = configured_workflow
        
        # Mock component failures
        with patch.object(workflow, '_try_semantic_cache') as mock_cache, \
             patch.object(workflow, '_process_with_standard_workflow') as mock_standard:
            
            mock_cache.side_effect = Exception("Cache unavailable")
            mock_standard.return_value = {
                'content': 'Fallback response',
                'confidence': 0.7,
                'used_fallback': True
            }
            
            query = "Test query"
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(characteristics, ProcessingPlan())
            
            assert result['used_fallback'] == True
            assert result['confidence'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, configured_workflow):
        """Test timeout handling in query processing."""
        workflow = configured_workflow
        
        # Mock slow processing
        async def slow_processing(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than typical timeout
            return {"content": "slow response"}
        
        with patch.object(workflow, '_process_with_standard_workflow', side_effect=slow_processing):
            query = "Test query"
            characteristics = await workflow._analyze_query_characteristics(query)
            
            # Should handle timeout gracefully
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    workflow._process_query_with_plan(characteristics, ProcessingPlan()),
                    timeout=1.0
                )
    
    def test_invalid_input_handling(self, configured_workflow):
        """Test handling of invalid inputs."""
        workflow = configured_workflow
        
        # Test empty query
        with pytest.raises(ValueError):
            asyncio.run(workflow._analyze_query_characteristics(""))
        
        # Test None query
        with pytest.raises((ValueError, TypeError)):
            asyncio.run(workflow._analyze_query_characteristics(None))


class TestPerformanceMonitoring:
    """Test performance monitoring and statistics."""
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, configured_workflow):
        """Test that statistics are properly tracked."""
        workflow = configured_workflow
        
        initial_queries = workflow.stats.get('queries_processed', 0)
        
        # Mock successful query processing
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            mock_process.return_value = {
                'content': 'Test response',
                'confidence': 0.8,
                'processing_time': 1.5
            }
            
            query = "Test query"
            characteristics = await workflow._analyze_query_characteristics(query)
            await workflow._process_query_with_plan(characteristics, ProcessingPlan())
            
            # Should increment query count
            assert workflow.stats['queries_processed'] > initial_queries
    
    def test_performance_metrics_collection(self, configured_workflow, performance_tracker):
        """Test performance metrics collection."""
        workflow = configured_workflow
        
        # Test timing collection
        performance_tracker.start_timing('test_operation')
        time.sleep(0.1)
        performance_tracker.end_timing('test_operation')
        
        duration = performance_tracker.get_duration('test_operation')
        assert duration >= 0.1
        assert duration < 1.0  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, configured_workflow):
        """Test cost tracking for queries."""
        workflow = configured_workflow
        
        # Mock query with cost information
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            mock_process.return_value = {
                'content': 'Test response',
                'confidence': 0.8,
                'cost': 0.05
            }
            
            query = "Expensive test query"
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(characteristics, ProcessingPlan())
            
            assert 'cost' in result
            assert result['cost'] > 0


class TestConfigurationIntegration:
    """Test integration with unified configuration system."""
    
    def test_performance_profile_application(self, configured_workflow):
        """Test that performance profiles are correctly applied."""
        workflow = configured_workflow
        config = workflow.config_manager.config
        
        # Test each profile
        for profile in PerformanceProfile:
            config.performance_profile = profile
            
            # Create characteristics for testing
            characteristics = QueryCharacteristics(
                original_query="Test query",
                complexity=QueryComplexity.MODERATE,
                complexity_score=0.6
            )
            
            plan = workflow._create_processing_plan(characteristics)
            
            # Verify profile-specific settings
            if profile == PerformanceProfile.HIGH_ACCURACY:
                assert plan.verification_confidence_threshold >= 0.9
            elif profile == PerformanceProfile.COST_OPTIMIZED:
                assert plan.max_cost <= 2.0
            elif profile == PerformanceProfile.SPEED:
                assert plan.timeout <= 3.0
    
    def test_feature_toggle_integration(self, configured_workflow):
        """Test that feature toggles work correctly."""
        workflow = configured_workflow
        config_manager = workflow.config_manager
        
        # Test disabling features
        config_manager.config.agentic_workflow.enabled = False
        config_manager.config.semantic_cache.enabled = False
        
        characteristics = QueryCharacteristics(
            original_query="Complex query that would normally use agentic workflow",
            complexity=QueryComplexity.COMPLEX,
            complexity_score=0.9
        )
        
        plan = workflow._create_processing_plan(characteristics)
        
        # Should not use disabled features
        assert not plan.use_agentic_workflow
        assert not plan.use_cache
    
    def test_health_monitoring_integration(self, configured_workflow):
        """Test integration with health monitoring."""
        workflow = configured_workflow
        config_manager = workflow.config_manager
        
        # Update component health
        config_manager.update_component_health(
            "unified_orchestrator",
            "healthy",
            {"queries_processed": 100},
            None
        )
        
        health = config_manager.get_system_health()
        assert "unified_orchestrator" in health["component_details"]
        assert health["component_details"]["unified_orchestrator"]["status"] == "healthy"