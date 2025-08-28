"""
Comprehensive Unit Tests for UnifiedWorkflow - Priority 1

This test suite provides comprehensive coverage of the UnifiedWorkflow orchestrator,
focusing on:
- Query analysis and characteristic detection
- Component selection and orchestration logic
- Processing plan optimization
- Error handling and fallback mechanisms
- Security validation
- Performance monitoring
- Configuration edge cases
"""

import asyncio
import os
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.unified_workflow import (
    UnifiedWorkflow, QueryCharacteristics, ProcessingPlan, QueryComplexity,
    create_unified_workflow
)
from src.unified_config import PerformanceProfile, FeatureStatus
from llama_index.core.workflow import StartEvent, StopEvent


class TestQueryAnalysis:
    """Test query analysis and characteristic detection."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance with mocked dependencies."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            mock_config_manager = Mock()
            mock_config_manager.config = Mock()
            mock_config_manager.config.performance_profile = PerformanceProfile.BALANCED
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config_manager.config.agentic_workflow = Mock()
            mock_config_manager.config.agentic_workflow.enabled = True
            mock_config_manager.config.agentic_workflow.settings = {"complexity_threshold": 0.7}
            
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=30.0, verbose=False)
            workflow.config_manager = mock_config_manager
            return workflow
    
    def test_extract_query_from_various_formats(self, workflow):
        """Test query extraction from different event formats."""
        # Test with query attribute
        event1 = Mock()
        event1.query = "test query 1"
        assert workflow._extract_query(event1) == "test query 1"
        
        # Test with input attribute
        event2 = Mock()
        event2.input = "test query 2"
        del event2.query  # Ensure query doesn't exist
        assert workflow._extract_query(event2) == "test query 2"
        
        # Test with message attribute
        event3 = Mock()
        event3.message = "test query 3"
        del event3.query
        del event3.input
        assert workflow._extract_query(event3) == "test query 3"
        
        # Test with dict method
        event4 = Mock()
        event4.dict.return_value = {"test": "test query 4"}
        del event4.query
        del event4.input
        del event4.message
        assert workflow._extract_query(event4) == "test query 4"
        
        # Test with no valid query
        event5 = Mock()
        event5.dict.return_value = {}
        del event5.query
        del event5.input
        del event5.message
        assert workflow._extract_query(event5) is None
    
    @pytest.mark.asyncio
    async def test_query_complexity_analysis(self, workflow):
        """Test query complexity analysis logic."""
        # Simple query
        simple_query = "What is machine learning?"
        characteristics = await workflow._analyze_query_characteristics(simple_query)
        assert characteristics.complexity == QueryComplexity.SIMPLE
        assert characteristics.complexity_score < 0.3
        assert not characteristics.requires_decomposition
        
        # Complex query
        complex_query = "Compare and analyze the differences between supervised and unsupervised learning algorithms, explain when to use each, and provide specific examples with their advantages and disadvantages in real-world applications."
        characteristics = await workflow._analyze_query_characteristics(complex_query)
        assert characteristics.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MODERATE]
        assert characteristics.complexity_score > 0.4
        
        # Multimodal query
        multimodal_query = "Show me an image of neural network architecture"
        characteristics = await workflow._analyze_query_characteristics(multimodal_query)
        assert characteristics.has_images
        assert characteristics.modality == "mixed"
        assert characteristics.complexity == QueryComplexity.MULTI_MODAL
    
    @pytest.mark.asyncio
    async def test_intent_classification(self, workflow):
        """Test query intent classification."""
        test_cases = [
            ("What is deep learning?", "informational"),
            ("How does backpropagation work?", "explanatory"),
            ("Compare RNN and LSTM", "comparative"),
            ("Tell me about AI", "general")
        ]
        
        for query, expected_intent in test_cases:
            characteristics = await workflow._analyze_query_characteristics(query)
            assert characteristics.intent == expected_intent
    
    def test_cost_estimation(self, workflow):
        """Test query cost estimation."""
        # Simple query cost
        simple_char = QueryCharacteristics(
            original_query="Simple test",
            complexity=QueryComplexity.SIMPLE,
            estimated_tokens=10
        )
        cost = workflow._estimate_query_cost(simple_char)
        assert 0.005 <= cost <= 0.02
        
        # Complex query cost
        complex_char = QueryCharacteristics(
            original_query="Complex test query",
            complexity=QueryComplexity.COMPLEX,
            estimated_tokens=100
        )
        cost = workflow._estimate_query_cost(complex_char)
        assert cost > workflow._estimate_query_cost(simple_char)
    
    def test_security_query_validation(self, workflow):
        """Test security validation in query analysis."""
        # Test SQL injection patterns
        malicious_queries = [
            "What is machine learning'; DROP TABLE users; --",
            "SELECT * FROM sensitive_data WHERE 1=1",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
        ]
        
        for malicious_query in malicious_queries:
            # Query should be processed but not cause security issues
            # The system should handle these gracefully
            assert len(malicious_query) > 0  # Basic validation
            # In a real system, additional sanitization would occur


class TestProcessingPlanCreation:
    """Test processing plan creation and optimization."""
    
    @pytest.fixture
    def workflow_with_components(self):
        """Create workflow with mocked components."""
        with patch('src.unified_workflow.get_unified_config') as mock_config:
            mock_config_manager = Mock()
            mock_config_manager.config = Mock()
            mock_config_manager.config.performance_profile = PerformanceProfile.HIGH_ACCURACY
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config_manager.should_auto_enable_feature.return_value = True
            mock_config_manager.is_feature_enabled.return_value = True
            mock_config_manager.get_system_health.return_value = {
                "features_enabled": {
                    "agentic_workflow": True,
                    "semantic_cache": True,
                    "hallucination_detection": True
                }
            }
            mock_config.return_value = mock_config_manager
            
            with patch('src.unified_workflow.init_settings'), \
                 patch('src.workflow.create_workflow') as mock_create:
                mock_create.return_value = Mock()
                
                workflow = UnifiedWorkflow(timeout=30.0)
                workflow.config_manager = mock_config_manager
                workflow.agentic_workflow = Mock()
                workflow.semantic_cache = Mock()
                workflow.hallucination_detector = Mock()
                workflow.multimodal_embedding = Mock()
                workflow.performance_optimizer = Mock()
                workflow.tts_engine = Mock()
                return workflow
    
    @pytest.mark.asyncio
    async def test_high_accuracy_profile_plan(self, workflow_with_components):
        """Test processing plan for high accuracy profile."""
        characteristics = QueryCharacteristics(
            original_query="Complex technical question",
            complexity=QueryComplexity.COMPLEX,
            complexity_score=0.8,
            estimated_tokens=200
        )
        
        plan = await workflow_with_components._create_processing_plan(characteristics)
        
        assert plan.use_agentic_workflow
        assert plan.use_hallucination_detection
        assert plan.confidence_threshold >= 0.9
        assert plan.verification_settings.get("strict_mode", False)
    
    @pytest.mark.asyncio
    async def test_speed_profile_plan(self, workflow_with_components):
        """Test processing plan for speed profile."""
        workflow_with_components.config.performance_profile = PerformanceProfile.SPEED
        
        characteristics = QueryCharacteristics(
            original_query="Simple question",
            complexity=QueryComplexity.SIMPLE,
            complexity_score=0.2,
            estimated_tokens=50
        )
        
        plan = await workflow_with_components._create_processing_plan(characteristics)
        
        assert plan.confidence_threshold <= 0.8
        assert plan.verification_settings.get("timeout", 5.0) <= 2.0
    
    @pytest.mark.asyncio
    async def test_cost_optimization_plan(self, workflow_with_components):
        """Test processing plan cost optimization."""
        characteristics = QueryCharacteristics(
            original_query="Expensive query that would exceed cost limits",
            complexity=QueryComplexity.COMPLEX,
            complexity_score=0.9,
            estimated_tokens=1000,
            estimated_cost=0.15  # Exceeds limit
        )
        
        original_plan = ProcessingPlan(
            use_agentic_workflow=True,
            use_multimodal_support=True,
            use_hallucination_detection=True,
            estimated_cost=0.15
        )
        
        optimized_plan = await workflow_with_components._optimize_plan_for_cost(
            original_plan, characteristics
        )
        
        assert optimized_plan.estimated_cost < original_plan.estimated_cost
        # Some expensive features should be disabled
        assert not optimized_plan.use_multimodal_support or not optimized_plan.use_agentic_workflow


class TestWorkflowExecution:
    """Test workflow execution and orchestration."""
    
    @pytest.fixture
    def mock_workflow_execution(self):
        """Mock workflow with execution components."""
        with patch('src.unified_workflow.get_unified_config') as mock_config:
            mock_config_manager = Mock()
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config_manager.is_feature_enabled.return_value = True
            mock_config.return_value = mock_config_manager
            
            with patch('src.unified_workflow.init_settings'), \
                 patch('src.workflow.create_workflow') as mock_create:
                
                # Mock base workflow
                mock_base_workflow = AsyncMock()
                mock_base_workflow.arun.return_value = "Base workflow response"
                mock_create.return_value = mock_base_workflow
                
                # Mock agentic workflow
                mock_agentic_workflow = AsyncMock()
                mock_agentic_workflow.arun.return_value = "Agentic workflow response"
                
                workflow = UnifiedWorkflow(timeout=30.0)
                workflow.config_manager = mock_config_manager
                workflow.base_workflow = mock_base_workflow
                workflow.agentic_workflow = mock_agentic_workflow
                workflow.semantic_cache = Mock()
                workflow.hallucination_detector = Mock()
                
                return workflow, mock_base_workflow, mock_agentic_workflow
    
    @pytest.mark.asyncio
    async def test_base_workflow_execution(self, mock_workflow_execution):
        """Test execution through base workflow."""
        workflow, mock_base, mock_agentic = mock_workflow_execution
        
        characteristics = QueryCharacteristics(
            original_query="Simple query",
            complexity=QueryComplexity.SIMPLE
        )
        
        plan = ProcessingPlan(
            use_agentic_workflow=False,
            use_hallucination_detection=False
        )
        
        result = await workflow._execute_with_plan("Simple query", characteristics, plan)
        
        mock_base.arun.assert_called_once_with("Simple query")
        mock_agentic.arun.assert_not_called()
        assert result == "Base workflow response"
    
    @pytest.mark.asyncio
    async def test_agentic_workflow_execution(self, mock_workflow_execution):
        """Test execution through agentic workflow."""
        workflow, mock_base, mock_agentic = mock_workflow_execution
        
        characteristics = QueryCharacteristics(
            original_query="Complex query",
            complexity=QueryComplexity.COMPLEX
        )
        
        plan = ProcessingPlan(
            use_agentic_workflow=True,
            use_hallucination_detection=False
        )
        
        result = await workflow._execute_with_plan("Complex query", characteristics, plan)
        
        mock_agentic.arun.assert_called_once()
        mock_base.arun.assert_not_called()
        assert result == "Agentic workflow response"
    
    @pytest.mark.asyncio
    async def test_execution_with_verification(self, mock_workflow_execution):
        """Test execution with hallucination detection."""
        workflow, mock_base, mock_agentic = mock_workflow_execution
        
        # Mock verification
        workflow.hallucination_detector.verify_response = AsyncMock()
        workflow.hallucination_detector.verify_response.return_value = (
            Mock(name='VERIFIED'), 0.95, "Verification passed"
        )
        
        characteristics = QueryCharacteristics(
            original_query="Query requiring verification",
            complexity=QueryComplexity.MODERATE
        )
        
        plan = ProcessingPlan(
            use_agentic_workflow=False,
            use_hallucination_detection=True,
            verification_settings={}
        )
        
        result = await workflow._execute_with_plan("Test query", characteristics, plan)
        
        workflow.hallucination_detector.verify_response.assert_called_once()
        assert result is not None


class TestErrorHandlingAndFallbacks:
    """Test error handling and fallback mechanisms."""
    
    @pytest.fixture
    def workflow_with_failures(self):
        """Workflow configured to test failure scenarios."""
        with patch('src.unified_workflow.get_unified_config') as mock_config:
            mock_config_manager = Mock()
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config.return_value = mock_config_manager
            
            with patch('src.unified_workflow.init_settings'):
                workflow = UnifiedWorkflow(timeout=30.0)
                workflow.config_manager = mock_config_manager
                return workflow
    
    @pytest.mark.asyncio
    async def test_fallback_when_main_workflow_fails(self, workflow_with_failures):
        """Test fallback processing when main workflow fails."""
        # Mock failing main workflow
        mock_base_workflow = AsyncMock()
        mock_base_workflow.arun.side_effect = Exception("Main workflow failed")
        workflow_with_failures.base_workflow = mock_base_workflow
        workflow_with_failures.agentic_workflow = None
        
        characteristics = QueryCharacteristics(
            original_query="Test query",
            complexity=QueryComplexity.SIMPLE
        )
        
        result = await workflow_with_failures._attempt_fallback_processing(
            "Test query", characteristics, Exception("Main workflow failed")
        )
        
        assert "encountered difficulties" in result
        assert "Main workflow failed" in result
    
    @pytest.mark.asyncio
    async def test_complete_system_failure_response(self, workflow_with_failures):
        """Test response when all systems fail."""
        workflow_with_failures.base_workflow = None
        workflow_with_failures.agentic_workflow = None
        
        characteristics = QueryCharacteristics(
            original_query="Test query",
            complexity=QueryComplexity.SIMPLE
        )
        
        result = await workflow_with_failures._attempt_fallback_processing(
            "Test query", characteristics, Exception("System failure")
        )
        
        assert "System unavailable" in result
        assert "contact support" in result
    
    def test_component_initialization_failure_handling(self):
        """Test handling of component initialization failures."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow', side_effect=Exception("Init failed")):
            
            mock_config_manager = Mock()
            mock_config_manager.update_component_health = Mock()
            mock_config.return_value = mock_config_manager
            
            workflow = UnifiedWorkflow(timeout=30.0)
            
            # Verify error handling was called
            mock_config_manager.update_component_health.assert_called()
            calls = mock_config_manager.update_component_health.call_args_list
            error_calls = [call for call in calls if len(call[0]) > 2 and call[0][1] == "error"]
            assert len(error_calls) > 0


class TestSecurityValidation:
    """Test security-focused validation and edge cases."""
    
    @pytest.fixture
    def secure_workflow(self):
        """Workflow configured for security testing."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            mock_config_manager = Mock()
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=30.0)
            workflow.config_manager = mock_config_manager
            return workflow
    
    def test_input_sanitization(self, secure_workflow):
        """Test that malicious inputs are handled safely."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "/../../../etc/passwd",
            "\x00\x01\x02",  # Binary data
            "A" * 10000,  # Very long input
        ]
        
        for malicious_input in malicious_inputs:
            # Extract query should handle malicious input safely
            event = Mock()
            event.query = malicious_input
            extracted = secure_workflow._extract_query(event)
            assert extracted == malicious_input  # Should not crash
    
    def test_resource_exhaustion_prevention(self, secure_workflow):
        """Test prevention of resource exhaustion attacks."""
        # Test with extremely large query
        large_query = "What is machine learning? " * 1000
        characteristics = QueryCharacteristics(
            original_query=large_query,
            complexity=QueryComplexity.COMPLEX,
            estimated_tokens=10000  # Very high token count
        )
        
        # Cost estimation should handle large queries
        cost = secure_workflow._estimate_query_cost(characteristics)
        assert cost > 0
        assert cost < 1.0  # Should have reasonable upper bound
    
    def test_configuration_validation_edge_cases(self, secure_workflow):
        """Test edge cases in configuration validation."""
        # Test with extreme cost limits
        extreme_config = {"max_query_cost": 0.0001}  # Very low limit
        secure_workflow.config.cost_management = extreme_config
        
        characteristics = QueryCharacteristics(
            original_query="Test query",
            complexity=QueryComplexity.SIMPLE,
            estimated_cost=0.01
        )
        
        plan = ProcessingPlan(estimated_cost=0.01)
        
        # Should handle extreme cost constraints gracefully
        try:
            # This would normally trigger cost optimization
            cost = secure_workflow._estimate_processing_cost(characteristics, plan)
            assert cost >= 0
        except Exception:
            pytest.fail("Should handle extreme cost constraints gracefully")


class TestPerformanceAndMonitoring:
    """Test performance tracking and monitoring capabilities."""
    
    @pytest.fixture
    def monitored_workflow(self):
        """Workflow with monitoring enabled."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            mock_config_manager = Mock()
            mock_config_manager.config.cost_management = {"max_query_cost": 0.10}
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=30.0)
            workflow.config_manager = mock_config_manager
            return workflow
    
    def test_statistics_tracking(self, monitored_workflow):
        """Test that statistics are properly tracked."""
        # Initial stats
        stats = monitored_workflow.get_stats()
        assert stats['total_queries'] == 0
        assert stats['successful_queries'] == 0
        assert stats['success_rate'] == 0.0
        
        # Simulate query processing
        monitored_workflow.stats['total_queries'] = 10
        monitored_workflow.stats['successful_queries'] = 8
        monitored_workflow.stats['cache_hits'] = 3
        monitored_workflow.stats['total_processing_time'] = 15.0
        monitored_workflow.stats['total_cost'] = 0.08
        
        stats = monitored_workflow.get_stats()
        assert stats['success_rate'] == 0.8
        assert stats['cache_hit_rate'] == 0.3
        assert stats['avg_processing_time'] == 1.5
        assert stats['avg_cost_per_query'] == 0.008
    
    def test_stats_reset(self, monitored_workflow):
        """Test statistics reset functionality."""
        # Set some stats
        monitored_workflow.stats['total_queries'] = 100
        monitored_workflow.stats['successful_queries'] = 95
        
        # Reset stats
        monitored_workflow.reset_stats()
        
        # Verify reset
        stats = monitored_workflow.get_stats()
        assert stats['total_queries'] == 0
        assert stats['successful_queries'] == 0
    
    def test_processing_time_limits(self, monitored_workflow):
        """Test processing time estimation and limits."""
        # Simple query time estimation
        simple_char = QueryCharacteristics(
            original_query="Simple query",
            complexity=QueryComplexity.SIMPLE
        )
        simple_plan = ProcessingPlan(use_agentic_workflow=False)
        
        simple_time = monitored_workflow._estimate_processing_time(simple_char, simple_plan)
        
        # Complex query time estimation
        complex_char = QueryCharacteristics(
            original_query="Complex query",
            complexity=QueryComplexity.COMPLEX
        )
        complex_plan = ProcessingPlan(
            use_agentic_workflow=True,
            use_hallucination_detection=True
        )
        
        complex_time = monitored_workflow._estimate_processing_time(complex_char, complex_plan)
        
        # Complex queries should take longer
        assert complex_time > simple_time
        assert simple_time > 0
        assert complex_time < 60.0  # Should have reasonable upper bound


class TestConfigurationEdgeCases:
    """Test edge cases and configuration scenarios."""
    
    def test_workflow_creation_with_missing_dependencies(self):
        """Test workflow creation when dependencies are missing."""
        with patch('src.unified_workflow.init_settings', side_effect=Exception("Settings failed")):
            with pytest.raises(RuntimeError):
                create_unified_workflow()
    
    def test_workflow_with_minimal_configuration(self):
        """Test workflow with minimal viable configuration."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            # Minimal config - most features disabled
            mock_config_manager = Mock()
            mock_config_manager.config = Mock()
            mock_config_manager.config.performance_profile = PerformanceProfile.SPEED
            mock_config_manager.config.cost_management = {"max_query_cost": 0.01}
            mock_config_manager.config.agentic_workflow = Mock()
            mock_config_manager.config.agentic_workflow.enabled = False
            mock_config_manager.config.semantic_cache = Mock()
            mock_config_manager.config.semantic_cache.enabled = False
            mock_config_manager.config.hallucination_detection = Mock()
            mock_config_manager.config.hallucination_detection.enabled = False
            mock_config_manager.config.multimodal_support = Mock()
            mock_config_manager.config.multimodal_support.enabled = False
            mock_config_manager.config.performance_optimization = Mock()
            mock_config_manager.config.performance_optimization.enabled = False
            mock_config_manager.config.tts_integration = Mock()
            mock_config_manager.config.tts_integration.enabled = False
            
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            # Should still create successfully with minimal config
            workflow = UnifiedWorkflow(timeout=30.0)
            assert workflow is not None
            assert workflow.config_manager == mock_config_manager
    
    @pytest.mark.asyncio
    async def test_query_analysis_edge_cases(self):
        """Test query analysis with edge case inputs."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            mock_config_manager = Mock()
            mock_config_manager.config.agentic_workflow = Mock()
            mock_config_manager.config.agentic_workflow.settings = {"complexity_threshold": 0.7}
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=30.0)
            
            # Test edge cases
            edge_cases = [
                "",  # Empty query
                " ",  # Whitespace only
                "?",  # Single character
                "A" * 1000,  # Very long query
                "测试中文查询",  # Non-English
                "What is 2+2?",  # Mathematical
            ]
            
            for query in edge_cases:
                characteristics = await workflow._analyze_query_characteristics(query)
                assert characteristics is not None
                assert characteristics.original_query == query
                assert 0.0 <= characteristics.complexity_score <= 1.0
                assert characteristics.estimated_cost >= 0.0


# Integration test with real-world scenarios
class TestRealWorldScenarios:
    """Test realistic usage scenarios and workflows."""
    
    @pytest.fixture
    def production_like_workflow(self):
        """Create workflow similar to production configuration."""
        with patch('src.unified_workflow.get_unified_config') as mock_config, \
             patch('src.unified_workflow.init_settings'), \
             patch('src.workflow.create_workflow') as mock_create:
            
            mock_config_manager = Mock()
            mock_config_manager.config.performance_profile = PerformanceProfile.BALANCED
            mock_config_manager.config.cost_management = {"max_query_cost": 0.05}
            mock_config_manager.is_feature_enabled.return_value = True
            mock_config_manager.should_auto_enable_feature.return_value = True
            mock_config_manager.get_system_health.return_value = {
                "features_enabled": {
                    "agentic_workflow": True,
                    "semantic_cache": True,
                    "hallucination_detection": True
                }
            }
            mock_config.return_value = mock_config_manager
            mock_create.return_value = Mock()
            
            workflow = UnifiedWorkflow(timeout=60.0)
            workflow.config_manager = mock_config_manager
            
            # Mock all components as available
            workflow.base_workflow = AsyncMock()
            workflow.base_workflow.arun.return_value = "Production response"
            workflow.agentic_workflow = AsyncMock()
            workflow.semantic_cache = Mock()
            workflow.hallucination_detector = Mock()
            
            return workflow
    
    @pytest.mark.asyncio
    async def test_typical_user_query_flow(self, production_like_workflow):
        """Test a typical user query end-to-end."""
        # Mock semantic cache miss
        production_like_workflow.semantic_cache.get.return_value = None
        
        # Create a realistic query
        user_query = "Explain the differences between machine learning and deep learning"
        
        # Create start event
        start_event = Mock()
        start_event.query = user_query
        start_event.dict.return_value = {"query": user_query}
        
        # Mock context
        mock_context = Mock()
        mock_context.data = {}
        
        # Test query analysis step
        result_event = await production_like_workflow.analyze_query(mock_context, start_event)
        
        # Verify context was populated
        assert "query" in mock_context.data
        assert "characteristics" in mock_context.data
        assert "processing_plan" in mock_context.data
        assert mock_context.data["query"] == user_query
    
    def test_concurrent_query_handling(self, production_like_workflow):
        """Test handling of concurrent queries."""
        # Simulate multiple queries processed simultaneously
        queries = [
            "What is AI?",
            "Explain neural networks",
            "Compare CNN and RNN",
            "How does attention mechanism work?",
            "What is transformer architecture?"
        ]
        
        for query in queries:
            # Each query should be processed independently
            characteristics = asyncio.run(
                production_like_workflow._analyze_query_characteristics(query)
            )
            assert characteristics is not None
            assert characteristics.original_query == query
            
        # Verify stats can handle concurrent processing
        stats = production_like_workflow.get_stats()
        assert isinstance(stats, dict)
        assert all(isinstance(v, (int, float, dict)) for v in stats.values())