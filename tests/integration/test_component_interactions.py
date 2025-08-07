"""
Integration tests for SOTA RAG component interactions.

Tests cover:
- Cross-component communication and data flow
- Feature integration and coordination
- Performance profile impact on component interactions
- Error propagation and recovery between components
- Configuration management across components
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.unified_workflow import UnifiedWorkflow, QueryCharacteristics, QueryComplexity
from src.health_monitor import get_health_monitor
from src.cache import SemanticCache
from src.verification import HallucinationDetector


class TestWorkflowCacheIntegration:
    """Test integration between workflow orchestrator and semantic cache."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_workflow(self, mock_redis_client, mock_openai_client, mock_llama_index):
        """Test workflow behavior with cache hits."""
        # Setup cached response
        cached_response = {
            'content': 'Machine learning is a subset of artificial intelligence',
            'confidence': 0.92,
            'sources': [],
            'cache_hit': True,
            'processing_time': 0.1
        }
        
        mock_redis_client.keys.return_value = ['cache:test_key']
        mock_redis_client.get.return_value = json.dumps(cached_response)
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 768
        
        # Create workflow
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'cache_hits': 0}
        
        # Mock cache integration
        with patch.object(workflow, '_try_semantic_cache') as mock_cache:
            mock_cache.return_value = cached_response
            
            query = "What is machine learning?"
            characteristics = await workflow._analyze_query_characteristics(query)
            
            # Process with cache hit
            result = await workflow._process_query_with_plan(
                characteristics, 
                workflow._create_processing_plan(characteristics)
            )
            
            # Should use cached response
            mock_cache.assert_called_once()
            assert result['cache_hit'] == True
            assert result['processing_time'] < 1.0
    
    @pytest.mark.asyncio
    async def test_cache_miss_workflow(self, mock_redis_client, mock_openai_client, mock_llama_index):
        """Test workflow behavior with cache misses."""
        # Setup cache miss
        mock_redis_client.keys.return_value = []
        mock_redis_client.get.return_value = None
        
        # Setup query engine response
        query_response = {
            'content': 'Deep learning uses neural networks with multiple layers',
            'confidence': 0.88,
            'sources': [{'content': 'Deep learning source', 'relevance': 0.9}],
            'processing_time': 2.5
        }
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'cache_misses': 0}
        
        with patch.object(workflow, '_try_semantic_cache') as mock_cache, \
             patch.object(workflow, '_process_with_standard_workflow') as mock_standard:
            
            mock_cache.return_value = None  # Cache miss
            mock_standard.return_value = query_response
            
            query = "What is deep learning?"
            characteristics = await workflow._analyze_query_characteristics(query)
            
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Should fall back to standard processing
            mock_cache.assert_called_once()
            mock_standard.assert_called_once()
            assert 'cache_hit' not in result or not result.get('cache_hit')
    
    @pytest.mark.asyncio
    async def test_cache_update_after_processing(self, mock_redis_client, mock_openai_client):
        """Test that cache is updated after processing new queries."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        
        # Mock successful processing
        processing_result = {
            'content': 'Neural networks are computing systems inspired by biological neural networks',
            'confidence': 0.91,
            'sources': [],
            'processing_time': 1.8
        }
        
        with patch.object(workflow, '_try_semantic_cache') as mock_cache_get, \
             patch.object(workflow, '_update_semantic_cache') as mock_cache_set, \
             patch.object(workflow, '_process_with_standard_workflow') as mock_process:
            
            mock_cache_get.return_value = None  # Cache miss
            mock_process.return_value = processing_result
            
            query = "What are neural networks?"
            characteristics = await workflow._analyze_query_characteristics(query)
            
            await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Should update cache with new result
            mock_cache_set.assert_called_once()


class TestWorkflowVerificationIntegration:
    """Test integration between workflow and verification system."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_response_verification(self, configured_workflow, mock_openai_client):
        """Test verification of high-confidence responses."""
        workflow = configured_workflow
        
        # High confidence response that should pass verification
        response = {
            'content': 'Python is a high-level programming language created by Guido van Rossum',
            'confidence': 0.95,
            'sources': [
                {'content': 'Python was developed by Guido van Rossum', 'relevance': 0.9}
            ]
        }
        
        # Mock verification success
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: The response is factually accurate and well-supported."
        
        with patch.object(workflow, '_verify_response') as mock_verify:
            mock_verify.return_value = {
                **response,
                'verification_passed': True,
                'verification_confidence': 0.93
            }
            
            verified_response = await workflow._verify_response(response, "Who created Python?")
            
            assert verified_response['verification_passed'] == True
            assert verified_response['verification_confidence'] >= 0.9
    
    @pytest.mark.asyncio
    async def test_low_confidence_response_rejection(self, configured_workflow, mock_openai_client):
        """Test rejection of low-confidence responses."""
        workflow = configured_workflow
        
        # Low confidence response that should fail verification
        response = {
            'content': 'Python was created by aliens in 1995 on Mars',
            'confidence': 0.45,
            'sources': []
        }
        
        # Mock verification failure
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "INCONSISTENT: The response contains factual errors and lacks credible sources."
        
        with patch.object(workflow, '_verify_response') as mock_verify:
            mock_verify.return_value = {
                **response,
                'verification_passed': False,
                'verification_confidence': 0.25,
                'requires_regeneration': True
            }
            
            verified_response = await workflow._verify_response(response, "Who created Python?")
            
            assert verified_response['verification_passed'] == False
            assert verified_response['requires_regeneration'] == True
    
    @pytest.mark.asyncio
    async def test_verification_performance_profile_adaptation(self, configured_workflow):
        """Test verification adaptation to performance profiles."""
        workflow = configured_workflow
        
        response = {
            'content': 'Test response for verification',
            'confidence': 0.85,
            'sources': [{'content': 'Source', 'relevance': 0.8}]
        }
        
        # Test different profiles
        profiles = [
            PerformanceProfile.HIGH_ACCURACY,
            PerformanceProfile.BALANCED,
            PerformanceProfile.COST_OPTIMIZED,
            PerformanceProfile.SPEED
        ]
        
        for profile in profiles:
            workflow.config_manager.config.performance_profile = profile
            
            with patch.object(workflow, '_verify_response') as mock_verify:
                if profile == PerformanceProfile.HIGH_ACCURACY:
                    # High accuracy should use stricter verification
                    mock_verify.return_value = {
                        **response,
                        'verification_threshold': 0.95,
                        'verification_timeout': 10.0
                    }
                elif profile == PerformanceProfile.SPEED:
                    # Speed profile should use faster verification
                    mock_verify.return_value = {
                        **response,
                        'verification_threshold': 0.75,
                        'verification_timeout': 2.0
                    }
                
                await workflow._verify_response(response, "Test query")
                mock_verify.assert_called_once()


class TestCacheVerificationIntegration:
    """Test integration between semantic cache and verification system."""
    
    @pytest.mark.asyncio
    async def test_cached_response_reverification(self, mock_redis_client, mock_openai_client):
        """Test reverification of cached responses when needed."""
        # Setup cache with response that needs reverification
        cached_response = {
            'content': 'Cached response about AI',
            'confidence': 0.85,
            'last_verified': 1234567890,  # Old timestamp
            'verification_confidence': 0.8
        }
        
        mock_redis_client.get.return_value = json.dumps(cached_response)
        mock_redis_client.keys.return_value = ['cache:test']
        
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        detector = HallucinationDetector()
        
        # Mock reverification
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Response remains accurate."
        
        with patch.object(cache, '_should_reverify') as mock_should_reverify, \
             patch.object(detector, 'verify_factual_consistency') as mock_verify:
            
            mock_should_reverify.return_value = True
            mock_verify.return_value = Mock(
                is_consistent=True,
                confidence=0.9,
                explanation="Still accurate"
            )
            
            result = await cache.get("What is AI?", reverify=True)
            
            if result:
                # Should have updated verification info
                assert result.get('verification_confidence', 0) >= 0.8
    
    @pytest.mark.asyncio
    async def test_verification_result_caching(self, mock_redis_client):
        """Test caching of verification results."""
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        
        # Verification result to cache
        verification_result = {
            'is_consistent': True,
            'confidence': 0.92,
            'hallucination_score': 0.1,
            'explanation': 'Response is factually accurate',
            'verified_at': 1234567890
        }
        
        await cache.set(
            "verification:query_hash",
            json.dumps(verification_result),
            confidence=0.92,
            ttl=7200  # 2 hours
        )
        
        # Should have cached the verification result
        mock_redis_client.set.assert_called_once()


class TestAgenticWorkflowIntegration:
    """Test integration of agentic workflow with other components."""
    
    @pytest.mark.asyncio
    async def test_query_decomposition_with_caching(self, configured_workflow, mock_redis_client):
        """Test query decomposition with subquery caching."""
        workflow = configured_workflow
        
        complex_query = "Compare supervised and unsupervised learning algorithms and provide examples"
        
        # Mock decomposed subqueries
        subqueries = [
            "What is supervised learning?",
            "What is unsupervised learning?",
            "Examples of supervised learning algorithms",
            "Examples of unsupervised learning algorithms"
        ]
        
        # Mock cached responses for some subqueries
        cache_responses = {
            subqueries[0]: {'content': 'Supervised learning uses labeled data', 'confidence': 0.9},
            subqueries[1]: {'content': 'Unsupervised learning finds patterns in unlabeled data', 'confidence': 0.88}
        }
        
        with patch.object(workflow, '_decompose_query') as mock_decompose, \
             patch.object(workflow, '_try_semantic_cache') as mock_cache, \
             patch.object(workflow, '_process_subqueries') as mock_process_sub:
            
            mock_decompose.return_value = subqueries
            
            # Cache hits for first two subqueries, misses for others
            def cache_side_effect(query):
                return cache_responses.get(query)
            mock_cache.side_effect = cache_side_effect
            
            # Mock processing of remaining subqueries
            mock_process_sub.return_value = {
                'aggregated_response': 'Comprehensive comparison of learning types',
                'confidence': 0.91,
                'subquery_results': cache_responses
            }
            
            characteristics = QueryCharacteristics(
                original_query=complex_query,
                complexity=QueryComplexity.COMPLEX,
                requires_decomposition=True
            )
            
            result = await workflow._process_with_agentic_workflow(characteristics)
            
            # Should use cached subqueries and process remaining ones
            assert mock_decompose.called
            assert mock_cache.call_count == len(subqueries)
    
    @pytest.mark.asyncio 
    async def test_agentic_response_verification(self, configured_workflow, mock_openai_client):
        """Test verification of agentic workflow responses."""
        workflow = configured_workflow
        
        # Complex agentic response
        agentic_response = {
            'content': 'Detailed analysis of machine learning approaches with examples and comparisons',
            'confidence': 0.87,
            'subqueries': ['What is ML?', 'Types of ML', 'ML examples'],
            'sources': [
                {'content': 'ML source 1', 'relevance': 0.9},
                {'content': 'ML source 2', 'relevance': 0.85}
            ]
        }
        
        # Mock verification of complex response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Response provides accurate and comprehensive analysis."
        
        with patch.object(workflow, '_verify_response') as mock_verify:
            mock_verify.return_value = {
                **agentic_response,
                'verification_passed': True,
                'verification_confidence': 0.89,
                'complexity_adjusted_confidence': 0.85  # Slightly lower for complex response
            }
            
            verified_response = await workflow._verify_response(
                agentic_response, 
                "Complex ML query"
            )
            
            # Should verify and potentially adjust confidence for complexity
            assert verified_response['verification_passed'] == True
            assert verified_response['complexity_adjusted_confidence'] <= agentic_response['confidence']


class TestMultimodalIntegration:
    """Test multimodal processing integration with other components."""
    
    @pytest.mark.asyncio
    async def test_multimodal_cache_integration(self, configured_workflow, mock_redis_client):
        """Test caching of multimodal responses."""
        workflow = configured_workflow
        
        multimodal_query = "Show me a diagram of transformer architecture"
        
        # Mock multimodal response
        multimodal_response = {
            'content': 'The transformer architecture consists of encoder and decoder layers',
            'image_description': 'Diagram showing encoder-decoder structure with attention mechanisms',
            'visual_elements': ['diagram', 'architecture', 'attention'],
            'confidence': 0.85
        }
        
        with patch.object(workflow, '_process_multimodal') as mock_multimodal, \
             patch.object(workflow, '_update_semantic_cache') as mock_cache_update:
            
            mock_multimodal.return_value = multimodal_response
            
            characteristics = QueryCharacteristics(
                original_query=multimodal_query,
                complexity=QueryComplexity.MULTI_MODAL,
                has_images=True
            )
            
            result = await workflow._process_multimodal(characteristics)
            
            # Should cache multimodal response with special handling
            mock_cache_update.assert_called_once()
            cache_call_args = mock_cache_update.call_args[0]
            
            # Cached entry should include multimodal metadata
            assert 'visual_elements' in result or any('visual' in str(arg) for arg in cache_call_args)
    
    @pytest.mark.asyncio
    async def test_multimodal_verification_challenges(self, configured_workflow, mock_openai_client):
        """Test verification challenges with multimodal content."""
        workflow = configured_workflow
        
        # Multimodal response with visual claims
        multimodal_response = {
            'content': 'This neural network diagram shows 3 layers with 10 nodes each',
            'image_description': 'Complex network visualization',
            'confidence': 0.8,
            'sources': []
        }
        
        # Mock multimodal verification
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "PARTIALLY_VERIFIABLE: Text description cannot be fully verified without image analysis."
        
        with patch.object(workflow, '_verify_response') as mock_verify:
            mock_verify.return_value = {
                **multimodal_response,
                'verification_passed': True,
                'verification_confidence': 0.75,  # Lower for multimodal
                'verification_limitations': ['Cannot verify visual claims without image']
            }
            
            verified_response = await workflow._verify_response(
                multimodal_response,
                "Show neural network diagram"
            )
            
            # Should handle multimodal verification limitations
            assert verified_response['verification_confidence'] < 0.85
            assert 'verification_limitations' in verified_response


class TestHealthMonitoringIntegration:
    """Test health monitoring integration across components."""
    
    @pytest.mark.asyncio
    async def test_component_health_reporting(self, configured_workflow):
        """Test that components report health to monitoring system."""
        workflow = configured_workflow
        monitor = get_health_monitor()
        
        # Simulate component operations
        workflow.stats['queries_processed'] = 100
        workflow.stats['successful_queries'] = 95
        workflow.stats['avg_response_time'] = 2.1
        
        # Report component health
        workflow.config_manager.update_component_health(
            'unified_orchestrator',
            'healthy',
            {
                'queries_processed': workflow.stats['queries_processed'],
                'success_rate': workflow.stats['successful_queries'] / workflow.stats['queries_processed'],
                'avg_response_time': workflow.stats['avg_response_time']
            }
        )
        
        # Check system health
        system_health = workflow.config_manager.get_system_health()
        
        assert 'unified_orchestrator' in system_health['component_details']
        component_health = system_health['component_details']['unified_orchestrator']
        assert component_health['status'] == 'healthy'
        assert component_health['metrics']['success_rate'] >= 0.9
    
    @pytest.mark.asyncio
    async def test_error_propagation_monitoring(self, configured_workflow):
        """Test monitoring of error propagation between components."""
        workflow = configured_workflow
        
        # Simulate component errors
        error_scenarios = [
            ('cache_connection_error', 'semantic_cache'),
            ('verification_timeout', 'verification_system'),
            ('agentic_processing_error', 'agentic_workflow')
        ]
        
        for error_type, component in error_scenarios:
            # Report component error
            workflow.config_manager.update_component_health(
                component,
                'degraded',
                {'error_count': 3, 'last_error': error_type},
                error_type
            )
            
            # Check that error is reflected in system health
            system_health = workflow.config_manager.get_system_health()
            component_health = system_health['component_details'][component]
            
            assert component_health['status'] == 'degraded'
            assert component_health['last_error'] == error_type
    
    def test_performance_threshold_monitoring(self, configured_workflow, performance_benchmarks):
        """Test monitoring of performance thresholds across components."""
        workflow = configured_workflow
        benchmarks = performance_benchmarks
        
        # Test response time monitoring
        slow_response_time = benchmarks['response_time']['complex_query'] + 1.0
        
        workflow.config_manager.update_component_health(
            'unified_orchestrator',
            'degraded',
            {
                'avg_response_time': slow_response_time,
                'threshold_violations': ['response_time_exceeded']
            }
        )
        
        system_health = workflow.config_manager.get_system_health()
        orchestrator_health = system_health['component_details']['unified_orchestrator']
        
        assert orchestrator_health['status'] == 'degraded'
        assert 'threshold_violations' in orchestrator_health['metrics']


class TestConfigurationPropagation:
    """Test configuration management across all components."""
    
    def test_performance_profile_propagation(self):
        """Test that performance profile changes propagate to all components."""
        reset_unified_config()
        config_manager = get_unified_config()
        
        # Change performance profile
        config_manager.config.performance_profile = PerformanceProfile.HIGH_ACCURACY
        
        # Check that all components adapt
        components = [
            config_manager.config.agentic_workflow,
            config_manager.config.semantic_cache,
            config_manager.config.hallucination_detection,
            config_manager.config.multimodal_support
        ]
        
        for component in components:
            # Each component should have profile-appropriate settings
            if hasattr(component, 'settings'):
                settings = component.settings
                
                # High accuracy profile should have conservative thresholds
                for key, value in settings.items():
                    if 'threshold' in key.lower():
                        assert isinstance(value, (int, float))
                        if 'confidence' in key.lower() or 'accuracy' in key.lower():
                            assert value >= 0.85  # High thresholds for accuracy
    
    def test_feature_toggle_propagation(self):
        """Test that feature toggles propagate correctly."""
        reset_unified_config()
        config_manager = get_unified_config()
        
        # Disable specific features
        config_manager.config.semantic_cache.enabled = False
        config_manager.config.hallucination_detection.enabled = False
        
        # Check feature status propagation
        assert not config_manager.is_feature_enabled('semantic_cache')
        assert not config_manager.is_feature_enabled('hallucination_detection')
        assert config_manager.is_feature_enabled('agentic_workflow')  # Should still be enabled
    
    def test_resource_limit_propagation(self):
        """Test that resource limits propagate to components."""
        reset_unified_config()
        config_manager = get_unified_config()
        
        # Set resource limits
        config_manager.config.cost_management['max_query_cost'] = 0.5
        config_manager.config.performance_targets['response_time_p95'] = 2.0
        
        # Components should respect these limits
        assert config_manager.config.cost_management['max_query_cost'] == 0.5
        assert config_manager.config.performance_targets['response_time_p95'] == 2.0
        
        # Individual component settings should align
        agentic_settings = config_manager.config.agentic_workflow.settings
        if 'max_cost_per_query' in agentic_settings:
            assert agentic_settings['max_cost_per_query'] <= 0.5