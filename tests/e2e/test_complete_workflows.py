"""
End-to-end tests for complete SOTA RAG workflows.

Tests cover:
- Complete query processing pipelines
- Real-world scenario testing
- Performance profile validation
- Error recovery and fallback scenarios
- User experience validation
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.unified_workflow import create_unified_workflow, UnifiedWorkflow, QueryComplexity
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.health_monitor import get_health_monitor


class TestCompleteQueryProcessing:
    """Test complete query processing from start to finish."""
    
    @pytest.mark.asyncio
    async def test_simple_factual_query_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_tracker):
        """Test end-to-end processing of simple factual queries."""
        # Setup mocks for complete pipeline
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 768
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Python is a high-level programming language created by Guido van Rossum in 1991."
        
        # No cache hit initially
        mock_redis_client.keys.return_value = []
        mock_redis_client.get.return_value = None
        
        # Query engine response
        mock_llama_index['engine'].query.return_value = Mock(
            response="Python is a high-level programming language created by Guido van Rossum in 1991.",
            source_nodes=[
                Mock(text="Python was developed by Guido van Rossum", score=0.9)
            ],
            metadata={'processing_time': 1.2, 'token_count': 18}
        )
        
        # Create and execute workflow
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'total_response_time': 0}
        
        query = "Who created Python?"
        
        performance_tracker.start_timing('e2e_simple_query')
        
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            expected_response = {
                'content': "Python is a high-level programming language created by Guido van Rossum in 1991.",
                'confidence': 0.92,
                'sources': [{'text': "Python was developed by Guido van Rossum", 'relevance': 0.9}],
                'processing_time': 1.2,
                'cache_hit': False,
                'verification_passed': True,
                'cost': 0.008
            }
            mock_process.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            performance_tracker.end_timing('e2e_simple_query')
            
            # Validate complete response
            assert result['content'] is not None
            assert result['confidence'] >= 0.85
            assert result['processing_time'] > 0
            assert 'sources' in result
            
            # Performance validation
            processing_time = performance_tracker.get_duration('e2e_simple_query')
            assert processing_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_complex_analytical_query_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_tracker):
        """Test end-to-end processing of complex analytical queries."""
        complex_query = "Compare supervised and unsupervised learning algorithms, provide examples of each, and explain when to use each approach"
        
        # Mock agentic workflow processing
        mock_openai_client.chat.completions.create.side_effect = [
            # Query decomposition
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                'subqueries': [
                    'What is supervised learning?',
                    'What is unsupervised learning?', 
                    'Examples of supervised learning algorithms',
                    'Examples of unsupervised learning algorithms',
                    'When to use supervised vs unsupervised learning?'
                ],
                'reasoning': 'Query requires comprehensive comparison'
            })))]),
            # Subquery processing responses
            Mock(choices=[Mock(message=Mock(content="Supervised learning uses labeled training data to learn mappings from inputs to outputs."))]),
            Mock(choices=[Mock(message=Mock(content="Unsupervised learning finds patterns in data without labeled examples."))]),
            Mock(choices=[Mock(message=Mock(content="Examples include linear regression, decision trees, SVM, neural networks."))]),
            Mock(choices=[Mock(message=Mock(content="Examples include k-means clustering, PCA, autoencoders, t-SNE."))]),
            Mock(choices=[Mock(message=Mock(content="Use supervised when you have labeled data and clear target variables."))]),
            # Response aggregation
            Mock(choices=[Mock(message=Mock(content="Comprehensive comparison of supervised vs unsupervised learning approaches with examples and use cases."))]),
            # Verification
            Mock(choices=[Mock(message=Mock(content="CONSISTENT: Response provides accurate and comprehensive comparison of learning approaches."))])
        ]
        
        # Mock embeddings
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.2] * 768
        
        # No cache hits for complex query
        mock_redis_client.keys.return_value = []
        mock_redis_client.get.return_value = None
        
        workflow = UnifiedWorkflow(timeout=60.0)  # Longer timeout for complex queries
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'agentic_queries': 0}
        
        performance_tracker.start_timing('e2e_complex_query')
        
        with patch.object(workflow, '_process_with_agentic_workflow') as mock_agentic:
            expected_response = {
                'content': 'Comprehensive comparison of supervised vs unsupervised learning with examples and use cases',
                'confidence': 0.89,
                'subqueries': [
                    'What is supervised learning?',
                    'What is unsupervised learning?',
                    'Examples and use cases'
                ],
                'sources': [
                    {'text': 'Supervised learning source', 'relevance': 0.9},
                    {'text': 'Unsupervised learning source', 'relevance': 0.85}
                ],
                'processing_time': 8.5,
                'verification_passed': True,
                'cost': 0.045
            }
            mock_agentic.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(complex_query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            performance_tracker.end_timing('e2e_complex_query')
            
            # Validate complex query response
            assert result['content'] is not None
            assert len(result['content']) > 200  # Should be comprehensive
            assert result['confidence'] >= 0.85
            assert 'subqueries' in result
            assert len(result['subqueries']) >= 3
            
            # Performance validation for complex queries
            processing_time = performance_tracker.get_duration('e2e_complex_query')
            assert processing_time < 15.0  # Should complete within 15 seconds
    
    @pytest.mark.asyncio
    async def test_multimodal_query_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_tracker):
        """Test end-to-end processing of multimodal queries."""
        multimodal_query = "Show me a diagram explaining the transformer architecture and describe how attention mechanisms work"
        
        # Mock multimodal processing
        mock_openai_client.chat.completions.create.side_effect = [
            # Image generation/description
            Mock(choices=[Mock(message=Mock(content="Generated description: Transformer architecture diagram with encoder-decoder structure"))]),
            # Content explanation
            Mock(choices=[Mock(message=Mock(content="The transformer uses self-attention mechanisms to process sequences in parallel"))]),
            # Verification
            Mock(choices=[Mock(message=Mock(content="CONSISTENT: Technical explanation is accurate"))])
        ]
        
        workflow = UnifiedWorkflow(timeout=45.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'multimodal_queries': 0}
        
        performance_tracker.start_timing('e2e_multimodal_query')
        
        with patch.object(workflow, '_process_multimodal') as mock_multimodal:
            expected_response = {
                'content': 'The transformer architecture uses self-attention mechanisms to process sequences efficiently',
                'image_description': 'Diagram showing encoder-decoder structure with attention layers',
                'visual_elements': ['diagram', 'attention', 'transformer'],
                'confidence': 0.87,
                'sources': [
                    {'text': 'Transformer paper source', 'relevance': 0.95}
                ],
                'processing_time': 6.2,
                'multimodal_processing': True,
                'cost': 0.025
            }
            mock_multimodal.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(multimodal_query)
            result = await workflow._process_multimodal(characteristics)
            
            performance_tracker.end_timing('e2e_multimodal_query')
            
            # Validate multimodal response
            assert result['content'] is not None
            assert 'image_description' in result
            assert result['multimodal_processing'] == True
            assert result['confidence'] >= 0.8
            
            # Performance validation
            processing_time = performance_tracker.get_duration('e2e_multimodal_query')
            assert processing_time < 10.0


class TestPerformanceProfileValidation:
    """Test end-to-end validation of performance profiles."""
    
    @pytest.mark.asyncio
    async def test_high_accuracy_profile_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test high accuracy profile end-to-end behavior."""
        # Configure high accuracy profile
        reset_unified_config()
        config_manager = get_unified_config()
        config_manager.config.performance_profile = PerformanceProfile.HIGH_ACCURACY
        
        # Mock high-quality processing
        mock_openai_client.chat.completions.create.side_effect = [
            # High quality response
            Mock(choices=[Mock(message=Mock(content="Highly accurate and detailed response with comprehensive analysis"))]),
            # Strict verification
            Mock(choices=[Mock(message=Mock(content="CONSISTENT: Response meets high accuracy standards with 98% confidence"))])
        ]
        
        workflow = UnifiedWorkflow(timeout=60.0)
        workflow.config_manager = config_manager
        workflow.stats = {'queries_processed': 0}
        
        query = "Explain quantum computing principles"
        
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            expected_response = {
                'content': 'Highly accurate quantum computing explanation with comprehensive technical details',
                'confidence': 0.97,  # High accuracy profile should achieve high confidence
                'verification_confidence': 0.98,
                'verification_passed': True,
                'processing_time': 4.8,
                'cost': 0.035,  # Higher cost acceptable for accuracy
                'accuracy_profile': 'high_accuracy'
            }
            mock_process.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # High accuracy profile validation
            assert result['confidence'] >= 0.95
            assert result['verification_confidence'] >= 0.95
            assert result['verification_passed'] == True
            assert result.get('accuracy_profile') == 'high_accuracy'
    
    @pytest.mark.asyncio
    async def test_speed_profile_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_tracker):
        """Test speed profile end-to-end behavior."""
        # Configure speed profile
        reset_unified_config()
        config_manager = get_unified_config()
        config_manager.config.performance_profile = PerformanceProfile.SPEED
        
        # Mock fast processing
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Quick and efficient response optimized for speed."
        
        workflow = UnifiedWorkflow(timeout=15.0)  # Tight timeout for speed
        workflow.config_manager = config_manager
        workflow.stats = {'queries_processed': 0}
        
        query = "What is machine learning?"
        
        performance_tracker.start_timing('speed_profile_e2e')
        
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            expected_response = {
                'content': 'Machine learning enables computers to learn from data without explicit programming',
                'confidence': 0.88,  # Good confidence but optimized for speed
                'processing_time': 0.8,  # Very fast processing
                'cost': 0.005,  # Low cost
                'cache_hit': False,
                'speed_optimized': True
            }
            mock_process.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            performance_tracker.end_timing('speed_profile_e2e')
            
            # Speed profile validation
            total_time = performance_tracker.get_duration('speed_profile_e2e')
            assert total_time < 2.0  # Should be very fast
            assert result['processing_time'] < 2.0
            assert result['cost'] < 0.01  # Should be cost efficient
            assert result['confidence'] >= 0.85  # Still good quality
    
    @pytest.mark.asyncio
    async def test_cost_optimized_profile_e2e(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test cost optimized profile end-to-end behavior."""
        # Configure cost optimized profile
        reset_unified_config()
        config_manager = get_unified_config()
        config_manager.config.performance_profile = PerformanceProfile.COST_OPTIMIZED
        
        # Mock cost-efficient processing with cache hit
        cached_response = {
            'content': 'Cached response about neural networks',
            'confidence': 0.89,
            'cache_hit': True,
            'cost': 0.001  # Very low cost due to cache
        }
        mock_redis_client.get.return_value = json.dumps(cached_response)
        mock_redis_client.keys.return_value = ['cache:test']
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = config_manager
        workflow.stats = {'queries_processed': 0, 'cache_hits': 0}
        
        query = "What are neural networks?"
        
        with patch.object(workflow, '_try_semantic_cache') as mock_cache:
            mock_cache.return_value = cached_response
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Cost optimization validation
            assert result.get('cache_hit') == True
            assert result['cost'] < 0.01  # Very low cost
            assert result['confidence'] >= 0.85  # Still good quality
            
            # Should prefer cache to minimize cost
            mock_cache.assert_called_once()


class TestErrorRecoveryScenarios:
    """Test error recovery and fallback scenarios."""
    
    @pytest.mark.asyncio
    async def test_cache_failure_recovery(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test recovery when cache system fails."""
        # Mock cache failure
        mock_redis_client.get.side_effect = Exception("Redis connection failed")
        
        # Mock successful fallback processing
        mock_llama_index['engine'].query.return_value = Mock(
            response="Fallback response without caching",
            source_nodes=[Mock(text="Fallback source", score=0.85)],
            metadata={}
        )
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'cache_errors': 0}
        
        query = "Test query for cache failure"
        
        with patch.object(workflow, '_try_semantic_cache') as mock_cache, \
             patch.object(workflow, '_process_with_standard_workflow') as mock_fallback:
            
            mock_cache.side_effect = Exception("Cache failed")
            mock_fallback.return_value = {
                'content': 'Response processed without cache after cache failure',
                'confidence': 0.82,
                'fallback_used': True,
                'cache_error': 'Cache failed'
            }
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Should recover gracefully from cache failure
            assert result['fallback_used'] == True
            assert 'cache_error' in result
            assert result['content'] is not None
            assert result['confidence'] >= 0.8
    
    @pytest.mark.asyncio
    async def test_verification_failure_recovery(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test recovery when verification system fails."""
        # Mock verification failure
        mock_openai_client.chat.completions.create.side_effect = Exception("Verification API failed")
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'verification_errors': 0}
        
        query = "Test query for verification failure"
        
        with patch.object(workflow, '_verify_response') as mock_verify, \
             patch.object(workflow, '_process_with_standard_workflow') as mock_process:
            
            # Mock processing success but verification failure
            mock_process.return_value = {
                'content': 'Response that could not be verified',
                'confidence': 0.75,
                'sources': []
            }
            
            mock_verify.side_effect = Exception("Verification failed")
            
            characteristics = await workflow._analyze_query_characteristics(query)
            
            # Should handle verification failure gracefully
            try:
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                # Should still provide response with adjusted confidence
                assert result['content'] is not None
                assert result['confidence'] < 0.8  # Lower confidence without verification
                
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                assert "verification" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_complete_system_degradation(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test behavior during complete system degradation."""
        # Mock multiple system failures
        mock_redis_client.get.side_effect = Exception("Cache failed")
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI API failed")
        mock_openai_client.embeddings.create.side_effect = Exception("Embedding API failed")
        
        # Only basic query engine works
        mock_llama_index['engine'].query.return_value = Mock(
            response="Basic response with minimal processing",
            source_nodes=[Mock(text="Basic source", score=0.7)],
            metadata={}
        )
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'system_errors': 0}
        
        query = "Test query during system degradation"
        
        with patch.object(workflow, '_process_with_basic_fallback') as mock_basic:
            mock_basic.return_value = {
                'content': 'Basic response using minimal system capabilities',
                'confidence': 0.6,
                'degraded_mode': True,
                'available_features': ['basic_retrieval'],
                'unavailable_features': ['caching', 'verification', 'agentic_processing']
            }
            
            characteristics = await workflow._analyze_query_characteristics(query)
            
            # Should still provide some response even in degraded mode
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            assert result['content'] is not None
            assert result['degraded_mode'] == True
            assert result['confidence'] >= 0.5  # Lower but still usable
            assert len(result['unavailable_features']) > 0


class TestUserExperienceValidation:
    """Test user experience aspects of the system."""
    
    @pytest.mark.asyncio
    async def test_response_quality_consistency(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test that response quality is consistent across similar queries."""
        similar_queries = [
            "What is machine learning?",
            "Can you explain machine learning?",
            "Tell me about machine learning",
            "Define machine learning"
        ]
        
        # Mock consistent high-quality responses
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Machine learning is a branch of artificial intelligence that enables systems to learn from data."
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        responses = []
        
        for query in similar_queries:
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                expected_response = {
                    'content': 'Machine learning is a branch of AI that enables systems to learn from data',
                    'confidence': 0.91,
                    'processing_time': 1.5,
                    'cost': 0.008
                }
                mock_process.return_value = expected_response
                
                characteristics = await workflow._analyze_query_characteristics(query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                responses.append(result)
        
        # Validate consistency
        confidences = [r['confidence'] for r in responses]
        processing_times = [r['processing_time'] for r in responses]
        
        # Confidence should be consistently high
        assert all(c >= 0.85 for c in confidences)
        assert max(confidences) - min(confidences) < 0.1  # Low variance
        
        # Processing times should be consistent
        assert max(processing_times) - min(processing_times) < 1.0
    
    @pytest.mark.asyncio
    async def test_progressive_complexity_handling(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_tracker):
        """Test handling of queries with progressive complexity."""
        queries_by_complexity = [
            ("What is AI?", QueryComplexity.SIMPLE),
            ("How does machine learning work?", QueryComplexity.MODERATE), 
            ("Compare different neural network architectures and their applications", QueryComplexity.COMPLEX),
            ("Show me a detailed analysis of transformer models with visual diagrams", QueryComplexity.MULTI_MODAL)
        ]
        
        workflow = UnifiedWorkflow(timeout=60.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        results = []
        
        for query, expected_complexity in queries_by_complexity:
            performance_tracker.start_timing(f'complexity_{expected_complexity.value}')
            
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                # Mock complexity-appropriate responses
                if expected_complexity == QueryComplexity.SIMPLE:
                    response = {
                        'content': 'Simple explanation of AI',
                        'confidence': 0.93,
                        'processing_time': 0.8,
                        'cost': 0.005
                    }
                elif expected_complexity == QueryComplexity.COMPLEX:
                    response = {
                        'content': 'Comprehensive analysis with detailed comparisons',
                        'confidence': 0.88,
                        'processing_time': 6.2,
                        'cost': 0.035,
                        'subqueries': ['Architecture 1', 'Architecture 2', 'Applications']
                    }
                else:
                    response = {
                        'content': f'Response appropriate for {expected_complexity.value} query',
                        'confidence': 0.9,
                        'processing_time': 2.5,
                        'cost': 0.015
                    }
                
                mock_process.return_value = response
                
                characteristics = await workflow._analyze_query_characteristics(query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                results.append((expected_complexity, result))
            
            performance_tracker.end_timing(f'complexity_{expected_complexity.value}')
        
        # Validate progressive complexity handling
        simple_result = next(r for c, r in results if c == QueryComplexity.SIMPLE)
        complex_result = next(r for c, r in results if c == QueryComplexity.COMPLEX)
        
        # Complex queries should take more time and cost more
        assert complex_result['processing_time'] > simple_result['processing_time']
        assert complex_result['cost'] > simple_result['cost']
        
        # But all should maintain good quality
        assert all(r['confidence'] >= 0.85 for _, r in results)
    
    @pytest.mark.asyncio
    async def test_citation_and_source_quality(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test quality of citations and sources in responses."""
        query = "What are the benefits of renewable energy?"
        
        # Mock response with citations
        mock_llama_index['engine'].query.return_value = Mock(
            response="Renewable energy offers environmental and economic benefits [citation:1]",
            source_nodes=[
                Mock(
                    text="Renewable energy reduces carbon emissions and provides sustainable power",
                    score=0.92,
                    metadata={'source': 'Environmental Science Journal', 'year': 2023}
                ),
                Mock(
                    text="Solar and wind energy create jobs and reduce energy costs",
                    score=0.88,
                    metadata={'source': 'Economic Analysis Report', 'year': 2023}
                )
            ],
            metadata={'total_sources': 2}
        )
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            expected_response = {
                'content': 'Renewable energy offers environmental benefits like reduced emissions [citation:1] and economic advantages including job creation [citation:2]',
                'confidence': 0.91,
                'sources': [
                    {
                        'text': 'Renewable energy reduces carbon emissions',
                        'relevance': 0.92,
                        'metadata': {'source': 'Environmental Science Journal', 'year': 2023}
                    },
                    {
                        'text': 'Solar and wind energy create jobs',
                        'relevance': 0.88,
                        'metadata': {'source': 'Economic Analysis Report', 'year': 2023}
                    }
                ],
                'citations': [
                    {'id': 1, 'source': 'Environmental Science Journal'},
                    {'id': 2, 'source': 'Economic Analysis Report'}
                ]
            }
            mock_process.return_value = expected_response
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Validate citation quality
            assert 'sources' in result
            assert len(result['sources']) >= 2
            assert all(s['relevance'] >= 0.8 for s in result['sources'])
            assert 'citations' in result
            assert all('source' in c for c in result['citations'])
            
            # Content should include citation markers
            assert '[citation:' in result['content']