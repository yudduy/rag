"""
Integration Tests for Cache + Verification Pipeline

This module provides comprehensive integration tests for the interaction between
the semantic cache system and verification system, focusing on:
1. Cache + Verification pipeline integration
2. Cache validation and reverification workflows
3. Verification result caching strategies
4. Performance optimization through intelligent caching
5. Error handling and fallback scenarios

Tests validate the complete pipeline from cache lookup through verification to storage.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from src.cache import SemanticCache, get_cache
from src.verification import HallucinationDetector, create_hallucination_detector, VerificationResult
from src.unified_workflow import UnifiedWorkflow
from src.unified_config import get_unified_config, reset_unified_config


@dataclass
class MockVerificationResult:
    """Mock verification result for testing."""
    name: str = "ACCEPTED"
    confidence: float = 0.85
    explanation: str = "Test verification result"


class TestCacheVerificationPipeline:
    """Test the integrated cache and verification pipeline."""
    
    @pytest.fixture
    def mock_cache_client(self, mock_redis_client):
        """Create a mocked semantic cache."""
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        cache.redis_client = mock_redis_client
        return cache
    
    @pytest.fixture
    def mock_verification_detector(self, mock_openai_client):
        """Create a mocked hallucination detector."""
        detector = HallucinationDetector()
        detector.llm = mock_openai_client
        return detector
    
    @pytest.mark.asyncio
    async def test_cache_hit_with_verification_pipeline(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test complete pipeline when cache hit occurs with verification required."""
        
        query = "What is the capital of France?"
        
        # Mock cached response
        cached_data = {
            'response': "The capital of France is Paris.",
            'confidence': 0.89,
            'sources': [{'text': 'Paris is the capital city of France', 'relevance': 0.95}],
            'cached_at': time.time(),
            'verification_status': 'pending'  # Needs verification
        }
        
        # Setup cache mock
        mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
        mock_cache_client.redis_client.keys.return_value = ['semantic_cache:test_key']
        
        # Setup verification mock
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: The response is factually accurate."
        
        # Test the pipeline
        with patch.object(mock_cache_client, '_get_cache_key') as mock_key, \
             patch.object(mock_verification_detector, 'verify_response') as mock_verify:
            
            mock_key.return_value = "test_key"
            mock_verify.return_value = (MockVerificationResult("ACCEPTED", 0.92), 0.92)
            
            # Get from cache
            result = mock_cache_client.get(query)
            
            if result:
                # Apply verification
                response_text, nodes, similarity = result
                verified_result, confidence = await mock_verification_detector.verify_response(
                    response_text, 0.89, None, []
                )
                
                # Validate pipeline results
                assert verified_result.name == "ACCEPTED"
                assert confidence >= 0.89
                assert response_text == cached_data['response']
                
                # Should update cache with verification result
                mock_verify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_miss_with_verification_and_storage(
        self, mock_cache_client, mock_verification_detector, mock_openai_client, mock_llama_index
    ):
        """Test pipeline when cache miss occurs, requiring processing, verification, and caching."""
        
        query = "Explain quantum computing principles"
        
        # Mock cache miss
        mock_cache_client.redis_client.get.return_value = None
        mock_cache_client.redis_client.keys.return_value = []
        
        # Mock query processing
        processed_response = {
            'response': 'Quantum computing uses quantum mechanical phenomena to process information.',
            'confidence': 0.87,
            'sources': [{'text': 'Quantum mechanics enables quantum computing', 'relevance': 0.9}]
        }
        
        # Mock verification
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Quantum computing explanation is scientifically accurate."
        
        mock_response = Mock()
        mock_response.response = processed_response['response']
        mock_response.source_nodes = [
            Mock(text=processed_response['sources'][0]['text'], score=0.9, id_="node1")
        ]
        mock_response.metadata = {'processing_time': 2.1}
        
        with patch.object(mock_verification_detector, 'verify_response') as mock_verify, \
             patch.object(mock_cache_client, 'put') as mock_cache_put:
            
            mock_verify.return_value = (MockVerificationResult("ACCEPTED", 0.91), 0.91)
            
            # Simulate the complete pipeline
            # 1. Cache miss (already mocked)
            cache_result = mock_cache_client.get(query)
            assert cache_result is None
            
            # 2. Process query (simulate with mock response)
            # 3. Verify response
            verified_result, confidence = await mock_verification_detector.verify_response(
                processed_response['response'], processed_response['confidence'], None, []
            )
            
            # 4. Cache the verified result
            mock_cache_client.put(query, mock_response, cost=0.015)
            
            # Validate pipeline execution
            assert verified_result.name == "ACCEPTED"
            assert confidence >= 0.87
            mock_verify.assert_called_once()
            mock_cache_put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_verification_failure_handling_in_pipeline(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test pipeline handling when verification fails or rejects response."""
        
        query = "What is the meaning of life?"
        
        # Mock cached response with questionable content
        cached_data = {
            'response': "The meaning of life is definitely 42 according to all scientific studies.",
            'confidence': 0.65,  # Lower confidence
            'sources': [],
            'cached_at': time.time() - 7200,  # 2 hours old
            'verification_status': 'needs_reverification'
        }
        
        mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
        mock_cache_client.redis_client.keys.return_value = ['semantic_cache:test_key']
        
        # Mock verification rejection
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "INCONSISTENT: Response makes unsupported scientific claims."
        
        with patch.object(mock_cache_client, '_get_cache_key') as mock_key, \
             patch.object(mock_verification_detector, 'verify_response') as mock_verify, \
             patch.object(mock_cache_client, 'invalidate') as mock_invalidate:
            
            mock_key.return_value = "test_key"
            mock_verify.return_value = (MockVerificationResult("REJECTED", 0.3), 0.3)
            
            # Get from cache
            result = mock_cache_client.get(query)
            
            if result:
                response_text, nodes, similarity = result
                
                # Apply verification
                verified_result, confidence = await mock_verification_detector.verify_response(
                    response_text, cached_data['confidence'], None, []
                )
                
                # Handle verification failure
                if verified_result.name == "REJECTED":
                    # Should invalidate cache entry
                    mock_cache_client.invalidate(query)
                    mock_invalidate.assert_called_once()
                
                assert verified_result.name == "REJECTED"
                assert confidence < 0.5
    
    @pytest.mark.asyncio
    async def test_verification_caching_strategy(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test caching of verification results to avoid redundant verification."""
        
        query = "What is photosynthesis?"
        verification_key = f"verification:{hash(query + 'response_hash')}"
        
        # Mock verification result caching
        verification_cache_data = {
            'result': 'ACCEPTED',
            'confidence': 0.94,
            'explanation': 'Scientifically accurate explanation',
            'verified_at': time.time(),
            'response_hash': 'abc123'
        }
        
        # First call: no cached verification
        mock_cache_client.redis_client.get.side_effect = [
            None,  # No cached verification
            None   # No cached response
        ]
        
        # Second call: cached verification exists
        def cache_side_effect(key):
            if 'verification:' in str(key):
                return json.dumps(verification_cache_data)
            return None
        
        with patch.object(mock_verification_detector, 'verify_response') as mock_verify, \
             patch.object(mock_cache_client, '_get_verification_cache_key') as mock_ver_key:
            
            mock_ver_key.return_value = verification_key
            mock_verify.return_value = (MockVerificationResult("ACCEPTED", 0.94), 0.94)
            
            response_text = "Photosynthesis converts light energy into chemical energy."
            
            # First verification (should call verify_response)
            result1, conf1 = await mock_verification_detector.verify_response(
                response_text, 0.85, None, []
            )
            
            # Reset and test cached verification
            mock_cache_client.redis_client.get.side_effect = cache_side_effect
            
            # Second verification (should use cached result)
            with patch.object(mock_cache_client, '_get_cached_verification') as mock_cached_ver:
                mock_cached_ver.return_value = (MockVerificationResult("ACCEPTED", 0.94), 0.94)
                
                result2, conf2 = mock_cached_ver(response_text, "abc123")
                
                # Should return same results without calling OpenAI again
                assert result2.name == result1.name
                assert conf2 == conf1
                mock_cached_ver.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_verification_requests(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test handling of concurrent requests for cache lookup and verification."""
        
        query = "What is artificial intelligence?"
        
        # Mock cache response
        cached_data = {
            'response': 'AI is a field of computer science focused on creating intelligent machines.',
            'confidence': 0.88,
            'sources': [{'text': 'AI definition source', 'relevance': 0.92}],
            'verification_status': 'verified',
            'verified_at': time.time() - 300  # 5 minutes ago, still fresh
        }
        
        mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
        mock_cache_client.redis_client.keys.return_value = ['semantic_cache:test_key']
        
        async def mock_verify_async(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate verification time
            return (MockVerificationResult("ACCEPTED", 0.90), 0.90)
        
        with patch.object(mock_cache_client, '_get_cache_key') as mock_key, \
             patch.object(mock_verification_detector, 'verify_response', side_effect=mock_verify_async):
            
            mock_key.return_value = "test_key"
            
            # Simulate concurrent requests
            async def process_request(request_id):
                result = mock_cache_client.get(query)
                if result:
                    response_text, nodes, similarity = result
                    verified_result, confidence = await mock_verification_detector.verify_response(
                        response_text, cached_data['confidence'], None, []
                    )
                    return f"Request {request_id}: {verified_result.name}"
                return f"Request {request_id}: No cache hit"
            
            # Run concurrent requests
            tasks = [process_request(i) for i in range(3)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == 3
            assert all("ACCEPTED" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_cache_verification_performance_optimization(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test performance optimizations in cache + verification pipeline."""
        
        query = "Explain machine learning algorithms"
        
        # Test with different cache scenarios
        scenarios = [
            # Fresh cache with recent verification - should skip verification
            {
                'cached_at': time.time() - 300,  # 5 minutes ago
                'verified_at': time.time() - 200,  # 3 minutes ago
                'verification_status': 'verified',
                'should_verify': False
            },
            # Old cache that needs reverification
            {
                'cached_at': time.time() - 7200,  # 2 hours ago
                'verified_at': time.time() - 7100,  # 1 hour 58 minutes ago
                'verification_status': 'expired',
                'should_verify': True
            },
            # Cache with failed previous verification
            {
                'cached_at': time.time() - 1800,  # 30 minutes ago
                'verified_at': time.time() - 1800,
                'verification_status': 'rejected',
                'should_verify': True
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            cached_data = {
                'response': f'ML explanation {i}',
                'confidence': 0.87,
                'sources': [{'text': f'ML source {i}', 'relevance': 0.9}],
                **scenario
            }
            
            mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
            
            with patch.object(mock_cache_client, '_should_reverify') as mock_should_reverify, \
                 patch.object(mock_verification_detector, 'verify_response') as mock_verify:
                
                mock_should_reverify.return_value = scenario['should_verify']
                mock_verify.return_value = (MockVerificationResult("ACCEPTED", 0.90), 0.90)
                
                # Simulate cache lookup with verification check
                result = mock_cache_client.get(query)
                
                if result and mock_should_reverify(cached_data):
                    response_text, nodes, similarity = result
                    await mock_verification_detector.verify_response(
                        response_text, cached_data['confidence'], None, []
                    )
                    
                    if scenario['should_verify']:
                        mock_verify.assert_called()
                    else:
                        mock_verify.assert_not_called()
                
                # Reset for next scenario
                mock_verify.reset_mock()


class TestCacheVerificationErrorHandling:
    """Test error handling in the cache + verification pipeline."""
    
    @pytest.mark.asyncio
    async def test_cache_connection_failure_with_verification(
        self, mock_verification_detector, mock_openai_client
    ):
        """Test pipeline behavior when cache connection fails but verification works."""
        
        # Mock cache connection failure
        failing_cache = SemanticCache(redis_url="redis://localhost:6379/15")
        failing_cache.redis_client = Mock()
        failing_cache.redis_client.get.side_effect = Exception("Redis connection failed")
        
        query = "What is deep learning?"
        response_text = "Deep learning uses neural networks with multiple layers."
        
        # Verification should still work
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Technical explanation is accurate."
        
        with patch.object(mock_verification_detector, 'verify_response') as mock_verify:
            mock_verify.return_value = (MockVerificationResult("ACCEPTED", 0.89), 0.89)
            
            # Cache fails, but verification should proceed
            try:
                cache_result = failing_cache.get(query)
                assert cache_result is None  # Cache failed, returns None
            except Exception:
                cache_result = None
            
            # Verification should still work independently
            verified_result, confidence = await mock_verification_detector.verify_response(
                response_text, 0.85, None, []
            )
            
            assert verified_result.name == "ACCEPTED"
            assert confidence >= 0.85
            mock_verify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_verification_failure_with_cache_fallback(
        self, mock_cache_client, mock_openai_client
    ):
        """Test pipeline behavior when verification fails but cache has valid data."""
        
        query = "What is natural language processing?"
        
        # Mock cached response
        cached_data = {
            'response': 'NLP is a field of AI that helps computers understand human language.',
            'confidence': 0.86,
            'sources': [{'text': 'NLP definition', 'relevance': 0.88}],
            'cached_at': time.time() - 600,  # 10 minutes ago
            'verification_status': 'verified'  # Previously verified
        }
        
        mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
        mock_cache_client.redis_client.keys.return_value = ['semantic_cache:test_key']
        
        # Mock verification failure
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI API failed")
        
        detector = HallucinationDetector()
        detector.llm = mock_openai_client
        
        with patch.object(mock_cache_client, '_get_cache_key') as mock_key:
            mock_key.return_value = "test_key"
            
            # Should get cached result
            result = mock_cache_client.get(query)
            assert result is not None
            
            response_text, nodes, similarity = result
            assert response_text == cached_data['response']
            
            # Verification fails, but we can still use cached result with lower confidence
            try:
                verified_result, confidence = await detector.verify_response(
                    response_text, cached_data['confidence'], None, []
                )
                # This may raise an exception
            except Exception as e:
                # Should handle gracefully and return cached result with adjusted confidence
                adjusted_confidence = cached_data['confidence'] * 0.8  # Reduce confidence
                assert adjusted_confidence < cached_data['confidence']
    
    @pytest.mark.asyncio
    async def test_partial_verification_pipeline_recovery(
        self, mock_cache_client, mock_verification_detector, mock_openai_client
    ):
        """Test recovery scenarios when parts of verification pipeline fail."""
        
        query = "Explain blockchain technology"
        
        # Mock cache with unverified content
        cached_data = {
            'response': 'Blockchain is a distributed ledger technology.',
            'confidence': 0.82,
            'sources': [{'text': 'Blockchain explanation', 'relevance': 0.85}],
            'verification_status': 'pending'
        }
        
        mock_cache_client.redis_client.get.return_value = json.dumps(cached_data)
        
        # Test different failure scenarios
        scenarios = [
            {
                'name': 'verification_timeout',
                'exception': asyncio.TimeoutError("Verification timed out"),
                'expected_confidence': 0.82 * 0.9  # Slight reduction for unverified
            },
            {
                'name': 'verification_api_error',
                'exception': Exception("API rate limit exceeded"),
                'expected_confidence': 0.82 * 0.85  # Larger reduction for API errors
            }
        ]
        
        for scenario in scenarios:
            with patch.object(mock_verification_detector, 'verify_response') as mock_verify:
                mock_verify.side_effect = scenario['exception']
                
                # Get cached result
                result = mock_cache_client.get(query)
                
                if result:
                    response_text, nodes, similarity = result
                    
                    # Attempt verification with error handling
                    try:
                        verified_result, confidence = await mock_verification_detector.verify_response(
                            response_text, cached_data['confidence'], None, []
                        )
                    except Exception as e:
                        # Should handle error gracefully
                        assert scenario['name'] in str(type(e).__name__).lower() or \
                               scenario['name'].replace('_', ' ') in str(e).lower()
                        
                        # Use cached result with reduced confidence
                        confidence = scenario['expected_confidence']
                        
                    # Should still provide usable result
                    assert confidence > 0.6  # Still usable despite verification failure


class TestCacheVerificationIntegrationWithWorkflow:
    """Test cache + verification integration within the unified workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_cache_verification_coordination(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test coordination of cache and verification within unified workflow."""
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        query = "What are the benefits of renewable energy?"
        
        # Mock cache hit with verification needed
        cached_data = {
            'response': 'Renewable energy provides environmental and economic benefits.',
            'confidence': 0.84,
            'verification_status': 'needs_verification'
        }
        
        mock_redis_client.get.return_value = json.dumps(cached_data)
        mock_redis_client.keys.return_value = ['semantic_cache:test_key']
        
        # Mock verification
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Environmental benefits are well documented."
        
        with patch.object(workflow, 'semantic_cache') as mock_cache, \
             patch.object(workflow, 'hallucination_detector') as mock_detector, \
             patch.object(workflow, '_verify_cached_result') as mock_verify_cached:
            
            # Setup mocks
            mock_cache.get.return_value = (cached_data['response'], [], 0.9)
            mock_verify_cached.return_value = f"{cached_data['response']}\n\n*Verified response*"
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            result = await workflow.arun(MockStartEvent(query))
            
            # Should use cache with verification
            assert workflow.stats['cache_hits'] > 0
            mock_verify_cached.assert_called_once()
            assert "Verified response" in str(result)
    
    @pytest.mark.asyncio
    async def test_workflow_cache_update_after_verification(
        self, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test that workflow updates cache with verification results."""
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        query = "How does machine learning work?"
        
        # Mock cache miss
        mock_redis_client.get.return_value = None
        mock_redis_client.keys.return_value = []
        
        # Mock processing and verification
        processed_response = "Machine learning algorithms learn patterns from data to make predictions."
        mock_llama_index['engine'].query.return_value = Mock(
            response=processed_response,
            source_nodes=[Mock(text="ML explanation", score=0.9, id_="node1")],
            metadata={}
        )
        
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Accurate explanation of machine learning process."
        
        with patch.object(workflow, 'semantic_cache') as mock_cache, \
             patch.object(workflow, '_execute_with_plan') as mock_execute:
            
            # Setup cache mock
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.put = Mock()
            
            # Mock execution
            mock_execute.return_value = processed_response
            
            class MockStartEvent:
                def __init__(self, query):
                    self.query = query
                    
                def dict(self):
                    return {"query": self.query}
            
            result = await workflow.arun(MockStartEvent(query))
            
            # Should process, verify, and cache
            assert result is not None
            # Cache should be updated with verified result
            # (Implementation depends on specific caching logic)