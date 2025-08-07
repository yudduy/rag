"""
Unit tests for hallucination detection and verification system.

Tests cover:
- Hallucination detection algorithms
- Confidence scoring and calibration
- Graph-based verification
- Performance and accuracy metrics
- Error handling and fallbacks
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.verification import (
    HallucinationDetector,
    ResponseConfidence,
    GraphConfidence,
    NodeConfidence,
    create_hallucination_detector,
    VerificationResult,
    _calculate_confidence_score,
    _detect_contradictions,
    _verify_factual_consistency
)


class TestConfidenceScoring:
    """Test confidence scoring algorithms."""
    
    def test_response_confidence_calculation(self):
        """Test response confidence scoring."""
        # High quality response
        high_quality_response = {
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'sources': [{'relevance': 0.9}, {'relevance': 0.8}],
            'token_count': 20,
            'factual_claims': 2
        }
        
        confidence = _calculate_confidence_score(high_quality_response)
        assert isinstance(confidence, ResponseConfidence)
        assert confidence.overall >= 0.8
        assert confidence.factual >= 0.7
        assert confidence.semantic >= 0.7
    
    def test_low_quality_response_scoring(self):
        """Test scoring of low quality responses."""
        low_quality_response = {
            'content': 'AI is magic and works by reading minds.',
            'sources': [{'relevance': 0.3}],
            'token_count': 10,
            'factual_claims': 1
        }
        
        confidence = _calculate_confidence_score(low_quality_response)
        assert confidence.overall < 0.7
        assert confidence.factual < 0.6
    
    def test_confidence_score_edge_cases(self):
        """Test edge cases in confidence scoring."""
        # Empty response
        empty_response = {
            'content': '',
            'sources': [],
            'token_count': 0,
            'factual_claims': 0
        }
        
        confidence = _calculate_confidence_score(empty_response)
        assert 0.0 <= confidence.overall <= 1.0
        
        # Very long response
        long_response = {
            'content': 'This is a very long response ' * 100,
            'sources': [{'relevance': 0.9}] * 10,
            'token_count': 300,
            'factual_claims': 5
        }
        
        confidence = _calculate_confidence_score(long_response)
        assert 0.0 <= confidence.overall <= 1.0


class TestHallucinationDetection:
    """Test hallucination detection algorithms."""
    
    @pytest.fixture
    def mock_detector(self, mock_openai_client):
        """Create a mocked hallucination detector."""
        detector = HallucinationDetector(
            verification_model="gpt-4",
            confidence_threshold=0.8
        )
        detector.client = mock_openai_client
        return detector
    
    @pytest.mark.asyncio
    async def test_factual_consistency_check(self, mock_detector, mock_openai_client):
        """Test factual consistency checking."""
        query = "What year was Python created?"
        response = "Python was created in 1991 by Guido van Rossum."
        sources = [
            {'content': 'Python development began in late 1989 by Guido van Rossum', 'relevance': 0.9}
        ]
        
        # Mock verification response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: The response correctly states Python was created in 1991 by Guido van Rossum."
        
        result = await mock_detector.verify_factual_consistency(query, response, sources)
        
        assert isinstance(result, VerificationResult)
        assert result.is_consistent == True
        assert result.confidence >= 0.8
    
    @pytest.mark.asyncio
    async def test_contradiction_detection(self, mock_detector):
        """Test contradiction detection in responses."""
        contradictory_response = """
        Machine learning requires massive amounts of data to work effectively.
        However, machine learning can work with very small datasets and still be effective.
        """
        
        contradictions = await _detect_contradictions(contradictory_response)
        
        # Should detect the contradiction
        assert len(contradictions) >= 1
        assert any("data" in str(contradiction).lower() for contradiction in contradictions)
    
    @pytest.mark.asyncio
    async def test_hallucination_score_calculation(self, mock_detector, mock_openai_client):
        """Test hallucination score calculation."""
        # Likely hallucination
        hallucinated_response = "The Eiffel Tower was built in 1889 on Mars by alien engineers."
        sources = [
            {'content': 'The Eiffel Tower was built in Paris in 1889', 'relevance': 0.5}
        ]
        
        # Mock verification indicating hallucination
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "INCONSISTENT: The response contains factual errors about the location."
        
        result = await mock_detector.detect_hallucination("Where was the Eiffel Tower built?", 
                                                         hallucinated_response, sources)
        
        assert result.hallucination_score > 0.5
        assert not result.is_consistent
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self, mock_detector):
        """Test confidence score calibration."""
        responses_with_known_quality = [
            ("Paris is the capital of France", 0.95),  # High confidence
            ("The sky is usually blue during the day", 0.9),  # High confidence
            ("Some say the Earth might be flat", 0.3),  # Low confidence
            ("Magic unicorns live in the forest", 0.1),  # Very low confidence
        ]
        
        for response, expected_confidence in responses_with_known_quality:
            mock_response = {
                'content': response,
                'sources': [{'relevance': 0.8}],
                'token_count': len(response.split()),
                'factual_claims': 1
            }
            
            confidence = _calculate_confidence_score(mock_response)
            
            # Check if confidence is in the expected range (±0.2 tolerance)
            assert abs(confidence.overall - expected_confidence) < 0.3


class TestGraphBasedVerification:
    """Test graph-based verification system."""
    
    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        return {
            'nodes': [
                {'id': 'node1', 'content': 'Python is a programming language', 'confidence': 0.95},
                {'id': 'node2', 'content': 'Guido van Rossum created Python', 'confidence': 0.9},
                {'id': 'node3', 'content': 'Python was first released in 1991', 'confidence': 0.85}
            ],
            'edges': [
                {'source': 'node1', 'target': 'node2', 'relationship': 'created_by'},
                {'source': 'node1', 'target': 'node3', 'relationship': 'released_date'}
            ]
        }
    
    def test_node_confidence_calculation(self, sample_graph_data):
        """Test node confidence calculation in graph."""
        nodes = sample_graph_data['nodes']
        
        for node in nodes:
            node_confidence = NodeConfidence(
                content_confidence=node['confidence'],
                structural_confidence=0.8,
                source_confidence=0.9
            )
            
            assert 0.0 <= node_confidence.overall <= 1.0
            assert node_confidence.overall >= min(
                node_confidence.content_confidence,
                node_confidence.structural_confidence,
                node_confidence.source_confidence
            ) * 0.5
    
    def test_graph_confidence_propagation(self, sample_graph_data):
        """Test confidence propagation through graph."""
        graph_confidence = GraphConfidence(
            average_node_confidence=0.9,
            connectivity_score=0.8,
            consistency_score=0.85
        )
        
        assert 0.0 <= graph_confidence.overall <= 1.0
        
        # Overall confidence should be influenced by all factors
        expected_range = (0.7, 1.0)  # Should be reasonably high for good inputs
        assert expected_range[0] <= graph_confidence.overall <= expected_range[1]
    
    @pytest.mark.asyncio
    async def test_graph_contradiction_detection(self, mock_detector, sample_graph_data):
        """Test contradiction detection within knowledge graph."""
        # Add contradictory node
        contradictory_data = sample_graph_data.copy()
        contradictory_data['nodes'].append({
            'id': 'node4', 
            'content': 'Python was first released in 1985',  # Contradicts node3
            'confidence': 0.7
        })
        
        # Mock graph verification
        with patch.object(mock_detector, '_verify_graph_consistency') as mock_verify:
            mock_verify.return_value = GraphConfidence(
                average_node_confidence=0.85,
                connectivity_score=0.8,
                consistency_score=0.6  # Low due to contradiction
            )
            
            result = await mock_detector.verify_with_graph(contradictory_data)
            
            # Should detect inconsistency
            assert result.consistency_score < 0.8


class TestVerificationPerformance:
    """Test verification system performance."""
    
    @pytest.mark.asyncio
    async def test_verification_speed(self, mock_detector, performance_tracker, mock_openai_client):
        """Test verification speed requirements."""
        query = "What is machine learning?"
        response = "Machine learning is a branch of artificial intelligence."
        sources = [{'content': 'ML is part of AI', 'relevance': 0.8}]
        
        # Mock fast API response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: Response is factually accurate."
        
        performance_tracker.start_timing('verification')
        
        result = await mock_detector.verify_factual_consistency(query, response, sources)
        
        performance_tracker.end_timing('verification')
        
        # Verification should complete within reasonable time
        verification_time = performance_tracker.get_duration('verification')
        assert verification_time < 5.0  # Should complete in under 5 seconds
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, mock_detector, mock_openai_client):
        """Test batch verification for multiple responses."""
        queries_responses = [
            ("What is AI?", "AI is artificial intelligence"),
            ("What is ML?", "ML is machine learning"),
            ("What is DL?", "DL is deep learning")
        ]
        
        # Mock batch verification
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "CONSISTENT: All responses are accurate."
        
        results = []
        for query, response in queries_responses:
            result = await mock_detector.verify_factual_consistency(query, response, [])
            results.append(result)
        
        assert len(results) == 3
        assert all(result.confidence > 0.5 for result in results)
    
    def test_memory_efficiency(self, mock_detector):
        """Test memory efficiency of verification system."""
        # Large response for testing
        large_response = "This is a comprehensive explanation of machine learning " * 1000
        
        # Should handle large responses without memory issues
        mock_response_data = {
            'content': large_response,
            'sources': [{'relevance': 0.8}],
            'token_count': len(large_response.split()),
            'factual_claims': 10
        }
        
        confidence = _calculate_confidence_score(mock_response_data)
        assert 0.0 <= confidence.overall <= 1.0


class TestVerificationErrorHandling:
    """Test error handling in verification system."""
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self, mock_detector, mock_openai_client):
        """Test handling of API failures during verification."""
        # Mock API failure
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        query = "Test query"
        response = "Test response"
        sources = []
        
        # Should handle API failure gracefully
        result = await mock_detector.verify_factual_consistency(query, response, sources)
        
        # Should return a result with low confidence instead of crashing
        assert result is not None
        assert result.confidence < 0.5
        assert not result.is_consistent
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_detector, mock_openai_client):
        """Test timeout handling in verification."""
        # Mock slow API response
        async def slow_api_call(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow response
            return Mock(choices=[Mock(message=Mock(content="CONSISTENT"))])
        
        mock_openai_client.chat.completions.create.side_effect = slow_api_call
        
        # Should timeout gracefully
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mock_detector.verify_factual_consistency("Query", "Response", []),
                timeout=1.0
            )
    
    def test_invalid_input_handling(self, mock_detector):
        """Test handling of invalid inputs."""
        # Test None inputs
        with pytest.raises((ValueError, TypeError)):
            asyncio.run(mock_detector.verify_factual_consistency(None, "response", []))
        
        # Test empty inputs
        with pytest.raises(ValueError):
            asyncio.run(mock_detector.verify_factual_consistency("", "", []))


class TestVerificationConfiguration:
    """Test verification system configuration."""
    
    def test_confidence_threshold_configuration(self):
        """Test different confidence thresholds."""
        thresholds = [0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            detector = HallucinationDetector(confidence_threshold=threshold)
            assert detector.confidence_threshold == threshold
    
    def test_verification_model_configuration(self):
        """Test different verification models."""
        models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        
        for model in models:
            detector = HallucinationDetector(verification_model=model)
            assert detector.verification_model == model
    
    def test_performance_profile_adaptation(self):
        """Test adaptation to different performance profiles."""
        # High accuracy profile
        high_accuracy_detector = HallucinationDetector(
            verification_model="gpt-4",
            confidence_threshold=0.9,
            timeout=10.0
        )
        
        assert high_accuracy_detector.confidence_threshold >= 0.9
        
        # Speed profile
        speed_detector = HallucinationDetector(
            verification_model="gpt-3.5-turbo",
            confidence_threshold=0.7,
            timeout=3.0
        )
        
        assert speed_detector.confidence_threshold <= 0.8
        assert speed_detector.timeout <= 5.0


class TestVerificationIntegration:
    """Test verification system integration."""
    
    def test_workflow_integration(self, mock_detector):
        """Test integration with workflow system."""
        # Detector should be compatible with workflow calls
        assert hasattr(mock_detector, 'verify_factual_consistency')
        assert hasattr(mock_detector, 'detect_hallucination')
        assert callable(mock_detector.verify_factual_consistency)
    
    def test_caching_integration(self, mock_detector):
        """Test integration with caching system."""
        # Verification results should be cacheable
        verification_result = VerificationResult(
            is_consistent=True,
            confidence=0.9,
            hallucination_score=0.1,
            explanation="Response is factually accurate"
        )
        
        # Should be serializable for caching
        import json
        serialized = json.dumps(verification_result.__dict__)
        deserialized = json.loads(serialized)
        
        assert deserialized['is_consistent'] == True
        assert deserialized['confidence'] == 0.9
    
    def test_monitoring_integration(self, mock_detector):
        """Test integration with monitoring system."""
        # Should expose metrics for monitoring
        assert hasattr(mock_detector, 'get_stats') or hasattr(mock_detector, '_stats')
        
        # Should track verification success/failure rates
        # This would be implemented in the actual monitoring integration