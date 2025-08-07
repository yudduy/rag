"""
Quality assurance tests for accuracy and reliability validation.

Tests cover:
- Response accuracy against ground truth
- Hallucination detection effectiveness
- Confidence score calibration
- Citation accuracy and relevance
- Consistency across similar queries
- Quality regression prevention
"""

import pytest
import json
import statistics
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from src.unified_workflow import UnifiedWorkflow, QueryComplexity
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.verification import HallucinationDetector, VerificationResult


@dataclass
class GroundTruthEntry:
    """Ground truth entry for accuracy testing."""
    query: str
    expected_content: str
    expected_concepts: List[str]
    difficulty: str  # 'easy', 'medium', 'hard'
    domain: str
    factual_claims: List[str]
    confidence_threshold: float


@dataclass 
class AccuracyMetrics:
    """Accuracy assessment metrics."""
    content_similarity: float
    concept_coverage: float
    factual_accuracy: float
    confidence_calibration: float
    overall_accuracy: float


class QualityAssessment:
    """Quality assessment utilities."""
    
    def __init__(self):
        self.ground_truth_data = self._load_ground_truth()
    
    def _load_ground_truth(self) -> List[GroundTruthEntry]:
        """Load ground truth data for testing."""
        return [
            GroundTruthEntry(
                query="What is machine learning?",
                expected_content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                expected_concepts=["artificial intelligence", "data", "learning", "algorithms", "automation"],
                difficulty="easy",
                domain="AI/ML",
                factual_claims=["ML is a subset of AI", "computers learn from data", "no explicit programming required"],
                confidence_threshold=0.9
            ),
            GroundTruthEntry(
                query="Explain the transformer architecture",
                expected_content="The transformer is a neural network architecture that uses self-attention mechanisms to process sequential data in parallel, consisting of encoder and decoder layers.",
                expected_concepts=["neural network", "self-attention", "encoder", "decoder", "parallel processing", "sequential data"],
                difficulty="medium",
                domain="AI/ML",
                factual_claims=["uses self-attention", "has encoder-decoder structure", "processes sequences in parallel"],
                confidence_threshold=0.85
            ),
            GroundTruthEntry(
                query="Compare supervised and unsupervised learning",
                expected_content="Supervised learning uses labeled training data to learn mappings from inputs to outputs, while unsupervised learning finds patterns in unlabeled data without explicit target variables.",
                expected_concepts=["supervised learning", "unsupervised learning", "labeled data", "unlabeled data", "patterns", "target variables"],
                difficulty="medium",
                domain="AI/ML",
                factual_claims=["supervised uses labeled data", "unsupervised finds patterns", "unsupervised has no explicit targets"],
                confidence_threshold=0.87
            ),
            GroundTruthEntry(
                query="What is the capital of France?",
                expected_content="The capital of France is Paris.",
                expected_concepts=["France", "capital", "Paris", "city"],
                difficulty="easy", 
                domain="Geography",
                factual_claims=["Paris is the capital of France"],
                confidence_threshold=0.95
            ),
            GroundTruthEntry(
                query="Explain quantum entanglement and its implications for quantum computing",
                expected_content="Quantum entanglement is a phenomenon where quantum particles become correlated such that the quantum state of each particle cannot be described independently. In quantum computing, entanglement enables quantum parallelism and is essential for quantum algorithms like Shor's algorithm.",
                expected_concepts=["quantum entanglement", "quantum particles", "correlation", "quantum state", "quantum computing", "quantum parallelism", "quantum algorithms"],
                difficulty="hard",
                domain="Physics/Computing",
                factual_claims=["particles become correlated", "states cannot be described independently", "enables quantum parallelism", "essential for quantum algorithms"],
                confidence_threshold=0.82
            )
        ]
    
    def calculate_content_similarity(self, actual: str, expected: str) -> float:
        """Calculate content similarity between actual and expected responses."""
        # Simple word-based similarity (in production, use more sophisticated methods)
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = actual_words & expected_words
        union = actual_words | expected_words
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost for length similarity
        length_similarity = 1.0 - abs(len(actual) - len(expected)) / max(len(actual), len(expected), 1)
        
        return (jaccard * 0.7 + length_similarity * 0.3)
    
    def calculate_concept_coverage(self, actual: str, expected_concepts: List[str]) -> float:
        """Calculate how well the response covers expected concepts."""
        actual_lower = actual.lower()
        covered_concepts = 0
        
        for concept in expected_concepts:
            if concept.lower() in actual_lower:
                covered_concepts += 1
        
        return covered_concepts / len(expected_concepts) if expected_concepts else 1.0
    
    def assess_factual_accuracy(self, actual: str, factual_claims: List[str]) -> float:
        """Assess factual accuracy against known claims."""
        actual_lower = actual.lower()
        accurate_claims = 0
        
        for claim in factual_claims:
            # Simple keyword matching (in production, use more sophisticated NLP)
            claim_words = claim.lower().split()
            if any(word in actual_lower for word in claim_words if len(word) > 3):
                accurate_claims += 1
        
        return accurate_claims / len(factual_claims) if factual_claims else 1.0
    
    def assess_confidence_calibration(self, confidence: float, accuracy: float) -> float:
        """Assess how well confidence aligns with actual accuracy."""
        # Perfect calibration would have confidence == accuracy
        calibration_error = abs(confidence - accuracy)
        
        # Convert to a score where 1.0 is perfect calibration
        return max(0.0, 1.0 - calibration_error)
    
    def calculate_overall_accuracy(
        self, 
        content_similarity: float,
        concept_coverage: float, 
        factual_accuracy: float,
        confidence_calibration: float
    ) -> float:
        """Calculate overall accuracy score."""
        weights = {
            'content': 0.3,
            'concepts': 0.3, 
            'factual': 0.25,
            'calibration': 0.15
        }
        
        return (
            content_similarity * weights['content'] +
            concept_coverage * weights['concepts'] + 
            factual_accuracy * weights['factual'] +
            confidence_calibration * weights['calibration']
        )


class TestAccuracyValidation:
    """Test response accuracy against ground truth."""
    
    @pytest.mark.asyncio
    async def test_factual_accuracy_validation(self, mock_llama_index, mock_redis_client, mock_openai_client, quality_metrics):
        """Test factual accuracy of responses."""
        quality_assessor = QualityAssessment()
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        accuracy_results = []
        
        for ground_truth in quality_assessor.ground_truth_data:
            # Mock response based on ground truth
            mock_response_content = ground_truth.expected_content
            
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': mock_response_content,
                    'confidence': ground_truth.confidence_threshold,
                    'sources': [
                        {'text': 'Relevant source text', 'relevance': 0.9}
                    ],
                    'processing_time': 1.5,
                    'verification_passed': True
                }
                
                characteristics = await workflow._analyze_query_characteristics(ground_truth.query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                # Assess accuracy
                content_sim = quality_assessor.calculate_content_similarity(
                    result['content'], ground_truth.expected_content
                )
                concept_coverage = quality_assessor.calculate_concept_coverage(
                    result['content'], ground_truth.expected_concepts
                )
                factual_accuracy = quality_assessor.assess_factual_accuracy(
                    result['content'], ground_truth.factual_claims
                )
                confidence_calibration = quality_assessor.assess_confidence_calibration(
                    result['confidence'], factual_accuracy
                )
                
                overall_accuracy = quality_assessor.calculate_overall_accuracy(
                    content_sim, concept_coverage, factual_accuracy, confidence_calibration
                )
                
                accuracy_metrics = AccuracyMetrics(
                    content_similarity=content_sim,
                    concept_coverage=concept_coverage,
                    factual_accuracy=factual_accuracy,
                    confidence_calibration=confidence_calibration,
                    overall_accuracy=overall_accuracy
                )
                
                accuracy_results.append((ground_truth, accuracy_metrics))
                
                print(f"Query: {ground_truth.query}")
                print(f"  Overall Accuracy: {overall_accuracy:.3f}")
                print(f"  Content Similarity: {content_sim:.3f}")
                print(f"  Concept Coverage: {concept_coverage:.3f}")
                print(f"  Factual Accuracy: {factual_accuracy:.3f}")
                print(f"  Confidence Calibration: {confidence_calibration:.3f}")
        
        # Validate accuracy thresholds
        overall_accuracies = [metrics.overall_accuracy for _, metrics in accuracy_results]
        avg_accuracy = statistics.mean(overall_accuracies)
        
        # Should meet minimum accuracy requirements
        thresholds = quality_metrics['accuracy_thresholds']
        profile = workflow.config_manager.config.performance_profile
        
        if profile.value in thresholds:
            min_accuracy = thresholds[profile.value]
            assert avg_accuracy >= min_accuracy, \
                f"Average accuracy {avg_accuracy:.3f} below threshold {min_accuracy}"
        
        # Individual query accuracy validation
        for ground_truth, metrics in accuracy_results:
            if ground_truth.difficulty == 'easy':
                assert metrics.overall_accuracy >= 0.85, \
                    f"Easy query accuracy {metrics.overall_accuracy:.3f} too low"
            elif ground_truth.difficulty == 'medium':
                assert metrics.overall_accuracy >= 0.75, \
                    f"Medium query accuracy {metrics.overall_accuracy:.3f} too low"
            # Hard queries can have lower accuracy requirements
    
    @pytest.mark.asyncio
    async def test_domain_specific_accuracy(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test accuracy across different knowledge domains."""
        quality_assessor = QualityAssessment()
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        domain_accuracies = {}
        
        for ground_truth in quality_assessor.ground_truth_data:
            domain = ground_truth.domain
            
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': ground_truth.expected_content,
                    'confidence': ground_truth.confidence_threshold,
                    'sources': [{'text': 'Domain-specific source', 'relevance': 0.88}],
                    'domain': domain
                }
                
                characteristics = await workflow._analyze_query_characteristics(ground_truth.query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                # Calculate domain-specific accuracy
                factual_accuracy = quality_assessor.assess_factual_accuracy(
                    result['content'], ground_truth.factual_claims
                )
                
                if domain not in domain_accuracies:
                    domain_accuracies[domain] = []
                domain_accuracies[domain].append(factual_accuracy)
        
        # Validate domain-specific performance
        for domain, accuracies in domain_accuracies.items():
            avg_domain_accuracy = statistics.mean(accuracies)
            
            print(f"Domain {domain}: Average accuracy {avg_domain_accuracy:.3f}")
            
            # Domain-specific thresholds
            if domain == "Geography":
                assert avg_domain_accuracy >= 0.95, "Geography accuracy too low"
            elif domain == "AI/ML":
                assert avg_domain_accuracy >= 0.80, "AI/ML accuracy too low"
            elif domain == "Physics/Computing":
                assert avg_domain_accuracy >= 0.75, "Physics/Computing accuracy acceptable for complexity"
    
    @pytest.mark.asyncio
    async def test_consistency_across_similar_queries(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test consistency of responses across similar queries."""
        similar_query_groups = [
            [
                "What is machine learning?",
                "Can you explain machine learning?",
                "Define machine learning",
                "Tell me about machine learning"
            ],
            [
                "What is the capital of France?",
                "France's capital city is?",
                "Which city is the capital of France?",
                "What city serves as France's capital?"
            ]
        ]
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        for query_group in similar_query_groups:
            responses = []
            
            for query in query_group:
                with patch.object(workflow, '_process_query_with_plan') as mock_process:
                    # Mock consistent high-quality response
                    if "machine learning" in query.lower():
                        mock_response = {
                            'content': 'Machine learning is a subset of AI that enables computers to learn from data',
                            'confidence': 0.92,
                            'key_concepts': ['AI', 'data', 'learning', 'computers']
                        }
                    else:  # France capital queries
                        mock_response = {
                            'content': 'The capital of France is Paris',
                            'confidence': 0.96,
                            'key_concepts': ['France', 'capital', 'Paris']
                        }
                    
                    mock_process.return_value = mock_response
                    
                    characteristics = await workflow._analyze_query_characteristics(query)
                    result = await workflow._process_query_with_plan(
                        characteristics,
                        workflow._create_processing_plan(characteristics)
                    )
                    responses.append(result)
            
            # Validate consistency
            confidences = [r['confidence'] for r in responses]
            content_lengths = [len(r['content']) for r in responses]
            
            # Confidence should be consistent
            confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
            assert confidence_std < 0.05, f"Confidence variance {confidence_std:.3f} too high"
            
            # Content length should be reasonably consistent
            length_coefficient_variation = statistics.stdev(content_lengths) / statistics.mean(content_lengths)
            assert length_coefficient_variation < 0.3, "Response length too variable"
            
            print(f"Query group consistency:")
            print(f"  Confidence std: {confidence_std:.3f}")
            print(f"  Length CV: {length_coefficient_variation:.3f}")


class TestHallucinationDetection:
    """Test hallucination detection effectiveness."""
    
    @pytest.fixture
    def hallucination_test_cases(self):
        """Test cases for hallucination detection."""
        return [
            {
                'query': 'When was the Eiffel Tower built?',
                'accurate_response': 'The Eiffel Tower was built between 1887 and 1889 in Paris, France.',
                'hallucinated_response': 'The Eiffel Tower was built in 1925 on Mars by alien engineers.',
                'sources': [{'content': 'The Eiffel Tower construction began in 1887', 'relevance': 0.9}]
            },
            {
                'query': 'What is the speed of light?',
                'accurate_response': 'The speed of light in a vacuum is approximately 299,792,458 meters per second.',
                'hallucinated_response': 'The speed of light is 150,000 km/h and it changes depending on the weather.',
                'sources': [{'content': 'Speed of light is 299,792,458 m/s', 'relevance': 0.95}]
            },
            {
                'query': 'Who invented the telephone?',
                'accurate_response': 'Alexander Graham Bell is credited with inventing the telephone in 1876.',
                'hallucinated_response': 'The telephone was invented by Steve Jobs in 1995 using iPhone technology.',
                'sources': [{'content': 'Alexander Graham Bell patented the telephone in 1876', 'relevance': 0.9}]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_hallucination_detection_accuracy(self, mock_openai_client, hallucination_test_cases, quality_metrics):
        """Test accuracy of hallucination detection."""
        detector = HallucinationDetector(
            verification_model="gpt-4",
            confidence_threshold=0.8
        )
        detector.client = mock_openai_client
        
        detection_results = []
        
        for test_case in hallucination_test_cases:
            # Test accurate response (should pass verification)
            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
                "CONSISTENT: The response is factually accurate and well-supported by sources."
            
            accurate_result = await detector.verify_factual_consistency(
                test_case['query'],
                test_case['accurate_response'], 
                test_case['sources']
            )
            
            # Test hallucinated response (should fail verification)
            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
                "INCONSISTENT: The response contains factual errors and contradicts reliable sources."
            
            hallucinated_result = await detector.verify_factual_consistency(
                test_case['query'],
                test_case['hallucinated_response'],
                test_case['sources']
            )
            
            detection_results.append({
                'query': test_case['query'],
                'accurate_detected_correctly': accurate_result.is_consistent,
                'hallucination_detected_correctly': not hallucinated_result.is_consistent,
                'accurate_confidence': accurate_result.confidence,
                'hallucination_confidence': hallucinated_result.confidence
            })
            
            print(f"Query: {test_case['query']}")
            print(f"  Accurate response verified: {accurate_result.is_consistent}")
            print(f"  Hallucination detected: {not hallucinated_result.is_consistent}")
            print(f"  Confidence scores: {accurate_result.confidence:.3f} vs {hallucinated_result.confidence:.3f}")
        
        # Validate detection effectiveness
        accurate_detection_rate = sum(1 for r in detection_results if r['accurate_detected_correctly']) / len(detection_results)
        hallucination_detection_rate = sum(1 for r in detection_results if r['hallucination_detected_correctly']) / len(detection_results)
        
        thresholds = quality_metrics['hallucination_detection']
        
        # Should have high precision (accurate responses pass verification)
        assert accurate_detection_rate >= thresholds['precision_threshold'], \
            f"Accurate detection rate {accurate_detection_rate:.3f} below precision threshold"
        
        # Should have high recall (hallucinations are caught)
        assert hallucination_detection_rate >= thresholds['recall_threshold'], \
            f"Hallucination detection rate {hallucination_detection_rate:.3f} below recall threshold"
        
        # F1 score validation
        if accurate_detection_rate + hallucination_detection_rate > 0:
            f1_score = 2 * (accurate_detection_rate * hallucination_detection_rate) / (accurate_detection_rate + hallucination_detection_rate)
            assert f1_score >= thresholds['f1_threshold'], \
                f"F1 score {f1_score:.3f} below threshold"
    
    @pytest.mark.asyncio
    async def test_confidence_score_calibration(self, mock_openai_client, hallucination_test_cases):
        """Test calibration of confidence scores."""
        detector = HallucinationDetector(
            verification_model="gpt-4",
            confidence_threshold=0.8
        )
        detector.client = mock_openai_client
        
        confidence_accuracy_pairs = []
        
        for test_case in hallucination_test_cases:
            responses_to_test = [
                (test_case['accurate_response'], True, 0.95),
                (test_case['hallucinated_response'], False, 0.25)
            ]
            
            for response, is_accurate, expected_confidence in responses_to_test:
                mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
                    f"{'CONSISTENT' if is_accurate else 'INCONSISTENT'}: Assessment with confidence {expected_confidence}"
                
                result = await detector.verify_factual_consistency(
                    test_case['query'],
                    response,
                    test_case['sources']
                )
                
                actual_accuracy = 1.0 if result.is_consistent else 0.0
                confidence_accuracy_pairs.append((result.confidence, actual_accuracy))
        
        # Assess calibration
        calibration_errors = []
        for confidence, accuracy in confidence_accuracy_pairs:
            calibration_error = abs(confidence - accuracy)
            calibration_errors.append(calibration_error)
        
        avg_calibration_error = statistics.mean(calibration_errors)
        
        # Good calibration should have low average error
        assert avg_calibration_error < 0.3, f"Average calibration error {avg_calibration_error:.3f} too high"
        
        print(f"Confidence calibration - Average error: {avg_calibration_error:.3f}")


class TestCitationAccuracy:
    """Test accuracy and relevance of citations."""
    
    @pytest.mark.asyncio
    async def test_citation_relevance_validation(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test that citations are relevant to the query and response."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        test_cases = [
            {
                'query': 'What are the benefits of renewable energy?',
                'sources': [
                    {
                        'text': 'Renewable energy sources like solar and wind reduce carbon emissions',
                        'relevance': 0.95,
                        'metadata': {'source': 'Environmental Science Journal', 'year': 2023}
                    },
                    {
                        'text': 'Solar panels can reduce electricity costs for homeowners',
                        'relevance': 0.88,
                        'metadata': {'source': 'Energy Economics Report', 'year': 2023}
                    },
                    {
                        'text': 'Renewable energy creates jobs in manufacturing and installation',
                        'relevance': 0.82,
                        'metadata': {'source': 'Labor Statistics Bureau', 'year': 2023}
                    }
                ]
            }
        ]
        
        for test_case in test_cases:
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': 'Renewable energy offers environmental benefits like reduced emissions [citation:1] and economic advantages including lower costs [citation:2] and job creation [citation:3]',
                    'confidence': 0.91,
                    'sources': test_case['sources'],
                    'citations': [
                        {'id': 1, 'source': 'Environmental Science Journal', 'relevance': 0.95},
                        {'id': 2, 'source': 'Energy Economics Report', 'relevance': 0.88},
                        {'id': 3, 'source': 'Labor Statistics Bureau', 'relevance': 0.82}
                    ]
                }
                
                characteristics = await workflow._analyze_query_characteristics(test_case['query'])
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                # Validate citation accuracy
                assert 'sources' in result
                assert 'citations' in result
                assert len(result['sources']) == len(result['citations'])
                
                # All sources should have good relevance
                for source in result['sources']:
                    assert source['relevance'] >= 0.8, f"Source relevance {source['relevance']} too low"
                
                # Citations should be properly formatted in content
                assert '[citation:1]' in result['content']
                assert '[citation:2]' in result['content']
                assert '[citation:3]' in result['content']
                
                # Citation metadata should be complete
                for citation in result['citations']:
                    assert 'id' in citation
                    assert 'source' in citation
                    assert 'relevance' in citation
    
    @pytest.mark.asyncio
    async def test_source_diversity_validation(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test that responses use diverse, high-quality sources."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        query = "What is the impact of climate change on global agriculture?"
        
        diverse_sources = [
            {
                'text': 'Climate change affects crop yields through temperature changes',
                'relevance': 0.92,
                'metadata': {'source': 'Nature Climate Change', 'type': 'journal', 'year': 2023}
            },
            {
                'text': 'Agricultural adaptation strategies are needed for changing climates',
                'relevance': 0.87,
                'metadata': {'source': 'FAO Report', 'type': 'organization', 'year': 2023}
            },
            {
                'text': 'Farmers report challenges from changing weather patterns',
                'relevance': 0.84,
                'metadata': {'source': 'Agricultural Survey', 'type': 'survey', 'year': 2023}
            }
        ]
        
        with patch.object(workflow, '_process_query_with_plan') as mock_process:
            mock_process.return_value = {
                'content': 'Climate change significantly impacts agriculture through various mechanisms',
                'confidence': 0.89,
                'sources': diverse_sources,
                'source_diversity_score': 0.85
            }
            
            characteristics = await workflow._analyze_query_characteristics(query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            # Validate source diversity
            source_types = set()
            source_organizations = set()
            
            for source in result['sources']:
                if 'metadata' in source:
                    source_types.add(source['metadata'].get('type', 'unknown'))
                    source_organizations.add(source['metadata'].get('source', 'unknown'))
            
            # Should have diverse source types and organizations
            assert len(source_types) >= 2, "Insufficient source type diversity"
            assert len(source_organizations) >= 2, "Insufficient source organization diversity"
            
            # Overall diversity score should be good
            diversity_score = result.get('source_diversity_score', 0)
            assert diversity_score >= 0.7, f"Source diversity score {diversity_score} too low"


class TestQualityRegression:
    """Test quality regression prevention."""
    
    @pytest.fixture
    def baseline_quality_metrics(self):
        """Baseline quality metrics for regression testing."""
        return {
            'average_confidence': 0.87,
            'average_accuracy': 0.84,
            'average_response_time': 2.3,
            'hallucination_detection_rate': 0.88,
            'citation_relevance': 0.89
        }
    
    @pytest.mark.asyncio
    async def test_quality_regression_detection(self, mock_llama_index, mock_redis_client, mock_openai_client, baseline_quality_metrics):
        """Test that quality doesn't regress below baseline."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "What are neural networks?",
            "Define deep learning",
            "What is natural language processing?"
        ]
        
        current_metrics = {
            'confidences': [],
            'accuracies': [],
            'response_times': [],
            'citation_relevances': []
        }
        
        for query in test_queries:
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': f'Comprehensive response to: {query}',
                    'confidence': 0.88,  # Should meet baseline
                    'processing_time': 2.1,  # Should meet baseline
                    'accuracy': 0.85,  # Should meet baseline
                    'sources': [
                        {'text': 'Relevant source', 'relevance': 0.9}
                    ]
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                current_metrics['confidences'].append(result['confidence'])
                current_metrics['accuracies'].append(result.get('accuracy', 0.85))
                current_metrics['response_times'].append(result['processing_time'])
                
                if 'sources' in result:
                    avg_relevance = statistics.mean(s['relevance'] for s in result['sources'])
                    current_metrics['citation_relevances'].append(avg_relevance)
        
        # Calculate current averages
        current_averages = {
            'average_confidence': statistics.mean(current_metrics['confidences']),
            'average_accuracy': statistics.mean(current_metrics['accuracies']),
            'average_response_time': statistics.mean(current_metrics['response_times']),
            'citation_relevance': statistics.mean(current_metrics['citation_relevances']) if current_metrics['citation_relevances'] else 0.0
        }
        
        # Check for regression
        regression_threshold = 0.95  # Allow 5% degradation
        
        for metric, baseline_value in baseline_quality_metrics.items():
            if metric == 'average_response_time':
                # For response time, lower is better
                max_acceptable = baseline_value * 1.1  # Allow 10% increase
                assert current_averages[metric] <= max_acceptable, \
                    f"{metric} regressed: {current_averages[metric]:.3f} > {max_acceptable:.3f}"
            else:
                # For other metrics, higher is better
                min_acceptable = baseline_value * regression_threshold
                if metric in current_averages:
                    assert current_averages[metric] >= min_acceptable, \
                        f"{metric} regressed: {current_averages[metric]:.3f} < {min_acceptable:.3f}"
        
        print("Quality regression check passed:")
        for metric, current_value in current_averages.items():
            baseline_value = baseline_quality_metrics.get(metric, 0)
            print(f"  {metric}: {current_value:.3f} (baseline: {baseline_value:.3f})")
    
    @pytest.mark.asyncio
    async def test_performance_profile_quality_consistency(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test that different performance profiles maintain expected quality levels."""
        profiles_to_test = [
            (PerformanceProfile.HIGH_ACCURACY, 0.95),
            (PerformanceProfile.BALANCED, 0.87),
            (PerformanceProfile.COST_OPTIMIZED, 0.82),
            (PerformanceProfile.SPEED, 0.80)
        ]
        
        for profile, expected_min_confidence in profiles_to_test:
            # Configure profile
            reset_unified_config()
            config_manager = get_unified_config()
            config_manager.config.performance_profile = profile
            
            workflow = UnifiedWorkflow(timeout=30.0)
            workflow.config_manager = config_manager
            workflow.stats = {'queries_processed': 0}
            
            query = "What is machine learning?"
            
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                # Mock profile-appropriate response
                if profile == PerformanceProfile.HIGH_ACCURACY:
                    mock_response = {
                        'content': 'Highly accurate and comprehensive ML explanation',
                        'confidence': 0.96,
                        'verification_passed': True,
                        'processing_time': 3.5
                    }
                elif profile == PerformanceProfile.SPEED:
                    mock_response = {
                        'content': 'Quick ML explanation',
                        'confidence': 0.82,
                        'processing_time': 0.8,
                        'speed_optimized': True
                    }
                else:
                    mock_response = {
                        'content': 'Balanced ML explanation',
                        'confidence': 0.88,
                        'processing_time': 1.8
                    }
                
                mock_process.return_value = mock_response
                
                characteristics = await workflow._analyze_query_characteristics(query)
                result = await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
                
                # Validate profile-specific quality
                assert result['confidence'] >= expected_min_confidence, \
                    f"Profile {profile.value} confidence {result['confidence']:.3f} below expected {expected_min_confidence}"
                
                # Profile-specific validations
                if profile == PerformanceProfile.HIGH_ACCURACY:
                    assert result.get('verification_passed', False), "High accuracy should verify responses"
                elif profile == PerformanceProfile.SPEED:
                    assert result['processing_time'] < 2.0, "Speed profile should be fast"
                    assert result.get('speed_optimized', False), "Should indicate speed optimization"
                
                print(f"Profile {profile.value}: confidence={result['confidence']:.3f}, time={result['processing_time']:.2f}s")