"""
Test data sets and mock services for comprehensive testing.

Provides:
- Curated test datasets for different scenarios
- Mock external service implementations
- Test data generators
- Benchmark datasets
- Ground truth data for validation
"""

import json
import random
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from unittest.mock import AsyncMock, Mock
from pathlib import Path
import tempfile
import os


@dataclass
class TestQuery:
    """Structured test query with expected outcomes."""
    id: str
    query: str
    category: str  # 'factual', 'analytical', 'conversational', 'multimodal'
    complexity: str  # 'simple', 'moderate', 'complex'
    domain: str  # 'AI/ML', 'Science', 'History', 'Geography', etc.
    expected_response_type: str  # 'direct', 'explanatory', 'comparative', 'list'
    expected_concepts: List[str]
    ground_truth_answer: str
    confidence_threshold: float
    processing_time_target: float  # seconds
    cost_target: float  # USD
    requires_decomposition: bool = False
    has_multimodal_elements: bool = False
    factual_claims: List[str] = None
    sources_required: int = 1


@dataclass
class TestDocument:
    """Test document with metadata."""
    id: str
    title: str
    content: str
    category: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    word_count: int
    key_concepts: List[str]
    factual_statements: List[str] = None
    author: str = "Test Author"
    publication_date: str = "2023"


@dataclass
class MockServiceResponse:
    """Mock service response structure."""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float
    cost: float
    sources: List[Dict[str, Any]] = None


class TestDatasets:
    """Comprehensive test datasets for SOTA RAG testing."""
    
    def __init__(self):
        self.queries = self._load_test_queries()
        self.documents = self._load_test_documents()
        self.benchmark_datasets = self._load_benchmark_datasets()
    
    def _load_test_queries(self) -> List[TestQuery]:
        """Load comprehensive test query dataset."""
        return [
            # Simple Factual Queries
            TestQuery(
                id="factual_001",
                query="What is the capital of France?",
                category="factual",
                complexity="simple",
                domain="Geography",
                expected_response_type="direct",
                expected_concepts=["France", "capital", "Paris", "city"],
                ground_truth_answer="The capital of France is Paris.",
                confidence_threshold=0.95,
                processing_time_target=1.0,
                cost_target=0.005,
                factual_claims=["Paris is the capital of France"]
            ),
            TestQuery(
                id="factual_002", 
                query="When was Python programming language created?",
                category="factual",
                complexity="simple",
                domain="Technology",
                expected_response_type="direct",
                expected_concepts=["Python", "programming language", "created", "1991", "Guido van Rossum"],
                ground_truth_answer="Python was created by Guido van Rossum in 1991.",
                confidence_threshold=0.92,
                processing_time_target=1.2,
                cost_target=0.006,
                factual_claims=["Python was created in 1991", "Guido van Rossum created Python"]
            ),
            
            # Moderate Complexity Queries  
            TestQuery(
                id="analytical_001",
                query="What is machine learning and how does it work?",
                category="analytical",
                complexity="moderate", 
                domain="AI/ML",
                expected_response_type="explanatory",
                expected_concepts=["machine learning", "artificial intelligence", "algorithms", "data", "patterns", "prediction"],
                ground_truth_answer="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by using algorithms to analyze data, identify patterns, and make predictions or decisions.",
                confidence_threshold=0.88,
                processing_time_target=2.0,
                cost_target=0.012,
                factual_claims=["ML is subset of AI", "computers learn from data", "uses algorithms to find patterns"],
                sources_required=2
            ),
            TestQuery(
                id="analytical_002",
                query="Explain the difference between supervised and unsupervised learning",
                category="analytical", 
                complexity="moderate",
                domain="AI/ML",
                expected_response_type="comparative",
                expected_concepts=["supervised learning", "unsupervised learning", "labeled data", "unlabeled data", "classification", "clustering"],
                ground_truth_answer="Supervised learning uses labeled training data to learn mappings from inputs to outputs, while unsupervised learning finds patterns in unlabeled data without explicit target variables.",
                confidence_threshold=0.85,
                processing_time_target=2.5,
                cost_target=0.015,
                factual_claims=["supervised uses labeled data", "unsupervised uses unlabeled data", "supervised learns input-output mappings"],
                sources_required=2
            ),
            
            # Complex Analytical Queries
            TestQuery(
                id="complex_001",
                query="Compare different neural network architectures, explain their strengths and weaknesses, and provide examples of when to use each",
                category="analytical",
                complexity="complex",
                domain="AI/ML", 
                expected_response_type="comparative",
                expected_concepts=["neural networks", "architectures", "CNN", "RNN", "transformer", "strengths", "weaknesses", "use cases"],
                ground_truth_answer="Neural networks have various architectures optimized for different tasks. CNNs excel at image processing, RNNs handle sequential data, and transformers are effective for language tasks. Each has specific strengths and optimal use cases.",
                confidence_threshold=0.82,
                processing_time_target=4.0,
                cost_target=0.035,
                requires_decomposition=True,
                factual_claims=["CNNs good for images", "RNNs handle sequences", "transformers for language"],
                sources_required=3
            ),
            TestQuery(
                id="complex_002",
                query="Analyze the environmental impact of renewable energy sources, compare their efficiency, and discuss the challenges in global adoption",
                category="analytical",
                complexity="complex", 
                domain="Environment/Energy",
                expected_response_type="analytical",
                expected_concepts=["renewable energy", "environmental impact", "efficiency", "solar", "wind", "hydroelectric", "challenges", "adoption"],
                ground_truth_answer="Renewable energy sources have varying environmental impacts and efficiencies. Solar and wind have minimal operational emissions but require significant materials for manufacturing. Hydroelectric is highly efficient but can disrupt ecosystems. Global adoption faces challenges including intermittency, storage, and infrastructure costs.",
                confidence_threshold=0.80,
                processing_time_target=5.0,
                cost_target=0.045,
                requires_decomposition=True,
                factual_claims=["renewable sources vary in impact", "solar/wind need manufacturing materials", "hydroelectric disrupts ecosystems"],
                sources_required=4
            ),
            
            # Multimodal Queries
            TestQuery(
                id="multimodal_001",
                query="Show me a diagram of the transformer architecture and explain how attention mechanisms work",
                category="multimodal",
                complexity="complex",
                domain="AI/ML",
                expected_response_type="explanatory",
                expected_concepts=["transformer", "architecture", "attention mechanism", "self-attention", "encoder", "decoder"],
                ground_truth_answer="The transformer architecture uses self-attention mechanisms to process sequences in parallel, consisting of encoder and decoder layers that can attend to different parts of the input sequence simultaneously.",
                confidence_threshold=0.78,
                processing_time_target=6.0,
                cost_target=0.055,
                has_multimodal_elements=True,
                factual_claims=["transformer uses self-attention", "processes sequences in parallel", "has encoder-decoder structure"],
                sources_required=2
            ),
            
            # Conversational Queries
            TestQuery(
                id="conversational_001",
                query="I'm confused about deep learning. Can you help me understand it better?",
                category="conversational",
                complexity="moderate",
                domain="AI/ML",
                expected_response_type="explanatory",
                expected_concepts=["deep learning", "neural networks", "layers", "learning", "artificial intelligence"],
                ground_truth_answer="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data. Think of it as a more sophisticated form of machine learning that can handle very complex tasks like image recognition and natural language processing.",
                confidence_threshold=0.85,
                processing_time_target=2.2,
                cost_target=0.018,
                factual_claims=["deep learning uses multiple layers", "models complex patterns", "handles images and NLP"],
                sources_required=2
            ),
            
            # Edge Cases and Challenging Queries
            TestQuery(
                id="edge_001",
                query="What is quantum entanglement and how might it be used in quantum computing?",
                category="analytical",
                complexity="complex",
                domain="Physics/Computing",
                expected_response_type="explanatory", 
                expected_concepts=["quantum entanglement", "quantum computing", "quantum particles", "correlation", "superposition"],
                ground_truth_answer="Quantum entanglement is a phenomenon where quantum particles become correlated such that measuring one instantly affects the other regardless of distance. In quantum computing, entanglement enables quantum parallelism and is essential for quantum algorithms.",
                confidence_threshold=0.75,  # Lower due to complexity
                processing_time_target=4.5,
                cost_target=0.040,
                requires_decomposition=True,
                factual_claims=["particles become correlated", "measurement affects both", "enables quantum parallelism"],
                sources_required=3
            ),
            
            # Ambiguous/Unclear Queries
            TestQuery(
                id="ambiguous_001", 
                query="How does it work?",
                category="ambiguous",
                complexity="simple",
                domain="General",
                expected_response_type="clarification",
                expected_concepts=["clarification", "context", "specify"],
                ground_truth_answer="I need more context to provide a helpful answer. Could you please specify what 'it' refers to?",
                confidence_threshold=0.90,  # Should be confident about needing clarification
                processing_time_target=1.0,
                cost_target=0.008,
                factual_claims=["needs more context"]
            )
        ]
    
    def _load_test_documents(self) -> List[TestDocument]:
        """Load test documents for indexing and retrieval."""
        return [
            TestDocument(
                id="doc_ml_001",
                title="Introduction to Machine Learning",
                content="""Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Some machine learning methods include supervised learning, unsupervised learning, and reinforcement learning. Supervised learning algorithms build a mathematical model based on training data that contains both inputs and desired outputs. Unsupervised learning algorithms build models from data that contains only inputs and no desired outputs. Reinforcement learning algorithms learn from their environment through trial and error.""",
                category="AI/ML",
                difficulty="beginner",
                word_count=156,
                key_concepts=["machine learning", "artificial intelligence", "supervised learning", "unsupervised learning", "reinforcement learning"],
                factual_statements=[
                    "Machine learning is a subset of artificial intelligence",
                    "Systems learn automatically without human intervention", 
                    "Supervised learning uses training data with inputs and outputs",
                    "Unsupervised learning uses data with only inputs",
                    "Reinforcement learning learns through trial and error"
                ]
            ),
            TestDocument(
                id="doc_dl_001", 
                title="Deep Learning Fundamentals",
                content="""Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, machine translation, natural language processing, and speech recognition.

The term "deep" in deep learning refers to the number of layers in the network. Traditional neural networks contain only 2-3 hidden layers, while deep networks can have as many as 150+ layers. Deep learning models are typically trained using large amounts of labeled data and neural network architectures that learn features directly from data without the need for manual feature extraction.

Applications of deep learning include autonomous vehicles, medical diagnosis, voice assistants, and image recognition systems. The field has seen remarkable progress since 2010, largely due to increases in computational power and the availability of large datasets.""",
                category="AI/ML", 
                difficulty="intermediate",
                word_count=162,
                key_concepts=["deep learning", "neural networks", "CNN", "RNN", "computer vision", "natural language processing"],
                factual_statements=[
                    "Deep learning is based on artificial neural networks",
                    "Deep refers to the number of layers in the network",
                    "Traditional networks have 2-3 hidden layers",
                    "Deep networks can have 150+ layers",
                    "Models learn features directly from data"
                ]
            ),
            TestDocument(
                id="doc_transformer_001",
                title="The Transformer Architecture",
                content="""The Transformer is a deep learning model introduced in 2017 that has become the foundation of many state-of-the-art natural language processing systems. Unlike previous sequence-to-sequence models that relied on recurrence or convolution, the Transformer is based entirely on attention mechanisms.

The key innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing each word. This enables the model to capture long-range dependencies more effectively than RNNs or LSTMs.

The Transformer architecture consists of an encoder and decoder, each composed of multiple layers. Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder layers additionally include a third sub-layer that performs multi-head attention over the encoder's output.

The attention mechanism can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Transformers have been successfully applied to machine translation, text summarization, question answering, and have formed the basis for large language models like GPT and BERT.""",
                category="AI/ML",
                difficulty="advanced", 
                word_count=228,
                key_concepts=["transformer", "attention mechanism", "self-attention", "encoder", "decoder", "multi-head attention"],
                factual_statements=[
                    "Transformer introduced in 2017",
                    "Based entirely on attention mechanisms",
                    "Self-attention weighs importance of words",
                    "Architecture has encoder and decoder",
                    "Each encoder layer has self-attention and feed-forward components",
                    "Formed basis for GPT and BERT"
                ]
            ),
            TestDocument(
                id="doc_renewable_001",
                title="Renewable Energy Sources",
                content="""Renewable energy comes from sources that are naturally replenished on a human timescale, such as sunlight, wind, rain, tides, waves, and geothermal heat. Unlike fossil fuels, renewable energy sources produce little to no greenhouse gas emissions during operation.

Solar energy harnesses sunlight using photovoltaic cells or solar thermal collectors. It's one of the fastest-growing energy sources globally, with costs decreasing significantly over the past decade. However, solar energy is intermittent and depends on weather conditions and time of day.

Wind energy uses turbines to convert kinetic energy from air currents into electrical power. Wind farms can be located on land (onshore) or in bodies of water (offshore). Offshore wind typically provides higher and more consistent wind speeds, making it more efficient.

Hydroelectric power generates electricity by using flowing water to turn turbines. It's one of the oldest and most established renewable energy sources, providing about 16% of global electricity. Large hydroelectric projects can impact local ecosystems and communities.

Geothermal energy taps into heat stored beneath the Earth's surface. It provides consistent, baseload power and has a small physical footprint compared to other renewable sources. However, it's geographically limited to areas with suitable geological conditions.

The transition to renewable energy faces challenges including energy storage, grid integration, and intermittency. Energy storage technologies like batteries are crucial for managing the variability of renewable sources.""",
                category="Environment/Energy",
                difficulty="intermediate",
                word_count=246,
                key_concepts=["renewable energy", "solar energy", "wind energy", "hydroelectric", "geothermal", "energy storage"],
                factual_statements=[
                    "Renewable sources are naturally replenished",
                    "Produce little to no greenhouse gas emissions",
                    "Solar energy costs have decreased significantly",
                    "Offshore wind provides higher wind speeds",
                    "Hydroelectric provides about 16% of global electricity",
                    "Geothermal has small physical footprint"
                ]
            ),
            TestDocument(
                id="doc_france_001",
                title="France: Geography and Government",
                content="""France, officially the French Republic, is a country located in Western Europe. It is bordered by Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Andorra, and Spain. France has coastlines on both the Atlantic Ocean and the Mediterranean Sea.

The capital and largest city of France is Paris, which is located in the north-central part of the country on the Seine River. Paris is home to approximately 2.1 million people in the city proper and over 12 million in the metropolitan area. It serves as the country's political, economic, and cultural center.

France has a semi-presidential republic form of government, with both a President and a Prime Minister. The President serves as the head of state and is elected for five-year terms. The Prime Minister serves as the head of government and is appointed by the President.

The country is divided into 18 administrative regions, including 13 regions in metropolitan France and 5 overseas regions. Each region is further divided into departments, of which there are 101 in total.

France has a rich cultural heritage and is known for its contributions to art, literature, philosophy, and cuisine. It is one of the world's leading tourist destinations, welcoming over 80 million international visitors annually.""",
                category="Geography",
                difficulty="beginner",
                word_count=209,
                key_concepts=["France", "Paris", "capital", "Western Europe", "government", "regions"],
                factual_statements=[
                    "France is located in Western Europe",
                    "Paris is the capital and largest city", 
                    "Paris has approximately 2.1 million people",
                    "France has semi-presidential republic government",
                    "President elected for five-year terms",
                    "Country divided into 18 administrative regions"
                ]
            )
        ]
    
    def _load_benchmark_datasets(self) -> Dict[str, List[TestQuery]]:
        """Load benchmark datasets for performance testing."""
        return {
            "speed_benchmark": [
                query for query in self.queries 
                if query.complexity == "simple" and query.processing_time_target <= 1.5
            ],
            "accuracy_benchmark": [
                query for query in self.queries
                if query.confidence_threshold >= 0.85 and query.factual_claims
            ],
            "complexity_benchmark": [
                query for query in self.queries
                if query.complexity == "complex" or query.requires_decomposition
            ],
            "cost_benchmark": [
                query for query in self.queries
                if query.cost_target <= 0.02  # Low-cost queries
            ],
            "multimodal_benchmark": [
                query for query in self.queries
                if query.has_multimodal_elements
            ]
        }
    
    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """Get queries filtered by category."""
        return [q for q in self.queries if q.category == category]
    
    def get_queries_by_complexity(self, complexity: str) -> List[TestQuery]:
        """Get queries filtered by complexity."""
        return [q for q in self.queries if q.complexity == complexity]
    
    def get_queries_by_domain(self, domain: str) -> List[TestQuery]:
        """Get queries filtered by domain.""" 
        return [q for q in self.queries if q.domain == domain]
    
    def get_documents_by_category(self, category: str) -> List[TestDocument]:
        """Get documents filtered by category."""
        return [d for d in self.documents if d.category == category]
    
    def export_dataset(self, filepath: str, dataset_type: str = "all"):
        """Export dataset to JSON file."""
        data = {}
        
        if dataset_type == "all" or dataset_type == "queries":
            data["queries"] = [asdict(q) for q in self.queries]
        
        if dataset_type == "all" or dataset_type == "documents": 
            data["documents"] = [asdict(d) for d in self.documents]
        
        if dataset_type == "all" or dataset_type == "benchmarks":
            data["benchmarks"] = {
                name: [asdict(q) for q in queries] 
                for name, queries in self.benchmark_datasets.items()
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class MockServices:
    """Mock implementations of external services for testing."""
    
    class MockOpenAIService:
        """Mock OpenAI API service."""
        
        def __init__(self):
            self.embedding_cache = {}
            self.completion_responses = {}
            self.api_calls = 0
            self.total_cost = 0.0
        
        async def create_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
            """Mock embedding creation."""
            self.api_calls += 1
            self.total_cost += 0.0001  # Mock cost per embedding
            
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Generate deterministic embedding based on text hash
            hash_val = hash(text)
            np.random.seed(abs(hash_val) % (2**31))
            embedding = np.random.normal(0, 1, 1536).tolist()  # Standard embedding size
            
            self.embedding_cache[text] = embedding
            return embedding
        
        async def create_completion(self, messages: List[Dict], model: str = "gpt-4", **kwargs) -> str:
            """Mock chat completion."""
            self.api_calls += 1
            
            # Mock cost based on model
            if "gpt-4" in model:
                self.total_cost += 0.03
            else:
                self.total_cost += 0.01
            
            # Generate response based on message content
            user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
            
            # Simple response generation based on keywords
            if "machine learning" in user_message.lower():
                return "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
            elif "transformer" in user_message.lower():
                return "The transformer is a neural network architecture based on attention mechanisms."
            elif "verification" in user_message.lower() or "consistent" in user_message.lower():
                return "CONSISTENT: The response is factually accurate and well-supported."
            elif "capital" in user_message.lower() and "france" in user_message.lower():
                return "The capital of France is Paris."
            else:
                return f"This is a mock response to: {user_message[:50]}..."
        
        def get_stats(self) -> Dict[str, Any]:
            """Get API usage statistics."""
            return {
                "total_api_calls": self.api_calls,
                "total_cost": self.total_cost,
                "cached_embeddings": len(self.embedding_cache)
            }
    
    class MockRedisService:
        """Mock Redis service for caching."""
        
        def __init__(self):
            self.data = {}
            self.access_count = {}
            self.total_operations = 0
        
        async def get(self, key: str) -> Optional[str]:
            """Mock Redis GET operation."""
            self.total_operations += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.data.get(key)
        
        async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
            """Mock Redis SET operation."""
            self.total_operations += 1
            self.data[key] = value
            return True
        
        async def exists(self, key: str) -> bool:
            """Mock Redis EXISTS operation."""
            self.total_operations += 1
            return key in self.data
        
        async def keys(self, pattern: str) -> List[str]:
            """Mock Redis KEYS operation."""
            self.total_operations += 1
            if pattern == "*":
                return list(self.data.keys())
            else:
                # Simple pattern matching
                return [k for k in self.data.keys() if pattern.replace("*", "") in k]
        
        async def ping(self) -> bool:
            """Mock Redis PING operation."""
            return True
        
        def get_stats(self) -> Dict[str, Any]:
            """Get cache usage statistics."""
            return {
                "total_keys": len(self.data),
                "total_operations": self.total_operations,
                "most_accessed_keys": sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:5]
            }
    
    class MockIndexService:
        """Mock LlamaIndex service."""
        
        def __init__(self, documents: List[TestDocument]):
            self.documents = {doc.id: doc for doc in documents}
            self.query_count = 0
        
        async def query(self, query_str: str, similarity_top_k: int = 5) -> MockServiceResponse:
            """Mock index query."""
            self.query_count += 1
            
            # Simple keyword matching for relevance
            relevant_docs = []
            query_words = set(query_str.lower().split())
            
            for doc in self.documents.values():
                doc_words = set(doc.content.lower().split())
                overlap = len(query_words & doc_words)
                if overlap > 0:
                    relevance = overlap / len(query_words | doc_words)
                    relevant_docs.append((doc, relevance))
            
            # Sort by relevance and take top k
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = relevant_docs[:similarity_top_k]
            
            # Generate response based on most relevant document
            if top_docs:
                best_doc, relevance = top_docs[0]
                response_content = best_doc.content[:200] + "..."  # Truncate for response
                
                sources = [
                    {
                        "text": doc.content[:150] + "...",
                        "relevance": rel,
                        "metadata": {"title": doc.title, "id": doc.id}
                    }
                    for doc, rel in top_docs
                ]
            else:
                response_content = "I don't have enough information to answer that query."
                sources = []
                relevance = 0.0
            
            return MockServiceResponse(
                content=response_content,
                confidence=min(0.9, relevance + 0.5),  # Boost confidence
                metadata={
                    "query_count": self.query_count,
                    "documents_searched": len(self.documents),
                    "relevant_documents": len(top_docs)
                },
                processing_time=random.uniform(0.5, 2.0),  # Random processing time
                cost=0.005,  # Mock cost
                sources=sources
            )
        
        def get_stats(self) -> Dict[str, Any]:
            """Get index usage statistics."""
            return {
                "total_queries": self.query_count,
                "total_documents": len(self.documents),
                "document_categories": list(set(doc.category for doc in self.documents.values()))
            }


class TestDataGenerator:
    """Generate synthetic test data for various scenarios."""
    
    @staticmethod
    def generate_query_variations(base_query: str, num_variations: int = 5) -> List[str]:
        """Generate variations of a base query."""
        variations = [base_query]
        
        # Simple variations (in production, would use more sophisticated methods)
        templates = [
            "Can you explain {}?",
            "What is {}?", 
            "Tell me about {}",
            "I want to know about {}",
            "Please describe {}"
        ]
        
        # Extract key terms
        key_terms = " ".join(base_query.split()[2:]) if len(base_query.split()) > 2 else base_query
        
        for template in templates[:num_variations-1]:
            variations.append(template.format(key_terms))
        
        return variations
    
    @staticmethod
    def generate_performance_test_queries(count: int = 100) -> List[TestQuery]:
        """Generate queries for performance testing."""
        templates = [
            ("What is {concept}?", "factual", "simple"),
            ("Explain {concept} and its applications", "analytical", "moderate"),
            ("Compare {concept1} and {concept2}", "analytical", "moderate"), 
            ("Analyze the impact of {concept} on {domain}", "analytical", "complex")
        ]
        
        concepts = ["AI", "machine learning", "deep learning", "neural networks", "algorithms"]
        domains = ["technology", "society", "business", "education", "healthcare"]
        
        queries = []
        for i in range(count):
            template, category, complexity = random.choice(templates)
            concept = random.choice(concepts)
            concept2 = random.choice([c for c in concepts if c != concept])
            domain = random.choice(domains)
            
            query_text = template.format(concept=concept, concept1=concept, concept2=concept2, domain=domain)
            
            queries.append(TestQuery(
                id=f"perf_{i:03d}",
                query=query_text,
                category=category,
                complexity=complexity,
                domain="AI/ML",
                expected_response_type="explanatory",
                expected_concepts=[concept],
                ground_truth_answer=f"Mock answer for {query_text}",
                confidence_threshold=0.8,
                processing_time_target=1.5 if complexity == "simple" else 3.0,
                cost_target=0.01 if complexity == "simple" else 0.03
            ))
        
        return queries
    
    @staticmethod
    def generate_stress_test_dataset(scale: str = "medium") -> Dict[str, List[TestQuery]]:
        """Generate datasets for stress testing."""
        scales = {
            "small": 50,
            "medium": 200, 
            "large": 1000,
            "xlarge": 5000
        }
        
        count = scales.get(scale, 200)
        
        return {
            "concurrent_queries": TestDataGenerator.generate_performance_test_queries(count // 4),
            "mixed_complexity": TestDataGenerator.generate_performance_test_queries(count // 4),
            "high_volume": TestDataGenerator.generate_performance_test_queries(count // 2)
        }


# Global test data instance
test_datasets = TestDatasets()
mock_services = MockServices()

# Export functions for easy access
def get_test_queries(category: str = None, complexity: str = None) -> List[TestQuery]:
    """Get test queries with optional filtering."""
    queries = test_datasets.queries
    
    if category:
        queries = [q for q in queries if q.category == category]
    if complexity:
        queries = [q for q in queries if q.complexity == complexity]
    
    return queries

def get_test_documents(category: str = None) -> List[TestDocument]:
    """Get test documents with optional filtering.""" 
    documents = test_datasets.documents
    
    if category:
        documents = [d for d in documents if d.category == category]
    
    return documents

def get_mock_openai_service() -> MockServices.MockOpenAIService:
    """Get mock OpenAI service instance."""
    return MockServices.MockOpenAIService()

def get_mock_redis_service() -> MockServices.MockRedisService:
    """Get mock Redis service instance."""
    return MockServices.MockRedisService()

def get_mock_index_service(documents: List[TestDocument] = None) -> MockServices.MockIndexService:
    """Get mock index service instance."""
    if documents is None:
        documents = test_datasets.documents
    return MockServices.MockIndexService(documents)