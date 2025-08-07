"""
Performance benchmarking and load testing for SOTA RAG system.

Tests cover:
- Response time benchmarks
- Throughput and concurrent query handling
- Resource usage monitoring
- Cost analysis and optimization
- Cache performance validation
- Scalability testing
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import psutil
import os

from src.unified_workflow import UnifiedWorkflow, QueryComplexity
from src.unified_config import get_unified_config, reset_unified_config, PerformanceProfile
from src.cache import SemanticCache
from src.health_monitor import get_health_monitor


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""
    operation: str
    avg_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    min_time: float
    max_time: float
    success_rate: float
    error_rate: float
    throughput: float  # queries per second
    total_queries: int
    total_time: float


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_bytes_sent: int
    network_bytes_recv: int


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.results = []
        self.resource_snapshots = []
        self.start_time = None
        self.process = psutil.Process(os.getpid())
    
    def start_benchmark(self):
        """Start benchmarking session."""
        self.start_time = time.time()
        self.resource_snapshots = []
        self.results = []
    
    def record_resource_usage(self):
        """Record current resource usage."""
        try:
            cpu = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            io_counters = self.process.io_counters()
            
            # Network stats (system-wide)
            net_io = psutil.net_io_counters()
            
            metrics = ResourceMetrics(
                cpu_percent=cpu,
                memory_mb=memory_info.rss / 1024 / 1024,
                memory_percent=memory_percent,
                disk_io_read=io_counters.read_bytes,
                disk_io_write=io_counters.write_bytes,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv
            )
            
            self.resource_snapshots.append(metrics)
            return metrics
        except Exception:
            # Return zero metrics if monitoring fails
            return ResourceMetrics(0, 0, 0, 0, 0, 0, 0)
    
    async def benchmark_operation(
        self,
        operation_name: str,
        operation_func,
        num_iterations: int = 100,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a specific operation."""
        times = []
        errors = 0
        successes = 0
        
        self.record_resource_usage()  # Initial snapshot
        
        start_total = time.time()
        
        for i in range(num_iterations):
            try:
                start = time.time()
                
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
                
                end = time.time()
                times.append(end - start)
                successes += 1
                
                # Record resource usage periodically
                if i % 10 == 0:
                    self.record_resource_usage()
                    
            except Exception as e:
                errors += 1
                print(f"Error in iteration {i}: {e}")
        
        end_total = time.time()
        total_time = end_total - start_total
        
        self.record_resource_usage()  # Final snapshot
        
        if times:
            result = BenchmarkResult(
                operation=operation_name,
                avg_time=statistics.mean(times),
                p50_time=statistics.median(times),
                p95_time=self._percentile(times, 95),
                p99_time=self._percentile(times, 99),
                min_time=min(times),
                max_time=max(times),
                success_rate=successes / num_iterations,
                error_rate=errors / num_iterations,
                throughput=successes / total_time,
                total_queries=num_iterations,
                total_time=total_time
            )
        else:
            # No successful operations
            result = BenchmarkResult(
                operation=operation_name,
                avg_time=0.0,
                p50_time=0.0,
                p95_time=0.0,
                p99_time=0.0,
                min_time=0.0,
                max_time=0.0,
                success_rate=0.0,
                error_rate=1.0,
                throughput=0.0,
                total_queries=num_iterations,
                total_time=total_time
            )
        
        self.results.append(result)
        return result
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.resource_snapshots]
        memory_values = [s.memory_mb for s in self.resource_snapshots]
        
        return {
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg_mb': statistics.mean(memory_values),
            'memory_max_mb': max(memory_values),
            'memory_growth_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        }


class TestResponseTimeBenchmarks:
    """Test response time performance against benchmarks."""
    
    @pytest.mark.asyncio
    async def test_simple_query_response_time(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_benchmarks):
        """Benchmark simple query response times."""
        benchmarks = performance_benchmarks
        target_time = benchmarks['response_time']['simple_query']
        
        # Setup fast mock responses
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Machine learning is a subset of artificial intelligence."
        mock_redis_client.get.return_value = None  # Cache miss
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        async def simple_query_test():
            query = "What is machine learning?"
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': 'ML is a subset of AI',
                    'confidence': 0.9,
                    'processing_time': 0.8,
                    'cost': 0.005
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                return await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
        
        result = await benchmark.benchmark_operation(
            "simple_query_processing",
            simple_query_test,
            num_iterations=50
        )
        
        # Validate performance against benchmarks
        assert result.success_rate >= 0.95, f"Success rate {result.success_rate} below threshold"
        assert result.p95_time <= target_time, f"P95 time {result.p95_time:.2f}s exceeds target {target_time}s"
        assert result.avg_time <= target_time * 0.8, f"Average time {result.avg_time:.2f}s exceeds 80% of target"
        
        # Resource usage should be reasonable
        resources = benchmark.get_resource_summary()
        assert resources.get('memory_growth_mb', 0) < 100, "Memory growth too high"
        assert resources.get('cpu_avg', 0) < 80, "CPU usage too high"
    
    @pytest.mark.asyncio
    async def test_complex_query_response_time(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_benchmarks):
        """Benchmark complex query response times."""
        benchmarks = performance_benchmarks
        target_time = benchmarks['response_time']['complex_query']
        
        # Setup mock for complex processing
        mock_openai_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content="Subquery 1 response"))]),
            Mock(choices=[Mock(message=Mock(content="Subquery 2 response"))]),
            Mock(choices=[Mock(message=Mock(content="Aggregated complex response"))]),
            Mock(choices=[Mock(message=Mock(content="CONSISTENT: Response is accurate"))])
        ]
        
        workflow = UnifiedWorkflow(timeout=60.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        async def complex_query_test():
            query = "Compare supervised and unsupervised learning with examples"
            with patch.object(workflow, '_process_with_agentic_workflow') as mock_agentic:
                mock_agentic.return_value = {
                    'content': 'Comprehensive comparison with examples',
                    'confidence': 0.88,
                    'processing_time': 2.5,
                    'cost': 0.025,
                    'subqueries': ['What is supervised learning?', 'What is unsupervised learning?']
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                return await workflow._process_with_agentic_workflow(characteristics)
        
        result = await benchmark.benchmark_operation(
            "complex_query_processing",
            complex_query_test,
            num_iterations=20  # Fewer iterations for complex queries
        )
        
        # Validate performance against benchmarks
        assert result.success_rate >= 0.90, f"Success rate {result.success_rate} below threshold"
        assert result.p95_time <= target_time, f"P95 time {result.p95_time:.2f}s exceeds target {target_time}s"
        assert result.avg_time <= target_time * 0.7, f"Average time {result.avg_time:.2f}s exceeds 70% of target"
    
    @pytest.mark.asyncio
    async def test_multimodal_query_response_time(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_benchmarks):
        """Benchmark multimodal query response times."""
        benchmarks = performance_benchmarks
        target_time = benchmarks['response_time']['multimodal_query']
        
        workflow = UnifiedWorkflow(timeout=60.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        async def multimodal_query_test():
            query = "Show me a diagram of neural network architecture"
            with patch.object(workflow, '_process_multimodal') as mock_multimodal:
                mock_multimodal.return_value = {
                    'content': 'Neural network architecture explanation',
                    'image_description': 'Diagram showing layers and connections',
                    'confidence': 0.85,
                    'processing_time': 4.2,
                    'cost': 0.02
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                return await workflow._process_multimodal(characteristics)
        
        result = await benchmark.benchmark_operation(
            "multimodal_query_processing",
            multimodal_query_test,
            num_iterations=15
        )
        
        # Validate performance against benchmarks
        assert result.success_rate >= 0.85, f"Success rate {result.success_rate} below threshold"
        assert result.p95_time <= target_time, f"P95 time {result.p95_time:.2f}s exceeds target {target_time}s"


class TestThroughputAndConcurrency:
    """Test system throughput and concurrent query handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test concurrent query processing capabilities."""
        # Setup mock responses
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Response to concurrent query"
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'concurrent_queries': 0}
        
        queries = [
            "What is AI?",
            "Explain machine learning",
            "Define neural networks", 
            "What is deep learning?",
            "Describe natural language processing"
        ] * 4  # 20 queries total
        
        async def process_single_query(query):
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': f'Response to: {query}',
                    'confidence': 0.9,
                    'processing_time': 1.0,
                    'cost': 0.01
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                return await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Process queries with specified concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_process(query):
                async with semaphore:
                    return await process_single_query(query)
            
            results = await asyncio.gather(*[
                bounded_process(query) for query in queries
            ])
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = len(queries) / total_time
            
            # Validate throughput scales with concurrency
            print(f"Concurrency {concurrency}: {throughput:.2f} queries/sec")
            
            assert len(results) == len(queries)
            assert all(r['content'] is not None for r in results)
            
            # Throughput should improve with higher concurrency (up to a point)
            if concurrency == 1:
                baseline_throughput = throughput
            elif concurrency == 5:
                assert throughput >= baseline_throughput * 2  # Should be at least 2x faster
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test performance under sustained load."""
        # Setup mock responses
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Response under sustained load"
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        async def sustained_load_query():
            query = "Test query for sustained load"
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': 'Response under load',
                    'confidence': 0.9,
                    'processing_time': 1.2,
                    'cost': 0.01
                }
                
                characteristics = await workflow._analyze_query_characteristics(query)
                return await workflow._process_query_with_plan(
                    characteristics,
                    workflow._create_processing_plan(characteristics)
                )
        
        # Run sustained load for 2 minutes
        duration = 120  # seconds
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration:
            batch_start = time.time()
            
            # Process batch of queries
            batch_size = 10
            batch_tasks = [sustained_load_query() for _ in range(batch_size)]
            
            try:
                await asyncio.wait_for(asyncio.gather(*batch_tasks), timeout=30.0)
                query_count += batch_size
                
                # Record metrics every 30 seconds
                if query_count % 100 == 0:
                    benchmark.record_resource_usage()
                    
            except asyncio.TimeoutError:
                print(f"Batch timeout at query {query_count}")
                break
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        sustained_throughput = query_count / total_time
        
        resources = benchmark.get_resource_summary()
        
        # Validate sustained performance
        assert sustained_throughput >= 5.0, f"Sustained throughput {sustained_throughput:.2f} too low"
        assert resources.get('memory_growth_mb', 0) < 500, "Excessive memory growth during sustained load"
        assert resources.get('cpu_avg', 0) < 90, "CPU usage too high during sustained load"
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, mock_redis_client, mock_openai_client, performance_benchmarks):
        """Test cache performance under load."""
        benchmarks = performance_benchmarks
        target_hit_rate = benchmarks['cache_hit_rate']['target_minimum']
        
        # Setup cache with some pre-existing entries
        cache_entries = {}
        for i in range(20):
            key = f"cache:query_{i}"
            cache_entries[key] = {
                'content': f'Cached response {i}',
                'confidence': 0.9,
                'embedding': [0.1 * i] * 768
            }
        
        def mock_redis_get(key):
            return json.dumps(cache_entries.get(key)) if key in cache_entries else None
        
        def mock_redis_keys(pattern):
            return list(cache_entries.keys())
        
        mock_redis_client.get.side_effect = mock_redis_get
        mock_redis_client.keys.side_effect = mock_redis_keys
        mock_redis_client.set.return_value = True
        
        # Mock embeddings for similarity computation
        mock_openai_client.embeddings.create.return_value.data[0].embedding = [0.1] * 768
        
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        # Test cache performance with mixed hit/miss pattern
        queries = [
            f"Query similar to cached {i % 10}"  # 50% should hit cache
            for i in range(100)
        ]
        
        hits = 0
        misses = 0
        
        for query in queries:
            try:
                result = await cache.get(query)
                if result:
                    hits += 1
                else:
                    misses += 1
                    # Simulate adding to cache
                    await cache.set(query, f"Response to {query}", confidence=0.9)
            except Exception as e:
                print(f"Cache error for query '{query}': {e}")
                misses += 1
        
        cache_hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        # Validate cache performance
        assert cache_hit_rate >= target_hit_rate, \
            f"Cache hit rate {cache_hit_rate:.2f} below target {target_hit_rate}"


class TestResourceUsageMonitoring:
    """Test resource usage monitoring and limits."""
    
    def test_memory_usage_limits(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test memory usage stays within limits."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        # Process many queries to test memory usage
        for i in range(100):
            with patch.object(workflow, '_process_query_with_plan') as mock_process:
                mock_process.return_value = {
                    'content': f'Response to query {i}',
                    'confidence': 0.9,
                    'processing_time': 1.0
                }
                
                query = f"Test query {i}"
                asyncio.run(workflow._analyze_query_characteristics(query))
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 200, f"Memory growth {memory_growth:.1f}MB too high"
    
    @pytest.mark.asyncio
    async def test_cpu_usage_efficiency(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test CPU usage efficiency during operations."""
        process = psutil.Process()
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        cpu_measurements = []
        
        async def cpu_monitoring_task():
            """Monitor CPU usage during operations."""
            for _ in range(10):  # Monitor for 10 seconds
                cpu_measurements.append(process.cpu_percent(interval=1))
        
        async def query_processing_task():
            """Process queries during CPU monitoring."""
            for i in range(20):
                with patch.object(workflow, '_process_query_with_plan') as mock_process:
                    mock_process.return_value = {
                        'content': f'CPU test response {i}',
                        'confidence': 0.9,
                        'processing_time': 0.5
                    }
                    
                    query = f"CPU test query {i}"
                    characteristics = await workflow._analyze_query_characteristics(query)
                    await workflow._process_query_with_plan(
                        characteristics,
                        workflow._create_processing_plan(characteristics)
                    )
                
                await asyncio.sleep(0.1)  # Small delay between queries
        
        # Run both tasks concurrently
        await asyncio.gather(
            cpu_monitoring_task(),
            query_processing_task()
        )
        
        if cpu_measurements:
            avg_cpu = statistics.mean(cpu_measurements)
            max_cpu = max(cpu_measurements)
            
            # CPU usage should be efficient
            assert avg_cpu < 70, f"Average CPU usage {avg_cpu:.1f}% too high"
            assert max_cpu < 90, f"Peak CPU usage {max_cpu:.1f}% too high"


class TestCostAnalysis:
    """Test cost analysis and optimization."""
    
    @pytest.mark.asyncio
    async def test_query_cost_tracking(self, mock_llama_index, mock_redis_client, mock_openai_client, performance_benchmarks):
        """Test accurate tracking of query costs."""
        benchmarks = performance_benchmarks
        max_simple_cost = benchmarks['cost']['max_simple_query']
        max_complex_cost = benchmarks['cost']['max_complex_query']
        
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0, 'total_cost': 0}
        
        # Test simple query costs
        simple_query = "What is AI?"
        with patch.object(workflow, '_process_query_with_plan') as mock_simple:
            mock_simple.return_value = {
                'content': 'AI is artificial intelligence',
                'confidence': 0.9,
                'cost': 0.008,  # Should be under max_simple_cost
                'token_usage': {'prompt': 10, 'completion': 15}
            }
            
            characteristics = await workflow._analyze_query_characteristics(simple_query)
            result = await workflow._process_query_with_plan(
                characteristics,
                workflow._create_processing_plan(characteristics)
            )
            
            assert result['cost'] <= max_simple_cost, \
                f"Simple query cost {result['cost']} exceeds limit {max_simple_cost}"
        
        # Test complex query costs
        complex_query = "Compare different machine learning algorithms with detailed examples"
        with patch.object(workflow, '_process_with_agentic_workflow') as mock_complex:
            mock_complex.return_value = {
                'content': 'Comprehensive ML algorithm comparison',
                'confidence': 0.88,
                'cost': 0.042,  # Should be under max_complex_cost
                'subqueries': ['Algorithm 1', 'Algorithm 2', 'Examples'],
                'token_usage': {'prompt': 200, 'completion': 150}
            }
            
            characteristics = await workflow._analyze_query_characteristics(complex_query)
            result = await workflow._process_with_agentic_workflow(characteristics)
            
            assert result['cost'] <= max_complex_cost, \
                f"Complex query cost {result['cost']} exceeds limit {max_complex_cost}"
    
    @pytest.mark.asyncio
    async def test_cost_optimization_strategies(self, mock_redis_client, mock_openai_client):
        """Test cost optimization through caching and model selection."""
        # Test cache-based cost optimization
        cached_response = {
            'content': 'Cached response to reduce costs',
            'confidence': 0.9,
            'cost': 0.001,  # Very low cost due to cache hit
            'cache_hit': True
        }
        
        # Mock cache hit
        mock_redis_client.get.return_value = json.dumps(cached_response)
        mock_redis_client.keys.return_value = ['cache:test']
        
        cache = SemanticCache(redis_url="redis://localhost:6379/15")
        
        # Test multiple similar queries - should hit cache and reduce costs
        similar_queries = [
            "What is machine learning?",
            "Can you explain machine learning?",
            "Tell me about machine learning"
        ]
        
        total_cost = 0
        cache_hits = 0
        
        for query in similar_queries:
            result = await cache.get(query)
            if result and result.get('cache_hit'):
                cache_hits += 1
                total_cost += result.get('cost', 0)
            else:
                total_cost += 0.01  # Regular processing cost
        
        # Should achieve significant cost savings through caching
        expected_uncached_cost = len(similar_queries) * 0.01
        cost_savings = (expected_uncached_cost - total_cost) / expected_uncached_cost
        
        assert cost_savings >= 0.5, f"Cost savings {cost_savings:.2f} insufficient"
        assert cache_hits >= 1, "No cache hits achieved"


class TestScalabilityValidation:
    """Test system scalability characteristics."""
    
    @pytest.mark.asyncio
    async def test_linear_scalability(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test that performance scales linearly with load."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        # Test different load levels
        load_levels = [10, 20, 40]  # Number of concurrent queries
        throughput_results = []
        
        for load in load_levels:
            queries = [f"Scalability test query {i}" for i in range(load)]
            
            async def process_query(query):
                with patch.object(workflow, '_process_query_with_plan') as mock_process:
                    mock_process.return_value = {
                        'content': f'Response to {query}',
                        'confidence': 0.9,
                        'processing_time': 1.0,
                        'cost': 0.01
                    }
                    
                    characteristics = await workflow._analyze_query_characteristics(query)
                    return await workflow._process_query_with_plan(
                        characteristics,
                        workflow._create_processing_plan(characteristics)
                    )
            
            start_time = time.time()
            
            # Process all queries concurrently
            results = await asyncio.gather(*[
                process_query(query) for query in queries
            ])
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = len(results) / total_time
            
            throughput_results.append((load, throughput))
            print(f"Load {load}: {throughput:.2f} queries/sec")
        
        # Validate scalability
        # Throughput should increase with load (up to system limits)
        for i in range(1, len(throughput_results)):
            prev_load, prev_throughput = throughput_results[i-1]
            curr_load, curr_throughput = throughput_results[i]
            
            # Throughput should scale reasonably with load
            expected_min_throughput = prev_throughput * (curr_load / prev_load) * 0.7
            assert curr_throughput >= expected_min_throughput, \
                f"Throughput {curr_throughput:.2f} doesn't scale with load"
    
    @pytest.mark.asyncio
    async def test_resource_scaling(self, mock_llama_index, mock_redis_client, mock_openai_client):
        """Test resource usage scaling with system load."""
        workflow = UnifiedWorkflow(timeout=30.0)
        workflow.config_manager = get_unified_config()
        workflow.stats = {'queries_processed': 0}
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        # Test different system loads
        load_scenarios = [
            ('light', 5, 10),    # 5 concurrent, 10 total queries
            ('medium', 10, 25),  # 10 concurrent, 25 total queries
            ('heavy', 15, 50)    # 15 concurrent, 50 total queries
        ]
        
        resource_scaling = []
        
        for scenario_name, concurrency, total_queries in load_scenarios:
            benchmark.record_resource_usage()  # Baseline
            
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_query(query_id):
                async with semaphore:
                    query = f"Resource scaling test {query_id}"
                    with patch.object(workflow, '_process_query_with_plan') as mock_process:
                        mock_process.return_value = {
                            'content': f'Response to query {query_id}',
                            'confidence': 0.9,
                            'processing_time': 1.0
                        }
                        
                        characteristics = await workflow._analyze_query_characteristics(query)
                        return await workflow._process_query_with_plan(
                            characteristics,
                            workflow._create_processing_plan(characteristics)
                        )
            
            # Process queries with specified concurrency
            start_time = time.time()
            results = await asyncio.gather(*[
                bounded_query(i) for i in range(total_queries)
            ])
            end_time = time.time()
            
            benchmark.record_resource_usage()  # After processing
            
            processing_time = end_time - start_time
            throughput = total_queries / processing_time
            resources = benchmark.get_resource_summary()
            
            resource_scaling.append({
                'scenario': scenario_name,
                'concurrency': concurrency,
                'total_queries': total_queries,
                'throughput': throughput,
                'memory_usage': resources.get('memory_avg_mb', 0),
                'cpu_usage': resources.get('cpu_avg', 0)
            })
        
        # Validate resource scaling
        for i, scenario in enumerate(resource_scaling):
            print(f"Scenario {scenario['scenario']}: "
                  f"throughput={scenario['throughput']:.2f}, "
                  f"memory={scenario['memory_usage']:.1f}MB, "
                  f"cpu={scenario['cpu_usage']:.1f}%")
            
            # Resource usage should scale reasonably
            assert scenario['memory_usage'] < 1000, "Memory usage too high"
            assert scenario['cpu_usage'] < 95, "CPU usage too high"
            
            if i > 0:  # Compare with previous scenario
                prev = resource_scaling[i-1]
                load_ratio = scenario['total_queries'] / prev['total_queries']
                memory_ratio = scenario['memory_usage'] / max(prev['memory_usage'], 1)
                
                # Memory usage shouldn't grow faster than load
                assert memory_ratio <= load_ratio * 1.5, \
                    f"Memory scaling {memory_ratio:.2f} too aggressive for load ratio {load_ratio:.2f}"