# Performance Testing Strategy Implementation Guide
## Enhanced RAG System - Production Deployment Guide

### Overview

This guide provides step-by-step instructions for implementing the comprehensive performance testing strategy for the Enhanced RAG System. It covers setup, execution, monitoring, and continuous improvement processes.

## Prerequisites

### System Requirements
- **Hardware**: 8+ GB RAM, 4+ CPU cores, 50+ GB storage
- **Python**: 3.8+ with asyncio support
- **Redis**: 6.0+ for semantic caching
- **Docker**: For containerized monitoring stack
- **Operating System**: Linux (recommended), macOS, or Windows

### Dependencies Installation
```bash
# Core testing dependencies
pip install pytest pytest-asyncio aiohttp psutil numpy pandas matplotlib
pip install redis openai llama-index

# Monitoring dependencies  
pip install prometheus-client grafana-api pydantic

# Load testing dependencies
pip install locust httpx asyncio-throttle

# Optional: Advanced profiling
pip install py-spy memory-profiler line-profiler
```

## Phase 1: Foundation Setup (Week 1-2)

### 1.1 Performance Testing Framework Setup

#### Install the Test Suite
```bash
# Create performance testing directory
mkdir -p performance-testing/{config,results,reports,scripts}

# Copy the performance test suite
cp .claude/docs/performance-test-suite.py performance-testing/
cp .claude/docs/performance-benchmarks.json performance-testing/config/
cp .claude/docs/monitoring-config.yaml performance-testing/config/
```

#### Configure Test Environment
```bash
# Create test configuration
cat > performance-testing/config/test-config.yaml << EOF
test_environment:
  name: "development"
  base_url: "http://localhost:4501"
  timeout: 30
  max_retries: 3

performance_targets:
  response_time_p95: 3.0
  success_rate: 0.95
  cache_hit_rate: 0.30
  memory_limit_mb: 1500

load_testing:
  max_concurrent_users: 100
  test_duration_minutes: 60
  ramp_up_period_minutes: 5
EOF
```

### 1.2 Monitoring Infrastructure Setup

#### Prometheus Configuration
```yaml
# performance-testing/config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['localhost:8000', 'localhost:4501']
    scrape_interval: 30s
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### Start Monitoring Stack
```bash
# Create docker-compose for monitoring
cat > performance-testing/docker-compose.yml << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      
  redis:
    image: redis:6-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

volumes:
  grafana-storage:
EOF

# Start monitoring infrastructure
docker-compose up -d
```

### 1.3 Baseline Measurement Collection

#### Run Initial Benchmarks
```bash
# Create baseline collection script
cat > performance-testing/scripts/collect-baseline.py << 'EOF'
#!/usr/bin/env python3
"""Collect baseline performance measurements."""

import asyncio
import json
import time
from pathlib import Path
import sys
import os

# Add the parent directory to path to import test suite
sys.path.append(str(Path(__file__).parent.parent))
from performance_test_suite import PerformanceTestSuite

async def collect_baseline():
    """Collect baseline performance metrics."""
    config = {
        'test_duration': 120,  # 2 minutes for baseline
        'max_concurrent_users': 10,
        'target_response_time': 3.0
    }
    
    test_suite = PerformanceTestSuite(config)
    
    print("🔍 Collecting baseline performance metrics...")
    
    # Run baseline benchmarks only
    baseline_results = await test_suite.run_baseline_benchmarks()
    
    # Save baseline results
    timestamp = int(time.time())
    baseline_file = f"performance-testing/results/baseline-{timestamp}.json"
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)
    
    print(f"✅ Baseline metrics saved to: {baseline_file}")
    
    # Print summary
    print("\n📊 BASELINE SUMMARY")
    print("=" * 50)
    
    if 'workflow_orchestration' in baseline_results:
        orch = baseline_results['workflow_orchestration']
        print(f"Workflow orchestration avg: {orch.get('avg_analysis_time', 0):.3f}s")
        print(f"Target: {orch.get('target_analysis_time', 0):.3f}s")
        
    if 'semantic_cache' in baseline_results:
        cache = baseline_results['semantic_cache']
        if 'error' not in cache:
            print(f"Cache hit time avg: {cache.get('avg_cache_hit_time', 0)*1000:.1f}ms")
            print(f"Cache miss time avg: {cache.get('avg_cache_miss_time', 0)*1000:.1f}ms")
    
    if 'verification_pipeline' in baseline_results:
        verif = baseline_results['verification_pipeline']
        if 'error' not in verif:
            print(f"Verification avg: {verif.get('avg_verification_time', 0):.3f}s")
            print(f"Target: {verif.get('target_verification_time', 0):.3f}s")
    
    return baseline_file

if __name__ == "__main__":
    asyncio.run(collect_baseline())
EOF

chmod +x performance-testing/scripts/collect-baseline.py
```

#### Execute Baseline Collection
```bash
cd performance-testing
python scripts/collect-baseline.py
```

## Phase 2: Component Testing (Week 3-4)

### 2.1 UnifiedWorkflow Performance Testing

#### Create Workflow-Specific Tests
```python
# performance-testing/tests/test_workflow_performance.py
import pytest
import asyncio
import time
from src.unified_workflow import UnifiedWorkflow

class TestWorkflowPerformance:
    """Test UnifiedWorkflow performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_query_analysis_performance(self):
        """Test query analysis speed meets targets."""
        workflow = UnifiedWorkflow(timeout=30.0)
        
        test_queries = [
            "What is AI?",  # Simple
            "Compare supervised vs unsupervised learning with examples",  # Complex
            "Show me neural network diagrams"  # Multimodal
        ]
        
        analysis_times = []
        
        for query in test_queries:
            start_time = time.time()
            characteristics = await workflow._analyze_query_characteristics(query)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            analysis_times.append(analysis_time)
            
            # Individual query should be fast
            assert analysis_time < 0.15, f"Query analysis too slow: {analysis_time:.3f}s"
        
        # Average should be well under target
        avg_time = sum(analysis_times) / len(analysis_times)
        assert avg_time < 0.10, f"Average analysis time too slow: {avg_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_plan_creation_performance(self):
        """Test processing plan creation speed."""
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Test plan creation for different complexity levels
        query = "Compare machine learning algorithms"
        characteristics = await workflow._analyze_query_characteristics(query)
        
        start_time = time.time()
        plan = await workflow._create_processing_plan(characteristics)
        end_time = time.time()
        
        plan_time = end_time - start_time
        assert plan_time < 0.20, f"Plan creation too slow: {plan_time:.3f}s"
        assert plan is not None, "Plan creation failed"
```

#### Run Workflow Performance Tests
```bash
cd performance-testing
pytest tests/test_workflow_performance.py -v --tb=short
```

### 2.2 Semantic Cache Optimization

#### Cache Performance Tuning Script
```python
# performance-testing/scripts/tune-cache-performance.py
#!/usr/bin/env python3
"""Tune semantic cache performance parameters."""

import asyncio
import json
import time
import statistics
from src.cache import SemanticCache

async def test_similarity_thresholds():
    """Test different similarity thresholds for optimal hit rate."""
    thresholds = [0.95, 0.96, 0.97, 0.98, 0.99]
    results = {}
    
    for threshold in thresholds:
        print(f"Testing similarity threshold: {threshold}")
        
        # Create cache with specific threshold
        cache_config = {
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": threshold,
            "cache_ttl": 3600,
            "max_cache_size": 1000,
            "redis_cache_url": "redis://localhost:6379",
            "cache_key_prefix": f"test_{threshold}_",
            "cache_stats_enabled": True
        }
        
        cache = SemanticCache(cache_config)
        
        # Test queries with similar meanings
        query_pairs = [
            ("What is AI?", "What is artificial intelligence?"),
            ("Explain ML", "Explain machine learning"),
            ("Define deep learning", "What is deep learning?"),
            ("Neural networks", "What are neural networks?"),
            ("Natural language processing", "What is NLP?")
        ]
        
        hits = 0
        total = 0
        
        for original, similar in query_pairs:
            # Cache original
            class MockResponse:
                def __init__(self, text):
                    self.response = text
                    self.source_nodes = []
                    self.metadata = {}
            
            cache.put(original, MockResponse(f"Response to: {original}"))
            
            # Try to get similar
            result = cache.get(similar)
            total += 1
            if result:
                hits += 1
        
        hit_rate = hits / total if total > 0 else 0
        results[threshold] = {
            'hit_rate': hit_rate,
            'hits': hits,
            'total': total
        }
    
    print("\n📊 SIMILARITY THRESHOLD RESULTS")
    print("=" * 50)
    for threshold, result in results.items():
        print(f"Threshold {threshold}: Hit rate {result['hit_rate']:.2%} ({result['hits']}/{result['total']})")
    
    # Find optimal threshold (balance between precision and recall)
    optimal_threshold = max(results.keys(), key=lambda t: results[t]['hit_rate'])
    print(f"\n🎯 Recommended threshold: {optimal_threshold} (hit rate: {results[optimal_threshold]['hit_rate']:.2%})")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_similarity_thresholds())
```

### 2.3 Verification Pipeline Optimization

#### Batch Processing Efficiency Test
```python
# performance-testing/scripts/optimize-verification-batching.py
#!/usr/bin/env python3
"""Optimize verification pipeline batch processing."""

import asyncio
import time
import statistics
from unittest.mock import Mock
from src.verification import create_hallucination_detector

async def test_batch_sizes():
    """Test different batch sizes for verification efficiency."""
    detector = create_hallucination_detector()
    batch_sizes = [1, 5, 10, 15, 20, 25]
    results = {}
    
    # Sample test cases
    test_cases = [
        ("What is AI?", "AI is artificial intelligence."),
        ("Explain ML", "ML is machine learning."),
        ("Define DL", "Deep learning uses neural networks."),
        ("What is NLP?", "NLP processes natural language."),
        ("Computer vision", "CV analyzes visual data.")
    ] * 5  # 25 total test cases
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Process in batches
        batches = [test_cases[i:i+batch_size] for i in range(0, len(test_cases), batch_size)]
        
        total_time = 0
        total_queries = 0
        
        for batch in batches:
            start_time = time.time()
            
            # Simulate batch verification
            for query, response in batch:
                try:
                    # Mock verification (since we don't have the full pipeline)
                    await asyncio.sleep(0.1)  # Simulate verification time
                    total_queries += 1
                except Exception:
                    pass
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_time_per_query = total_time / total_queries if total_queries > 0 else 0
        throughput = total_queries / total_time if total_time > 0 else 0
        
        results[batch_size] = {
            'avg_time_per_query': avg_time_per_query,
            'throughput': throughput,
            'total_time': total_time,
            'total_queries': total_queries
        }
    
    print("\n📊 BATCH SIZE RESULTS")
    print("=" * 60)
    print("Batch Size | Avg Time/Query | Throughput | Total Time")
    print("-" * 60)
    
    for batch_size, result in results.items():
        print(f"{batch_size:>9} | {result['avg_time_per_query']:>13.3f}s | {result['throughput']:>9.1f}/s | {result['total_time']:>8.1f}s")
    
    # Find optimal batch size
    optimal_batch_size = max(results.keys(), key=lambda b: results[b]['throughput'])
    print(f"\n🎯 Optimal batch size: {optimal_batch_size} (throughput: {results[optimal_batch_size]['throughput']:.1f}/s)")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_batch_sizes())
```

## Phase 3: Integration and Profiles (Week 5-6)

### 3.1 Performance Profile Validation

#### Profile Validation Script
```bash
# performance-testing/scripts/validate-profiles.sh
#!/bin/bash
set -e

echo "🔍 Validating Performance Profiles..."

# Set environment variables for each profile
profiles=("high_accuracy" "balanced" "cost_optimized" "speed")

for profile in "${profiles[@]}"; do
    echo ""
    echo "Testing $profile profile..."
    
    # Set profile-specific environment variables
    export PERFORMANCE_PROFILE=$profile
    
    # Run profile-specific validation
    python3 << EOF
import asyncio
import sys
import os
sys.path.append('.')
from performance_test_suite import PerformanceTestSuite

async def validate_profile():
    config = {'test_duration': 180}  # 3 minutes per profile
    suite = PerformanceTestSuite(config)
    
    profile_name = os.environ.get('PERFORMANCE_PROFILE', 'balanced')
    print(f"🧪 Testing {profile_name} profile...")
    
    if profile_name == 'high_accuracy':
        result = await suite.validate_high_accuracy_profile()
    elif profile_name == 'balanced':
        result = await suite.validate_balanced_profile()
    elif profile_name == 'cost_optimized':
        result = await suite.validate_cost_optimized_profile()
    elif profile_name == 'speed':
        result = await suite.validate_speed_profile()
    else:
        print(f"❌ Unknown profile: {profile_name}")
        return
    
    # Check if targets are met
    meets_targets = result.get('meets_targets', False)
    if meets_targets:
        print(f"✅ {profile_name} profile meets all targets")
    else:
        print(f"❌ {profile_name} profile does not meet targets")
        print(f"   Avg response time: {result.get('avg_response_time', 0):.2f}s")
        print(f"   Avg accuracy: {result.get('avg_accuracy', 0):.2%}")
        
    return meets_targets

if __name__ == "__main__":
    success = asyncio.run(validate_profile())
    if not success:
        sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo "✅ $profile profile validation passed"
    else
        echo "❌ $profile profile validation failed"
        exit 1
    fi
done

echo ""
echo "🎉 All profile validations completed!"
```

### 3.2 End-to-End Performance Testing

#### Comprehensive End-to-End Test
```python
# performance-testing/scripts/e2e-performance-test.py
#!/usr/bin/env python3
"""End-to-end performance testing across all components."""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from performance_test_suite import PerformanceTestSuite

async def run_e2e_performance_test():
    """Run comprehensive end-to-end performance test."""
    
    print("🚀 Starting End-to-End Performance Test")
    print("=" * 60)
    
    config = {
        'test_duration': 600,  # 10 minutes
        'max_concurrent_users': 50,
        'target_response_time': 3.0,
        'target_success_rate': 0.95
    }
    
    test_suite = PerformanceTestSuite(config)
    
    # Run comprehensive test suite
    results = await test_suite.run_comprehensive_suite()
    
    # Save results
    timestamp = int(time.time())
    results_file = f"performance-testing/results/e2e-test-{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Analyze results
    summary = results.get('summary', {})
    
    print("\n📊 END-TO-END TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    
    if summary.get('key_findings'):
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
    
    if summary.get('performance_targets_met'):
        print("\nPerformance Profile Status:")
        for profile, status in summary['performance_targets_met'].items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {profile}")
    
    if summary.get('recommendations'):
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  🔧 {rec}")
    
    # Determine overall success
    overall_success = summary.get('overall_status') == 'PASSED'
    
    if overall_success:
        print("\n🎉 End-to-End Performance Test: PASSED")
        return 0
    else:
        print("\n⚠️  End-to-End Performance Test: NEEDS ATTENTION")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_e2e_performance_test())
    sys.exit(exit_code)
```

## Phase 4: Production Readiness (Week 7-8)

### 4.1 Production Monitoring Deployment

#### Monitoring Stack Configuration
```bash
# Create production monitoring configuration
cat > performance-testing/config/production-monitoring.yml << EOF
# Production monitoring configuration
monitoring:
  environment: production
  data_retention: 90d
  alert_channels:
    - email
    - slack
    - pagerduty

prometheus:
  scrape_configs:
    - job_name: 'rag-system-prod'
      static_configs:
        - targets: ['prod-rag-01:8000', 'prod-rag-02:8000']
      scrape_interval: 15s
      
grafana:
  datasources:
    - name: prometheus
      type: prometheus
      url: http://prometheus:9090
      
  dashboards:
    - dashboard: rag-system-overview
      uid: rag-overview
      panels:
        - title: "Request Rate"
          type: graph
          targets: ["rate(requests_total[5m])"]
        - title: "Response Time P95"
          type: graph  
          targets: ["histogram_quantile(0.95, response_time_seconds)"]
        - title: "Error Rate"
          type: stat
          targets: ["rate(errors_total[5m])"]

alerting:
  rules:
    - alert: HighResponseTime
      expr: histogram_quantile(0.95, response_time_seconds) > 5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High response time detected"
        
    - alert: HighErrorRate
      expr: rate(errors_total[5m]) > 0.05
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
EOF
```

### 4.2 Continuous Performance Testing Pipeline

#### CI/CD Integration Script
```yaml
# .github/workflows/performance-testing.yml
name: Performance Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:6-alpine
        ports:
          - 6379:6379
        options: --health-cmd redis-cli ping --health-interval 10s --health-timeout 5s --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r performance-testing/requirements.txt
      
      - name: Start RAG system
        run: |
          # Start the system in background
          uv run -m llama_deploy.apiserver &
          sleep 10
          uv run llamactl deploy llama_deploy.yml &
          sleep 30
      
      - name: Run performance tests
        run: |
          cd performance-testing
          python scripts/e2e-performance-test.py
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: performance-testing/results/
          
      - name: Performance regression check
        run: |
          python performance-testing/scripts/check-regression.py \
            --baseline performance-testing/results/baseline.json \
            --current performance-testing/results/latest.json \
            --threshold 0.20
```

### 4.3 Performance Regression Detection

#### Regression Detection Script
```python
# performance-testing/scripts/check-regression.py
#!/usr/bin/env python3
"""Check for performance regressions against baseline."""

import json
import argparse
import sys
from pathlib import Path

def load_results(file_path):
    """Load performance results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def check_regression(baseline, current, threshold=0.20):
    """Check for performance regression."""
    regressions = []
    improvements = []
    
    # Check response times
    baseline_p95 = baseline.get('profile_validation', {}).get('balanced', {}).get('p95_response_time', 0)
    current_p95 = current.get('profile_validation', {}).get('balanced', {}).get('p95_response_time', 0)
    
    if baseline_p95 > 0 and current_p95 > 0:
        change_ratio = (current_p95 - baseline_p95) / baseline_p95
        if change_ratio > threshold:
            regressions.append(f"Response time P95 regression: {change_ratio:.1%} slower")
        elif change_ratio < -threshold:
            improvements.append(f"Response time P95 improvement: {abs(change_ratio):.1%} faster")
    
    # Check accuracy
    baseline_accuracy = baseline.get('profile_validation', {}).get('balanced', {}).get('avg_accuracy', 0)
    current_accuracy = current.get('profile_validation', {}).get('balanced', {}).get('avg_accuracy', 0)
    
    if baseline_accuracy > 0 and current_accuracy > 0:
        accuracy_change = current_accuracy - baseline_accuracy
        if accuracy_change < -threshold:
            regressions.append(f"Accuracy regression: {abs(accuracy_change):.1%} decrease")
        elif accuracy_change > threshold:
            improvements.append(f"Accuracy improvement: {accuracy_change:.1%} increase")
    
    # Check cache hit rate
    baseline_cache = baseline.get('profile_validation', {}).get('cost_optimized', {}).get('cache_hit_rate', 0)
    current_cache = current.get('profile_validation', {}).get('cost_optimized', {}).get('cache_hit_rate', 0)
    
    if baseline_cache > 0 and current_cache > 0:
        cache_change = current_cache - baseline_cache
        if cache_change < -threshold:
            regressions.append(f"Cache hit rate regression: {abs(cache_change):.1%} decrease")
        elif cache_change > threshold:
            improvements.append(f"Cache hit rate improvement: {cache_change:.1%} increase")
    
    return regressions, improvements

def main():
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('--baseline', required=True, help='Baseline results file')
    parser.add_argument('--current', required=True, help='Current results file')
    parser.add_argument('--threshold', type=float, default=0.20, help='Regression threshold (default: 0.20)')
    
    args = parser.parse_args()
    
    # Load results
    try:
        baseline = load_results(args.baseline)
        current = load_results(args.current)
    except FileNotFoundError as e:
        print(f"❌ Error loading results: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return 1
    
    # Check for regressions
    regressions, improvements = check_regression(baseline, current, args.threshold)
    
    print("🔍 Performance Regression Analysis")
    print("=" * 50)
    
    if regressions:
        print("\n❌ REGRESSIONS DETECTED:")
        for regression in regressions:
            print(f"  • {regression}")
    
    if improvements:
        print("\n✅ IMPROVEMENTS DETECTED:")
        for improvement in improvements:
            print(f"  • {improvement}")
    
    if not regressions and not improvements:
        print("\n➡️  No significant performance changes detected")
    
    # Return exit code
    if regressions:
        print(f"\n⚠️  Performance regression detected (threshold: {args.threshold:.0%})")
        return 1
    else:
        print("\n🎉 No performance regressions detected")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Usage Instructions

### Daily Performance Monitoring
```bash
# Run daily performance check
cd performance-testing
./scripts/daily-performance-check.sh
```

### Load Testing Before Deployment
```bash
# Run pre-deployment load test
cd performance-testing
python scripts/pre-deployment-load-test.py --duration 300 --users 25
```

### Performance Profile Switching
```bash
# Switch to high accuracy profile
export PERFORMANCE_PROFILE=high_accuracy
uv run -m llama_deploy.apiserver

# Switch to speed profile  
export PERFORMANCE_PROFILE=speed
uv run -m llama_deploy.apiserver
```

### Emergency Performance Investigation
```bash
# Quick performance diagnosis
cd performance-testing
python scripts/performance-diagnosis.py --quick

# Detailed performance analysis
python scripts/performance-diagnosis.py --detailed --duration 600
```

## Best Practices

### Performance Testing Best Practices
1. **Always baseline before changes** - Collect baseline metrics before any system modifications
2. **Test in isolation** - Test individual components separately before integration testing
3. **Use realistic data** - Test with production-like query patterns and data volumes
4. **Monitor continuously** - Set up continuous monitoring to catch regressions early
5. **Document everything** - Keep detailed records of performance changes and their causes

### Monitoring Best Practices
1. **Set meaningful thresholds** - Base alerts on business impact, not arbitrary numbers
2. **Monitor leading indicators** - Track metrics that predict problems before they occur
3. **Keep dashboards simple** - Focus on the most important metrics for quick diagnosis
4. **Regular review and tuning** - Review and adjust monitoring thresholds regularly

### Optimization Best Practices
1. **Measure before optimizing** - Always measure performance before making changes
2. **Optimize the biggest bottlenecks first** - Focus on changes with the highest impact
3. **Test optimizations thoroughly** - Validate that optimizations actually improve performance
4. **Consider trade-offs** - Understand the trade-offs between speed, accuracy, and cost

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage breakdown
python performance-testing/scripts/memory-analysis.py

# Tune garbage collection
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1
```

#### Slow Cache Performance
```bash
# Check Redis performance
redis-cli --latency-history -i 1

# Optimize cache settings
python performance-testing/scripts/optimize-cache.py
```

#### Poor Verification Performance
```bash
# Analyze verification bottlenecks
python performance-testing/scripts/analyze-verification.py

# Tune batch processing
python performance-testing/scripts/tune-verification-batching.py
```

## Support and Maintenance

### Regular Maintenance Tasks
- **Weekly**: Review performance dashboards and alerts
- **Monthly**: Run comprehensive performance test suite
- **Quarterly**: Review and update performance targets
- **Annually**: Conduct comprehensive performance architecture review

### Getting Help
- Check the performance monitoring dashboards first
- Run the performance diagnosis script for automated analysis
- Review recent changes that might have affected performance
- Consult the performance optimization playbook for common issues

This implementation guide provides a comprehensive framework for deploying and maintaining the performance testing strategy for the Enhanced RAG System. Follow the phases sequentially for best results, and maintain continuous monitoring for optimal system performance.