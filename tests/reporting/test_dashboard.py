"""
Test reporting and metrics dashboard for SOTA RAG system.

Provides:
- Comprehensive test result aggregation
- Performance metrics visualization
- Quality trend analysis
- Coverage reporting
- Interactive dashboard generation
"""

import json
import sqlite3
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import html

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class TestResult:
    """Individual test result record."""
    test_id: str
    test_name: str
    category: str  # 'unit', 'integration', 'e2e', 'performance', 'quality'
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None


@dataclass
class TestSuite:
    """Test suite results summary."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    timestamp: datetime
    coverage_percentage: float = 0.0
    performance_score: float = 0.0
    quality_score: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    throughput_qps: float
    cache_hit_rate: float
    accuracy_score: float
    cost_per_query: float
    error_rate: float
    availability: float


class TestDatabase:
    """SQLite database for storing test results and metrics."""
    
    def __init__(self, db_path: str = "test_results.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                duration REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                error_message TEXT,
                metrics TEXT
            )
        """)
        
        # Test suites table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_suites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_name TEXT NOT NULL,
                total_tests INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                errors INTEGER NOT NULL,
                total_duration REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                coverage_percentage REAL DEFAULT 0.0,
                performance_score REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                response_time_p50 REAL NOT NULL,
                response_time_p95 REAL NOT NULL,
                response_time_p99 REAL NOT NULL,
                throughput_qps REAL NOT NULL,
                cache_hit_rate REAL NOT NULL,
                accuracy_score REAL NOT NULL,
                cost_per_query REAL NOT NULL,
                error_rate REAL NOT NULL,
                availability REAL NOT NULL
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_results_timestamp ON test_results(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_suites_timestamp ON test_suites(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
        
        conn.commit()
        conn.close()
    
    def store_test_result(self, result: TestResult):
        """Store a single test result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_results 
            (test_id, test_name, category, status, duration, timestamp, error_message, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_id,
            result.test_name,
            result.category,
            result.status,
            result.duration,
            result.timestamp,
            result.error_message,
            json.dumps(result.metrics) if result.metrics else None
        ))
        
        conn.commit()
        conn.close()
    
    def store_test_suite(self, suite: TestSuite):
        """Store test suite summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_suites 
            (suite_name, total_tests, passed, failed, skipped, errors, 
             total_duration, timestamp, coverage_percentage, performance_score, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            suite.suite_name,
            suite.total_tests,
            suite.passed,
            suite.failed,
            suite.skipped,
            suite.errors,
            suite.total_duration,
            suite.timestamp,
            suite.coverage_percentage,
            suite.performance_score,
            suite.quality_score
        ))
        
        conn.commit()
        conn.close()
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics 
            (timestamp, response_time_p50, response_time_p95, response_time_p99,
             throughput_qps, cache_hit_rate, accuracy_score, cost_per_query,
             error_rate, availability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp,
            metrics.response_time_p50,
            metrics.response_time_p95,
            metrics.response_time_p99,
            metrics.throughput_qps,
            metrics.cache_hit_rate,
            metrics.accuracy_score,
            metrics.cost_per_query,
            metrics.error_rate,
            metrics.availability
        ))
        
        conn.commit()
        conn.close()
    
    def get_test_results(self, category: str = None, days: int = 7) -> List[TestResult]:
        """Get test results with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM test_results WHERE timestamp >= ?"
        params = [datetime.now() - timedelta(days=days)]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            metrics = json.loads(row[8]) if row[8] else None
            results.append(TestResult(
                test_id=row[1],
                test_name=row[2],
                category=row[3],
                status=row[4],
                duration=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                error_message=row[7],
                metrics=metrics
            ))
        
        return results
    
    def get_test_suites(self, days: int = 7) -> List[TestSuite]:
        """Get test suite summaries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM test_suites 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        """, (datetime.now() - timedelta(days=days),))
        
        rows = cursor.fetchall()
        conn.close()
        
        suites = []
        for row in rows:
            suites.append(TestSuite(
                suite_name=row[1],
                total_tests=row[2],
                passed=row[3],
                failed=row[4],
                skipped=row[5],
                errors=row[6],
                total_duration=row[7],
                timestamp=datetime.fromisoformat(row[8]),
                coverage_percentage=row[9],
                performance_score=row[10],
                quality_score=row[11]
            ))
        
        return suites
    
    def get_system_metrics(self, days: int = 7) -> List[SystemMetrics]:
        """Get system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM system_metrics 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        """, (datetime.now() - timedelta(days=days),))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in rows:
            metrics.append(SystemMetrics(
                timestamp=datetime.fromisoformat(row[1]),
                response_time_p50=row[2],
                response_time_p95=row[3],
                response_time_p99=row[4],
                throughput_qps=row[5],
                cache_hit_rate=row[6],
                accuracy_score=row[7],
                cost_per_query=row[8],
                error_rate=row[9],
                availability=row[10]
            ))
        
        return metrics


class TestReporter:
    """Generate comprehensive test reports."""
    
    def __init__(self, database: TestDatabase):
        self.db = database
    
    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate summary report of test results."""
        test_results = self.db.get_test_results(days=days)
        test_suites = self.db.get_test_suites(days=days)
        system_metrics = self.db.get_system_metrics(days=days)
        
        # Test results summary
        results_by_category = {}
        for result in test_results:
            if result.category not in results_by_category:
                results_by_category[result.category] = {
                    'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0
                }
            results_by_category[result.category]['total'] += 1
            results_by_category[result.category][result.status] += 1
        
        # Latest test suite results
        latest_suite = test_suites[0] if test_suites else None
        
        # System performance trends
        performance_trend = {}
        if system_metrics:
            recent_metrics = system_metrics[:10]  # Last 10 data points
            performance_trend = {
                'avg_response_time': statistics.mean(m.response_time_p95 for m in recent_metrics),
                'avg_throughput': statistics.mean(m.throughput_qps for m in recent_metrics),
                'avg_cache_hit_rate': statistics.mean(m.cache_hit_rate for m in recent_metrics),
                'avg_accuracy': statistics.mean(m.accuracy_score for m in recent_metrics),
                'avg_cost': statistics.mean(m.cost_per_query for m in recent_metrics),
                'avg_error_rate': statistics.mean(m.error_rate for m in recent_metrics)
            }
        
        return {
            'timestamp': datetime.now(),
            'period_days': days,
            'test_results_summary': results_by_category,
            'latest_test_suite': asdict(latest_suite) if latest_suite else None,
            'performance_trends': performance_trend,
            'total_tests_run': len(test_results),
            'overall_pass_rate': len([r for r in test_results if r.status == 'passed']) / len(test_results) if test_results else 0
        }
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate detailed performance report."""
        performance_results = self.db.get_test_results(category='performance', days=days)
        system_metrics = self.db.get_system_metrics(days=days)
        
        # Performance test analysis
        performance_analysis = {}
        if performance_results:
            durations = [r.duration for r in performance_results]
            performance_analysis = {
                'total_performance_tests': len(performance_results),
                'avg_test_duration': statistics.mean(durations),
                'p95_test_duration': sorted(durations)[int(0.95 * len(durations))] if durations else 0,
                'failed_performance_tests': len([r for r in performance_results if r.status == 'failed'])
            }
        
        # System performance trends
        performance_trends = []
        for metric in system_metrics[-24:]:  # Last 24 data points
            performance_trends.append({
                'timestamp': metric.timestamp.isoformat(),
                'response_time_p95': metric.response_time_p95,
                'throughput': metric.throughput_qps,
                'cache_hit_rate': metric.cache_hit_rate,
                'accuracy': metric.accuracy_score,
                'cost_per_query': metric.cost_per_query
            })
        
        return {
            'timestamp': datetime.now(),
            'performance_test_analysis': performance_analysis,
            'performance_trends': performance_trends,
            'benchmark_validation': self._validate_benchmarks(system_metrics)
        }
    
    def _validate_benchmarks(self, metrics: List[SystemMetrics]) -> Dict[str, bool]:
        """Validate system performance against benchmarks."""
        if not metrics:
            return {}
        
        latest = metrics[0]
        
        return {
            'response_time_p95_under_3s': latest.response_time_p95 < 3.0,
            'throughput_above_5_qps': latest.throughput_qps >= 5.0,
            'cache_hit_rate_above_25pct': latest.cache_hit_rate >= 0.25,
            'accuracy_above_90pct': latest.accuracy_score >= 0.90,
            'error_rate_below_1pct': latest.error_rate < 0.01,
            'availability_above_99pct': latest.availability >= 0.99
        }
    
    def generate_quality_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate quality assurance report."""
        quality_results = self.db.get_test_results(category='quality', days=days)
        
        quality_metrics = {}
        accuracy_scores = []
        
        for result in quality_results:
            if result.metrics:
                if 'accuracy_score' in result.metrics:
                    accuracy_scores.append(result.metrics['accuracy_score'])
                
                # Collect other quality metrics
                for metric, value in result.metrics.items():
                    if metric not in quality_metrics:
                        quality_metrics[metric] = []
                    quality_metrics[metric].append(value)
        
        # Calculate quality summary
        quality_summary = {}
        for metric, values in quality_metrics.items():
            if isinstance(values[0], (int, float)):
                quality_summary[metric] = {
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return {
            'timestamp': datetime.now(),
            'total_quality_tests': len(quality_results),
            'passed_quality_tests': len([r for r in quality_results if r.status == 'passed']),
            'quality_metrics_summary': quality_summary,
            'accuracy_trend': accuracy_scores[-10:] if accuracy_scores else []  # Last 10 measurements
        }


class DashboardGenerator:
    """Generate HTML dashboard for test results."""
    
    def __init__(self, database: TestDatabase):
        self.db = database
        self.reporter = TestReporter(database)
    
    def generate_html_dashboard(self, output_path: str = "test_dashboard.html"):
        """Generate comprehensive HTML dashboard."""
        summary_report = self.reporter.generate_summary_report()
        performance_report = self.reporter.generate_performance_report()
        quality_report = self.reporter.generate_quality_report()
        
        html_content = self._generate_html_template(
            summary_report, performance_report, quality_report
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard generated: {output_path}")
        return output_path
    
    def _generate_html_template(self, summary_report: Dict, performance_report: Dict, quality_report: Dict) -> str:
        """Generate HTML template with embedded data."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOTA RAG Test Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 25px;
            border-left: 4px solid #667eea;
        }}
        .card h3 {{
            margin-top: 0;
            color: #333;
            font-size: 1.3em;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .metric-label {{
            font-weight: 500;
            color: #555;
        }}
        .metric-value {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .metric-value.success {{ color: #28a745; }}
        .metric-value.warning {{ color: #ffc107; }}
        .metric-value.danger {{ color: #dc3545; }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .status-item {{
            text-align: center;
            padding: 15px;
            border-radius: 6px;
            background: #f8f9fa;
        }}
        .status-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .status-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .chart-placeholder {{
            height: 200px;
            background: #f8f9fa;
            border: 2px dashed #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-style: italic;
            margin: 20px 0;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .benchmark-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 8px;
        }}
        .benchmark-pass {{ background-color: #28a745; }}
        .benchmark-fail {{ background-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SOTA RAG Test Dashboard</h1>
            <div class="subtitle">Generated on {summary_report['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
        </div>
        
        <div class="dashboard-grid">
            <!-- Test Results Summary -->
            <div class="card">
                <h3>Test Results Summary</h3>
                <div class="metric">
                    <span class="metric-label">Total Tests Run</span>
                    <span class="metric-value">{summary_report['total_tests_run']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Overall Pass Rate</span>
                    <span class="metric-value {'success' if summary_report['overall_pass_rate'] > 0.9 else 'warning' if summary_report['overall_pass_rate'] > 0.8 else 'danger'}">{summary_report['overall_pass_rate']:.1%}</span>
                </div>
                
                <div class="status-grid">
                    {self._generate_test_category_stats(summary_report['test_results_summary'])}
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div class="card">
                <h3>Performance Metrics</h3>
                {self._generate_performance_metrics(summary_report.get('performance_trends', {}))}
                
                <h4>Benchmark Validation</h4>
                {self._generate_benchmark_indicators(performance_report.get('benchmark_validation', {}))}
            </div>
            
            <!-- Quality Metrics -->
            <div class="card">
                <h3>Quality Assurance</h3>
                <div class="metric">
                    <span class="metric-label">Quality Tests</span>
                    <span class="metric-value">{quality_report['total_quality_tests']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality Pass Rate</span>
                    <span class="metric-value {'success' if quality_report['passed_quality_tests'] / max(quality_report['total_quality_tests'], 1) > 0.9 else 'warning'}">{quality_report['passed_quality_tests'] / max(quality_report['total_quality_tests'], 1):.1%}</span>
                </div>
                
                {self._generate_quality_metrics(quality_report.get('quality_metrics_summary', {}))}
            </div>
            
            <!-- Performance Trends -->
            <div class="card">
                <h3>Performance Trends</h3>
                <div class="chart-placeholder">
                    Performance trend charts would be displayed here
                    <br>(Requires visualization library integration)
                </div>
            </div>
            
            <!-- Test Coverage -->
            <div class="card">
                <h3>Test Coverage</h3>
                {self._generate_coverage_info(summary_report.get('latest_test_suite'))}
            </div>
            
            <!-- System Health -->
            <div class="card">
                <h3>System Health</h3>
                {self._generate_system_health(summary_report.get('performance_trends', {}))}
            </div>
        </div>
        
        <div class="footer">
            <p>SOTA RAG Testing Framework | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>Automated testing ensures system reliability, performance, and quality standards.</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _generate_test_category_stats(self, results_summary: Dict) -> str:
        """Generate HTML for test category statistics."""
        html = ""
        for category, stats in results_summary.items():
            success_rate = stats['passed'] / max(stats['total'], 1)
            color_class = 'success' if success_rate > 0.9 else 'warning' if success_rate > 0.8 else 'danger'
            
            html += f"""
            <div class="status-item">
                <div class="status-number {color_class}">{stats['total']}</div>
                <div class="status-label">{category.title()}</div>
                <div style="font-size: 0.8em; margin-top: 5px;">{success_rate:.1%} pass</div>
            </div>
            """
        return html
    
    def _generate_performance_metrics(self, trends: Dict) -> str:
        """Generate HTML for performance metrics."""
        if not trends:
            return '<div class="metric"><span class="metric-label">No performance data available</span></div>'
        
        return f"""
        <div class="metric">
            <span class="metric-label">Avg Response Time</span>
            <span class="metric-value {'success' if trends.get('avg_response_time', 0) < 2.0 else 'warning'}">{trends.get('avg_response_time', 0):.2f}s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Throughput</span>
            <span class="metric-value">{trends.get('avg_throughput', 0):.1f} qps</span>
        </div>
        <div class="metric">
            <span class="metric-label">Cache Hit Rate</span>
            <span class="metric-value {'success' if trends.get('avg_cache_hit_rate', 0) > 0.3 else 'warning'}">{trends.get('avg_cache_hit_rate', 0):.1%}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Accuracy</span>
            <span class="metric-value {'success' if trends.get('avg_accuracy', 0) > 0.9 else 'warning'}">{trends.get('avg_accuracy', 0):.1%}</span>
        </div>
        """
    
    def _generate_benchmark_indicators(self, benchmarks: Dict) -> str:
        """Generate benchmark validation indicators."""
        if not benchmarks:
            return '<div class="metric"><span class="metric-label">No benchmark data available</span></div>'
        
        html = ""
        for benchmark, passed in benchmarks.items():
            indicator_class = "benchmark-pass" if passed else "benchmark-fail"
            benchmark_label = benchmark.replace('_', ' ').title()
            
            html += f"""
            <div class="metric">
                <span class="metric-label">{benchmark_label}</span>
                <span class="metric-value">
                    {"PASS" if passed else "FAIL"}
                    <span class="benchmark-indicator {indicator_class}"></span>
                </span>
            </div>
            """
        
        return html
    
    def _generate_quality_metrics(self, quality_summary: Dict) -> str:
        """Generate quality metrics HTML."""
        if not quality_summary:
            return '<div class="metric"><span class="metric-label">No quality data available</span></div>'
        
        html = ""
        for metric, data in quality_summary.items():
            if isinstance(data, dict) and 'average' in data:
                metric_label = metric.replace('_', ' ').title()
                value = data['average']
                
                # Determine color based on metric type
                color_class = 'success'
                if 'accuracy' in metric or 'confidence' in metric:
                    color_class = 'success' if value > 0.9 else 'warning' if value > 0.8 else 'danger'
                elif 'error' in metric or 'hallucination' in metric:
                    color_class = 'success' if value < 0.1 else 'warning' if value < 0.2 else 'danger'
                
                html += f"""
                <div class="metric">
                    <span class="metric-label">{metric_label}</span>
                    <span class="metric-value {color_class}">{value:.3f}</span>
                </div>
                """
        
        return html
    
    def _generate_coverage_info(self, latest_suite: Optional[Dict]) -> str:
        """Generate test coverage information."""
        if not latest_suite:
            return '<div class="metric"><span class="metric-label">No coverage data available</span></div>'
        
        coverage = latest_suite.get('coverage_percentage', 0)
        color_class = 'success' if coverage > 80 else 'warning' if coverage > 60 else 'danger'
        
        return f"""
        <div class="metric">
            <span class="metric-label">Code Coverage</span>
            <span class="metric-value {color_class}">{coverage:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Performance Score</span>
            <span class="metric-value">{latest_suite.get('performance_score', 0):.1f}/100</span>
        </div>
        <div class="metric">
            <span class="metric-label">Quality Score</span>
            <span class="metric-value">{latest_suite.get('quality_score', 0):.1f}/100</span>
        </div>
        """
    
    def _generate_system_health(self, trends: Dict) -> str:
        """Generate system health indicators."""
        if not trends:
            return '<div class="metric"><span class="metric-label">No health data available</span></div>'
        
        error_rate = trends.get('avg_error_rate', 0)
        error_color = 'success' if error_rate < 0.01 else 'warning' if error_rate < 0.05 else 'danger'
        
        return f"""
        <div class="metric">
            <span class="metric-label">Error Rate</span>
            <span class="metric-value {error_color}">{error_rate:.2%}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Cost per Query</span>
            <span class="metric-value">${trends.get('avg_cost', 0):.4f}</span>
        </div>
        """


class TestMetricsCollector:
    """Collect and aggregate test metrics from various sources."""
    
    def __init__(self, database: TestDatabase):
        self.db = database
    
    def collect_pytest_results(self, junit_xml_path: str) -> List[TestResult]:
        """Parse pytest JUnit XML results."""
        import xml.etree.ElementTree as ET
        
        if not Path(junit_xml_path).exists():
            return []
        
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()
        
        results = []
        
        for testsuite in root.findall('testsuite'):
            for testcase in testsuite.findall('testcase'):
                test_name = testcase.get('name', '')
                classname = testcase.get('classname', '')
                duration = float(testcase.get('time', 0))
                
                # Determine status
                status = 'passed'
                error_message = None
                
                if testcase.find('failure') is not None:
                    status = 'failed'
                    failure = testcase.find('failure')
                    error_message = failure.get('message', '')
                elif testcase.find('error') is not None:
                    status = 'error'
                    error = testcase.find('error')
                    error_message = error.get('message', '')
                elif testcase.find('skipped') is not None:
                    status = 'skipped'
                
                # Extract category from classname
                category = 'unit'
                if 'integration' in classname.lower():
                    category = 'integration'
                elif 'e2e' in classname.lower():
                    category = 'e2e'
                elif 'performance' in classname.lower():
                    category = 'performance'
                elif 'quality' in classname.lower():
                    category = 'quality'
                
                result = TestResult(
                    test_id=f"{classname}::{test_name}",
                    test_name=test_name,
                    category=category,
                    status=status,
                    duration=duration,
                    timestamp=datetime.now(),
                    error_message=error_message
                )
                
                results.append(result)
                self.db.store_test_result(result)
        
        return results
    
    def collect_coverage_data(self, coverage_xml_path: str) -> float:
        """Extract coverage percentage from coverage.xml."""
        import xml.etree.ElementTree as ET
        
        if not Path(coverage_xml_path).exists():
            return 0.0
        
        try:
            tree = ET.parse(coverage_xml_path)
            root = tree.getroot()
            
            # Find coverage element and extract line-rate
            coverage_elem = root.find('coverage')
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get('line-rate', 0))
                return line_rate * 100  # Convert to percentage
        except Exception:
            pass
        
        return 0.0
    
    def collect_performance_metrics(
        self, 
        response_times: List[float],
        throughput: float,
        cache_hit_rate: float,
        accuracy_score: float,
        cost_per_query: float,
        error_rate: float = 0.0,
        availability: float = 1.0
    ):
        """Collect and store system performance metrics."""
        if not response_times:
            return
        
        response_times_sorted = sorted(response_times)
        n = len(response_times_sorted)
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            response_time_p50=response_times_sorted[int(0.5 * n)],
            response_time_p95=response_times_sorted[int(0.95 * n)],
            response_time_p99=response_times_sorted[int(0.99 * n)] if n > 1 else response_times_sorted[0],
            throughput_qps=throughput,
            cache_hit_rate=cache_hit_rate,
            accuracy_score=accuracy_score,
            cost_per_query=cost_per_query,
            error_rate=error_rate,
            availability=availability
        )
        
        self.db.store_system_metrics(metrics)
        return metrics


# CLI interface for dashboard generation
def main():
    """Main CLI interface for test dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SOTA RAG Test Dashboard')
    parser.add_argument('--db-path', default='test_results.db', help='Path to test results database')
    parser.add_argument('--output', default='test_dashboard.html', help='Output HTML file path')
    parser.add_argument('--junit-xml', help='Path to JUnit XML results file')
    parser.add_argument('--coverage-xml', help='Path to coverage XML file')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to include')
    
    args = parser.parse_args()
    
    # Initialize database and components
    db = TestDatabase(args.db_path)
    collector = TestMetricsCollector(db)
    generator = DashboardGenerator(db)
    
    # Collect test results if provided
    if args.junit_xml:
        print(f"Collecting test results from {args.junit_xml}")
        results = collector.collect_pytest_results(args.junit_xml)
        print(f"Collected {len(results)} test results")
        
        # Create test suite summary
        if results:
            suite = TestSuite(
                suite_name="Latest Test Run",
                total_tests=len(results),
                passed=len([r for r in results if r.status == 'passed']),
                failed=len([r for r in results if r.status == 'failed']),
                skipped=len([r for r in results if r.status == 'skipped']),
                errors=len([r for r in results if r.status == 'error']),
                total_duration=sum(r.duration for r in results),
                timestamp=datetime.now(),
                coverage_percentage=collector.collect_coverage_data(args.coverage_xml) if args.coverage_xml else 0.0
            )
            db.store_test_suite(suite)
    
    # Generate dashboard
    print(f"Generating dashboard for last {args.days} days")
    output_path = generator.generate_html_dashboard(args.output)
    print(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()