#!/usr/bin/env python3
"""
Comprehensive Integration Test Runner for Enhanced RAG System

This script provides a comprehensive test runner for all integration tests,
with detailed reporting, performance metrics, and failure analysis.

Usage:
    python run_integration_tests.py [options]

Options:
    --suite SUITE_NAME      Run specific test suite (orchestration, cache, health, api, dataflow)
    --parallel              Run tests in parallel
    --coverage              Generate coverage reports
    --performance           Include performance benchmarking
    --html-report           Generate HTML test report
    --verbose               Enable verbose output
    --timeout SECONDS       Set test timeout (default: 300)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import importlib

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test suite configurations
TEST_SUITES = {
    "orchestration": {
        "name": "Unified Workflow Orchestration Tests",
        "module": "tests.integration.test_unified_workflow_orchestration",
        "description": "Tests for workflow orchestration, component selection, and coordination",
        "estimated_time": 120,  # seconds
        "requirements": ["mock_openai_client", "mock_redis_client", "mock_llama_index"]
    },
    "cache": {
        "name": "Cache + Verification Pipeline Tests", 
        "module": "tests.integration.test_cache_verification_pipeline",
        "description": "Tests for semantic cache and verification system integration",
        "estimated_time": 90,
        "requirements": ["mock_redis_client", "mock_openai_client"]
    },
    "health": {
        "name": "System Health and Monitoring Tests",
        "module": "tests.integration.test_system_health_monitoring", 
        "description": "Tests for health monitoring, performance tracking, and alerting",
        "estimated_time": 100,
        "requirements": ["health_monitor"]
    },
    "api": {
        "name": "LlamaDeploy API and Service Tests",
        "module": "tests.integration.test_llamadeploy_api_services",
        "description": "Tests for LlamaDeploy integration, API endpoints, and service communication",
        "estimated_time": 80,
        "requirements": ["mock_llamadeploy_client"]
    },
    "dataflow": {
        "name": "Data Flow Validation Tests",
        "module": "tests.integration.test_data_flow_validation",
        "description": "Tests for complete data flow from indexing to response generation",
        "estimated_time": 110,
        "requirements": ["temp_document_dir", "mock_openai_client"]
    }
}


class IntegrationTestRunner:
    """Comprehensive integration test runner with reporting and metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0,
            "suites": {},
            "overall_stats": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0
            },
            "performance_metrics": {},
            "coverage_data": {},
            "system_info": self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for test context."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "working_directory": str(Path.cwd()),
            "timestamp": time.time()
        }
    
    def run_all_suites(self) -> Dict[str, Any]:
        """Run all integration test suites."""
        print("🚀 Starting Enhanced RAG System Integration Tests")
        print("=" * 70)
        
        self.results["start_time"] = time.time()
        
        # Print system info
        self._print_system_info()
        
        # Determine suites to run
        if self.config.get("suite"):
            suites_to_run = [self.config["suite"]]
        else:
            suites_to_run = list(TEST_SUITES.keys())
        
        print(f"\n📋 Test Suites to Run: {', '.join(suites_to_run)}")
        print(f"⚙️  Configuration: {self._format_config()}")
        print("-" * 70)
        
        # Run each suite
        for suite_name in suites_to_run:
            if suite_name not in TEST_SUITES:
                print(f"❌ Unknown test suite: {suite_name}")
                continue
            
            try:
                suite_result = self._run_test_suite(suite_name)
                self.results["suites"][suite_name] = suite_result
                self._update_overall_stats(suite_result)
                
            except Exception as e:
                print(f"❌ Error running suite '{suite_name}': {e}")
                self.results["suites"][suite_name] = {
                    "status": "error",
                    "error": str(e),
                    "tests": {},
                    "duration": 0
                }
        
        self.results["end_time"] = time.time()
        self.results["total_duration"] = self.results["end_time"] - self.results["start_time"]
        
        # Generate reports
        self._generate_summary_report()
        
        if self.config.get("html_report"):
            self._generate_html_report()
        
        if self.config.get("coverage"):
            self._generate_coverage_report()
        
        return self.results
    
    def _print_system_info(self):
        """Print system information."""
        info = self.results["system_info"]
        print(f"\n🖥️  System Information:")
        print(f"   Platform: {info['platform']}")
        print(f"   Python: {info['python_version']}")
        print(f"   CPU Cores: {info['cpu_count']}")
        print(f"   Memory: {info['memory_gb']} GB")
    
    def _format_config(self) -> str:
        """Format configuration for display."""
        config_items = []
        if self.config.get("parallel"):
            config_items.append("parallel")
        if self.config.get("performance"):
            config_items.append("performance")
        if self.config.get("coverage"):
            config_items.append("coverage")
        if self.config.get("verbose"):
            config_items.append("verbose")
        
        return ", ".join(config_items) if config_items else "default"
    
    def _run_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite."""
        suite_config = TEST_SUITES[suite_name]
        
        print(f"\n🧪 Running: {suite_config['name']}")
        print(f"   Description: {suite_config['description']}")
        print(f"   Estimated Time: {suite_config['estimated_time']}s")
        
        start_time = time.time()
        
        try:
            # Build pytest command
            pytest_cmd = self._build_pytest_command(suite_name, suite_config)
            
            # Run tests
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 300)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse pytest output
            suite_result = self._parse_pytest_output(result, suite_name, duration)
            
            # Print suite results
            self._print_suite_results(suite_name, suite_result)
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            
            error_result = {
                "status": "timeout",
                "duration": duration,
                "tests": {},
                "stdout": "",
                "stderr": f"Test suite timed out after {self.config.get('timeout', 300)}s",
                "return_code": -1
            }
            
            print(f"   ⏰ TIMEOUT after {duration:.1f}s")
            return error_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            error_result = {
                "status": "error",
                "duration": duration,
                "tests": {},
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
            
            print(f"   ❌ ERROR: {e}")
            return error_result
    
    def _build_pytest_command(self, suite_name: str, suite_config: Dict[str, Any]) -> List[str]:
        """Build pytest command for test suite."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test module
        module_path = suite_config["module"].replace(".", "/") + ".py"
        cmd.append(module_path)
        
        # Add pytest options
        cmd.extend([
            "-v",  # Verbose
            "--tb=short",  # Short traceback format
            "--disable-warnings",  # Reduce noise
            f"--timeout={self.config.get('timeout', 300)}"
        ])
        
        # Parallel execution
        if self.config.get("parallel"):
            cmd.extend(["-n", "auto"])
        
        # Coverage
        if self.config.get("coverage"):
            cmd.extend([
                "--cov=src",
                f"--cov-report=term-missing",
                f"--cov-report=html:coverage_{suite_name}"
            ])
        
        # Performance benchmarking
        if self.config.get("performance"):
            cmd.extend([
                "--benchmark-only",
                "--benchmark-sort=mean"
            ])
        
        # Output format
        cmd.extend([
            "--json-report",
            f"--json-report-file=test_results_{suite_name}.json"
        ])
        
        return cmd
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess, 
                           suite_name: str, duration: float) -> Dict[str, Any]:
        """Parse pytest output and extract test results."""
        
        # Try to load JSON report
        json_report_file = f"test_results_{suite_name}.json"
        test_details = {}
        
        if os.path.exists(json_report_file):
            try:
                with open(json_report_file, 'r') as f:
                    json_data = json.load(f)
                    
                # Extract test details from JSON report
                for test in json_data.get("tests", []):
                    test_details[test["nodeid"]] = {
                        "status": test["outcome"],
                        "duration": test.get("setup", {}).get("duration", 0) + 
                                  test.get("call", {}).get("duration", 0) + 
                                  test.get("teardown", {}).get("duration", 0),
                        "error": test.get("call", {}).get("longrepr", "") if test["outcome"] == "failed" else None
                    }
                    
            except Exception as e:
                print(f"   ⚠️  Could not parse JSON report: {e}")
        
        # Determine overall status
        if result.returncode == 0:
            status = "passed"
        elif "FAILURES" in result.stdout or "ERRORS" in result.stdout:
            status = "failed"
        else:
            status = "error"
        
        return {
            "status": status,
            "duration": duration,
            "tests": test_details,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def _print_suite_results(self, suite_name: str, suite_result: Dict[str, Any]):
        """Print results for a test suite."""
        status = suite_result["status"]
        duration = suite_result["duration"]
        
        # Status icon
        status_icons = {
            "passed": "✅",
            "failed": "❌", 
            "error": "💥",
            "timeout": "⏰"
        }
        
        icon = status_icons.get(status, "❓")
        
        print(f"   {icon} {status.upper()} in {duration:.1f}s")
        
        # Test breakdown
        tests = suite_result.get("tests", {})
        if tests:
            passed = sum(1 for t in tests.values() if t["status"] == "passed")
            failed = sum(1 for t in tests.values() if t["status"] == "failed")
            errors = sum(1 for t in tests.values() if t["status"] == "error")
            skipped = sum(1 for t in tests.values() if t["status"] == "skipped")
            
            print(f"   📊 Tests: {len(tests)} total, {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
            
            # Show failed tests
            if failed > 0 or errors > 0:
                print(f"   💔 Failures:")
                for test_name, test_info in tests.items():
                    if test_info["status"] in ["failed", "error"]:
                        short_name = test_name.split("::")[-1]
                        error_preview = test_info.get("error", "")[:100] + "..." if test_info.get("error") else ""
                        print(f"      - {short_name}: {error_preview}")
        
        if status == "error" and suite_result.get("stderr"):
            print(f"   ⚠️  Error: {suite_result['stderr'][:200]}...")
    
    def _update_overall_stats(self, suite_result: Dict[str, Any]):
        """Update overall statistics."""
        tests = suite_result.get("tests", {})
        
        for test_info in tests.values():
            self.results["overall_stats"]["total_tests"] += 1
            
            status = test_info["status"]
            if status == "passed":
                self.results["overall_stats"]["passed"] += 1
            elif status == "failed":
                self.results["overall_stats"]["failed"] += 1
            elif status == "error":
                self.results["overall_stats"]["errors"] += 1
            elif status == "skipped":
                self.results["overall_stats"]["skipped"] += 1
    
    def _generate_summary_report(self):
        """Generate and display summary report."""
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        # Overall stats
        stats = self.results["overall_stats"]
        total_duration = self.results["total_duration"]
        
        print(f"⏱️  Total Duration: {total_duration:.1f}s")
        print(f"🧪 Total Tests: {stats['total_tests']}")
        print(f"✅ Passed: {stats['passed']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"💥 Errors: {stats['errors']}")
        print(f"⏭️  Skipped: {stats['skipped']}")
        
        if stats['total_tests'] > 0:
            success_rate = (stats['passed'] / stats['total_tests']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Suite breakdown
        print(f"\n📋 Suite Results:")
        for suite_name, suite_result in self.results["suites"].items():
            suite_config = TEST_SUITES[suite_name]
            status = suite_result["status"]
            duration = suite_result["duration"]
            
            status_icon = {"passed": "✅", "failed": "❌", "error": "💥", "timeout": "⏰"}.get(status, "❓")
            
            print(f"   {status_icon} {suite_config['name']}: {status} ({duration:.1f}s)")
        
        # Overall result
        if stats['failed'] == 0 and stats['errors'] == 0:
            print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED - Review failures above")
        
        print("=" * 70)
    
    def _generate_html_report(self):
        """Generate HTML test report."""
        try:
            html_content = self._build_html_report()
            
            report_path = Path("integration_test_report.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📄 HTML Report: {report_path.absolute()}")
            
        except Exception as e:
            print(f"⚠️  Could not generate HTML report: {e}")
    
    def _build_html_report(self) -> str:
        """Build HTML report content."""
        stats = self.results["overall_stats"]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced RAG System - Integration Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: #f9f9f9; padding: 15px; border-radius: 8px; text-align: center; }}
                .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
                .passed {{ border-left: 5px solid #28a745; }}
                .failed {{ border-left: 5px solid #dc3545; }}
                .error {{ border-left: 5px solid #ffc107; }}
                .test-list {{ margin-top: 10px; font-size: 0.9em; }}
                .test-item {{ padding: 5px; margin: 5px 0; }}
                .test-passed {{ background: #d4edda; }}
                .test-failed {{ background: #f8d7da; }}
                .test-error {{ background: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced RAG System - Integration Test Report</h1>
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Duration: {self.results['total_duration']:.1f} seconds</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{stats['total_tests']}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['passed']}</h3>
                    <p>Passed</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['failed']}</h3>
                    <p>Failed</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['errors']}</h3>
                    <p>Errors</p>
                </div>
            </div>
        """
        
        # Add suite details
        for suite_name, suite_result in self.results["suites"].items():
            suite_config = TEST_SUITES[suite_name]
            status_class = suite_result["status"]
            
            html += f"""
            <div class="suite {status_class}">
                <h3>{suite_config['name']} ({suite_result['status']})</h3>
                <p>{suite_config['description']}</p>
                <p>Duration: {suite_result['duration']:.1f}s</p>
            """
            
            # Add test details if available
            if suite_result.get("tests"):
                html += '<div class="test-list">'
                for test_name, test_info in suite_result["tests"].items():
                    short_name = test_name.split("::")[-1]
                    test_class = f"test-{test_info['status']}"
                    
                    html += f'<div class="test-item {test_class}">'
                    html += f'{short_name} ({test_info["status"]}) - {test_info["duration"]:.2f}s'
                    html += '</div>'
                html += '</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_coverage_report(self):
        """Generate coverage report summary."""
        print(f"\n📈 Coverage reports generated in coverage_* directories")
        
        # Try to summarize coverage if data exists
        for suite_name in self.results["suites"].keys():
            coverage_dir = Path(f"coverage_{suite_name}")
            if coverage_dir.exists():
                print(f"   📊 {suite_name}: {coverage_dir}")


def main():
    """Main entry point for integration test runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG System Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--suite", 
        choices=list(TEST_SUITES.keys()),
        help="Run specific test suite"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Generate coverage reports"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Include performance benchmarking"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Set test timeout in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    config = {
        "suite": args.suite,
        "parallel": args.parallel,
        "coverage": args.coverage,
        "performance": args.performance,
        "html_report": args.html_report,
        "verbose": args.verbose,
        "timeout": args.timeout
    }
    
    runner = IntegrationTestRunner(config)
    
    # Run tests
    try:
        results = runner.run_all_suites()
        
        # Exit with appropriate code
        if results["overall_stats"]["failed"] > 0 or results["overall_stats"]["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()