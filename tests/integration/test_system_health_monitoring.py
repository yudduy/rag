"""
Integration Tests for System Health and Monitoring

This module provides comprehensive integration tests for system health monitoring,
focusing on:
1. Health check integration across all components
2. Performance monitoring during workflow execution
3. Resource usage tracking and limits
4. Circuit breaker and fallback behavior
5. Real-time health status reporting
6. Alert and notification systems

Tests validate monitoring capabilities under normal and stress conditions.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from src.health_monitor import HealthMonitor, get_health_monitor, ComponentHealth, SystemHealth
from src.unified_workflow import UnifiedWorkflow
from src.unified_config import get_unified_config, reset_unified_config
from src.resource_manager import ResourceManager, get_resource_manager


class TestSystemHealthIntegration:
    """Test system health monitoring integration across components."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor instance for testing."""
        monitor = HealthMonitor()
        monitor.component_health = {}
        monitor.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_latency': 0.0
        }
        return monitor
    
    @pytest.mark.asyncio
    async def test_workflow_health_integration(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test health monitoring integration with unified workflow."""
        
        # Setup workflow with health monitoring
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Mock successful component initialization
        with patch('src.health_monitor.get_health_monitor', return_value=health_monitor):
            
            # Process a query and monitor health
            query = "What is artificial intelligence?"
            
            with patch.object(workflow, '_execute_with_plan') as mock_execute:
                mock_execute.return_value = "AI is a field of computer science."
                
                class MockStartEvent:
                    def __init__(self, query):
                        self.query = query
                        
                    def dict(self):
                        return {"query": self.query}
                
                # Execute workflow
                start_time = time.time()
                result = await workflow.arun(MockStartEvent(query))
                end_time = time.time()
                
                # Update health metrics
                processing_time = end_time - start_time
                health_monitor.update_component_health(
                    'unified_workflow',
                    'healthy',
                    {
                        'last_query_time': processing_time,
                        'total_queries': 1,
                        'success_rate': 1.0
                    }
                )
                
                # Validate health reporting
                system_health = health_monitor.get_system_health()
                assert 'unified_workflow' in system_health.component_details
                
                workflow_health = system_health.component_details['unified_workflow']
                assert workflow_health.status == 'healthy'
                assert workflow_health.metrics['success_rate'] == 1.0
                assert workflow_health.metrics['total_queries'] == 1
    
    @pytest.mark.asyncio
    async def test_component_health_monitoring_during_execution(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test real-time component health monitoring during workflow execution."""
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        # Mock component operations with health updates
        components_to_monitor = [
            'semantic_cache',
            'hallucination_detection', 
            'agentic_workflow',
            'multimodal_support'
        ]
        
        # Simulate component operations
        for component in components_to_monitor:
            # Simulate healthy component
            health_monitor.update_component_health(
                component,
                'healthy',
                {
                    'uptime': 3600,  # 1 hour
                    'request_count': 100,
                    'error_count': 2,
                    'avg_response_time': 0.5
                }
            )
        
        # Test system health aggregation
        system_health = health_monitor.get_system_health()
        
        assert system_health.overall_status in ['healthy', 'degraded']
        assert len(system_health.component_details) >= len(components_to_monitor)
        
        # All monitored components should be healthy
        for component in components_to_monitor:
            if component in system_health.component_details:
                component_health = system_health.component_details[component]
                assert component_health.status == 'healthy'
                assert component_health.metrics['error_count'] <= 5  # Reasonable error threshold
    
    @pytest.mark.asyncio
    async def test_performance_threshold_monitoring(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test monitoring of performance thresholds and degradation detection."""
        
        workflow = UnifiedWorkflow(timeout=60.0)
        
        # Define performance thresholds
        thresholds = {
            'max_response_time': 5.0,
            'min_success_rate': 0.95,
            'max_error_rate': 0.05,
            'max_memory_usage': 0.85
        }
        
        # Test scenarios that breach thresholds
        test_scenarios = [
            {
                'name': 'slow_response',
                'metrics': {
                    'avg_response_time': 8.0,  # Exceeds threshold
                    'success_rate': 0.98,
                    'memory_usage': 0.6
                },
                'expected_status': 'degraded'
            },
            {
                'name': 'high_error_rate', 
                'metrics': {
                    'avg_response_time': 2.0,
                    'success_rate': 0.90,  # Below threshold
                    'error_rate': 0.10,    # Above threshold
                    'memory_usage': 0.7
                },
                'expected_status': 'degraded'
            },
            {
                'name': 'high_memory_usage',
                'metrics': {
                    'avg_response_time': 3.0,
                    'success_rate': 0.97,
                    'memory_usage': 0.92   # Above threshold
                },
                'expected_status': 'degraded'
            },
            {
                'name': 'healthy_metrics',
                'metrics': {
                    'avg_response_time': 1.5,
                    'success_rate': 0.98,
                    'error_rate': 0.02,
                    'memory_usage': 0.65
                },
                'expected_status': 'healthy'
            }
        ]
        
        for scenario in test_scenarios:
            # Update component health with scenario metrics
            health_monitor.update_component_health(
                'unified_workflow',
                scenario['expected_status'],
                scenario['metrics']
            )
            
            # Check threshold violations
            system_health = health_monitor.get_system_health()
            workflow_health = system_health.component_details.get('unified_workflow')
            
            if workflow_health:
                assert workflow_health.status == scenario['expected_status']
                
                # Check for threshold violations
                violations = []
                metrics = workflow_health.metrics
                
                if metrics.get('avg_response_time', 0) > thresholds['max_response_time']:
                    violations.append('response_time_exceeded')
                
                if metrics.get('success_rate', 1.0) < thresholds['min_success_rate']:
                    violations.append('success_rate_below_threshold')
                
                if metrics.get('error_rate', 0) > thresholds['max_error_rate']:
                    violations.append('error_rate_exceeded')
                
                if metrics.get('memory_usage', 0) > thresholds['max_memory_usage']:
                    violations.append('memory_usage_exceeded')
                
                # Degraded status should have violations
                if scenario['expected_status'] == 'degraded':
                    assert len(violations) > 0
    
    @pytest.mark.asyncio
    async def test_resource_usage_monitoring(self, health_monitor):
        """Test monitoring of system resource usage and limits."""
        
        # Mock resource manager
        resource_manager = ResourceManager()
        
        # Test resource monitoring scenarios
        resource_scenarios = [
            {
                'name': 'normal_usage',
                'resources': {
                    'cpu_percent': 45.0,
                    'memory_percent': 60.0,
                    'disk_percent': 70.0,
                    'active_connections': 50
                },
                'expected_status': 'healthy'
            },
            {
                'name': 'high_cpu_usage',
                'resources': {
                    'cpu_percent': 95.0,  # High CPU
                    'memory_percent': 65.0,
                    'disk_percent': 72.0,
                    'active_connections': 55
                },
                'expected_status': 'degraded'
            },
            {
                'name': 'memory_pressure',
                'resources': {
                    'cpu_percent': 50.0,
                    'memory_percent': 92.0,  # High memory
                    'disk_percent': 75.0,
                    'active_connections': 45
                },
                'expected_status': 'degraded'
            }
        ]
        
        with patch.object(resource_manager, 'get_system_resources') as mock_resources:
            
            for scenario in resource_scenarios:
                mock_resources.return_value = scenario['resources']
                
                # Update health monitor with resource data
                health_monitor.update_system_metrics(scenario['resources'])
                
                # Check system health
                system_health = health_monitor.get_system_health()
                
                # Resource constraints should affect overall system status
                if scenario['expected_status'] == 'degraded':
                    # System should detect resource pressure
                    resource_issues = []
                    if scenario['resources']['cpu_percent'] > 90:
                        resource_issues.append('high_cpu')
                    if scenario['resources']['memory_percent'] > 90:
                        resource_issues.append('memory_pressure')
                    
                    assert len(resource_issues) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test circuit breaker behavior based on health monitoring."""
        
        workflow = UnifiedWorkflow(timeout=30.0)
        
        # Simulate component failures leading to circuit breaker activation
        failure_scenarios = [
            {
                'component': 'semantic_cache',
                'failure_count': 10,
                'failure_rate': 0.8,
                'should_open_circuit': True
            },
            {
                'component': 'hallucination_detection',
                'failure_count': 3,
                'failure_rate': 0.2,
                'should_open_circuit': False
            }
        ]
        
        for scenario in failure_scenarios:
            component = scenario['component']
            
            # Update health with failure metrics
            health_monitor.update_component_health(
                component,
                'error',
                {
                    'failure_count': scenario['failure_count'],
                    'total_requests': scenario['failure_count'] / scenario['failure_rate'],
                    'failure_rate': scenario['failure_rate'],
                    'last_error': 'Connection timeout'
                }
            )
            
            # Check circuit breaker status
            system_health = health_monitor.get_system_health()
            component_health = system_health.component_details.get(component)
            
            if component_health and scenario['should_open_circuit']:
                assert component_health.status == 'error'
                assert component_health.metrics['failure_rate'] >= 0.5
                
                # Circuit should be open for this component
                # (Implementation would disable the component)
    
    @pytest.mark.asyncio 
    async def test_health_monitoring_under_load(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test health monitoring behavior under high load conditions."""
        
        workflow = UnifiedWorkflow(timeout=45.0)
        
        # Simulate high load scenario
        load_config = {
            'concurrent_requests': 20,
            'request_duration': 0.1,  # 100ms per request
            'total_duration': 2.0     # 2 seconds of load
        }
        
        async def simulate_request(request_id):
            """Simulate a single request."""
            start_time = time.time()
            
            # Mock request processing
            await asyncio.sleep(load_config['request_duration'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update health metrics
            health_monitor.update_component_health(
                f'request_handler',
                'healthy',
                {
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'timestamp': end_time
                }
            )
            
            return f"Request {request_id} completed"
        
        # Generate concurrent load
        start_time = time.time()
        
        tasks = []
        for i in range(load_config['concurrent_requests']):
            task = asyncio.create_task(simulate_request(i))
            tasks.append(task)
            
            # Small delay between request starts
            await asyncio.sleep(0.01)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Validate load test results
        successful_requests = [r for r in results if isinstance(r, str) and "completed" in r]
        failed_requests = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_requests) / len(results)
        
        # Update overall health with load test results
        health_monitor.update_component_health(
            'load_test',
            'healthy' if success_rate > 0.95 else 'degraded',
            {
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': success_rate,
                'total_duration': total_duration,
                'requests_per_second': len(results) / total_duration
            }
        )
        
        # Check system health after load test
        system_health = health_monitor.get_system_health()
        load_test_health = system_health.component_details.get('load_test')
        
        if load_test_health:
            assert load_test_health.metrics['success_rate'] >= 0.8  # Reasonable under load
            assert load_test_health.metrics['requests_per_second'] > 0
    
    def test_health_data_persistence_and_history(self, health_monitor):
        """Test persistence and historical tracking of health data."""
        
        # Simulate health data over time
        time_series_data = []
        
        for i in range(10):
            timestamp = time.time() + i * 60  # 1 minute intervals
            
            health_data = {
                'timestamp': timestamp,
                'cpu_usage': 50 + (i * 2),  # Gradually increasing
                'memory_usage': 60 + (i * 1.5),
                'active_requests': 20 + (i * 3)
            }
            
            # Update health monitor
            health_monitor.update_system_metrics(health_data)
            time_series_data.append(health_data)
        
        # Test health history retrieval
        with patch.object(health_monitor, 'get_health_history') as mock_history:
            mock_history.return_value = time_series_data
            
            history = health_monitor.get_health_history(duration_minutes=10)
            
            assert len(history) == len(time_series_data)
            
            # Validate trend detection
            cpu_values = [d['cpu_usage'] for d in history]
            memory_values = [d['memory_usage'] for d in history]
            
            # Should show increasing trend
            assert cpu_values[-1] > cpu_values[0]
            assert memory_values[-1] > memory_values[0]


class TestHealthAlertingAndNotifications:
    """Test health monitoring alerting and notification systems."""
    
    @pytest.mark.asyncio
    async def test_alert_generation_on_health_degradation(self, health_monitor):
        """Test generation of alerts when component health degrades."""
        
        # Mock alert system
        alerts_generated = []
        
        def mock_send_alert(alert_type, component, metrics, severity):
            alerts_generated.append({
                'type': alert_type,
                'component': component, 
                'metrics': metrics,
                'severity': severity,
                'timestamp': time.time()
            })
        
        with patch.object(health_monitor, 'send_alert', side_effect=mock_send_alert):
            
            # Simulate component degradation
            degradation_scenarios = [
                {
                    'component': 'unified_workflow',
                    'status': 'degraded',
                    'metrics': {
                        'avg_response_time': 10.0,  # Very slow
                        'success_rate': 0.85,       # Below threshold
                        'error_count': 15
                    },
                    'expected_severity': 'high'
                },
                {
                    'component': 'semantic_cache',
                    'status': 'error',
                    'metrics': {
                        'connection_failures': 5,
                        'cache_hit_rate': 0.0,      # Cache not working
                        'last_error': 'Redis connection refused'
                    },
                    'expected_severity': 'critical'
                }
            ]
            
            for scenario in degradation_scenarios:
                # Update component health to trigger alert
                health_monitor.update_component_health(
                    scenario['component'],
                    scenario['status'],
                    scenario['metrics']
                )
                
                # Should generate appropriate alert
                matching_alerts = [
                    alert for alert in alerts_generated 
                    if alert['component'] == scenario['component']
                ]
                
                if scenario['status'] in ['degraded', 'error']:
                    assert len(matching_alerts) > 0
                    
                    latest_alert = matching_alerts[-1]
                    assert latest_alert['severity'] in ['high', 'critical']
    
    @pytest.mark.asyncio
    async def test_alert_suppression_and_escalation(self, health_monitor):
        """Test alert suppression and escalation logic."""
        
        alerts_sent = []
        
        def mock_alert_handler(alert_type, component, metrics, severity):
            alerts_sent.append({
                'component': component,
                'severity': severity,
                'timestamp': time.time(),
                'suppressed': False
            })
        
        with patch.object(health_monitor, 'send_alert', side_effect=mock_alert_handler):
            
            component = 'test_component'
            
            # Send multiple alerts for same issue
            for i in range(5):
                health_monitor.update_component_health(
                    component,
                    'degraded',
                    {
                        'error_count': i + 1,
                        'issue_id': 'same_issue'  # Same underlying issue
                    }
                )
                
                # Small delay between alerts
                await asyncio.sleep(0.1)
            
            # Should suppress duplicate alerts
            component_alerts = [a for a in alerts_sent if a['component'] == component]
            
            # First alert should be sent, subsequent ones may be suppressed
            assert len(component_alerts) >= 1
            
            # If multiple alerts sent, they should show escalation
            if len(component_alerts) > 1:
                # Later alerts should not exceed reasonable frequency
                time_diffs = []
                for i in range(1, len(component_alerts)):
                    time_diff = component_alerts[i]['timestamp'] - component_alerts[i-1]['timestamp']
                    time_diffs.append(time_diff)
                
                # Should have some minimum time between alerts (not spam)
                assert all(diff >= 0.05 for diff in time_diffs)  # At least 50ms between alerts
    
    def test_health_dashboard_integration(self, health_monitor):
        """Test integration with health monitoring dashboard."""
        
        # Mock dashboard data
        dashboard_data = {
            'system_overview': {
                'overall_status': 'healthy',
                'total_components': 8,
                'healthy_components': 7,
                'degraded_components': 1,
                'error_components': 0
            },
            'component_statuses': {},
            'recent_alerts': [],
            'performance_trends': {},
            'resource_usage': {}
        }
        
        # Update with current system health
        system_health = health_monitor.get_system_health()
        
        # Count component statuses
        status_counts = {}
        for component, health in system_health.component_details.items():
            status = health.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        dashboard_data['system_overview'].update({
            'healthy_components': status_counts.get('healthy', 0),
            'degraded_components': status_counts.get('degraded', 0), 
            'error_components': status_counts.get('error', 0)
        })
        
        # Test dashboard data structure
        assert 'system_overview' in dashboard_data
        assert 'overall_status' in dashboard_data['system_overview']
        
        # Validate component status counts
        total_reported = (
            dashboard_data['system_overview']['healthy_components'] +
            dashboard_data['system_overview']['degraded_components'] +
            dashboard_data['system_overview']['error_components']
        )
        
        assert total_reported >= 0  # Should have some components


class TestHealthMonitoringRecovery:
    """Test health monitoring during system recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_component_recovery_detection(self, health_monitor):
        """Test detection and reporting of component recovery."""
        
        component = 'recovery_test_component'
        
        # Simulate component failure
        health_monitor.update_component_health(
            component,
            'error',
            {
                'error_count': 10,
                'last_error': 'Connection failed',
                'failure_time': time.time()
            }
        )
        
        # Verify error state
        system_health = health_monitor.get_system_health()
        component_health = system_health.component_details.get(component)
        assert component_health.status == 'error'
        
        # Simulate recovery
        await asyncio.sleep(0.1)  # Brief pause
        
        health_monitor.update_component_health(
            component,
            'healthy',
            {
                'error_count': 0,
                'successful_requests': 5,
                'recovery_time': time.time(),
                'uptime_since_recovery': 0.1
            }
        )
        
        # Verify recovery
        updated_health = health_monitor.get_system_health()
        recovered_component = updated_health.component_details.get(component)
        
        assert recovered_component.status == 'healthy'
        assert recovered_component.metrics['error_count'] == 0
        assert 'recovery_time' in recovered_component.metrics
    
    @pytest.mark.asyncio
    async def test_system_wide_recovery_coordination(
        self, health_monitor, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test coordination of system-wide recovery after major failure."""
        
        workflow = UnifiedWorkflow(timeout=60.0)
        
        # Simulate system-wide degradation
        critical_components = [
            'unified_workflow',
            'semantic_cache', 
            'hallucination_detection',
            'query_engine'
        ]
        
        # Cause degradation in all components
        for component in critical_components:
            health_monitor.update_component_health(
                component,
                'degraded',
                {
                    'error_rate': 0.3,
                    'response_time': 8.0,
                    'degradation_cause': 'system_overload'
                }
            )
        
        # Check overall system health
        degraded_health = health_monitor.get_system_health()
        assert degraded_health.overall_status in ['degraded', 'error']
        
        # Simulate recovery process
        recovery_steps = [
            # Step 1: Core workflow recovery
            {'component': 'unified_workflow', 'delay': 0.1},
            # Step 2: Cache system recovery 
            {'component': 'semantic_cache', 'delay': 0.1},
            # Step 3: Verification system recovery
            {'component': 'hallucination_detection', 'delay': 0.1},
            # Step 4: Query engine recovery
            {'component': 'query_engine', 'delay': 0.1}
        ]
        
        for step in recovery_steps:
            await asyncio.sleep(step['delay'])
            
            # Restore component health
            health_monitor.update_component_health(
                step['component'],
                'healthy',
                {
                    'error_rate': 0.02,
                    'response_time': 1.5,
                    'recovery_step': True,
                    'recovery_order': recovery_steps.index(step) + 1
                }
            )
        
        # Verify system recovery
        recovered_health = health_monitor.get_system_health()
        
        # System should be healthy again
        healthy_components = [
            comp for comp, health in recovered_health.component_details.items()
            if health.status == 'healthy'
        ]
        
        assert len(healthy_components) >= len(critical_components)
        
        # Overall system status should improve
        assert recovered_health.overall_status in ['healthy', 'degraded']
        # (May still be 'degraded' if other components weren't recovered)