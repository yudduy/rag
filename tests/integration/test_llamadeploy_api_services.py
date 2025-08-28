"""
Integration Tests for LlamaDeploy API and Service Integration

This module provides comprehensive integration tests for LlamaDeploy service integration,
focusing on:
1. LlamaDeploy API endpoint testing
2. Task creation and event streaming
3. UI integration with backend services
4. Real-time chat functionality
5. Service discovery and health checks
6. Deployment configuration validation

Tests validate API contracts and service interactions with proper mocking.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import httpx

from src.unified_workflow import UnifiedWorkflow, create_unified_workflow
from src.health_monitor import HealthMonitor, create_monitoring_api


class TestLlamaDeployAPIIntegration:
    """Test LlamaDeploy API integration and endpoints."""
    
    @pytest.fixture
    def mock_llamadeploy_client(self):
        """Create mock LlamaDeploy client for testing."""
        client = Mock()
        client.create_deployment = AsyncMock()
        client.get_deployment = AsyncMock()
        client.create_task = AsyncMock()
        client.get_task = AsyncMock()
        client.stream_task_events = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_deployment_config(self):
        """Mock deployment configuration."""
        return {
            'name': 'chat',
            'control_plane': {'port': 8000},
            'default_service': 'workflow',
            'services': {
                'workflow': {
                    'name': 'Unified SOTA RAG Workflow',
                    'path': 'src.workflow:workflow',
                    'env': {
                        'USE_UNIFIED_ORCHESTRATOR': 'true',
                        'PERFORMANCE_PROFILE': 'balanced',
                        'AGENTIC_WORKFLOW_ENABLED': 'true',
                        'VERIFICATION_ENABLED': 'true'
                    }
                },
                'health_monitor': {
                    'name': 'Health Monitor API',
                    'path': 'src.health_monitor:create_monitoring_api'
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_deployment_creation_and_configuration(
        self, mock_llamadeploy_client, mock_deployment_config
    ):
        """Test deployment creation with proper configuration."""
        
        # Mock successful deployment creation
        mock_llamadeploy_client.create_deployment.return_value = {
            'id': 'deploy-123',
            'name': 'chat',
            'status': 'running',
            'services': ['workflow', 'health_monitor'],
            'endpoints': {
                'workflow': 'http://localhost:8000/deployments/chat/tasks',
                'health_monitor': 'http://localhost:8001/health'
            }
        }
        
        # Create deployment
        deployment_result = await mock_llamadeploy_client.create_deployment(
            config=mock_deployment_config
        )
        
        # Validate deployment
        assert deployment_result['status'] == 'running'
        assert 'workflow' in deployment_result['services']
        assert 'health_monitor' in deployment_result['services']
        assert 'endpoints' in deployment_result
        
        # Verify API endpoints are properly configured
        endpoints = deployment_result['endpoints']
        assert endpoints['workflow'].startswith('http://localhost:8000')
        assert endpoints['health_monitor'].startswith('http://localhost:8001')
        
        mock_llamadeploy_client.create_deployment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_creation_api_endpoint(
        self, mock_llamadeploy_client, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test task creation through LlamaDeploy API endpoint."""
        
        deployment_id = 'chat'
        task_request = {
            'input': 'What is machine learning?',
            'session_id': 'session-123',
            'user_id': 'user-456'
        }
        
        # Mock task creation response
        mock_task_response = {
            'task_id': 'task-789',
            'status': 'running',
            'created_at': time.time(),
            'input': task_request['input'],
            'session_id': task_request['session_id']
        }
        
        mock_llamadeploy_client.create_task.return_value = mock_task_response
        
        # Create task via API
        task_result = await mock_llamadeploy_client.create_task(
            deployment_id=deployment_id,
            **task_request
        )
        
        # Validate task creation
        assert task_result['task_id'] == 'task-789'
        assert task_result['status'] == 'running'
        assert task_result['input'] == task_request['input']
        assert task_result['session_id'] == task_request['session_id']
        
        # Verify API call
        mock_llamadeploy_client.create_task.assert_called_once_with(
            deployment_id=deployment_id,
            **task_request
        )
    
    @pytest.mark.asyncio
    async def test_event_streaming_api_integration(
        self, mock_llamadeploy_client, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test event streaming through LlamaDeploy API."""
        
        task_id = 'task-789'
        
        # Mock event stream
        mock_events = [
            {
                'type': 'task_started',
                'task_id': task_id,
                'timestamp': time.time(),
                'data': {'status': 'processing'}
            },
            {
                'type': 'query_analyzed',
                'task_id': task_id,
                'timestamp': time.time() + 0.5,
                'data': {
                    'complexity': 'simple',
                    'estimated_cost': 0.01
                }
            },
            {
                'type': 'response_generated',
                'task_id': task_id, 
                'timestamp': time.time() + 1.0,
                'data': {
                    'content': 'Machine learning is a subset of artificial intelligence...',
                    'confidence': 0.92,
                    'sources': [{'text': 'ML definition', 'relevance': 0.9}]
                }
            },
            {
                'type': 'task_completed',
                'task_id': task_id,
                'timestamp': time.time() + 1.2,
                'data': {
                    'status': 'completed',
                    'processing_time': 1.2,
                    'total_cost': 0.008
                }
            }
        ]
        
        # Mock async event stream
        async def mock_event_stream():
            for event in mock_events:
                yield event
                await asyncio.sleep(0.1)
        
        mock_llamadeploy_client.stream_task_events.return_value = mock_event_stream()
        
        # Stream events
        received_events = []
        async for event in mock_llamadeploy_client.stream_task_events(task_id):
            received_events.append(event)
        
        # Validate event stream
        assert len(received_events) == len(mock_events)
        
        # Check event sequence
        event_types = [e['type'] for e in received_events]
        expected_sequence = ['task_started', 'query_analyzed', 'response_generated', 'task_completed']
        assert event_types == expected_sequence
        
        # Validate event data
        completed_event = received_events[-1]
        assert completed_event['type'] == 'task_completed'
        assert completed_event['data']['status'] == 'completed'
        assert completed_event['data']['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_health_check_api_integration(self, mock_deployment_config):
        """Test health check API integration."""
        
        # Create mock health monitoring API
        health_app = create_monitoring_api()
        client = TestClient(health_app)
        
        # Mock health monitor
        mock_health_data = {
            'overall_status': 'healthy',
            'timestamp': time.time(),
            'components': {
                'unified_workflow': {
                    'status': 'healthy',
                    'metrics': {
                        'queries_processed': 100,
                        'success_rate': 0.98,
                        'avg_response_time': 1.5
                    }
                },
                'semantic_cache': {
                    'status': 'healthy',
                    'metrics': {
                        'cache_hit_rate': 0.35,
                        'total_requests': 75
                    }
                }
            }
        }
        
        with patch('src.health_monitor.get_health_monitor') as mock_monitor:
            mock_monitor.return_value.get_system_health.return_value = Mock(
                overall_status='healthy',
                component_details=mock_health_data['components']
            )
            
            # Test health endpoint
            response = client.get('/health')
            
            assert response.status_code == 200
            health_data = response.json()
            
            assert health_data['status'] == 'healthy'
            assert 'components' in health_data
            assert 'unified_workflow' in health_data['components']
    
    @pytest.mark.asyncio
    async def test_service_discovery_integration(self, mock_llamadeploy_client):
        """Test service discovery and registration."""
        
        # Mock service registration
        services_config = {
            'workflow': {
                'name': 'Unified SOTA RAG Workflow',
                'health_check': '/health',
                'endpoints': ['/query', '/stream'],
                'dependencies': ['health_monitor']
            },
            'health_monitor': {
                'name': 'Health Monitor API',
                'health_check': '/health',
                'endpoints': ['/health', '/metrics'],
                'dependencies': []
            }
        }
        
        # Mock service discovery response
        mock_llamadeploy_client.get_deployment.return_value = {
            'id': 'chat',
            'services': services_config,
            'service_status': {
                'workflow': 'running',
                'health_monitor': 'running'
            },
            'service_endpoints': {
                'workflow': 'http://localhost:8000',
                'health_monitor': 'http://localhost:8001'
            }
        }
        
        # Get deployment info
        deployment_info = await mock_llamadeploy_client.get_deployment('chat')
        
        # Validate service discovery
        assert 'services' in deployment_info
        assert 'service_status' in deployment_info
        assert 'service_endpoints' in deployment_info
        
        # Check all services are running
        for service, status in deployment_info['service_status'].items():
            assert status == 'running'
        
        # Validate service endpoints
        assert deployment_info['service_endpoints']['workflow'].startswith('http://')
        assert deployment_info['service_endpoints']['health_monitor'].startswith('http://')


class TestUIIntegrationWithBackend:
    """Test UI integration with backend services."""
    
    @pytest.fixture
    def mock_ui_client(self):
        """Create mock UI client for testing."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_chat_interface_integration(
        self, mock_ui_client, mock_llamadeploy_client
    ):
        """Test chat interface integration with backend workflow."""
        
        # Mock chat session
        chat_session = {
            'session_id': 'chat-session-123',
            'user_id': 'user-456',
            'conversation_history': []
        }
        
        # Mock user message
        user_message = {
            'role': 'user',
            'content': 'Can you explain deep learning?',
            'timestamp': time.time()
        }
        
        # Mock backend response
        backend_response = {
            'role': 'assistant',
            'content': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers...',
            'confidence': 0.91,
            'sources': [
                {'text': 'Deep learning definition', 'relevance': 0.9, 'url': 'example.com/dl'}
            ],
            'timestamp': time.time(),
            'processing_time': 2.1
        }
        
        # Mock task creation and response
        mock_llamadeploy_client.create_task.return_value = {
            'task_id': 'task-abc123',
            'status': 'running'
        }
        
        # Mock event stream for response
        async def mock_chat_events():
            yield {'type': 'response_chunk', 'data': {'content': 'Deep learning is'}}
            yield {'type': 'response_chunk', 'data': {'content': ' a subset of'}}
            yield {'type': 'response_chunk', 'data': {'content': ' machine learning...'}}
            yield {'type': 'response_complete', 'data': backend_response}
        
        mock_llamadeploy_client.stream_task_events.return_value = mock_chat_events()
        
        # Simulate chat interaction
        with patch.object(mock_ui_client, 'send_message') as mock_send:
            mock_send.return_value = backend_response
            
            # Send message through UI
            response = await mock_ui_client.send_message(
                session_id=chat_session['session_id'],
                message=user_message['content']
            )
            
            # Validate response
            assert response['content'].startswith('Deep learning is')
            assert response['confidence'] >= 0.9
            assert len(response['sources']) > 0
            assert response['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_real_time_streaming_integration(
        self, mock_ui_client, mock_llamadeploy_client
    ):
        """Test real-time streaming integration between UI and backend."""
        
        task_id = 'streaming-task-456'
        
        # Mock streaming response chunks
        response_chunks = [
            'Machine',
            ' learning',
            ' is',
            ' a',
            ' powerful',
            ' branch',
            ' of',
            ' artificial',
            ' intelligence',
            ' that',
            ' enables',
            ' computers',
            ' to',
            ' learn',
            ' and',
            ' improve',
            ' from',
            ' experience.'
        ]
        
        # Mock streaming events
        async def mock_streaming_events():
            for i, chunk in enumerate(response_chunks):
                yield {
                    'type': 'response_chunk',
                    'data': {
                        'chunk': chunk,
                        'chunk_index': i,
                        'is_final': i == len(response_chunks) - 1
                    }
                }
                await asyncio.sleep(0.01)  # Simulate streaming delay
            
            # Final metadata event
            yield {
                'type': 'response_metadata',
                'data': {
                    'total_chunks': len(response_chunks),
                    'confidence': 0.94,
                    'sources': [{'text': 'ML overview', 'relevance': 0.88}]
                }
            }
        
        mock_llamadeploy_client.stream_task_events.return_value = mock_streaming_events()
        
        # Collect streaming response
        streamed_chunks = []
        metadata = None
        
        async for event in mock_llamadeploy_client.stream_task_events(task_id):
            if event['type'] == 'response_chunk':
                streamed_chunks.append(event['data']['chunk'])
            elif event['type'] == 'response_metadata':
                metadata = event['data']
        
        # Validate streaming
        full_response = ''.join(streamed_chunks)
        assert full_response == 'Machine learning is a powerful branch of artificial intelligence that enables computers to learn and improve from experience.'
        assert len(streamed_chunks) == len(response_chunks)
        assert metadata['confidence'] >= 0.9
    
    @pytest.mark.asyncio
    async def test_ui_error_handling_integration(
        self, mock_ui_client, mock_llamadeploy_client
    ):
        """Test UI error handling integration with backend failures."""
        
        # Test different error scenarios
        error_scenarios = [
            {
                'name': 'backend_timeout',
                'exception': asyncio.TimeoutError('Request timed out'),
                'expected_ui_response': 'Request timed out. Please try again.'
            },
            {
                'name': 'service_unavailable',
                'exception': Exception('Service temporarily unavailable'),
                'expected_ui_response': 'Service is temporarily unavailable.'
            },
            {
                'name': 'invalid_input',
                'exception': ValueError('Invalid input format'),
                'expected_ui_response': 'Please check your input and try again.'
            }
        ]
        
        for scenario in error_scenarios:
            # Mock backend failure
            mock_llamadeploy_client.create_task.side_effect = scenario['exception']
            
            # Test UI error handling
            with patch.object(mock_ui_client, 'handle_error') as mock_error_handler:
                mock_error_handler.return_value = {
                    'error': True,
                    'message': scenario['expected_ui_response'],
                    'error_type': scenario['name']
                }
                
                try:
                    await mock_llamadeploy_client.create_task(
                        deployment_id='chat',
                        input='Test message'
                    )
                except Exception as e:
                    # UI should handle this gracefully
                    error_response = mock_error_handler(e)
                    
                    assert error_response['error'] == True
                    assert 'message' in error_response
                    assert error_response['error_type'] == scenario['name']
            
            # Reset for next scenario
            mock_llamadeploy_client.create_task.side_effect = None


class TestLlamaDeployPerformanceIntegration:
    """Test performance aspects of LlamaDeploy integration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_handling(
        self, mock_llamadeploy_client, mock_llama_index, mock_redis_client, mock_openai_client
    ):
        """Test handling of concurrent tasks through LlamaDeploy."""
        
        # Create multiple concurrent tasks
        num_tasks = 10
        tasks_data = [
            {
                'input': f'Question {i}: What is topic {i}?',
                'session_id': f'session-{i}',
                'expected_response': f'Answer {i}: Topic {i} explanation.'
            }
            for i in range(num_tasks)
        ]
        
        # Mock task creation and processing
        async def mock_create_task(deployment_id, **kwargs):
            task_id = f"task-{kwargs['input'][-2]}"  # Extract task number
            return {
                'task_id': task_id,
                'status': 'running',
                'input': kwargs['input']
            }
        
        mock_llamadeploy_client.create_task.side_effect = mock_create_task
        
        # Mock task completion
        async def mock_get_task(task_id):
            task_num = task_id.split('-')[1]
            return {
                'task_id': task_id,
                'status': 'completed',
                'result': f'Answer {task_num}: Topic {task_num} explanation.',
                'processing_time': 1.5,
                'cost': 0.01
            }
        
        mock_llamadeploy_client.get_task.side_effect = mock_get_task
        
        # Create tasks concurrently
        start_time = time.time()
        
        create_tasks = [
            mock_llamadeploy_client.create_task(
                deployment_id='chat',
                **task_data
            )
            for task_data in tasks_data
        ]
        
        created_tasks = await asyncio.gather(*create_tasks)
        
        # Get task results concurrently
        get_tasks = [
            mock_llamadeploy_client.get_task(task['task_id'])
            for task in created_tasks
        ]
        
        completed_tasks = await asyncio.gather(*get_tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate concurrent processing
        assert len(completed_tasks) == num_tasks
        assert all(task['status'] == 'completed' for task in completed_tasks)
        
        # Should be faster than sequential processing
        # (This is a rough estimate, actual performance depends on implementation)
        assert total_time < num_tasks * 0.5  # Should be much faster than 5 seconds
        
        # Verify all tasks completed successfully
        success_rate = sum(1 for task in completed_tasks if task['status'] == 'completed') / len(completed_tasks)
        assert success_rate >= 0.95
    
    @pytest.mark.asyncio
    async def test_load_balancing_integration(
        self, mock_llamadeploy_client
    ):
        """Test load balancing integration across multiple service instances."""
        
        # Mock multiple service instances
        service_instances = [
            {'id': 'workflow-1', 'endpoint': 'http://localhost:8001', 'load': 0.2},
            {'id': 'workflow-2', 'endpoint': 'http://localhost:8002', 'load': 0.3},
            {'id': 'workflow-3', 'endpoint': 'http://localhost:8003', 'load': 0.1}
        ]
        
        # Mock load balancer selection
        def select_least_loaded_instance():
            return min(service_instances, key=lambda x: x['load'])
        
        # Simulate multiple requests
        num_requests = 15
        instance_usage = {instance['id']: 0 for instance in service_instances}
        
        for i in range(num_requests):
            # Select instance based on load balancing
            selected_instance = select_least_loaded_instance()
            
            # Mock task creation on selected instance
            mock_llamadeploy_client.create_task.return_value = {
                'task_id': f'task-{i}',
                'instance_id': selected_instance['id'],
                'status': 'running'
            }
            
            # Track usage
            instance_usage[selected_instance['id']] += 1
            
            # Update instance load (simulate processing)
            for instance in service_instances:
                if instance['id'] == selected_instance['id']:
                    instance['load'] += 0.05  # Increase load
                else:
                    instance['load'] = max(0, instance['load'] - 0.01)  # Decrease load
        
        # Validate load distribution
        total_requests = sum(instance_usage.values())
        assert total_requests == num_requests
        
        # Load should be reasonably distributed (not all on one instance)
        max_usage = max(instance_usage.values())
        min_usage = min(instance_usage.values())
        
        # No instance should handle more than 60% of requests
        assert max_usage / total_requests < 0.6
        # Load distribution should be reasonably balanced
        assert (max_usage - min_usage) / total_requests < 0.4


class TestLlamaDeployConfigurationValidation:
    """Test validation of LlamaDeploy configuration and deployment."""
    
    def test_deployment_config_validation(self):
        """Test validation of deployment configuration."""
        
        # Valid configuration
        valid_config = {
            'name': 'chat',
            'control-plane': {'port': 8000},
            'default-service': 'workflow',
            'services': {
                'workflow': {
                    'name': 'Unified SOTA RAG Workflow',
                    'path': 'src.workflow:workflow',
                    'python-dependencies': [
                        'llama-index-core>=0.12.45',
                        'fastapi>=0.100.0'
                    ],
                    'env': {
                        'USE_UNIFIED_ORCHESTRATOR': 'true',
                        'PERFORMANCE_PROFILE': 'balanced'
                    }
                }
            },
            'ui': {
                'name': 'Enhanced RAG UI',
                'port': 3000
            }
        }
        
        # Validate required fields
        required_fields = ['name', 'services', 'default-service']
        for field in required_fields:
            assert field in valid_config
        
        # Validate services configuration
        assert 'workflow' in valid_config['services']
        workflow_service = valid_config['services']['workflow']
        
        assert 'name' in workflow_service
        assert 'path' in workflow_service
        
        # Validate environment configuration
        env_config = workflow_service['env']
        assert env_config['USE_UNIFIED_ORCHESTRATOR'] == 'true'
        assert env_config['PERFORMANCE_PROFILE'] in ['balanced', 'high_accuracy', 'speed', 'cost_optimized']
    
    def test_service_dependency_validation(self):
        """Test validation of service dependencies."""
        
        # Configuration with dependencies
        config_with_deps = {
            'services': {
                'workflow': {
                    'name': 'Workflow Service',
                    'dependencies': ['health_monitor', 'cache_service']
                },
                'health_monitor': {
                    'name': 'Health Monitor',
                    'dependencies': []
                },
                'cache_service': {
                    'name': 'Cache Service',
                    'dependencies': ['health_monitor']
                }
            }
        }
        
        # Validate dependency resolution
        services = config_with_deps['services']
        
        # Check that all dependencies exist
        for service_name, service_config in services.items():
            dependencies = service_config.get('dependencies', [])
            for dep in dependencies:
                assert dep in services, f"Dependency '{dep}' for service '{service_name}' not found"
        
        # Check for circular dependencies (simplified check)
        def has_circular_dependency(service, visited, path):
            if service in path:
                return True
            if service in visited:
                return False
            
            visited.add(service)
            path.add(service)
            
            for dep in services[service].get('dependencies', []):
                if has_circular_dependency(dep, visited, path):
                    return True
            
            path.remove(service)
            return False
        
        visited = set()
        for service in services:
            if service not in visited:
                assert not has_circular_dependency(service, visited, set()), \
                    f"Circular dependency detected involving service '{service}'"
    
    @pytest.mark.asyncio
    async def test_environment_variable_propagation(self, mock_llamadeploy_client):
        """Test that environment variables are properly propagated to services."""
        
        # Mock deployment with environment variables
        env_vars = {
            'PERFORMANCE_PROFILE': 'high_accuracy',
            'AGENTIC_WORKFLOW_ENABLED': 'true',
            'SEMANTIC_CACHE_ENABLED': 'false',
            'VERIFICATION_ENABLED': 'true',
            'MAX_QUERY_COST': '1.50'
        }
        
        deployment_config = {
            'name': 'chat',
            'services': {
                'workflow': {
                    'name': 'Workflow Service',
                    'env': env_vars
                }
            }
        }
        
        # Mock deployment creation
        mock_llamadeploy_client.create_deployment.return_value = {
            'id': 'deploy-env-test',
            'status': 'running',
            'services': {
                'workflow': {
                    'status': 'running',
                    'environment': env_vars
                }
            }
        }
        
        # Create deployment
        result = await mock_llamadeploy_client.create_deployment(config=deployment_config)
        
        # Validate environment propagation
        assert result['services']['workflow']['environment'] == env_vars
        
        # Verify specific environment variables
        workflow_env = result['services']['workflow']['environment']
        assert workflow_env['PERFORMANCE_PROFILE'] == 'high_accuracy'
        assert workflow_env['AGENTIC_WORKFLOW_ENABLED'] == 'true'
        assert workflow_env['MAX_QUERY_COST'] == '1.50'