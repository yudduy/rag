"""System health monitoring and metrics collection.

Tracks component health, collects performance metrics, generates alerts,
and provides API endpoints for monitoring the RAG system.
"""

import asyncio
import logging
import time
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPException = None
    JSONResponse = None

from src.unified_config import get_unified_config, ComponentHealth, FeatureStatus
from src.cache import get_cache
from src.verification import create_hallucination_detector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """How urgent an alert is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """A system alert about a potential issue."""
    id: str
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    timestamp: float
    uptime: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time: float
    p95_response_time: float
    cache_hit_rate: float
    verification_success_rate: float
    error_rate: float
    cost_per_query: float
    daily_cost: float
    component_status: Dict[str, str]
    performance_profile: str


class HealthMonitor:
    """
    Comprehensive health monitoring system for SOTA RAG components.
    
    Provides:
    - Real-time health checks for all components
    - Performance metrics collection
    - Alert generation and management
    - System diagnostics
    - API endpoints for monitoring
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.config_manager = get_unified_config()
        self.config = self.config_manager.config
        
        # Initialize monitoring state
        self.alerts: List[HealthAlert] = []
        self.metrics_history: List[SystemMetrics] = []
        self.last_health_check = 0.0
        self.start_time = time.time()
        
        # Performance tracking
        self.query_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'response_times': [],
            'costs': [],
            'cache_hits': 0,
            'verification_successes': 0,
            'verification_attempts': 0
        }
        
        # Component instances for health checks
        self._component_instances = {}
        self._initialize_component_instances()
        
        # Start background monitoring if enabled
        if self.config.monitoring_enabled:
            self._start_background_monitoring()
        
        logger.info("HealthMonitor initialized")
    
    def _initialize_component_instances(self):
        """Initialize component instances for health checking."""
        try:
            # Initialize cache instance
            if self.config.semantic_cache.enabled:
                self._component_instances['semantic_cache'] = get_cache()
        except Exception as e:
            logger.warning(f"Failed to initialize cache instance for health checks: {e}")
        
        try:
            # Initialize verification instance
            if self.config.hallucination_detection.enabled:
                self._component_instances['hallucination_detection'] = create_hallucination_detector()
        except Exception as e:
            logger.warning(f"Failed to initialize verification instance for health checks: {e}")
        
        # Add other component instances as needed
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        if hasattr(asyncio, 'create_task'):
            try:
                # Run health checks periodically
                asyncio.create_task(self._periodic_health_checks())
                logger.info("Background health monitoring started")
            except Exception as e:
                logger.warning(f"Failed to start background monitoring: {e}")
    
    async def _periodic_health_checks(self):
        """Perform periodic health checks."""
        while self.config.monitoring_enabled:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Periodic health check failed: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all system components.
        
        Returns:
            Dict with health check results
        """
        check_time = time.time()
        health_results = {
            'timestamp': check_time,
            'overall_status': 'healthy',
            'components': {},
            'alerts_generated': 0
        }
        
        # Check each component
        components_to_check = [
            ('unified_config', self._check_config_health),
            ('semantic_cache', self._check_cache_health),
            ('hallucination_detection', self._check_verification_health),
            ('base_workflow', self._check_workflow_health),
            ('agentic_workflow', self._check_agentic_health),
            ('multimodal_support', self._check_multimodal_health),
            ('tts_integration', self._check_tts_health),
            ('performance_optimization', self._check_performance_health)
        ]
        
        alerts_generated = 0
        critical_issues = 0
        degraded_components = 0
        
        for component_name, check_function in components_to_check:
            try:
                component_health = await check_function()
                health_results['components'][component_name] = component_health
                
                # Update component health in config
                status = component_health['status']
                metrics = component_health.get('metrics', {})
                error_message = component_health.get('error_message')
                
                self.config_manager.update_component_health(
                    component_name, status, metrics, error_message
                )
                
                # Generate alerts if needed
                if status in ['error', 'critical']:
                    critical_issues += 1
                    alert = await self._generate_alert(
                        component_name, 
                        AlertSeverity.HIGH if status == 'error' else AlertSeverity.CRITICAL,
                        f"Component {component_name} is {status}",
                        component_health
                    )
                    if alert:
                        alerts_generated += 1
                elif status == 'degraded':
                    degraded_components += 1
                    alert = await self._generate_alert(
                        component_name,
                        AlertSeverity.MEDIUM,
                        f"Component {component_name} is degraded",
                        component_health
                    )
                    if alert:
                        alerts_generated += 1
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                health_results['components'][component_name] = {
                    'status': 'error',
                    'error_message': str(e),
                    'check_time': check_time
                }
                critical_issues += 1
        
        # Determine overall system status
        total_components = len(components_to_check)
        if critical_issues > 0:
            health_results['overall_status'] = 'critical' if critical_issues > total_components // 2 else 'degraded'
        elif degraded_components > 0:
            health_results['overall_status'] = 'degraded'
        else:
            health_results['overall_status'] = 'healthy'
        
        health_results['alerts_generated'] = alerts_generated
        self.last_health_check = check_time
        
        logger.info(f"Health check completed: {health_results['overall_status']} "
                   f"({alerts_generated} alerts generated)")
        
        return health_results
    
    async def _check_config_health(self) -> Dict[str, Any]:
        """Check unified configuration health."""
        try:
            system_health = self.config_manager.get_system_health()
            
            return {
                'status': system_health['overall_status'],
                'check_time': time.time(),
                'metrics': {
                    'healthy_components': system_health['healthy_components'],
                    'total_components': system_health['total_components'],
                    'features_enabled': sum(system_health['features_enabled'].values()),
                    'performance_profile': system_health['performance_profile']
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check semantic cache health."""
        if not self.config.semantic_cache.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            cache_instance = self._component_instances.get('semantic_cache')
            if not cache_instance:
                return {'status': 'error', 'error_message': 'Cache instance not available', 'check_time': time.time()}
            
            health_check = cache_instance.health_check()
            cache_stats = cache_instance.get_stats()
            
            status = 'healthy'
            if not health_check.get('redis_available', False):
                status = 'degraded'  # Using fallback cache
            elif health_check.get('redis_error'):
                status = 'error'
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': {
                    'hit_rate': cache_instance.get_hit_rate(),
                    'cache_size': cache_stats.cache_size,
                    'total_queries': cache_stats.total_queries,
                    'redis_available': health_check.get('redis_available', False)
                },
                'details': health_check
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_verification_health(self) -> Dict[str, Any]:
        """Check hallucination detection health."""
        if not self.config.hallucination_detection.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            verification_instance = self._component_instances.get('hallucination_detection')
            if not verification_instance:
                return {'status': 'error', 'error_message': 'Verification instance not available', 'check_time': time.time()}
            
            # Test verification functionality
            test_start = time.time()
            
            # Simple test of verification components
            status = 'healthy'
            metrics = {}
            
            # Check if verification model is accessible
            if hasattr(verification_instance, 'verification_llm'):
                try:
                    # Simple test call
                    test_response = await verification_instance.verification_llm.acomplete("Test")
                    response_time = time.time() - test_start
                    metrics['response_time'] = response_time
                    
                    if response_time > 5.0:
                        status = 'degraded'  # Slow response
                except Exception as model_error:
                    status = 'error'
                    return {
                        'status': status,
                        'error_message': f"Verification model error: {str(model_error)}",
                        'check_time': time.time()
                    }
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': metrics
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_workflow_health(self) -> Dict[str, Any]:
        """Check base workflow health."""
        try:
            # Basic workflow health - check if it can be imported
            from src.workflow import create_legacy_workflow
            
            # Test creation (but don't actually create to avoid overhead)
            status = 'healthy'
            
            # Check if index is available
            from src.index import get_index
            index = get_index()
            if index is None:
                status = 'error'
                return {
                    'status': status,
                    'error_message': 'Index not available - run `uv run generate`',
                    'check_time': time.time()
                }
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': {
                    'index_available': True
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_agentic_health(self) -> Dict[str, Any]:
        """Check agentic workflow health."""
        if not self.config.agentic_workflow.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            # Check if agentic components are importable
            from src.agentic_workflow import AgenticWorkflow
            from src.agentic import QueryClassifier, QueryDecomposer
            
            status = 'healthy'
            metrics = {}
            
            # Basic component availability checks
            try:
                classifier = QueryClassifier()
                decomposer = QueryDecomposer()
                metrics['components_available'] = True
            except Exception as component_error:
                status = 'degraded'
                metrics['component_error'] = str(component_error)
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': metrics
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_multimodal_health(self) -> Dict[str, Any]:
        """Check multimodal support health."""
        if not self.config.multimodal_support.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            # Check CLIP availability
            import clip
            status = 'healthy'
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': {
                    'clip_available': True
                }
            }
        except ImportError:
            return {
                'status': 'error',
                'error_message': 'CLIP not available',
                'check_time': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_tts_health(self) -> Dict[str, Any]:
        """Check TTS integration health."""
        if not self.config.tts_integration.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            # Check TTS engine availability
            engine = self.config.tts_integration.settings.get('engine', 'pyttsx3')
            
            if engine == 'pyttsx3':
                import pyttsx3
                status = 'healthy'
            elif engine == 'gtts':
                from gtts import gTTS
                status = 'healthy'
            else:
                status = 'error'
                return {
                    'status': status,
                    'error_message': f'Unknown TTS engine: {engine}',
                    'check_time': time.time()
                }
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': {
                    'engine': engine,
                    'engine_available': True
                }
            }
        except ImportError as e:
            return {
                'status': 'error',
                'error_message': f'TTS engine not available: {str(e)}',
                'check_time': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """Check performance optimization health."""
        if not self.config.performance_optimization.enabled:
            return {'status': 'disabled', 'check_time': time.time()}
        
        try:
            # Basic performance checks
            status = 'healthy'
            metrics = {
                'optimization_enabled': True
            }
            
            # Check if performance module is available
            from src.performance import get_performance_optimizer
            optimizer = get_performance_optimizer()
            metrics['optimizer_available'] = True
            
            return {
                'status': status,
                'check_time': time.time(),
                'metrics': metrics
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'check_time': time.time()
            }
    
    async def _generate_alert(self, component: str, severity: AlertSeverity, 
                            message: str, details: Dict[str, Any]) -> Optional[HealthAlert]:
        """Generate and store a health alert."""
        if not self.config.error_alerting_enabled:
            return None
        
        alert_id = f"{component}_{int(time.time())}"
        alert = HealthAlert(
            id=alert_id,
            timestamp=time.time(),
            severity=severity,
            component=component,
            message=message,
            details=details
        )
        
        # Check if similar alert already exists
        existing_alert = self._find_similar_alert(alert)
        if existing_alert:
            logger.debug(f"Similar alert already exists for {component}")
            return None
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = time.time() - 86400
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        logger.warning(f"Generated {severity.value} alert for {component}: {message}")
        return alert
    
    def _find_similar_alert(self, new_alert: HealthAlert) -> Optional[HealthAlert]:
        """Find similar existing alert."""
        for alert in self.alerts:
            if (alert.component == new_alert.component and
                alert.severity == new_alert.severity and
                not alert.resolved and
                time.time() - alert.timestamp < 3600):  # Within last hour
                return alert
        return None
    
    def record_query_metrics(self, success: bool, response_time: float, cost: float = 0.0,
                            cache_hit: bool = False, verification_success: bool = False):
        """Record metrics for a processed query."""
        self.query_metrics['total_queries'] += 1
        
        if success:
            self.query_metrics['successful_queries'] += 1
        else:
            self.query_metrics['failed_queries'] += 1
        
        self.query_metrics['response_times'].append(response_time)
        self.query_metrics['costs'].append(cost)
        
        if cache_hit:
            self.query_metrics['cache_hits'] += 1
        
        if verification_success:
            self.query_metrics['verification_successes'] += 1
        
        # Keep only recent metrics (last 1000 queries)
        if len(self.query_metrics['response_times']) > 1000:
            self.query_metrics['response_times'] = self.query_metrics['response_times'][-1000:]
            self.query_metrics['costs'] = self.query_metrics['costs'][-1000:]
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate response time metrics
        response_times = self.query_metrics['response_times']
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        p95_response_time = 0.0
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        # Calculate rates
        total_queries = self.query_metrics['total_queries']
        cache_hit_rate = (self.query_metrics['cache_hits'] / total_queries) if total_queries > 0 else 0.0
        error_rate = (self.query_metrics['failed_queries'] / total_queries) if total_queries > 0 else 0.0
        
        verification_attempts = max(self.query_metrics['verification_successes'], 1)  # Avoid division by zero
        verification_success_rate = (self.query_metrics['verification_successes'] / verification_attempts)
        
        # Calculate cost metrics
        costs = self.query_metrics['costs']
        cost_per_query = sum(costs) / len(costs) if costs else 0.0
        
        # Daily cost (rough estimate based on current session)
        hours_running = uptime / 3600
        daily_cost = (sum(costs) / max(hours_running, 1)) * 24 if costs else 0.0
        
        # Get component status
        system_health = self.config_manager.get_system_health()
        component_status = {
            name: health.get('status', 'unknown')
            for name, health in system_health['component_details'].items()
        }
        
        return SystemMetrics(
            timestamp=current_time,
            uptime=uptime,
            total_queries=total_queries,
            successful_queries=self.query_metrics['successful_queries'],
            failed_queries=self.query_metrics['failed_queries'],
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            cache_hit_rate=cache_hit_rate,
            verification_success_rate=verification_success_rate,
            error_rate=error_rate,
            cost_per_query=cost_per_query,
            daily_cost=daily_cost,
            component_status=component_status,
            performance_profile=self.config.performance_profile.value
        )
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None, 
                   resolved: Optional[bool] = None) -> List[HealthAlert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        metrics = self.get_system_metrics()
        
        if format.lower() == "json":
            return json.dumps(asdict(metrics), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


# FastAPI endpoints (if FastAPI is available)
def create_monitoring_api() -> Optional[FastAPI]:
    """Create FastAPI application with monitoring endpoints."""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, monitoring API disabled")
        return None
    
    app = FastAPI(title="SOTA RAG Health Monitor", version="1.0.0")
    health_monitor = get_health_monitor()
    
    @app.get("/health")
    async def health_check():
        """Get system health status."""
        try:
            health_results = await health_monitor.perform_health_check()
            return JSONResponse(content=health_results)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Health check failed: {str(e)}"}
            )
    
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        try:
            metrics = health_monitor.get_system_metrics()
            return JSONResponse(content=asdict(metrics))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to get metrics: {str(e)}"}
            )
    
    @app.get("/alerts")
    async def get_alerts(severity: Optional[str] = None, resolved: Optional[bool] = None):
        """Get system alerts."""
        try:
            alert_severity = None
            if severity:
                alert_severity = AlertSeverity(severity.lower())
            
            alerts = health_monitor.get_alerts(alert_severity, resolved)
            return JSONResponse(content=[asdict(alert) for alert in alerts])
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to get alerts: {str(e)}"}
            )
    
    @app.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str):
        """Acknowledge an alert."""
        if health_monitor.acknowledge_alert(alert_id):
            return JSONResponse(content={"message": f"Alert {alert_id} acknowledged"})
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Alert {alert_id} not found"}
            )
    
    @app.post("/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str):
        """Resolve an alert."""
        if health_monitor.resolve_alert(alert_id):
            return JSONResponse(content={"message": f"Alert {alert_id} resolved"})
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Alert {alert_id} not found"}
            )
    
    @app.get("/config")
    async def get_config():
        """Get current system configuration."""
        try:
            config = health_monitor.config_manager.export_config()
            return JSONResponse(content=config)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to get config: {str(e)}"}
            )
    
    @app.get("/stats")
    async def get_stats():
        """Get processing statistics."""
        try:
            # This would come from the unified workflow if available
            stats = {
                "query_metrics": health_monitor.query_metrics,
                "uptime": time.time() - health_monitor.start_time,
                "last_health_check": health_monitor.last_health_check,
                "active_alerts": len([a for a in health_monitor.alerts if not a.resolved])
            }
            return JSONResponse(content=stats)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to get stats: {str(e)}"}
            )
    
    logger.info("Monitoring API created with endpoints: /health, /metrics, /alerts, /config, /stats")
    return app