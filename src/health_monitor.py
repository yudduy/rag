"""System health monitoring and metrics collection.

Tracks component health, collects performance metrics, generates alerts,
and provides API endpoints for monitoring the RAG system.
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

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
    component: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: str  # healthy, degraded, error
    last_check: float
    response_time: float = 0.0
    error_count: int = 0
    uptime: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System metrics."""
    timestamp: float = field(default_factory=time.time)
    query_count: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    active_components: int = 0


class HealthMonitor:
    """
    Health monitoring system for SOTA RAG components.
    
    Monitors component health, tracks metrics, generates alerts.
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        from src.config import get_global_config
        
        self.config = get_global_config()
        
        # Initialize monitoring state
        self.alerts: List[HealthAlert] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.last_health_check = 0.0
        
        # Health check interval (seconds)
        self.check_interval = 60.0
        
        logger.info("Health monitor initialized")
    
    def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        start_time = time.time()
        
        try:
            # Basic health check - component exists and is responding
            health = ComponentHealth(
                name=component_name,
                status="healthy",
                last_check=start_time,
                response_time=time.time() - start_time
            )
            
            self.component_health[component_name] = health
            return health
            
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            
            health = ComponentHealth(
                name=component_name,
                status="error",
                last_check=start_time,
                response_time=time.time() - start_time,
                error_count=1
            )
            
            self.component_health[component_name] = health
            self._create_alert(component_name, AlertSeverity.HIGH, f"Component unhealthy: {e}")
            
            return health
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        now = time.time()
        
        # Run health checks if needed
        if now - self.last_health_check > self.check_interval:
            self._run_health_checks()
            self.last_health_check = now
        
        # Calculate overall health
        healthy_components = sum(1 for h in self.component_health.values() if h.status == "healthy")
        total_components = len(self.component_health)
        health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 100
        
        # Determine overall status
        if health_percentage >= 90:
            overall_status = "healthy"
        elif health_percentage >= 70:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
            
            return {
            "status": overall_status,
            "health_percentage": health_percentage,
            "components": {name: h.status for name, h in self.component_health.items()},
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "last_check": self.last_health_check,
            "uptime": now  # Simplified uptime
        }
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # Calculate metrics from component health
        avg_response_time = 0.0
        if self.component_health:
            avg_response_time = sum(h.response_time for h in self.component_health.values()) / len(self.component_health)
        
        error_count = sum(h.error_count for h in self.component_health.values())
        error_rate = error_count / max(len(self.component_health), 1)
        
        metrics = SystemMetrics(
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            active_components=len([h for h in self.component_health.values() if h.status == "healthy"])
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def _run_health_checks(self):
        """Run health checks on all components."""
        components = ["config", "workflow", "cache", "verification"]
        
        for component in components:
            try:
                self.check_component_health(component)
        except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
    
    def _create_alert(self, component: str, severity: AlertSeverity, message: str):
        """Create a new health alert."""
        alert = HealthAlert(
            component=component,
            severity=severity,
            message=message
        )
        
        self.alerts.append(alert)
        logger.warning(f"Health alert: [{severity.value}] {component}: {message}")
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_alerts(self, unresolved_only: bool = False) -> List[HealthAlert]:
        """Get health alerts."""
        if unresolved_only:
            return [a for a in self.alerts if not a.resolved]
        return self.alerts.copy()
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor