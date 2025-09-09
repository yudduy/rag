"""Simple system health monitoring."""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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
    timestamp: float
    resolved: bool = False


class HealthMonitor:
    """Simple health monitoring system."""
    
    def __init__(self):
        """Initialize the health monitor."""
        self.alerts: List[HealthAlert] = []
        self.last_check = 0.0
        logger.info("Health monitor initialized")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return {
            "status": "healthy",
            "components": {"config": "healthy", "workflow": "healthy"},
            "active_alerts": len(self.alerts),
            "last_check": time.time()
        }
    
    def get_alerts(self) -> List[HealthAlert]:
        """Get health alerts."""
        return self.alerts.copy()


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
