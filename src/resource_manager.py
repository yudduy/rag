"""
Resource Management and Connection Security Module.

Provides comprehensive resource management, connection pooling, timeout handling,
and security monitoring for the RAG system.
"""

import os
import time
import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_threads: int = 50
    max_connections: int = 100
    max_query_time_seconds: int = 30
    max_file_size_mb: int = 50
    max_cache_size_mb: int = 512


@dataclass
class ConnectionConfig:
    """Connection configuration with security settings."""
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    max_idle_time_seconds: int = 300
    enable_keepalive: bool = True
    ssl_verify: bool = True
    connection_pool_size: int = 10


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    active_threads: int = 0
    active_connections: int = 0
    disk_usage_mb: float = 0.0
    network_io_bytes: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout_seconds:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - request blocked")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e


class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.metrics_history: List[ResourceMetrics] = []
        self.alert_callbacks: List[Callable] = []
        self._monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start resource monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 entries
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Check for resource violations
                self._check_resource_violations(metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval_seconds)
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            cpu_percent = self.process.cpu_percent()
            active_threads = self.process.num_threads()
            
            # Get disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            disk_usage_mb = (disk_usage.total - disk_usage.free) / (1024 * 1024)
            
            # Get network I/O
            try:
                net_io = psutil.net_io_counters()
                network_io_bytes = net_io.bytes_sent + net_io.bytes_recv
            except Exception:
                network_io_bytes = 0
            
            return ResourceMetrics(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                active_threads=active_threads,
                disk_usage_mb=disk_usage_mb,
                network_io_bytes=network_io_bytes,
            )
            
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            return ResourceMetrics()
    
    def _check_resource_violations(self, metrics: ResourceMetrics):
        """Check for resource limit violations and trigger alerts."""
        violations = []
        
        if metrics.memory_mb > self.limits.max_memory_mb:
            violations.append(f"Memory usage exceeded: {metrics.memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")
        
        if metrics.cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"CPU usage exceeded: {metrics.cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
        
        if metrics.active_threads > self.limits.max_threads:
            violations.append(f"Thread count exceeded: {metrics.active_threads} > {self.limits.max_threads}")
        
        if violations:
            for violation in violations:
                logger.warning(f"Resource violation: {violation}")
                for callback in self.alert_callbacks:
                    try:
                        callback(violation, metrics)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, ResourceMetrics], None]):
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        current = self.get_current_metrics()
        
        # Calculate averages from history
        if self.metrics_history:
            avg_memory = sum(m.memory_mb for m in self.metrics_history) / len(self.metrics_history)
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
            max_memory = max(m.memory_mb for m in self.metrics_history)
            max_cpu = max(m.cpu_percent for m in self.metrics_history)
        else:
            avg_memory = avg_cpu = max_memory = max_cpu = 0
        
        return {
            "current": {
                "memory_mb": current.memory_mb,
                "cpu_percent": current.cpu_percent,
                "active_threads": current.active_threads,
                "disk_usage_mb": current.disk_usage_mb,
            },
            "averages": {
                "memory_mb": avg_memory,
                "cpu_percent": avg_cpu,
            },
            "peaks": {
                "memory_mb": max_memory,
                "cpu_percent": max_cpu,
            },
            "limits": {
                "memory_mb": self.limits.max_memory_mb,
                "cpu_percent": self.limits.max_cpu_percent,
                "threads": self.limits.max_threads,
            },
            "violations": {
                "memory": current.memory_mb > self.limits.max_memory_mb,
                "cpu": current.cpu_percent > self.limits.max_cpu_percent,
                "threads": current.active_threads > self.limits.max_threads,
            }
        }


class ConnectionManager:
    """Secure connection manager with pooling and monitoring."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.active_connections: Dict[str, Any] = {}
        self.connection_metrics: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    @contextmanager
    def get_connection(self, connection_id: str, connection_factory: Callable):
        """Get a managed connection with automatic cleanup."""
        connection = None
        start_time = time.time()
        
        try:
            with self._lock:
                # Check if connection exists and is healthy
                if connection_id in self.active_connections:
                    connection = self.active_connections[connection_id]
                    if self._is_connection_healthy(connection):
                        yield connection
                        return
                    else:
                        # Remove unhealthy connection
                        del self.active_connections[connection_id]
                
                # Create new connection with circuit breaker
                circuit_breaker = self.circuit_breakers.setdefault(
                    connection_id, 
                    CircuitBreaker()
                )
                
                connection = circuit_breaker.call(connection_factory)
                self.active_connections[connection_id] = connection
                
                # Initialize metrics
                self.connection_metrics[connection_id] = {
                    "created_at": time.time(),
                    "last_used": time.time(),
                    "use_count": 0,
                    "total_time": 0,
                }
            
            yield connection
            
        except Exception as e:
            logger.error(f"Connection error for {connection_id}: {e}")
            raise
        
        finally:
            # Update metrics
            if connection_id in self.connection_metrics:
                duration = time.time() - start_time
                self.connection_metrics[connection_id]["last_used"] = time.time()
                self.connection_metrics[connection_id]["use_count"] += 1
                self.connection_metrics[connection_id]["total_time"] += duration
    
    def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy."""
        try:
            # Basic health check - could be customized per connection type
            if hasattr(connection, 'ping'):
                connection.ping()
            elif hasattr(connection, 'is_connected') and not connection.is_connected():
                return False
            
            return True
        except Exception:
            return False
    
    def cleanup_idle_connections(self):
        """Clean up idle connections."""
        current_time = time.time()
        to_remove = []
        
        with self._lock:
            for conn_id, metrics in self.connection_metrics.items():
                idle_time = current_time - metrics["last_used"]
                if idle_time > self.config.max_idle_time_seconds:
                    to_remove.append(conn_id)
            
            for conn_id in to_remove:
                if conn_id in self.active_connections:
                    try:
                        connection = self.active_connections[conn_id]
                        if hasattr(connection, 'close'):
                            connection.close()
                    except Exception as e:
                        logger.error(f"Error closing connection {conn_id}: {e}")
                    
                    del self.active_connections[conn_id]
                    del self.connection_metrics[conn_id]
                    
                    logger.info(f"Closed idle connection: {conn_id}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        with self._lock:
            active_count = len(self.active_connections)
            
            total_uses = sum(m["use_count"] for m in self.connection_metrics.values())
            total_time = sum(m["total_time"] for m in self.connection_metrics.values())
            avg_time = total_time / max(total_uses, 1)
            
            return {
                "active_connections": active_count,
                "total_connection_uses": total_uses,
                "average_connection_time": avg_time,
                "circuit_breakers": {
                    cb_id: cb.state for cb_id, cb in self.circuit_breakers.items()
                }
            }


class SecureResourceManager:
    """Main resource manager with security features."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None, 
                 connection_config: Optional[ConnectionConfig] = None):
        self.limits = limits or ResourceLimits(
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "2048")),
            max_cpu_percent=float(os.getenv("MAX_CPU_PERCENT", "80.0")),
            max_threads=int(os.getenv("MAX_THREADS", "50")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "100")),
            max_query_time_seconds=int(os.getenv("MAX_QUERY_TIME_SECONDS", "30")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            max_cache_size_mb=int(os.getenv("MAX_CACHE_SIZE_MB", "512")),
        )
        
        self.connection_config = connection_config or ConnectionConfig(
            timeout_seconds=int(os.getenv("CONNECTION_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("CONNECTION_RETRY_ATTEMPTS", "3")),
            retry_delay_seconds=float(os.getenv("CONNECTION_RETRY_DELAY", "1.0")),
            max_idle_time_seconds=int(os.getenv("CONNECTION_MAX_IDLE_TIME", "300")),
        )
        
        self.monitor = ResourceMonitor(self.limits)
        self.connection_manager = ConnectionManager(self.connection_config)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.limits.max_threads)
        
        # Set up cleanup timer
        self.cleanup_timer = None
        self._setup_cleanup_timer()
        
        # Add alert callback for resource violations
        self.monitor.add_alert_callback(self._handle_resource_alert)
        
        logger.info("SecureResourceManager initialized")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop resource monitoring and cleanup."""
        self.monitor.stop_monitoring()
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        self.thread_pool.shutdown(wait=True)
    
    def _setup_cleanup_timer(self):
        """Set up periodic cleanup timer."""
        def cleanup():
            try:
                self.connection_manager.cleanup_idle_connections()
                gc.collect()  # Force garbage collection
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            finally:
                # Schedule next cleanup
                self.cleanup_timer = threading.Timer(300, cleanup)  # Every 5 minutes
                self.cleanup_timer.daemon = True
                self.cleanup_timer.start()
        
        self.cleanup_timer = threading.Timer(300, cleanup)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def _handle_resource_alert(self, violation: str, metrics: ResourceMetrics):
        """Handle resource violation alerts."""
        logger.warning(f"Resource alert: {violation}")
        
        # Implement emergency measures if needed
        if metrics.memory_mb > self.limits.max_memory_mb * 0.9:
            logger.warning("Emergency memory cleanup triggered")
            gc.collect()
            
            # Could implement more aggressive cleanup here
            # such as clearing caches, closing connections, etc.
    
    @contextmanager
    def execute_with_timeout(self, timeout_seconds: Optional[int] = None):
        """Execute code with timeout protection."""
        timeout_seconds = timeout_seconds or self.limits.max_query_time_seconds
        
        def timeout_handler():
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()
        
        try:
            yield
        finally:
            timer.cancel()
    
    def execute_async_with_timeout(self, coro, timeout_seconds: Optional[int] = None):
        """Execute async operation with timeout."""
        timeout_seconds = timeout_seconds or self.limits.max_query_time_seconds
        
        async def _execute():
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async operation timed out after {timeout_seconds} seconds")
        
        return _execute()
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to thread pool with resource monitoring."""
        if self.monitor.get_current_metrics().active_threads >= self.limits.max_threads:
            raise Exception("Thread limit exceeded - task rejected")
        
        return self.thread_pool.submit(func, *args, **kwargs)
    
    def get_secure_connection(self, connection_id: str, connection_factory: Callable):
        """Get secure connection with monitoring."""
        return self.connection_manager.get_connection(connection_id, connection_factory)
    
    def validate_file_size(self, file_size_bytes: int) -> bool:
        """Validate file size against limits."""
        file_size_mb = file_size_bytes / (1024 * 1024)
        return file_size_mb <= self.limits.max_file_size_mb
    
    def validate_memory_usage(self) -> Tuple[bool, str]:
        """Validate current memory usage."""
        metrics = self.monitor.get_current_metrics()
        if metrics.memory_mb > self.limits.max_memory_mb:
            return False, f"Memory usage {metrics.memory_mb:.1f}MB exceeds limit {self.limits.max_memory_mb}MB"
        return True, ""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        resource_summary = self.monitor.get_resource_summary()
        connection_stats = self.connection_manager.get_connection_stats()
        
        # Determine overall health
        health_score = 1.0
        issues = []
        
        if resource_summary["violations"]["memory"]:
            health_score -= 0.3
            issues.append("Memory usage high")
        
        if resource_summary["violations"]["cpu"]:
            health_score -= 0.2
            issues.append("CPU usage high")
        
        if resource_summary["violations"]["threads"]:
            health_score -= 0.2
            issues.append("Thread count high")
        
        # Check circuit breakers
        open_breakers = [
            cb_id for cb_id, state in connection_stats["circuit_breakers"].items()
            if state == "OPEN"
        ]
        
        if open_breakers:
            health_score -= 0.3
            issues.append(f"Circuit breakers open: {', '.join(open_breakers)}")
        
        health_status = "healthy" if health_score > 0.7 else "degraded" if health_score > 0.3 else "critical"
        
        return {
            "status": health_status,
            "health_score": max(0.0, health_score),
            "issues": issues,
            "resources": resource_summary,
            "connections": connection_stats,
            "limits": {
                "memory_mb": self.limits.max_memory_mb,
                "cpu_percent": self.limits.max_cpu_percent,
                "threads": self.limits.max_threads,
                "connections": self.limits.max_connections,
                "query_time_seconds": self.limits.max_query_time_seconds,
            }
        }


# Global resource manager instance
_resource_manager: Optional[SecureResourceManager] = None


def get_resource_manager() -> SecureResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = SecureResourceManager()
        _resource_manager.start_monitoring()
    return _resource_manager


def cleanup_resources():
    """Cleanup global resources."""
    global _resource_manager
    if _resource_manager:
        _resource_manager.stop_monitoring()
        _resource_manager = None