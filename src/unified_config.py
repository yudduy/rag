"""
Unified Configuration Management for SOTA RAG System

This module provides centralized configuration management for all SOTA RAG components,
including performance profiles, feature toggles, health monitoring, and validation.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class PerformanceProfile(Enum):
    """Performance profiles for different use cases."""
    HIGH_ACCURACY = "high_accuracy"       # Maximum quality, cost optimized
    BALANCED = "balanced"                 # Balance of quality, speed, and cost
    COST_OPTIMIZED = "cost_optimized"     # Minimize costs while maintaining quality
    SPEED = "speed"                       # Maximum speed, quality optimized


class FeatureStatus(Enum):
    """Status of individual features."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"                         # Enabled based on query characteristics
    ERROR = "error"                       # Feature has errors but system continues


@dataclass
class FeatureConfig:
    """Configuration for individual SOTA features."""
    enabled: bool = True
    status: FeatureStatus = FeatureStatus.ENABLED
    auto_enable_threshold: float = 0.0    # Threshold for auto-enabling
    fallback_enabled: bool = True         # Whether to fall back if feature fails
    error_count: int = 0                  # Track error count for health monitoring
    max_errors: int = 5                   # Max errors before disabling
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status of system components."""
    component_name: str
    status: str = "healthy"               # healthy, degraded, error, unknown
    last_check: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedConfig:
    """
    Unified configuration for all SOTA RAG components.
    
    This configuration provides centralized management of all features,
    performance profiles, health monitoring, and system settings.
    """
    
    # Performance profile
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # Core RAG Configuration
    rag_config: Dict[str, Any] = field(default_factory=dict)
    
    # Feature Configurations
    agentic_workflow: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=True,
        settings={
            "agent_routing_enabled": True,
            "query_decomposition_enabled": True,
            "parallel_execution_enabled": True,
            "max_subqueries": 3,
            "routing_threshold": 0.8,
            "complexity_threshold": 0.7,
            "routing_model": "gpt-3.5-turbo",
            "decomposition_model": "gpt-3.5-turbo",
            "aggregation_model": "gpt-4o-mini"
        }
    ))
    
    semantic_cache: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=False,  # Disabled by default since it requires Redis
        settings={
            "similarity_threshold": 0.97,
            "ttl": 3600,
            "max_size": 10000,
            "redis_url": "redis://localhost:6379",
            "warming_enabled": False,
            "stats_enabled": True
        }
    ))
    
    hallucination_detection: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=True,
        settings={
            "verification_threshold": 0.8,
            "verification_model": "gpt-4o-mini",
            "multi_level_confidence": True,
            "ensemble_verification": True,
            "batch_processing": True,
            "batch_size": 5,
            "timeout": 5.0,
            "caching_enabled": True
        }
    ))
    
    multimodal_support: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=False,  # Disabled by default since it requires CLIP
        settings={
            "clip_model_name": "ViT-B/32",
            "image_indexing_enabled": False,
            "cross_modal_search": False,
            "max_image_size_mb": 10,
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
            "ocr_enabled": True,
            "quality_threshold": 0.5
        }
    ))
    
    performance_optimization: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=True,
        settings={
            "advanced_similarity_enabled": True,
            "batch_embedding": True,
            "connection_pooling": True,
            "memory_optimization": True,
            "similarity_levels": ["lexical", "semantic", "intent"]
        }
    ))
    
    tts_integration: FeatureConfig = field(default_factory=lambda: FeatureConfig(
        enabled=False,  # Disabled by default
        settings={
            "engine": "pyttsx3",
            "voice_speed": 150,
            "quality": "medium"
        }
    ))
    
    # System Configuration
    monitoring_enabled: bool = True
    health_check_interval: float = 60.0
    error_alerting_enabled: bool = True
    debug_mode: bool = False
    
    # Component Health
    component_health: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    # Performance Targets
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "response_time_p95": 3.0,      # 95th percentile response time in seconds
        "accuracy_target": 0.96,       # Target accuracy for balanced profile
        "cache_hit_rate": 0.3,         # Target cache hit rate
        "verification_success_rate": 0.95,  # Target verification success rate
        "error_rate_threshold": 0.05   # Maximum acceptable error rate
    })
    
    # Cost Management
    cost_management: Dict[str, Any] = field(default_factory=lambda: {
        "max_query_cost": 2.00,
        "daily_cost_limit": 50.00,
        "cost_monitoring_enabled": True,
        "cost_alerts_enabled": True,
        "optimization_enabled": True
    })


class UnifiedConfigManager:
    """
    Manager for unified configuration with validation, health monitoring,
    and dynamic feature management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config = UnifiedConfig()
        self._load_environment()
        self._initialize_performance_profiles()
        self._validate_configuration()
        
        logger.info(f"UnifiedConfigManager initialized with profile: {self.config.performance_profile.value}")
    
    def _load_environment(self):
        """Load configuration from environment variables."""
        # Load environment variables from the correct location
        if not os.getenv("OPENAI_API_KEY"):
            project_root = Path(__file__).parent.parent.absolute()
            env_path = project_root / "src" / ".env"
            
            if env_path.exists():
                load_dotenv(env_path, override=True)
                logger.info(f"Loaded environment from: {env_path}")
            else:
                alt_env_path = project_root / ".env"
                if alt_env_path.exists():
                    load_dotenv(alt_env_path, override=True)
                    logger.info(f"Loaded environment from: {alt_env_path}")
        
        # Set performance profile
        profile_name = os.getenv("PERFORMANCE_PROFILE", "balanced")
        try:
            self.config.performance_profile = PerformanceProfile(profile_name)
        except ValueError:
            logger.warning(f"Invalid performance profile: {profile_name}, using balanced")
            self.config.performance_profile = PerformanceProfile.BALANCED
        
        # Load feature configurations from environment
        self._load_feature_config_from_env()
        
        # Load system settings
        self.config.monitoring_enabled = os.getenv("UNIFIED_MONITORING_ENABLED", "true").lower() == "true"
        self.config.health_check_interval = float(os.getenv("HEALTH_CHECK_INTERVAL", "60.0"))
        self.config.error_alerting_enabled = os.getenv("ERROR_ALERTING_ENABLED", "true").lower() == "true"
        self.config.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    def _load_feature_config_from_env(self):
        """Load feature configurations from environment variables."""
        
        # Agentic workflow configuration
        self.config.agentic_workflow.enabled = os.getenv("AGENTIC_WORKFLOW_ENABLED", "true").lower() == "true"
        self.config.agentic_workflow.settings.update({
            "agent_routing_enabled": os.getenv("AGENT_ROUTING_ENABLED", "true").lower() == "true",
            "query_decomposition_enabled": os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true",
            "parallel_execution_enabled": os.getenv("PARALLEL_EXECUTION_ENABLED", "true").lower() == "true",
            "max_subqueries": int(os.getenv("MAX_SUBQUERIES", "3")),
            "routing_threshold": float(os.getenv("AGENT_ROUTING_THRESHOLD", "0.8")),
            "complexity_threshold": float(os.getenv("QUERY_COMPLEXITY_THRESHOLD", "0.7")),
            "routing_model": os.getenv("ROUTING_MODEL", "gpt-3.5-turbo"),
            "decomposition_model": os.getenv("DECOMPOSITION_MODEL", "gpt-3.5-turbo"),
            "aggregation_model": os.getenv("SUBQUERY_AGGREGATION_MODEL", "gpt-4o-mini")
        })
        
        # Semantic cache configuration
        self.config.semantic_cache.enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true"
        self.config.semantic_cache.settings.update({
            "similarity_threshold": float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.97")),
            "ttl": int(os.getenv("CACHE_TTL", "3600")),
            "max_size": int(os.getenv("MAX_CACHE_SIZE", "10000")),
            "redis_url": os.getenv("REDIS_CACHE_URL", "redis://localhost:6379"),
            "warming_enabled": os.getenv("CACHE_WARMING_ENABLED", "false").lower() == "true",
            "stats_enabled": os.getenv("CACHE_STATS_ENABLED", "true").lower() == "true"
        })
        
        # Hallucination detection configuration
        self.config.hallucination_detection.enabled = os.getenv("VERIFICATION_ENABLED", "true").lower() == "true"
        self.config.hallucination_detection.settings.update({
            "verification_threshold": float(os.getenv("VERIFICATION_THRESHOLD", "0.8")),
            "verification_model": os.getenv("VERIFICATION_MODEL", "gpt-4o-mini"),
            "multi_level_confidence": os.getenv("MULTI_LEVEL_CONFIDENCE", "true").lower() == "true",
            "ensemble_verification": os.getenv("ENSEMBLE_VERIFICATION", "true").lower() == "true",
            "batch_processing": os.getenv("VERIFICATION_BATCH_PROCESSING", "true").lower() == "true",
            "batch_size": int(os.getenv("VERIFICATION_BATCH_SIZE", "5")),
            "timeout": float(os.getenv("VERIFICATION_TIMEOUT", "5.0")),
            "caching_enabled": os.getenv("VERIFICATION_CACHING_ENABLED", "true").lower() == "true"
        })
        
        # Multimodal support configuration
        self.config.multimodal_support.enabled = os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true"
        self.config.multimodal_support.settings.update({
            "clip_model_name": os.getenv("CLIP_MODEL_NAME", "ViT-B/32"),
            "image_indexing_enabled": os.getenv("IMAGE_INDEXING_ENABLED", "false").lower() == "true",
            "cross_modal_search": os.getenv("CROSS_MODAL_SEARCH_ENABLED", "false").lower() == "true",
            "max_image_size_mb": int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
            "supported_formats": os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,bmp,tiff").split(","),
            "ocr_enabled": os.getenv("OCR_ENABLED", "true").lower() == "true",
            "quality_threshold": float(os.getenv("MIN_IMAGE_QUALITY_SCORE", "0.5"))
        })
        
        # Performance optimization configuration
        self.config.performance_optimization.enabled = os.getenv("PERFORMANCE_OPTIMIZATION_ENABLED", "true").lower() == "true"
        self.config.performance_optimization.settings.update({
            "advanced_similarity_enabled": os.getenv("ADVANCED_SIMILARITY_ENABLED", "true").lower() == "true",
            "batch_embedding": os.getenv("BATCH_EMBEDDING_ENABLED", "true").lower() == "true",
            "connection_pooling": os.getenv("CONNECTION_POOLING_ENABLED", "true").lower() == "true",
            "memory_optimization": os.getenv("MEMORY_OPTIMIZATION_ENABLED", "true").lower() == "true"
        })
        
        # TTS integration configuration
        self.config.tts_integration.enabled = os.getenv("TTS_INTEGRATION_ENABLED", "false").lower() == "true"
        self.config.tts_integration.settings.update({
            "engine": os.getenv("TTS_ENGINE", "pyttsx3"),
            "voice_speed": int(os.getenv("VOICE_SPEED", "150")),
            "quality": os.getenv("TTS_QUALITY", "medium")
        })
        
        # Cost management configuration
        self.config.cost_management.update({
            "max_query_cost": float(os.getenv("MAX_QUERY_COST", "2.00")),
            "daily_cost_limit": float(os.getenv("DAILY_COST_LIMIT", "50.00")),
            "cost_monitoring_enabled": os.getenv("COST_MONITORING_ENABLED", "true").lower() == "true",
            "cost_alerts_enabled": os.getenv("COST_ALERTS_ENABLED", "true").lower() == "true",
            "optimization_enabled": os.getenv("COST_OPTIMIZATION_ENABLED", "true").lower() == "true"
        })
    
    def _initialize_performance_profiles(self):
        """Initialize settings based on performance profile."""
        profile = self.config.performance_profile
        
        if profile == PerformanceProfile.HIGH_ACCURACY:
            # Maximum accuracy settings
            self._apply_high_accuracy_profile()
        elif profile == PerformanceProfile.BALANCED:
            # Balanced settings (default)
            self._apply_balanced_profile()
        elif profile == PerformanceProfile.COST_OPTIMIZED:
            # Cost-optimized settings
            self._apply_cost_optimized_profile()
        elif profile == PerformanceProfile.SPEED:
            # Speed-optimized settings
            self._apply_speed_profile()
        
        logger.info(f"Applied {profile.value} performance profile")
    
    def _apply_high_accuracy_profile(self):
        """Apply high accuracy performance profile."""
        # Enable all quality features
        self.config.agentic_workflow.enabled = True
        self.config.hallucination_detection.enabled = True
        self.config.performance_optimization.enabled = True
        
        # High quality settings
        self.config.agentic_workflow.settings.update({
            "routing_threshold": 0.9,
            "complexity_threshold": 0.6,  # Lower threshold = more decomposition
            "max_subqueries": 5
        })
        
        self.config.hallucination_detection.settings.update({
            "verification_threshold": 0.9,
            "ensemble_verification": True,
            "multi_level_confidence": True
        })
        
        # Performance targets
        self.config.performance_targets.update({
            "response_time_p95": 5.0,  # Allow longer for accuracy
            "accuracy_target": 0.98,
            "verification_success_rate": 0.98
        })
        
        # Cost settings
        self.config.cost_management.update({
            "max_query_cost": 5.00,    # Higher cost allowed
            "optimization_enabled": False  # Don't compromise quality for cost
        })
    
    def _apply_balanced_profile(self):
        """Apply balanced performance profile (default)."""
        # Enable most features with balanced settings
        self.config.agentic_workflow.enabled = True
        self.config.hallucination_detection.enabled = True
        self.config.performance_optimization.enabled = True
        
        # Balanced settings (already set in defaults)
        self.config.performance_targets.update({
            "response_time_p95": 3.0,
            "accuracy_target": 0.96,
            "verification_success_rate": 0.95
        })
    
    def _apply_cost_optimized_profile(self):
        """Apply cost-optimized performance profile."""
        # Enable cost-saving features
        self.config.semantic_cache.enabled = True  # Enable caching to save costs
        self.config.performance_optimization.enabled = True
        
        # Cost-optimized settings
        self.config.agentic_workflow.settings.update({
            "routing_threshold": 0.7,  # Less strict routing
            "complexity_threshold": 0.8,  # Higher threshold = less decomposition
            "max_subqueries": 2,
            "routing_model": "gpt-3.5-turbo",  # Use cheaper model
            "aggregation_model": "gpt-3.5-turbo"
        })
        
        self.config.hallucination_detection.settings.update({
            "verification_model": "gpt-3.5-turbo",  # Use cheaper model
            "ensemble_verification": False,  # Disable expensive ensemble
            "batch_processing": True  # Use batching for efficiency
        })
        
        # Performance targets
        self.config.performance_targets.update({
            "accuracy_target": 0.92,  # Slightly lower accuracy acceptable
            "cache_hit_rate": 0.5     # Higher cache hit rate target
        })
        
        # Strict cost limits
        self.config.cost_management.update({
            "max_query_cost": 1.00,   # Lower cost limit
            "optimization_enabled": True
        })
    
    def _apply_speed_profile(self):
        """Apply speed-optimized performance profile."""
        # Enable performance features, disable slow ones
        self.config.performance_optimization.enabled = True
        self.config.semantic_cache.enabled = True  # Enable caching for speed
        
        # Disable or reduce slow features
        self.config.agentic_workflow.settings.update({
            "parallel_execution_enabled": True,  # Keep parallel for speed
            "complexity_threshold": 0.9,  # Much higher threshold = less decomposition
            "max_subqueries": 2
        })
        
        self.config.hallucination_detection.settings.update({
            "ensemble_verification": False,  # Disable slow ensemble
            "batch_processing": True,  # Use batching
            "timeout": 2.0  # Shorter timeout
        })
        
        # Performance targets
        self.config.performance_targets.update({
            "response_time_p95": 1.5,  # Aggressive speed target
            "accuracy_target": 0.90,   # Lower accuracy for speed
            "cache_hit_rate": 0.6      # High cache hit rate for speed
        })
    
    def _validate_configuration(self):
        """Validate the unified configuration."""
        errors = []
        warnings = []
        
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY is required")
        
        # Validate feature dependencies
        if self.config.semantic_cache.enabled:
            try:
                import redis
            except ImportError:
                warnings.append("Redis not available, semantic cache will use fallback")
                self.config.semantic_cache.status = FeatureStatus.ERROR
        
        if self.config.multimodal_support.enabled:
            try:
                import clip
            except ImportError:
                warnings.append("CLIP not available, multimodal support disabled")
                self.config.multimodal_support.enabled = False
                self.config.multimodal_support.status = FeatureStatus.ERROR
        
        if self.config.tts_integration.enabled:
            try:
                import pyttsx3
            except ImportError:
                warnings.append("TTS engine not available, TTS integration disabled")
                self.config.tts_integration.enabled = False
                self.config.tts_integration.status = FeatureStatus.ERROR
        
        # Validate thresholds
        for feature_name, feature_config in [
            ("agentic_workflow", self.config.agentic_workflow),
            ("hallucination_detection", self.config.hallucination_detection),
            ("semantic_cache", self.config.semantic_cache),
        ]:
            for setting_name, value in feature_config.settings.items():
                if "threshold" in setting_name and isinstance(value, (int, float)):
                    if not 0.0 <= value <= 1.0:
                        warnings.append(f"{feature_name}.{setting_name} ({value}) should be between 0.0-1.0")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        logger.info("Configuration validation completed")
    
    def get_feature_config(self, feature_name: str) -> Optional[FeatureConfig]:
        """Get configuration for a specific feature."""
        return getattr(self.config, feature_name, None)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled and healthy."""
        feature_config = self.get_feature_config(feature_name)
        if not feature_config:
            return False
        
        return (feature_config.enabled and 
                feature_config.status not in [FeatureStatus.ERROR] and
                feature_config.error_count < feature_config.max_errors)
    
    def should_auto_enable_feature(self, feature_name: str, query_characteristics: Dict[str, Any]) -> bool:
        """Determine if a feature should be auto-enabled based on query characteristics."""
        feature_config = self.get_feature_config(feature_name)
        if not feature_config or feature_config.status == FeatureStatus.ERROR:
            return False
        
        if feature_config.status != FeatureStatus.AUTO:
            return feature_config.enabled
        
        # Auto-enable logic based on query characteristics
        complexity_score = query_characteristics.get("complexity_score", 0.0)
        query_type = query_characteristics.get("query_type", "simple")
        
        if feature_name == "agentic_workflow":
            return complexity_score >= feature_config.auto_enable_threshold or query_type == "complex"
        elif feature_name == "hallucination_detection":
            return complexity_score >= feature_config.auto_enable_threshold
        
        return feature_config.enabled
    
    def update_component_health(self, component_name: str, status: str, 
                              metrics: Optional[Dict[str, Any]] = None,
                              error_message: Optional[str] = None):
        """Update health status of a component."""
        import time
        
        if component_name not in self.config.component_health:
            self.config.component_health[component_name] = ComponentHealth(component_name)
        
        health = self.config.component_health[component_name]
        health.status = status
        health.last_check = time.time()
        health.error_message = error_message
        if metrics:
            health.metrics.update(metrics)
        
        # Update feature error counts
        if status == "error" and component_name in ["agentic_workflow", "semantic_cache", 
                                                   "hallucination_detection", "multimodal_support",
                                                   "performance_optimization", "tts_integration"]:
            feature_config = self.get_feature_config(component_name)
            if feature_config:
                feature_config.error_count += 1
                if feature_config.error_count >= feature_config.max_errors:
                    feature_config.status = FeatureStatus.ERROR
                    logger.warning(f"Feature {component_name} disabled due to excessive errors")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_components = sum(1 for h in self.config.component_health.values() if h.status == "healthy")
        total_components = len(self.config.component_health)
        
        overall_status = "healthy"
        if total_components == 0:
            overall_status = "unknown"
        elif healthy_components == 0:
            overall_status = "critical"
        elif healthy_components < total_components * 0.7:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "performance_profile": self.config.performance_profile.value,
            "features_enabled": {
                name: self.is_feature_enabled(name) 
                for name in ["agentic_workflow", "semantic_cache", "hallucination_detection",
                            "multimodal_support", "performance_optimization", "tts_integration"]
            },
            "component_details": {
                name: {
                    "status": health.status,
                    "last_check": health.last_check,
                    "error_message": health.error_message,
                    "metrics": health.metrics
                }
                for name, health in self.config.component_health.items()
            }
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return asdict(self.config)
    
    def save_config(self, path: Optional[str] = None) -> str:
        """Save configuration to file."""
        config_data = self.export_config()
        save_path = path or self.config_path or "unified_config.json"
        
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to: {save_path}")
        return save_path
    
    def reset_feature_errors(self, feature_name: str):
        """Reset error count for a feature."""
        feature_config = self.get_feature_config(feature_name)
        if feature_config:
            feature_config.error_count = 0
            if feature_config.status == FeatureStatus.ERROR:
                feature_config.status = FeatureStatus.ENABLED if feature_config.enabled else FeatureStatus.DISABLED
            logger.info(f"Reset errors for feature: {feature_name}")


# Global configuration manager
_unified_config_manager: Optional[UnifiedConfigManager] = None


def get_unified_config() -> UnifiedConfigManager:
    """
    Get the global unified configuration manager instance.
    
    Returns:
        UnifiedConfigManager instance
    """
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedConfigManager()
    return _unified_config_manager


def reset_unified_config():
    """Reset the global configuration manager (useful for testing)."""
    global _unified_config_manager
    _unified_config_manager = None