"""
SOTA RAG Configuration - Simplified configuration management.

All configuration is handled through environment variables for simplicity.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for the SOTA RAG system."""
    
    # Core settings
    openai_api_key: str
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Features (enabled/disabled via environment variables)
    semantic_cache_enabled: bool = True
    verification_enabled: bool = True
    query_decomposition_enabled: bool = False
    multimodal_enabled: bool = False
    
    # Performance settings
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: float = 120.0
    
    # Cache settings
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    similarity_threshold: float = 0.85
    
    # Cost controls
    max_query_cost: float = 2.00
    cost_monitoring_enabled: bool = True
    
    # Debug settings
    verbose: bool = False
    debug_mode: bool = False


def get_config() -> RAGConfig:
    """Get configuration from environment variables."""
    
    # Required settings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return RAGConfig(
        # Core settings
        openai_api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        
        # Features
        semantic_cache_enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true",
        verification_enabled=os.getenv("VERIFICATION_ENABLED", "true").lower() == "true",
        query_decomposition_enabled=os.getenv("QUERY_DECOMPOSITION_ENABLED", "false").lower() == "true",
        multimodal_enabled=os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true",
        
        # Performance
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        timeout=float(os.getenv("TIMEOUT", "120.0")),
        
        # Cache
        redis_url=os.getenv("REDIS_URL"),
        cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        
        # Cost controls
        max_query_cost=float(os.getenv("MAX_QUERY_COST", "2.00")),
        cost_monitoring_enabled=os.getenv("COST_MONITORING_ENABLED", "true").lower() == "true",
        
        # Debug
        verbose=os.getenv("VERBOSE", "false").lower() == "true",
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true"
    )


# Global config instance
_config: Optional[RAGConfig] = None


def init_config() -> RAGConfig:
    """Initialize and return global configuration."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def get_global_config() -> RAGConfig:
    """Get the global configuration instance."""
    if _config is None:
        return init_config()
    return _config
