"""
SOTA RAG Settings - Simplified settings initialization.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def init_settings():
    """Set up LlamaIndex with OpenAI models and validate configuration."""
    
    # Load environment variables if not already set
        project_root = Path(__file__).parent.parent.absolute()
    env_path = project_root / ".env"
    
    if env_path.exists() and not os.getenv("OPENAI_API_KEY"):
        load_dotenv(env_path)
    
    # Check that we have a valid OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    if api_key in {"your_api_key_here", "sk-example", "REDACTED_SK_KEY"}:
        raise ValueError("Please set a valid OpenAI API key")
    
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MAX_TOKENS", "2048"))
    )
    
        Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    
    # Set chunk size
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    logger.info("LlamaIndex settings initialized successfully")


def get_agentic_config():
    """Get agentic configuration from environment variables."""
    return {
        "routing_enabled": os.getenv("AGENT_ROUTING_ENABLED", "false").lower() == "true",
        "decomposition_enabled": os.getenv("QUERY_DECOMPOSITION_ENABLED", "false").lower() == "true",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "timeout": float(os.getenv("TIMEOUT", "120.0")),
        "verbose": os.getenv("WORKFLOW_VERBOSE", "false").lower() == "true"
    }