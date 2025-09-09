import os
import logging
from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

def init_settings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    logger.info("Settings initialized")

def get_agentic_config():
    return {"enabled": False}

def get_cache_config():
    return {
        "semantic_cache_enabled": os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true",
        "redis_url": os.getenv("REDIS_URL"),
        "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
        "cache_similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        "max_cache_size": int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
        "cache_key_prefix": "sota_rag:",
        "cache_stats_enabled": True
    }

def get_verification_config():
    return {"enabled": os.getenv("VERIFICATION_ENABLED", "true").lower() == "true"}
