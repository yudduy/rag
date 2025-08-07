import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import Settings
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    try:
        from llama_index.embeddings import OpenAIEmbedding
    except ImportError:
        # Fallback for different versions
        OpenAIEmbedding = None
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def init_settings():
    """Initialize LlamaIndex settings with comprehensive configuration validation."""
    
    # Ensure environment variables are loaded from the correct location
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
    
    # Validate required environment variables
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise RuntimeError("OPENAI_API_KEY is missing or not configured in environment variables. Please set it in src/.env")
    
    # Model configuration with cost optimization
    main_model = os.getenv("MODEL", "gpt-4o")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    logger.info(f"Initializing with models: LLM={main_model}, Embeddings={embedding_model}")
    
    # Configure main LLM with optimized settings
    Settings.llm = OpenAI(
        model=main_model,
        temperature=float(os.getenv("TEMPERATURE", "0.1")),  # Lower temperature for consistency
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),  # Reasonable limit for cost control
    )
    
    # Configure embedding model
    if OpenAIEmbedding is not None:
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "100")),  # Batch embeddings for efficiency
        )
    else:
        logger.warning("OpenAI embeddings not available, using default embedding model")
    
    # Set global chunk size for optimal performance
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Validate configuration parameters
    _validate_configuration()
    _validate_agentic_configuration()
    _validate_cache_configuration()
    _validate_verification_configuration()
    _validate_multimodal_configuration()
    
    # Configure logging level
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    logger.info("Settings initialized successfully with enhanced RAG and agentic configuration")


def _validate_configuration():
    """Validate all RAG configuration parameters."""
    
    # Sentence window configuration
    sentence_window_size = int(os.getenv("SENTENCE_WINDOW_SIZE", "3"))
    if sentence_window_size < 1 or sentence_window_size > 10:
        logger.warning(f"SENTENCE_WINDOW_SIZE ({sentence_window_size}) should be between 1-10")
        
    # Retrieval configuration
    top_k = int(os.getenv("TOP_K", "10"))
    if top_k < 1 or top_k > 100:
        raise ValueError(f"TOP_K ({top_k}) must be between 1-100")
        
    # Reranking configuration
    rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))
    if rerank_top_n < 1 or rerank_top_n > top_k:
        logger.warning(f"RERANK_TOP_N ({rerank_top_n}) should be between 1 and TOP_K ({top_k})")
        
    # Similarity threshold
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        logger.warning(f"SIMILARITY_THRESHOLD ({similarity_threshold}) should be between 0.0-1.0")
        
    # Temperature validation
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    if temperature < 0.0 or temperature > 2.0:
        logger.warning(f"TEMPERATURE ({temperature}) should be between 0.0-2.0")
        
    # Chunk size validation
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    if chunk_size < 100 or chunk_size > 2000:
        logger.warning(f"CHUNK_SIZE ({chunk_size}) should be between 100-2000")
        
    logger.info("Configuration validation completed")


def _validate_agentic_configuration():
    """Validate agentic workflow configuration parameters."""
    
    # Agent routing configuration
    routing_enabled = os.getenv("AGENT_ROUTING_ENABLED", "true").lower() == "true"
    routing_threshold = float(os.getenv("AGENT_ROUTING_THRESHOLD", "0.8"))
    if routing_threshold < 0.0 or routing_threshold > 1.0:
        logger.warning(f"AGENT_ROUTING_THRESHOLD ({routing_threshold}) should be between 0.0-1.0")
    
    # Query decomposition configuration
    decomposition_enabled = os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"
    max_subqueries = int(os.getenv("MAX_SUBQUERIES", "3"))
    if max_subqueries < 1 or max_subqueries > 10:
        logger.warning(f"MAX_SUBQUERIES ({max_subqueries}) should be between 1-10")
    
    # Query complexity threshold
    complexity_threshold = float(os.getenv("QUERY_COMPLEXITY_THRESHOLD", "0.7"))
    if complexity_threshold < 0.0 or complexity_threshold > 1.0:
        logger.warning(f"QUERY_COMPLEXITY_THRESHOLD ({complexity_threshold}) should be between 0.0-1.0")
    
    # Routing model configuration
    routing_model = os.getenv("ROUTING_MODEL", "gpt-3.5-turbo")
    decomposition_model = os.getenv("DECOMPOSITION_MODEL", "gpt-3.5-turbo")
    
    logger.info(f"Agentic configuration: Routing={routing_enabled}, Decomposition={decomposition_enabled}")
    logger.info(f"Routing threshold={routing_threshold}, Max subqueries={max_subqueries}")
    logger.info("Agentic configuration validation completed")


def _validate_cache_configuration():
    """Validate semantic cache configuration parameters."""
    
    # Cache similarity threshold
    cache_similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.97"))
    if cache_similarity_threshold < 0.8 or cache_similarity_threshold > 1.0:
        logger.warning(f"CACHE_SIMILARITY_THRESHOLD ({cache_similarity_threshold}) should be between 0.8-1.0 for effective caching")
    
    # Cache TTL validation
    cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    if cache_ttl < 300 or cache_ttl > 86400:  # 5 minutes to 24 hours
        logger.warning(f"CACHE_TTL ({cache_ttl}) should be between 300-86400 seconds")
    
    # Max cache size validation
    max_cache_size = int(os.getenv("MAX_CACHE_SIZE", "10000"))
    if max_cache_size < 100 or max_cache_size > 100000:
        logger.warning(f"MAX_CACHE_SIZE ({max_cache_size}) should be between 100-100000")
    
    # Cache enabled validation
    semantic_cache_enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true"
    redis_url = os.getenv("REDIS_CACHE_URL", "redis://localhost:6379")
    
    logger.info(f"Semantic cache configuration: Enabled={semantic_cache_enabled}")
    logger.info(f"Cache threshold={cache_similarity_threshold}, TTL={cache_ttl}s, Max size={max_cache_size}")
    logger.info("Cache configuration validation completed")


def _validate_verification_configuration():
    """Validate hallucination detection and verification configuration parameters."""
    
    # Verification enabled/disabled
    verification_enabled = os.getenv("VERIFICATION_ENABLED", "true").lower() == "true"
    multi_level_confidence = os.getenv("MULTI_LEVEL_CONFIDENCE", "true").lower() == "true"
    
    # Verification threshold validation
    verification_threshold = float(os.getenv("VERIFICATION_THRESHOLD", "0.8"))
    if verification_threshold < 0.0 or verification_threshold > 1.0:
        logger.warning(f"VERIFICATION_THRESHOLD ({verification_threshold}) should be between 0.0-1.0")
    
    # Verification model validation
    verification_model = os.getenv("VERIFICATION_MODEL", "gpt-4o-mini")
    valid_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
    if verification_model not in valid_models:
        logger.warning(f"VERIFICATION_MODEL ({verification_model}) not in recommended models: {valid_models}")
    
    # Ensemble verification
    ensemble_verification = os.getenv("ENSEMBLE_VERIFICATION", "true").lower() == "true"
    
    # Debate augmentation (optional for critical domains)
    debate_augmentation_enabled = os.getenv("DEBATE_AUGMENTATION_ENABLED", "false").lower() == "true"
    
    # Confidence calculation parameters
    node_confidence_weight = float(os.getenv("NODE_CONFIDENCE_WEIGHT", "0.3"))
    graph_confidence_weight = float(os.getenv("GRAPH_CONFIDENCE_WEIGHT", "0.25"))
    generation_consistency_weight = float(os.getenv("GENERATION_CONSISTENCY_WEIGHT", "0.25"))
    citation_accuracy_weight = float(os.getenv("CITATION_ACCURACY_WEIGHT", "0.2"))
    
    # Validate weights sum to 1.0
    total_weight = (node_confidence_weight + graph_confidence_weight + 
                   generation_consistency_weight + citation_accuracy_weight)
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Confidence weights sum to {total_weight:.2f}, should sum to 1.0")
    
    # Performance thresholds
    max_verification_time = float(os.getenv("MAX_VERIFICATION_TIME", "2.0"))
    if max_verification_time < 0.5 or max_verification_time > 10.0:
        logger.warning(f"MAX_VERIFICATION_TIME ({max_verification_time}) should be between 0.5-10.0 seconds")
    
    # Alert thresholds
    low_confidence_alert_threshold = float(os.getenv("LOW_CONFIDENCE_ALERT_THRESHOLD", "0.6"))
    if low_confidence_alert_threshold < 0.0 or low_confidence_alert_threshold > 1.0:
        logger.warning(f"LOW_CONFIDENCE_ALERT_THRESHOLD ({low_confidence_alert_threshold}) should be between 0.0-1.0")
    
    # Performance and optimization configuration
    max_verification_time = float(os.getenv("MAX_VERIFICATION_TIME", "2.0"))
    verification_timeout = float(os.getenv("VERIFICATION_TIMEOUT", "5.0"))
    verification_caching_enabled = os.getenv("VERIFICATION_CACHING_ENABLED", "true").lower() == "true"
    smart_verification_routing = os.getenv("SMART_VERIFICATION_ROUTING", "true").lower() == "true"
    
    # Batch processing configuration
    verification_batch_processing = os.getenv("VERIFICATION_BATCH_PROCESSING", "true").lower() == "true"
    verification_batch_size = int(os.getenv("VERIFICATION_BATCH_SIZE", "5"))
    
    # Monitoring configuration
    verification_metrics_enabled = os.getenv("VERIFICATION_METRICS_ENABLED", "true").lower() == "true"
    hallucination_alerts_enabled = os.getenv("HALLUCINATION_ALERTS_ENABLED", "true").lower() == "true"
    
    # Validate performance settings
    if verification_timeout < 1.0 or verification_timeout > 30.0:
        logger.warning(f"VERIFICATION_TIMEOUT ({verification_timeout}) should be between 1.0-30.0 seconds")
    
    if verification_batch_size < 1 or verification_batch_size > 20:
        logger.warning(f"VERIFICATION_BATCH_SIZE ({verification_batch_size}) should be between 1-20")
    
    logger.info(f"Verification configuration: Enabled={verification_enabled}, Multi-level={multi_level_confidence}")
    logger.info(f"Verification threshold={verification_threshold}, Model={verification_model}")
    logger.info(f"Ensemble={ensemble_verification}, Debate augmentation={debate_augmentation_enabled}")
    logger.info(f"Caching={verification_caching_enabled}, Smart routing={smart_verification_routing}")
    logger.info(f"Timeout={verification_timeout}s, Batch processing={verification_batch_processing}")
    logger.info("Verification configuration validation completed")


def get_rag_config() -> dict:
    """Get current RAG configuration as a dictionary for monitoring/debugging."""
    return {
        "sentence_window_size": int(os.getenv("SENTENCE_WINDOW_SIZE", "3")),
        "top_k": int(os.getenv("TOP_K", "10")),
        "hybrid_search_enabled": os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true",
        "rerank_enabled": os.getenv("RERANK_ENABLED", "true").lower() == "true",
        "rerank_top_n": int(os.getenv("RERANK_TOP_N", "5")),
        "rerank_model": os.getenv("RERANK_MODEL", "gpt-3.5-turbo"),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
        "main_model": os.getenv("MODEL", "gpt-4o"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "2048")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
    }


def get_agentic_config() -> dict:
    """Get current agentic workflow configuration as a dictionary for monitoring/debugging."""
    return {
        "agent_routing_enabled": os.getenv("AGENT_ROUTING_ENABLED", "true").lower() == "true",
        "query_decomposition_enabled": os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true",
        "agent_routing_threshold": float(os.getenv("AGENT_ROUTING_THRESHOLD", "0.8")),
        "max_subqueries": int(os.getenv("MAX_SUBQUERIES", "3")),
        "query_complexity_threshold": float(os.getenv("QUERY_COMPLEXITY_THRESHOLD", "0.7")),
        "routing_model": os.getenv("ROUTING_MODEL", "gpt-3.5-turbo"),
        "decomposition_model": os.getenv("DECOMPOSITION_MODEL", "gpt-3.5-turbo"),
        "parallel_execution_enabled": os.getenv("PARALLEL_EXECUTION_ENABLED", "true").lower() == "true",
        "subquery_aggregation_model": os.getenv("SUBQUERY_AGGREGATION_MODEL", "gpt-4o-mini"),
    }


def get_cache_config() -> dict:
    """Get current semantic cache configuration as a dictionary for monitoring/debugging."""
    return {
        "semantic_cache_enabled": os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true",
        "cache_similarity_threshold": float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.97")),
        "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
        "redis_cache_url": os.getenv("REDIS_CACHE_URL", "redis://localhost:6379"),
        "max_cache_size": int(os.getenv("MAX_CACHE_SIZE", "10000")),
        "cache_key_prefix": os.getenv("CACHE_KEY_PREFIX", "rag_semantic:"),
        "cache_stats_enabled": os.getenv("CACHE_STATS_ENABLED", "true").lower() == "true",
        "cache_warming_enabled": os.getenv("CACHE_WARMING_ENABLED", "false").lower() == "true",
    }


def get_verification_config() -> dict:
    """Get current verification configuration as a dictionary for monitoring/debugging."""
    return {
        "verification_enabled": os.getenv("VERIFICATION_ENABLED", "true").lower() == "true",
        "multi_level_confidence": os.getenv("MULTI_LEVEL_CONFIDENCE", "true").lower() == "true",
        "verification_threshold": float(os.getenv("VERIFICATION_THRESHOLD", "0.8")),
        "verification_model": os.getenv("VERIFICATION_MODEL", "gpt-4o-mini"),
        "ensemble_verification": os.getenv("ENSEMBLE_VERIFICATION", "true").lower() == "true",
        "debate_augmentation_enabled": os.getenv("DEBATE_AUGMENTATION_ENABLED", "false").lower() == "true",
        "node_confidence_weight": float(os.getenv("NODE_CONFIDENCE_WEIGHT", "0.3")),
        "graph_confidence_weight": float(os.getenv("GRAPH_CONFIDENCE_WEIGHT", "0.25")),
        "generation_consistency_weight": float(os.getenv("GENERATION_CONSISTENCY_WEIGHT", "0.25")),
        "citation_accuracy_weight": float(os.getenv("CITATION_ACCURACY_WEIGHT", "0.2")),
        "max_verification_time": float(os.getenv("MAX_VERIFICATION_TIME", "2.0")),
        "verification_timeout": float(os.getenv("VERIFICATION_TIMEOUT", "5.0")),
        "verification_caching_enabled": os.getenv("VERIFICATION_CACHING_ENABLED", "true").lower() == "true",
        "smart_verification_routing": os.getenv("SMART_VERIFICATION_ROUTING", "true").lower() == "true",
        "verification_batch_processing": os.getenv("VERIFICATION_BATCH_PROCESSING", "true").lower() == "true",
        "verification_batch_size": int(os.getenv("VERIFICATION_BATCH_SIZE", "5")),
        "low_confidence_alert_threshold": float(os.getenv("LOW_CONFIDENCE_ALERT_THRESHOLD", "0.6")),
        "verification_metrics_enabled": os.getenv("VERIFICATION_METRICS_ENABLED", "true").lower() == "true",
        "hallucination_alerts_enabled": os.getenv("HALLUCINATION_ALERTS_ENABLED", "true").lower() == "true",
    }


def _validate_multimodal_configuration():
    """Validate multimodal CLIP and TTS configuration parameters."""
    
    # Multimodal feature flags
    multimodal_enabled = os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true"
    image_indexing_enabled = os.getenv("IMAGE_INDEXING_ENABLED", "false").lower() == "true"
    tts_integration_enabled = os.getenv("TTS_INTEGRATION_ENABLED", "false").lower() == "true"
    
    # CLIP model configuration
    clip_model_name = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    valid_clip_models = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16", 
        "openai/clip-vit-large-patch14"
    ]
    
    if clip_model_name not in valid_clip_models:
        logger.warning(f"CLIP_MODEL_NAME ({clip_model_name}) not in recommended models: {valid_clip_models}")
    
    # Cross-modal retrieval threshold
    multimodal_threshold = float(os.getenv("MULTIMODAL_THRESHOLD", "0.6"))
    if multimodal_threshold < 0.0 or multimodal_threshold > 1.0:
        logger.warning(f"MULTIMODAL_THRESHOLD ({multimodal_threshold}) should be between 0.0-1.0")
    
    # Image processing configuration
    max_image_size = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    if max_image_size < 1 or max_image_size > 100:
        logger.warning(f"MAX_IMAGE_SIZE_MB ({max_image_size}) should be between 1-100")
    
    # Image formats
    supported_image_formats = os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,bmp,tiff").split(",")
    expected_formats = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    
    # TTS configuration
    tts_engine = os.getenv("TTS_ENGINE", "pyttsx3")
    valid_tts_engines = ["pyttsx3", "gtts"]
    
    if tts_engine not in valid_tts_engines:
        logger.warning(f"TTS_ENGINE ({tts_engine}) not in supported engines: {valid_tts_engines}")
    
    # Voice configuration
    voice_speed = int(os.getenv("VOICE_SPEED", "150"))
    if voice_speed < 50 or voice_speed > 300:
        logger.warning(f"VOICE_SPEED ({voice_speed}) should be between 50-300 words per minute")
    
    # Multimodal embedding dimensions
    text_embedding_dim = int(os.getenv("TEXT_EMBEDDING_DIM", "1536"))  # OpenAI text-embedding-3-large
    clip_embedding_dim = int(os.getenv("CLIP_EMBEDDING_DIM", "512"))   # CLIP standard dimension
    
    # Performance configuration
    image_batch_size = int(os.getenv("IMAGE_BATCH_SIZE", "8"))
    if image_batch_size < 1 or image_batch_size > 64:
        logger.warning(f"IMAGE_BATCH_SIZE ({image_batch_size}) should be between 1-64")
    
    # OCR configuration for image text extraction
    ocr_enabled = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_language = os.getenv("OCR_LANGUAGE", "eng")
    
    # Cross-modal search configuration
    cross_modal_search_enabled = os.getenv("CROSS_MODAL_SEARCH_ENABLED", "false").lower() == "true"
    text_image_weight_ratio = float(os.getenv("TEXT_IMAGE_WEIGHT_RATIO", "0.7"))  # 0.7 text, 0.3 image
    
    if text_image_weight_ratio < 0.0 or text_image_weight_ratio > 1.0:
        logger.warning(f"TEXT_IMAGE_WEIGHT_RATIO ({text_image_weight_ratio}) should be between 0.0-1.0")
    
    # Quality thresholds
    min_image_quality_score = float(os.getenv("MIN_IMAGE_QUALITY_SCORE", "0.5"))
    if min_image_quality_score < 0.0 or min_image_quality_score > 1.0:
        logger.warning(f"MIN_IMAGE_QUALITY_SCORE ({min_image_quality_score}) should be between 0.0-1.0")
    
    # Performance optimization settings
    multimodal_cache_enabled = os.getenv("MULTIMODAL_CACHE_ENABLED", "true").lower() == "true"
    clip_model_cache_dir = os.getenv("CLIP_MODEL_CACHE_DIR", "./models/clip")
    
    logger.info(f"Multimodal configuration: Enabled={multimodal_enabled}")
    logger.info(f"Image indexing={image_indexing_enabled}, TTS={tts_integration_enabled}")
    logger.info(f"CLIP model={clip_model_name}, Threshold={multimodal_threshold}")
    logger.info(f"Cross-modal search={cross_modal_search_enabled}, Text/Image ratio={text_image_weight_ratio}")
    logger.info(f"OCR enabled={ocr_enabled}, Language={ocr_language}")
    logger.info(f"TTS engine={tts_engine}, Voice speed={voice_speed}wpm")
    logger.info("Multimodal configuration validation completed")


def get_multimodal_config() -> dict:
    """Get current multimodal configuration as a dictionary for monitoring/debugging."""
    return {
        "multimodal_enabled": os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true",
        "image_indexing_enabled": os.getenv("IMAGE_INDEXING_ENABLED", "false").lower() == "true", 
        "tts_integration_enabled": os.getenv("TTS_INTEGRATION_ENABLED", "false").lower() == "true",
        "clip_model_name": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        "multimodal_threshold": float(os.getenv("MULTIMODAL_THRESHOLD", "0.6")),
        "max_image_size_mb": int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
        "supported_image_formats": os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,bmp,tiff").split(","),
        "tts_engine": os.getenv("TTS_ENGINE", "pyttsx3"),
        "voice_speed": int(os.getenv("VOICE_SPEED", "150")),
        "text_embedding_dim": int(os.getenv("TEXT_EMBEDDING_DIM", "1536")),
        "clip_embedding_dim": int(os.getenv("CLIP_EMBEDDING_DIM", "512")),
        "image_batch_size": int(os.getenv("IMAGE_BATCH_SIZE", "8")),
        "ocr_enabled": os.getenv("OCR_ENABLED", "true").lower() == "true",
        "ocr_language": os.getenv("OCR_LANGUAGE", "eng"),
        "cross_modal_search_enabled": os.getenv("CROSS_MODAL_SEARCH_ENABLED", "false").lower() == "true",
        "text_image_weight_ratio": float(os.getenv("TEXT_IMAGE_WEIGHT_RATIO", "0.7")),
        "min_image_quality_score": float(os.getenv("MIN_IMAGE_QUALITY_SCORE", "0.5")),
        "multimodal_cache_enabled": os.getenv("MULTIMODAL_CACHE_ENABLED", "true").lower() == "true",
        "clip_model_cache_dir": os.getenv("CLIP_MODEL_CACHE_DIR", "./models/clip"),
    }
