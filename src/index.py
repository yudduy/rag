import logging
import os
from pathlib import Path
from typing import Optional

from llama_index.core.indices import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex

# Import multimodal components
from src.multimodal import (
    get_multimodal_embedding_model,
    is_multimodal_enabled,
    is_image_indexing_enabled
)

logger = logging.getLogger("uvicorn")

# Use absolute path to ensure it works regardless of working directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
STORAGE_DIR = str(PROJECT_ROOT / "src" / "storage")


def get_index():
    """
    Load existing index with multimodal support if enabled.
    
    Returns:
        VectorStoreIndex: The loaded index, or None if no index exists
    """
    # Check if storage exists
    if not os.path.exists(STORAGE_DIR):
        logger.info(f"No existing index found at {STORAGE_DIR}")
        return None
    
    try:
        logger.info(f"Loading index from {STORAGE_DIR}...")
        
        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        
        # Check for multimodal embedding model
        multimodal_model = None
        if is_multimodal_enabled():
            try:
                multimodal_model = get_multimodal_embedding_model()
                if multimodal_model:
                    logger.info("Multimodal embedding model loaded for index")
            except Exception as e:
                logger.warning(f"Failed to load multimodal model, using standard embeddings: {e}")
        
        # Load the index
        index = load_index_from_storage(
            storage_context,
            embed_model=multimodal_model
        )
        
        # Log index information
        num_nodes = len(list(index.docstore.docs.values()))
        logger.info(f"Loaded index with {num_nodes} nodes from {STORAGE_DIR}")
        
        # Check for multimodal nodes if enabled
        if is_multimodal_enabled():
            image_nodes = sum(1 for node in index.docstore.docs.values() 
                            if node.metadata.get("modality") == "image")
            text_nodes = num_nodes - image_nodes
            logger.info(f"Index contains {text_nodes} text nodes and {image_nodes} image nodes")
        
        return index
        
    except Exception as e:
        logger.error(f"Failed to load index from {STORAGE_DIR}: {e}")
        return None


def get_multimodal_index(nodes: list, show_progress: bool = True) -> Optional[VectorStoreIndex]:
    """
    Create a new multimodal index from nodes.
    
    Args:
        nodes: List of nodes to index
        show_progress: Whether to show progress during indexing
        
    Returns:
        VectorStoreIndex: The created index, or None if creation fails
    """
    try:
        if not nodes:
            logger.warning("No nodes provided for multimodal index creation")
            return None
        
        # Get multimodal embedding model if enabled
        embed_model = None
        if is_multimodal_enabled():
            embed_model = get_multimodal_embedding_model()
            if embed_model:
                logger.info(f"Creating multimodal index with {embed_model.model_name}")
            else:
                logger.info("Creating standard index (multimodal model unavailable)")
        else:
            logger.info("Creating standard text-only index")
        
        # Create index
        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            show_progress=show_progress
        )
        
        logger.info(f"Successfully created index with {len(nodes)} nodes")
        return index
        
    except Exception as e:
        logger.error(f"Failed to create multimodal index: {e}")
        return None


def persist_multimodal_index(index: VectorStoreIndex, persist_dir: Optional[str] = None) -> bool:
    """
    Persist multimodal index to storage.
    
    Args:
        index: Index to persist
        persist_dir: Directory to persist to (defaults to STORAGE_DIR)
        
    Returns:
        bool: True if persistence succeeded
    """
    try:
        storage_dir = persist_dir or STORAGE_DIR
        
        # Ensure storage directory exists
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist the index
        index.storage_context.persist(storage_dir)
        
        logger.info(f"Index persisted successfully to {storage_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to persist index: {e}")
        return False


def get_index_info(index: VectorStoreIndex) -> dict:
    """
    Get comprehensive information about the index.
    
    Args:
        index: Index to analyze
        
    Returns:
        dict: Index information
    """
    try:
        nodes = list(index.docstore.docs.values())
        total_nodes = len(nodes)
        
        # Count different node types
        text_nodes = sum(1 for node in nodes if node.metadata.get("modality", "text") == "text")
        image_nodes = sum(1 for node in nodes if node.metadata.get("modality") == "image")
        
        # Calculate size information
        total_text_length = sum(len(node.text or "") for node in nodes)
        avg_text_length = total_text_length / total_nodes if total_nodes > 0 else 0
        
        # Check embedding information
        embedded_nodes = sum(1 for node in nodes if hasattr(node, 'embedding') and node.embedding)
        embedding_coverage = embedded_nodes / total_nodes if total_nodes > 0 else 0
        
        # Multimodal specific info
        multimodal_info = {}
        if is_multimodal_enabled():
            multimodal_info = {
                "multimodal_enabled": True,
                "image_indexing_enabled": is_image_indexing_enabled(),
                "clip_embeddings": sum(1 for node in nodes 
                                     if node.metadata.get("embedding_model") == "clip"),
            }
        else:
            multimodal_info = {"multimodal_enabled": False}
        
        return {
            "total_nodes": total_nodes,
            "text_nodes": text_nodes,
            "image_nodes": image_nodes,
            "total_text_length": total_text_length,
            "avg_text_length": round(avg_text_length, 2),
            "embedding_coverage": round(embedding_coverage, 3),
            "storage_dir": STORAGE_DIR,
            **multimodal_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get index info: {e}")
        return {"error": str(e)}
