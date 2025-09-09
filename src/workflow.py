"""
SOTA RAG Workflow - Main entry point for workflow creation.

This module provides a simple factory function for creating RAG workflows.
"""

import logging
from dotenv import load_dotenv
from pathlib import Path

from src.rag_workflow import create_workflow as create_rag_workflow
from src.settings import init_settings

logger = logging.getLogger(__name__)


def create_workflow():
    """
    Create the main SOTA RAG workflow.
    
    Returns:
        RAGWorkflow: The main workflow instance
    """
    try:
        # Load environment variables
        project_root = Path(__file__).parent.parent.absolute()
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
        
        # Initialize settings
        init_settings()
        
        # Create the workflow
        workflow = create_rag_workflow()
        logger.info("SOTA RAG workflow created successfully")
        
        return workflow
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise RuntimeError(f"Workflow creation failed: {e}") from e