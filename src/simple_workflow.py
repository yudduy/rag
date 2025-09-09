"""
Simple RAG Workflow for testing functionality.
"""

import os
import logging
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI

from src.index import get_index
from src.query import get_query_engine_tool
from src.citation import CITATION_SYSTEM_PROMPT
from src.settings import init_settings

logger = logging.getLogger(__name__)


def create_simple_workflow():
    """Create a basic working RAG workflow for testing."""
    
    # Initialize settings
    init_settings()
    
    # Get index and query engine
    index = get_index()
    query_engine_tool = get_query_engine_tool(index)
    
    # Create basic agent workflow
    workflow = AgentWorkflow(
        llm=OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        tools=[query_engine_tool],
        system_prompt=CITATION_SYSTEM_PROMPT,
        verbose=os.getenv("WORKFLOW_VERBOSE", "false").lower() == "true"
    )
    
    logger.info("Simple RAG workflow created successfully")
    return workflow
