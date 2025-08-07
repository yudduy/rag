from dotenv import load_dotenv
import logging
import os
from pathlib import Path

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.core.workflow import Workflow

from src.index import get_index
from src.query import get_query_engine_tool
from src.citation import CITATION_SYSTEM_PROMPT, enable_citation
from src.settings import init_settings, get_agentic_config
from src.agentic_workflow import AgenticWorkflow
from src.unified_config import get_unified_config
from src.unified_workflow import create_unified_workflow

logger = logging.getLogger(__name__)


def create_workflow() -> Workflow:
    """
    Create the unified SOTA RAG workflow orchestrator.
    
    This function serves as the main entry point for creating workflows.
    It checks the configuration and returns either:
    1. UnifiedWorkflow (SOTA orchestrator) - when unified features are enabled
    2. AgenticWorkflow (enhanced agent) - when only agentic features are enabled  
    3. AgentWorkflow (base agent) - when using basic RAG only
    
    Returns:
        Workflow: The most appropriate workflow based on configuration
        
    Raises:
        RuntimeError: If workflow creation fails
        ValueError: If configuration is invalid
    """
    try:
        # Load environment variables from the correct location
        project_root = Path(__file__).parent.parent.absolute()
        env_path = project_root / "src" / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from: {env_path}")
        else:
            # Try alternative location
            alt_env_path = project_root / ".env"
            if alt_env_path.exists():
                load_dotenv(alt_env_path)
                logger.info(f"Loaded environment from: {alt_env_path}")
            else:
                logger.warning(f"No .env file found at {env_path} or {alt_env_path}")
                # Still try to load from default location
                load_dotenv()
        
        # Initialize settings
        init_settings()
        
        # Check if unified orchestrator should be used
        use_unified_orchestrator = os.getenv("USE_UNIFIED_ORCHESTRATOR", "true").lower() == "true"
        
        if use_unified_orchestrator:
            logger.info("Creating unified SOTA RAG workflow orchestrator")
            try:
                workflow = create_unified_workflow()
                logger.info("Unified workflow orchestrator created successfully")
                return workflow
            except Exception as unified_error:
                logger.error(f"Failed to create unified workflow: {unified_error}")
                logger.info("Falling back to enhanced agentic workflow")
                # Fall through to create enhanced agentic workflow
        
        # Fallback: Create enhanced agentic workflow or base workflow
        logger.info("Initializing enhanced RAG workflow with multi-stage retrieval")
        
        # Get the index with validation
        index = get_index()
        if index is None:
            raise RuntimeError(
                "Index not found! Please run `uv run generate` to index the data first."
            )
        
        logger.info("Index loaded successfully")
        
        # Create query tool with advanced features
        query_tool = get_query_engine_tool(index=index)
        
        # Enable citation system
        query_tool = enable_citation(query_tool)
        
        logger.info("Created query engine with sentence-windowing, hybrid search, and citation support")

        # Enhanced system prompt optimized for production RAG
        system_prompt = (
            "You are an advanced AI assistant with access to a sophisticated retrieval system. "
            "You can answer questions by searching through documents using multiple retrieval strategies "
            "including semantic search, keyword matching, and intelligent reranking. "
            "Always provide accurate, well-cited responses based on the retrieved information. "
            "If you cannot find relevant information in the knowledge base, clearly state this limitation."
        )
        system_prompt += CITATION_SYSTEM_PROMPT

        # Validate LLM is configured
        if Settings.llm is None:
            raise ValueError("LLM not configured. Please check your settings.")

        # Create base agent workflow
        base_workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[query_tool],
            llm=Settings.llm,
            system_prompt=system_prompt,
        )
        
        # Check if agentic features are enabled
        agentic_config = get_agentic_config()
        if (agentic_config.get("agent_routing_enabled", True) or 
            agentic_config.get("query_decomposition_enabled", True)):
            
            logger.info("Creating enhanced agentic workflow with routing and decomposition")
            workflow = AgenticWorkflow(
                agent_workflow=base_workflow,
                timeout=120.0,
                verbose=bool(os.getenv("WORKFLOW_VERBOSE", "false").lower() == "true")
            )
        else:
            logger.info("Using standard agent workflow (agentic features disabled)")
            workflow = base_workflow
        
        logger.info("RAG workflow created successfully")
        return workflow
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise RuntimeError(f"Workflow creation failed: {str(e)}") from e


def create_legacy_workflow() -> AgentWorkflow:
    """
    Create the legacy agent workflow for backward compatibility.
    
    This function creates the original AgentWorkflow without SOTA enhancements.
    Used for testing, debugging, or when SOTA features are not available.
    
    Returns:
        AgentWorkflow: Legacy workflow without SOTA enhancements
    """
    logger.info("Creating legacy agent workflow")
    
    # Initialize basic settings
    init_settings()
    
    # Get the index
    index = get_index()
    if index is None:
        raise RuntimeError("Index not found! Please run `uv run generate` to index the data first.")
    
    # Create basic query tool
    query_tool = get_query_engine_tool(index=index)
    query_tool = enable_citation(query_tool)
    
    # Basic system prompt
    system_prompt = (
        "You are an AI assistant with access to a document retrieval system. "
        "Answer questions based on the retrieved information and provide citations."
    ) + CITATION_SYSTEM_PROMPT
    
    # Create and return basic workflow
    workflow = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
    )
    
    logger.info("Legacy workflow created successfully")
    return workflow


# Create the main workflow instance
workflow = create_workflow()
