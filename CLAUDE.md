# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LlamaIndex RAG (Retrieval-Augmented Generation) application that uses Workflows deployed with LlamaDeploy. It provides a chat interface with document retrieval capabilities and citation support.

## Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables
# Create src/.env with:
# OPENAI_API_KEY=your_api_key_here
# MODEL=gpt-4.1 (optional, defaults to gpt-4.1)
# EMBEDDING_MODEL=text-embedding-3-large (optional)
```

### Generate Index
```bash
# Index documents in ui/data directory
uv run generate
```

### Run Services
```bash
# Start API server (in one terminal)
uv run -m llama_deploy.apiserver

# Deploy the workflow (in another terminal)
uv run llamactl deploy llama_deploy.yml
```

### UI Development
```bash
# Run the UI development server
cd ui && npm run dev
```

### Type Checking
```bash
# Run mypy type checking
uv run mypy src
```

### Testing
```bash
# Run pytest tests
uv run pytest
```

## Architecture

### Core Components

1. **Workflow System** (`src/workflow.py`): Creates an AgentWorkflow with citation-enabled query tools. The workflow handles chat interactions and document retrieval.

2. **Index Management** (`src/index.py`, `src/generate.py`): 
   - Documents stored in `ui/data/` are indexed using OpenAI embeddings
   - Index persisted to `src/storage/` for reuse

3. **Query Engine** (`src/query.py`): Provides retrieval functionality with configurable similarity_top_k parameter.

4. **Citation System** (`src/citation.py`): 
   - Adds citation IDs to retrieved chunks
   - Formats responses with `[citation:id]` references
   - Citation prompt template guides LLM to include proper citations

5. **LlamaDeploy Configuration** (`llama_deploy.yml`):
   - Deploys workflow as service on port 8000
   - UI served on port 3000
   - Default service: workflow

6. **UI Server** (`ui/index.ts`): TypeScript server using @llamaindex/server package to provide chat interface connected to LlamaDeploy.

### Key Files

- `src/settings.py`: Configures OpenAI LLM and embedding models
- `src/workflow.py`: Main workflow definition with agent setup
- `src/citation.py`: Citation processing and formatting
- `ui/data/`: Directory for documents to be indexed
- `src/storage/`: Persisted vector index location

## API Endpoints

- Create task: `POST http://localhost:4501/deployments/chat/tasks/create`
- Stream events: `GET http://localhost:4501/deployments/chat/tasks/{task_id}/events`
- UI: `http://localhost:4501/deployments/chat/ui`
- API docs: `http://localhost:4501/docs`

## Development Notes

- Always run `uv run generate` after adding new documents to `ui/data/`
- The index must exist before starting the workflow (check `src/storage/` directory)
- Environment variables are loaded from `src/.env` using python-dotenv
- UI configuration can be modified in `ui/index.ts`