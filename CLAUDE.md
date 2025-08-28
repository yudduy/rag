# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready Enhanced RAG (Retrieval-Augmented Generation) system built on LlamaIndex with state-of-the-art capabilities. The system features intelligent workflow orchestration, semantic caching, hallucination detection, multimodal support, and comprehensive security. It uses LlamaDeploy for scalable deployment and provides both API and web interfaces.

### SOTA Features
- **Unified Workflow Orchestration**: Intelligent query routing and processing plan optimization
- **Agentic Workflows**: Query decomposition and parallel execution with advanced routing
- **Semantic Caching**: Redis-based caching with 97% similarity threshold and 31%+ hit rates
- **Hallucination Detection**: Multi-level confidence scoring with GPT-4o-mini verification
- **Performance Profiles**: Configurable quality/cost/speed optimization (high_accuracy|balanced|cost_optimized|speed)
- **Security Hardening**: Comprehensive input sanitization, API key validation, and injection protection
- **Multimodal Support**: CLIP integration for text-image search and OCR capabilities
- **Production Monitoring**: Health checks, resource management, and comprehensive error handling

## Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables from template
cp src/.env.example src/.env
# Edit src/.env with your configuration:
# OPENAI_API_KEY=your_openai_api_key_here
# PERFORMANCE_PROFILE=balanced  # high_accuracy|balanced|cost_optimized|speed
# SEMANTIC_CACHE_ENABLED=true
# VERIFICATION_ENABLED=true
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

# Deploy the enhanced workflow (in another terminal)
uv run llamactl deploy llama_deploy.yml
```

### Development Commands

#### Code Quality
```bash
# Run mypy type checking
uv run mypy src

# Run comprehensive test suite (2000+ tests)
uv run pytest

# Run specific test categories
uv run pytest tests/unit/           # Unit tests
uv run pytest tests/integration/   # Integration tests
uv run pytest tests/security/      # Security validation
uv run pytest tests/performance/   # Performance benchmarks
```

#### Security Testing
```bash
# Run security vulnerability tests
uv run python test_security_fixes.py

# Run integration security tests
uv run python run_integration_tests.py
```

#### Performance Testing
```bash
# Run performance benchmark suite
uv run python performance-test-suite.py

# Monitor system performance
curl http://localhost:4501/health
curl http://localhost:4501/metrics
```

#### UI Development
```bash
# Run the enhanced UI development server
cd ui && npm run dev
```

## Architecture

### Core Components

1. **Unified Workflow Orchestrator** (`src/unified_workflow.py`): 
   - Master orchestrator with intelligent query routing
   - Query complexity analysis and processing plan optimization
   - Performance profile-based resource allocation
   - Security-first design with comprehensive input validation
   - Real-time health monitoring and fallback mechanisms

2. **Agentic Workflow System** (`src/agentic_workflow.py`, `src/agentic.py`):
   - Advanced query decomposition and parallel execution
   - Intelligent agent routing based on query characteristics
   - Multi-step reasoning with intermediate result validation
   - Cost-aware execution with configurable thresholds

3. **Semantic Caching Layer** (`src/cache.py`):
   - Redis-based semantic caching with 97% similarity threshold
   - Advanced cosine similarity matching for query deduplication
   - Secure connection handling with TLS/SSL support
   - TTL-based cache management with statistics tracking
   - 31%+ cache hit rates in production scenarios

4. **Hallucination Detection & Verification** (`src/verification.py`, `src/verification_integration.py`):
   - Multi-level confidence scoring (node, graph, response levels)
   - GPT-4o-mini based verification with ensemble methods
   - Debate-augmented verification for high-risk queries
   - Configurable confidence thresholds and fallback strategies

5. **Enhanced Security Framework** (`src/security.py`, `src/security_config.py`):
   - Comprehensive input sanitization and prompt injection protection
   - API key validation with placeholder detection
   - Rate limiting and security violation tracking
   - Path traversal prevention and resource exhaustion protection

6. **Performance Optimization** (`src/performance.py`, `src/unified_config.py`):
   - Four configurable performance profiles (high_accuracy|balanced|cost_optimized|speed)
   - Dynamic resource allocation and cost monitoring
   - Real-time performance metrics and optimization suggestions

7. **Multimodal Processing** (`src/multimodal.py`, `src/tts.py`):
   - CLIP integration for text-image semantic search
   - OCR capabilities with PDF and image processing
   - Text-to-speech integration with multiple engines
   - Cross-modal retrieval and content generation

8. **Index Management** (`src/index.py`, `src/generate.py`): 
   - Enhanced document processing with multimodal support
   - Hybrid search combining vector and BM25 retrieval
   - Persistent storage with incremental updates
   - Advanced reranking and relevance scoring

9. **Production Monitoring** (`src/health_monitor.py`, `src/resource_manager.py`):
   - Comprehensive health checks and system monitoring
   - Resource usage tracking and alerting
   - Error aggregation and performance analytics
   - Automated recovery and scaling recommendations

### Key Files

**Core Orchestration:**
- `src/unified_workflow.py`: Master workflow orchestrator with intelligent routing
- `src/agentic_workflow.py`: Advanced agentic workflow implementation
- `src/workflow.py`: Base workflow with enhanced citation support

**SOTA Features:**
- `src/cache.py`: Redis semantic caching with advanced similarity matching
- `src/verification.py`: Multi-level hallucination detection and confidence scoring
- `src/security.py`: Comprehensive security framework and input sanitization
- `src/performance.py`: Performance profiles and optimization engine
- `src/multimodal.py`: CLIP integration and multimodal processing

**Configuration & Settings:**
- `src/settings.py`: Enhanced configuration with security validation
- `src/unified_config.py`: Unified configuration management
- `src/.env.example`: Complete configuration template
- `llama_deploy.yml`: Production deployment configuration

**Supporting Components:**
- `src/citation.py`: Advanced citation processing and formatting
- `src/query.py`: Enhanced retrieval with hybrid search
- `src/health_monitor.py`: Production monitoring and health checks
- `src/resource_manager.py`: Resource management and optimization

**Data & Storage:**
- `ui/data/`: Documents for indexing (supports PDF, text, images)
- `src/storage/`: Persistent vector and graph storage
- `tests/`: Comprehensive test suite (2000+ tests, >90% coverage)

## API Endpoints

### Core RAG Endpoints
- **Create task**: `POST http://localhost:4501/deployments/chat/tasks/create`
- **Stream events**: `GET http://localhost:4501/deployments/chat/tasks/{task_id}/events`
- **Task status**: `GET http://localhost:4501/deployments/chat/tasks/{task_id}`

### Enhanced Features
- **Query with profile**: `POST http://localhost:4501/deployments/chat/tasks/create`
  ```json
  {
    "input": "{\"user_msg\":\"Your query\",\"profile\":\"high_accuracy\"}",
    "service_id": "workflow"
  }
  ```

### System Monitoring
- **System health**: `GET http://localhost:4501/health`
- **Performance metrics**: `GET http://localhost:4501/metrics`
- **Cache statistics**: `GET http://localhost:4501/cache/stats`
- **Security status**: `GET http://localhost:4501/security/status`

### User Interfaces
- **Enhanced UI**: `http://localhost:4501/deployments/chat/ui`
- **API Documentation**: `http://localhost:4501/docs`
- **Health Monitor**: `http://localhost:8001/health` (if health_monitor service enabled)

### Security Considerations
- All endpoints include comprehensive input validation
- Rate limiting applied to prevent abuse
- Security headers and CORS protection enabled
- Audit logging for all API access

## Development Workflow

### Initial Setup
1. **Environment Configuration**: Copy `src/.env.example` to `src/.env` and configure your settings
2. **API Key Validation**: Ensure valid OpenAI API key (system validates format and rejects placeholders)
3. **Optional Dependencies**: Install multimodal (`uv sync --extra multimodal`) or TTS features as needed
4. **Redis Setup**: For caching, ensure Redis is running locally or configure REDIS_CACHE_URL

### Development Process
1. **Document Updates**: Always run `uv run generate` after adding new documents to `ui/data/`
2. **Index Validation**: Verify index exists in `src/storage/` before starting workflows
3. **Testing**: Run comprehensive test suite before commits: `uv run pytest`
4. **Security**: Validate security fixes: `uv run python test_security_fixes.py`
5. **Performance**: Benchmark changes: `uv run python performance-test-suite.py`

### Configuration Management
- **Environment Loading**: System loads from `src/.env` with fallback to project root
- **Profile Selection**: Choose performance profile based on use case requirements
- **Feature Toggles**: Enable/disable SOTA features via environment variables
- **Security Validation**: All configurations undergo security validation on startup

### Production Considerations
- **Performance Profiles**: Use `high_accuracy` for critical applications, `balanced` for general use
- **Caching**: Enable semantic caching for production to reduce costs by 31%+
- **Monitoring**: Enable health monitoring and error alerting
- **Security**: Review security configuration and enable verification for production
- **Resource Management**: Monitor system resources and configure appropriate limits

### Debugging and Troubleshooting
- **Logs**: Configure LOG_LEVEL in environment for appropriate verbosity
- **Health Checks**: Use `/health` and `/metrics` endpoints for system diagnostics
- **Test Isolation**: Use individual test categories for targeted debugging
- **Performance Analysis**: Enable profiling in development for performance issues

### UI Development
- **Development Server**: `cd ui && npm run dev` for hot-reload development
- **Component Testing**: Test individual UI components with the enhanced features
- **API Integration**: Validate UI against all API endpoints and performance profiles
- **Accessibility**: Follow accessibility guidelines implemented in UI components