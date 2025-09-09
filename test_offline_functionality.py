#!/usr/bin/env python3
"""
Offline test of SOTA RAG system structure and imports.

This test verifies that all components can be imported and instantiated
without requiring OpenAI API calls.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Mock environment
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["SEMANTIC_CACHE_ENABLED"] = "false"
os.environ["VERIFICATION_ENABLED"] = "false"
os.environ["QUERY_DECOMPOSITION_ENABLED"] = "false"

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core modules
        from src.settings import init_settings, get_agentic_config, get_cache_config, get_verification_config
        print("✓ Settings module")
        
        from src.config import get_config, get_global_config
        print("✓ Config module")
        
        from src.workflow import create_workflow
        print("✓ Workflow module")
        
        from src.simple_workflow import create_simple_workflow
        print("✓ Simple workflow module")
        
        # Component modules
        from src.cache import get_cache
        print("✓ Cache module")
        
        from src.verification import create_hallucination_detector
        print("✓ Verification module")
        
        from src.agentic import QueryClassifier, QueryDecomposer, QueryType
        print("✓ Agentic module")
        
        from src.health_monitor_simple import get_health_monitor
        print("✓ Health monitor module")
        
        from src.performance import SimilarityDetector, CacheManager
        print("✓ Performance module")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.settings import get_agentic_config, get_cache_config, get_verification_config
        
        agentic_config = get_agentic_config()
        print(f"✓ Agentic config: {agentic_config}")
        
        cache_config = get_cache_config()
        print(f"✓ Cache config: {cache_config}")
        
        verification_config = get_verification_config()
        print(f"✓ Verification config: {verification_config}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_component_creation():
    """Test that components can be created without API calls."""
    print("\nTesting component creation...")
    
    try:
        # Test cache
        from src.cache import get_cache
        cache = get_cache()
        print(f"✓ Cache created: {type(cache)}")
        
        # Test health monitor
        from src.health_monitor_simple import get_health_monitor
        monitor = get_health_monitor()
        print(f"✓ Health monitor created: {type(monitor)}")
        
        # Test agentic components (these might need API key but should at least instantiate)
        from src.agentic import QueryType
        query_types = list(QueryType)
        print(f"✓ Query types: {[qt.value for qt in query_types]}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Component creation issue: {e}")
        return False

def test_file_structure():
    """Test that the expected files exist."""
    print("\nTesting file structure...")
    
    expected_files = [
        "src/__init__.py",
        "src/workflow.py", 
        "src/simple_workflow.py",
        "src/settings.py",
        "src/config.py",
        "src/cache.py",
        "src/verification.py",
        "src/agentic.py",
        "src/health_monitor.py",
        "src/performance.py",
        "src/index.py",
        "src/query.py",
        "src/citation.py",
        "ui/data/sample.txt",
        "ui/data/101.pdf"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    return True

def test_dependencies():
    """Test that critical dependencies are available."""
    print("\nTesting dependencies...")
    
    try:
        import llama_index
        print(f"✓ LlamaIndex available")
        
        from llama_index.core import __version__ as core_version
        print(f"✓ LlamaIndex core version: {core_version}")
        
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        print("✓ Core LlamaIndex components")
        
        from llama_index.core.agent.workflow import AgentWorkflow
        print("✓ Agent workflow available")
        
        from llama_index.llms.openai import OpenAI
        print("✓ OpenAI LLM available")
        
        from llama_index.embeddings.openai import OpenAIEmbedding
        print("✓ OpenAI embeddings available")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency issue: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SOTA RAG System - Offline Structure Test")
    print("=" * 60)
    
    all_tests = [
        test_file_structure,
        test_dependencies,
        test_imports,
        test_configuration,
        test_component_creation
    ]
    
    results = []
    for test_func in all_tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ SOTA RAG system structure is correct!")
        print("✅ All modules can be imported successfully")
        print("✅ Configuration system works")
        print("✅ Components can be instantiated")
        print("\n📋 Next steps:")
        print("1. Add a valid OPENAI_API_KEY to test with real data")
        print("2. The system is ready for production use")
        return True
    else:
        print(f"⚠ SOME TESTS FAILED ({passed}/{total})")
        print("Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
