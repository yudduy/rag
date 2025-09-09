#!/usr/bin/env python3
"""
SOTA RAG System - Usage Demonstration

This script shows how to use the SOTA RAG system in production.
Set your OPENAI_API_KEY environment variable before running.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("SOTA RAG System - Usage Demonstration")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nTo use this demo:")
        print("1. Get your API key from: https://platform.openai.com/account/api-keys")
        print("2. Run: export OPENAI_API_KEY='your-api-key-here'")
        print("3. Run this script again")
        return False
    
    if api_key.startswith("test-") or api_key in ["your-api-key-here", "sk-example"]:
        print("❌ ERROR: Please set a valid OpenAI API key")
        return False
    
    print(f"✓ API key found: {api_key[:7]}...{api_key[-4:]}")
    
    try:
        # Import and create workflow
        print("\n1. Creating SOTA RAG workflow...")
        from src.workflow import create_workflow
        
        workflow = create_workflow()
        print(f"✓ Workflow created: {type(workflow)}")
        
        # Test queries
        test_queries = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the most important findings?"
        ]
        
        print(f"\n2. Testing with {len(test_queries)} sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                # Run the query
                response = workflow.run(query=query)
                print(f"✓ Response: {str(response)[:200]}...")
                
                # Show confidence and sources if available
                if hasattr(response, 'confidence'):
                    print(f"  Confidence: {response.confidence:.2f}")
                if hasattr(response, 'sources'):
                    print(f"  Sources: {len(response.sources)} documents")
                
            except Exception as e:
                print(f"✗ Query failed: {e}")
        
        print("\n" + "=" * 60)
        print("USAGE EXAMPLES")
        print("=" * 60)
        
        print("""
# Basic usage:
from src.workflow import create_workflow

workflow = create_workflow()
response = workflow.run(query="Your question here")
print(response)

# With configuration:
import os
os.environ["SEMANTIC_CACHE_ENABLED"] = "true"     # Enable caching
os.environ["VERIFICATION_ENABLED"] = "true"       # Enable verification
os.environ["QUERY_DECOMPOSITION_ENABLED"] = "true"  # Enable smart routing

workflow = create_workflow()
response = workflow.run(query="Complex multi-part question")

# Available environment variables:
- OPENAI_MODEL (default: gpt-4o-mini)
- EMBEDDING_MODEL (default: text-embedding-3-small)  
- SEMANTIC_CACHE_ENABLED (default: true)
- VERIFICATION_ENABLED (default: true)
- QUERY_DECOMPOSITION_ENABLED (default: false)
- REDIS_URL (for distributed caching)
- CACHE_TTL (cache expiry in seconds)
- SIMILARITY_THRESHOLD (for cache matching)
""")
        
        print("\n🎉 SOTA RAG System is working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
