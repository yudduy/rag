/**
 * Debug script to test RAG functionality
 * Run with: node scripts/debug-rag.js
 */

const { getPineconeRAGCore } = require('../lib/pinecone-rag-core');

async function testRAGSystem() {
  console.log('🧪 Testing RAG System...\n');

  try {
    const ragCore = getPineconeRAGCore();
    
    // Test 1: Check RAG status
    console.log('1. Checking RAG status...');
    const status = await ragCore.getStatus();
    console.log('RAG Status:', status);
    
    // Test 2: Test with a dummy user ID (you'll need to replace this with a real user ID)
    const testUserId = 'test-user-123'; // Replace with actual user ID from your database
    
    console.log('\n2. Testing document retrieval...');
    const docs = await ragCore.getUserDocuments(testUserId);
    console.log(`Found ${docs.length} documents for user ${testUserId}`);
    
    if (docs.length > 0) {
      console.log('Documents:');
      docs.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. ${doc.filename} (${doc.chunks} chunks)`);
      });
      
      // Test 3: Try retrieving with a simple query
      console.log('\n3. Testing document search...');
      const searchResults = await ragCore.retrieveDocuments(
        'resume experience skills',
        testUserId,
        { maxDocs: 3, threshold: 0.5 }
      );
      
      console.log(`Search returned ${searchResults.length} results`);
      searchResults.forEach((result, idx) => {
        console.log(`  Result ${idx + 1}: ${result.source} (score: ${result.relevance_score})`);
        console.log(`    Content preview: ${result.content.substring(0, 100)}...`);
      });
    } else {
      console.log('❌ No documents found. This could mean:');
      console.log('   - No documents have been uploaded yet');
      console.log('   - Documents failed to index properly');
      console.log('   - Wrong user ID being used');
    }
    
  } catch (error) {
    console.error('❌ RAG system error:', error);
    console.log('\nPossible issues:');
    console.log('- Pinecone API key not set or invalid');
    console.log('- Pinecone index not created or wrong name');
    console.log('- HuggingFace embeddings not working');
    console.log('- Network connectivity issues');
  }
}

// Check environment variables
function checkEnvironment() {
  console.log('🔧 Environment Check:\n');
  
  const requiredVars = [
    'PINECONE_API_KEY',
    'PINECONE_INDEX_NAME',
    'HUGGINGFACE_API_KEY'
  ];
  
  requiredVars.forEach(varName => {
    const value = process.env[varName];
    if (value) {
      console.log(`✅ ${varName}: ${value.substring(0, 10)}...`);
    } else {
      console.log(`❌ ${varName}: Not set`);
    }
  });
  
  console.log('');
}

// Run tests
checkEnvironment();
testRAGSystem().then(() => {
  console.log('\n✨ RAG debugging complete!');
}).catch(error => {
  console.error('\n💥 Debug script failed:', error);
});
