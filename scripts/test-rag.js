import { config } from 'dotenv';
import { getPineconeRAGCore } from '../lib/pinecone-rag-core.js';

// Load environment variables
config({ path: '.env.local' });

async function testRAG() {
  console.log('🧪 Testing RAG pipeline...');
  console.log('📋 Environment check:');
  console.log('  PINECONE_API_KEY:', process.env.PINECONE_API_KEY ? '✅ SET' : '❌ MISSING');
  console.log('  HUGGINGFACE_API_KEY:', process.env.HUGGINGFACE_API_KEY ? '✅ SET' : '❌ MISSING');
  console.log('  POSTGRES_URL:', process.env.POSTGRES_URL ? '✅ SET' : '❌ MISSING');
  console.log('');

  try {
    const ragCore = getPineconeRAGCore();
    const result = await ragCore.testPipeline();
    
    console.log('🎯 RAG Test Results:');
    console.log(JSON.stringify(result, null, 2));
    
    if (result.success) {
      console.log('✅ RAG pipeline is working correctly!');
    } else {
      console.log('⚠️  RAG pipeline has some issues - check the details above');
    }
  } catch (error) {
    console.error('❌ RAG Test Error:', error.message);
    process.exit(1);
  }
}

testRAG();
