# 🚀 Pinecone + HuggingFace Migration Guide

This guide walks you through migrating from the legacy MemoryVectorStore to a cost-optimized **Pinecone + HuggingFace** setup for zero monthly costs.

## 🎯 **Migration Benefits**

- **💰 Zero Cost**: Pinecone Serverless free tier + HuggingFace free tier = $0/month
- **🔧 Native Deletion**: Proper vector cleanup (no more rebuilding!)
- **📊 Better Isolation**: Namespace-based user separation
- **⚡ Improved Performance**: Distributed vector storage with auto-scaling
- **🛡️ Production Ready**: Battle-tested infrastructure

## 📋 **Prerequisites**

### 1. Create Pinecone Account
1. Go to [pinecone.io](https://www.pinecone.io/)
2. Sign up for free account
3. Create a new project
4. Get your API key from the dashboard

### 2. Get HuggingFace API Key (Optional but Recommended)
1. Go to [huggingface.co](https://huggingface.co/)
2. Sign up/login
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions

> **Note**: HuggingFace API key is optional. Without it, you'll use the public inference endpoint with rate limits.

## 🔧 **Environment Setup**

Add these environment variables to your `.env.local`:

```bash
# Pinecone Configuration (REQUIRED)
PINECONE_API_KEY="your-pinecone-api-key-here"
PINECONE_INDEX_NAME="rag-documents"

# HuggingFace Configuration (OPTIONAL)
HUGGINGFACE_API_KEY="your-huggingface-api-key-here"

# Migration Control
USE_PINECONE_RAG=true

# Keep existing variables:
# POSTGRES_URL, AUTH_SECRET, GOOGLE_GENERATIVE_AI_API_KEY, etc.
```

## 🏃‍♂️ **Migration Steps**

### Step 1: Test the New System

```bash
npm run rag:test
```

This will test:
- ✅ HuggingFace embeddings generation
- ✅ Pinecone connection and index creation
- ✅ End-to-end pipeline

### Step 2: Run the Migration

```bash
npm run rag:migrate
```

This will:
1. **Extract** all documents from MemoryVectorStore
2. **Convert** to HuggingFace embeddings (384 dimensions)
3. **Upload** to Pinecone with user namespaces
4. **Validate** migration success

### Step 3: Verify Migration

The migration script will show:
```
🚀 RAG Migration: MemoryVectorStore → Pinecone + HuggingFace
================================================

1. Testing systems...
   Old system: ✅
   New system: ✅

2. Migrating data...
   Users migrated: 3
   Documents migrated: 12
   Chunks migrated: 156

3. Migration Results:
   Success: ✅

4. Validating migration...
   Validation: ✅
   Old system docs: 12
   New system vectors: 156

✅ Migration completed!
```

### Step 4: Test Your Application

1. Start your development server: `npm run dev`
2. Upload a test document
3. Ask questions about the document
4. Verify search results are working

## 🎛️ **Configuration Options**

### RAG Settings
```bash
RAG_CHUNK_SIZE=1000              # Text chunk size
RAG_CHUNK_OVERLAP=200            # Overlap between chunks
RAG_MAX_DOCS=5                   # Max documents returned
RAG_SIMILARITY_THRESHOLD=0.7     # Similarity threshold
```

### Embedding Model Options
The system uses `sentence-transformers/all-MiniLM-L6-v2` by default (384 dimensions).

You can change this in `lib/huggingface-embeddings.ts`:
```typescript
// Other options:
// 'sentence-transformers/all-mpnet-base-v2'  (768 dim, better quality)
// 'sentence-transformers/all-distilroberta-v1' (768 dim)
```

## 📊 **Free Tier Limits**

### Pinecone Serverless (Free Tier)
- **Storage**: 2GB (~300K vectors with 384 dimensions)
- **Operations**: 2M writes + 1M reads per month
- **Projects**: 1 project, 5 indexes
- **Perfect for**: <10 users, experimental projects

### HuggingFace Inference API
- **Rate Limits**: ~1000 requests/hour (public endpoint)
- **With API Key**: Higher rate limits, better reliability
- **Models**: All sentence-transformer models available

## 🔧 **Troubleshooting**

### Migration Issues

**"New system is not ready"**
- Check `PINECONE_API_KEY` is set correctly
- Verify Pinecone account has free tier available
- Wait 1-2 minutes for index creation

**"HuggingFace API rate limit"**
- Add `HUGGINGFACE_API_KEY` for higher limits
- Reduce batch size in migration script
- Wait and retry

**"Old system has no documents"**
- This is normal if you haven't uploaded documents yet
- Skip migration and start using new system directly

### Runtime Issues

**"Embedding generation failed"**
```bash
# Test embeddings directly:
npm run rag:test
```

**"Pinecone connection timeout"**
- Check your internet connection
- Verify API key is correct
- Try different region in `pinecone-rag-core.ts`

## 🧹 **Cleanup (Optional)**

After successful migration, you can remove OpenAI dependencies:

```bash
npm uninstall @langchain/openai
```

Remove from your `.env.local`:
```bash
# OPENAI_API_KEY="..." # No longer needed
```

## 🚀 **Next Steps**

1. **Monitor Usage**: Check Pinecone dashboard for usage metrics
2. **Optimize Performance**: Adjust similarity thresholds based on results  
3. **Scale Up**: Upgrade to paid tiers when you exceed free limits
4. **Advanced Features**: Implement metadata filtering, hybrid search

## 📈 **Performance Comparison**

| Feature | MemoryVectorStore | Pinecone + HuggingFace |
|---------|-------------------|------------------------|
| **Cost** | $20+/month (OpenAI) | $0/month (Free tiers) |
| **Deletion** | Rebuild entire store | Native deletion |
| **Persistence** | In-memory only | Persistent storage |
| **Scalability** | Limited by RAM | Auto-scaling |
| **User Isolation** | Map-based | Namespaces |
| **Search Quality** | High (OpenAI) | Good (384-dim) |

## 🆘 **Support**

If you encounter issues:

1. Check the [Pinecone documentation](https://docs.pinecone.io/guides/get-started/overview)
2. Review [HuggingFace Inference API docs](https://huggingface.co/docs/api-inference/index)
3. Run `npm run rag:test` for diagnostics
4. Check application logs for detailed error messages

---

**🎉 Congratulations!** You've successfully migrated to a cost-effective, production-ready RAG system!
