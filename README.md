# 🚀 Zero-Cost RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with **Next.js**, **Pinecone**, and **HuggingFace**. This system provides intelligent document-based conversations with **zero monthly costs** using free tier services.

## ✨ Features

- **📄 Document Intelligence** - Upload and chat with PDF, DOCX, MD, and TXT files
- **💰 Zero Cost Operation** - Pinecone Serverless + HuggingFace free tiers
- **🔒 Multi-User Support** - Secure namespace-based user isolation
- **⚡ Real-time Chat** - Streaming responses with document context
- **🎯 Smart Search** - Semantic document retrieval with relevance scoring
- **🛡️ Production Ready** - Authentication, error handling, and monitoring
- **📱 Modern UI** - Responsive design with drag-and-drop file uploads
- **🔧 Developer Friendly** - TypeScript, comprehensive error handling

## 🏗️ Architecture

```mermaid
graph TD
    A[User] --> B[Next.js Frontend]
    B --> C[API Routes]
    C --> D[Authentication]
    D --> E[Document Processing]
    E --> F[HuggingFace Embeddings]
    F --> G[Pinecone Vector Store]
    G --> H[RAG Context]
    H --> I[Gemini AI]
    I --> J[Streaming Response]
```

## 🚀 Quick Start

### Prerequisites

1. **Node.js 18+** and **npm**
2. **Pinecone Account** (free tier)
3. **HuggingFace Account** (optional, for higher rate limits)
4. **Google AI API Key** (for Gemini)
5. **PostgreSQL Database** (local or cloud)

### Step 1: Clone and Install

```bash
# Clone the repository
git clone <your-repository-url>
cd rag

# Install dependencies
npm install
```

### Step 2: Set Up Free Tier Accounts

#### 2.1 Pinecone Setup (Required)

1. **Create Account**: Go to [pinecone.io](https://www.pinecone.io/) and sign up
2. **Create Project**: Create a new project in the dashboard
3. **Get API Key**: 
   - Go to "API Keys" in the left sidebar
   - Copy your API key
   - Note: The system will auto-create your index on first use

#### 2.2 HuggingFace Setup (Optional but Recommended)

1. **Create Account**: Go to [huggingface.co](https://huggingface.co/) and sign up
2. **Generate Token**:
   - Go to Settings → Access Tokens
   - Create a new token with "Read" permissions
   - Copy the token

#### 2.3 Google AI Setup (Required)

1. **Get API Key**: Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Create Key**: Generate a new API key for Gemini

### Step 3: Database Setup

#### Option A: Local PostgreSQL
```bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Create database
createdb rag_chatbot
```

#### Option B: Cloud Database (Recommended)
- **Vercel Postgres**: Free tier with 60 hours compute/month
- **Supabase**: Free tier with 500MB storage
- **Neon**: Free tier with 3GB storage

### Step 4: Environment Configuration

Create `.env.local` in the project root:

```bash
# Database (Required)
POSTGRES_URL="postgresql://username:password@localhost:5432/rag_chatbot"
# OR for cloud databases:
# POSTGRES_URL="your-cloud-database-connection-string"

# Authentication (Required)
AUTH_SECRET="your-random-secret-key-min-32-chars"

# AI Models (Required)
GOOGLE_GENERATIVE_AI_API_KEY="your-google-ai-api-key"

# Pinecone (Required)
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="rag-documents"

# HuggingFace (Optional - for higher rate limits)
HUGGINGFACE_API_KEY="your-huggingface-token"

# RAG Configuration (Optional - uses smart defaults)
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_MAX_DOCS=5
RAG_SIMILARITY_THRESHOLD=0.7
RAG_QUERY_EXPANSION=false
```

### Step 5: Database Migration

```bash
# Run database migrations
npm run db:migrate

# Optional: Push schema changes
npm run db:push
```

### Step 6: Test the RAG System

```bash
# Test Pinecone + HuggingFace integration
npm run rag:test
```

Expected output:
```json
{
  "success": true,
  "embeddings_test": { "success": true, "dimensions": 384 },
  "pinecone_test": { "pinecone_connected": true, "index_name": "rag-documents" }
}
```

### Step 7: Start the Application

```bash
# Development mode
npm run dev

# Production build
npm run build
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📖 How to Use

### 1. User Registration/Login

1. Navigate to the application
2. Click "Register" to create an account
3. Verify your email (if configured)
4. Login with your credentials

### 2. Upload Documents

1. **Click the Document Manager** (📁 icon in the navbar)
2. **Upload Files**:
   - Drag & drop files or click "Choose Files"
   - Supported: PDF, DOCX, MD, TXT (max 10MB each)
   - Files are automatically processed and indexed
3. **View Progress**: See upload status and chunk counts
4. **Manage Documents**: View, delete, or re-upload files

### 3. Chat with Your Documents

1. **Start a Conversation**: Type your question in the chat
2. **Automatic Context**: The system automatically finds relevant documents
3. **Manual Search**: Use the `searchDocuments` tool for specific queries
4. **Source Attribution**: See which documents informed the response

#### Example Conversations:

```
You: "What are the main findings in the research paper?"
AI: Based on your uploaded research paper "AI_Study_2024.pdf", the main findings include...
[Source: AI_Study_2024.pdf]

You: "Search for information about machine learning algorithms"
AI: *searches documents* Found 3 relevant sections about ML algorithms...
```

### 4. Advanced Features

#### Document Search Tool
```
You: "Search my documents for 'quarterly revenue'"
AI: *uses searchDocuments tool* Found 2 documents mentioning quarterly revenue:
1. Q3_Report.pdf - Revenue increased 15%...
2. Annual_Summary.docx - Quarterly breakdown shows...
```

#### Multi-Document Conversations
```
You: "Compare the findings between document A and document B"
AI: *analyzes both documents* Comparing the two documents:
Document A suggests... while Document B indicates...
```

## ⚙️ Configuration Guide

### Free Tier Limits

| Service | Free Tier Limit | Your Usage |
|---------|----------------|------------|
| **Pinecone** | 2GB storage (~300K vectors) | Perfect for <50 documents |
| **HuggingFace** | 1000 requests/hour | Sufficient for personal use |
| **Gemini** | 15 requests/minute | Great for chat interactions |
| **Vercel Postgres** | 60 hours/month | Adequate for development |

### Performance Tuning

#### For Better Accuracy:
```bash
RAG_SIMILARITY_THRESHOLD=0.8  # Higher threshold
RAG_MAX_DOCS=7               # More context
```

#### For Faster Responses:
```bash
RAG_SIMILARITY_THRESHOLD=0.6  # Lower threshold
RAG_MAX_DOCS=3               # Less context
```

#### For Cost Optimization:
```bash
# Don't set HUGGINGFACE_API_KEY to use public endpoint
RAG_CHUNK_SIZE=800           # Smaller chunks
```

### Environment Profiles

#### Development
```bash
NODE_ENV=development
RAG_SIMILARITY_THRESHOLD=0.6
```

#### Production
```bash
NODE_ENV=production
RAG_SIMILARITY_THRESHOLD=0.75
HUGGINGFACE_API_KEY=your-token  # For reliability
```

## 🛠️ Development

### Project Structure

```
rag/
├── app/                    # Next.js App Router
│   ├── (auth)/            # Authentication pages
│   ├── (chat)/            # Chat interface & API
│   └── api/               # API endpoints
├── components/            # React components
│   ├── custom/           # App-specific components
│   ├── flights/          # Demo flight components
│   └── ui/               # Reusable UI components
├── lib/                  # Core libraries
│   ├── pinecone-rag-core.ts    # Pinecone RAG implementation
│   ├── huggingface-embeddings.ts # HF embeddings
│   ├── document-processor.ts    # File processing
│   └── utils.ts          # Utilities
├── db/                   # Database
│   ├── schema.ts         # Drizzle schema
│   ├── queries.ts        # Database queries
│   └── migrate.ts        # Migration runner
└── public/               # Static assets
```

### Adding New Document Types

1. **Update file validation** in `lib/document-processor.ts`:
```typescript
const ALLOWED_TYPES = [
  "text/plain",
  "text/markdown", 
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "your-new-mime-type" // Add here
];
```

2. **Add processing logic** for the new file type
3. **Update frontend validation** in `components/custom/document-manager.tsx`

### Custom Embedding Models

Replace the default model in `lib/huggingface-embeddings.ts`:

```typescript
// Current: all-MiniLM-L6-v2 (384 dimensions)
this.model = 'sentence-transformers/all-MiniLM-L6-v2';

// Alternatives:
// 'sentence-transformers/all-mpnet-base-v2' (768 dim, better quality)
// 'sentence-transformers/all-distilroberta-v1' (768 dim)
```

**Important**: If changing dimensions, update Pinecone index configuration.

### Running Tests

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Database operations
npm run db:generate  # Generate migrations
npm run db:push      # Push schema changes

# RAG system testing
npm run rag:test     # Test embeddings + Pinecone
```

## 🚀 Production Deployment

### Vercel Deployment (Recommended)

1. **Connect Repository**: Import your GitHub repo to Vercel
2. **Environment Variables**: Add all `.env.local` variables to Vercel
3. **Database**: Use Vercel Postgres or external provider
4. **Deploy**: Automatic deployment on push

```bash
# Optional: Deploy via CLI
npm i -g vercel
vercel --prod
```

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables Checklist

Production deployment requires:
- ✅ `POSTGRES_URL` - Database connection
- ✅ `AUTH_SECRET` - Session encryption
- ✅ `GOOGLE_GENERATIVE_AI_API_KEY` - Gemini API
- ✅ `PINECONE_API_KEY` - Vector database
- ✅ `PINECONE_INDEX_NAME` - Index name
- ⚠️ `HUGGINGFACE_API_KEY` - Optional but recommended
- ⚠️ `NODE_ENV=production` - Production mode

## 🔧 Troubleshooting

### Common Issues

#### "Pinecone connection failed"
```bash
# Check API key format
echo $PINECONE_API_KEY

# Test connection
npm run rag:test
```

#### "HuggingFace rate limit exceeded"
```bash
# Add API key for higher limits
HUGGINGFACE_API_KEY=your-token

# Or reduce batch size in code
```

#### "Database connection error"
```bash
# Check connection string
npm run db:push

# Test connection
psql $POSTGRES_URL -c "SELECT 1"
```

#### "Document upload fails"
- Check file size (10MB limit)
- Verify file type is supported
- Check server logs for processing errors

#### "No search results"
- Verify documents are uploaded and indexed
- Lower similarity threshold: `RAG_SIMILARITY_THRESHOLD=0.6`
- Check document content quality

### Performance Optimization

#### Slow Document Upload
1. **Reduce chunk size**: `RAG_CHUNK_SIZE=800`
2. **Check HuggingFace rate limits**
3. **Use HuggingFace API key**

#### Slow Chat Responses
1. **Lower similarity threshold**: `RAG_SIMILARITY_THRESHOLD=0.6`
2. **Reduce max documents**: `RAG_MAX_DOCS=3`
3. **Check Pinecone region** (use `us-east-1`)

### Monitoring

#### System Health
```bash
# Check all services
npm run rag:test

# Database health
npm run db:push --dry-run

# Check logs
tail -f .next/server.log
```

#### Cost Monitoring
- **Pinecone**: Monitor vector count in dashboard
- **HuggingFace**: Check request usage
- **Gemini**: Monitor API usage in Google Cloud Console

## 📊 Usage Analytics

### Free Tier Capacity

With default settings, you can handle:
- **Documents**: ~50-100 typical PDFs (depends on length)
- **Users**: Unlimited (namespace isolation)
- **Queries**: ~1000/hour (HuggingFace limit)
- **Storage**: 2GB vectors + unlimited database

### Scaling Up

When you exceed free tiers:
1. **Pinecone**: $70/month for 20GB storage
2. **HuggingFace**: $9/month for Inference Endpoints
3. **Database**: $20/month for managed PostgreSQL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test: `npm run rag:test`
4. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/rag.git
cd rag

# Install dependencies
npm install

# Set up environment
cp .env.example .env.local
# Edit .env.local with your API keys

# Run tests
npm run type-check
npm run lint
npm run rag:test

# Start development
npm run dev
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [Pinecone](https://www.pinecone.io/) - Vector database
- [HuggingFace](https://huggingface.co/) - Embeddings
- [Google Gemini](https://ai.google.dev/) - Language model
- [Drizzle ORM](https://orm.drizzle.team/) - Database toolkit
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

**🎉 Happy building!** If you have questions or need help, please open an issue or start a discussion.