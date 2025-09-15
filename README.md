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

### Step 3: Vercel Postgres Setup

Based on the [Vercel Postgres documentation](https://vercel.com/docs/postgres), set up your database through the Vercel Marketplace:

#### 3.1 Create Vercel Postgres Database

1. **Login to Vercel**: Go to [vercel.com](https://vercel.com) and sign in
2. **Navigate to Storage**: In your project dashboard, click the **Storage** tab
3. **Create Database**: 
   - Click **Create Database**
   - Select **Postgres** from the marketplace
   - Choose a database name (e.g., `rag-chatbot-db`)
   - Select your preferred region (closest to your users)
   - Click **Create**

#### 3.2 Connect Database to Project

1. **Connect Project**: After database creation, click **Connect Project**
2. **Select Project**: Choose your RAG chatbot project from the dropdown
3. **Connect**: Click **Connect** - Vercel automatically injects environment variables
4. **Verify Connection**: Check that `POSTGRES_URL` appears in your project's environment variables

#### 3.3 Pull Environment Variables Locally

```bash
# Install Vercel CLI (if not already installed)
npm install -g vercel

# Login to Vercel
vercel login

# Link your local project (if not already linked)
vercel link

# Pull environment variables to local development
vercel env pull .env.local
```

This will automatically create/update your `.env.local` file with the Vercel Postgres connection string.

### Step 4: Environment Configuration

After pulling Vercel environment variables, your `.env.local` should contain the Vercel Postgres connection. Add the remaining required variables:

```bash
# Database (Automatically added by Vercel)
POSTGRES_URL="vercel-postgres-connection-string-from-vercel-env-pull"

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
# Run database migrations to set up tables in Vercel Postgres
npm run db:migrate

# Optional: Push schema changes directly to Vercel Postgres
npm run db:push
```

**Note**: The database operations will run against your Vercel Postgres instance using the `POSTGRES_URL` environment variable.

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
| **Vercel Postgres** | 60 compute hours/month, 256MB storage | Ideal for development and small projects |

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
2. **Database Integration**: Vercel Postgres is already connected and environment variables are automatically injected
3. **Additional Environment Variables**: Add the remaining variables (API keys, secrets) to Vercel project settings
4. **Deploy**: Automatic deployment on push with seamless database connectivity

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
- ✅ `POSTGRES_URL` - Automatically injected by Vercel Postgres
- ✅ `AUTH_SECRET` - Session encryption (add manually)
- ✅ `GOOGLE_GENERATIVE_AI_API_KEY` - Gemini API (add manually)
- ✅ `PINECONE_API_KEY` - Vector database (add manually)
- ✅ `PINECONE_INDEX_NAME` - Index name (add manually)
- ⚠️ `HUGGINGFACE_API_KEY` - Optional but recommended (add manually)
- ⚠️ `NODE_ENV=production` - Production mode (automatically set)

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
# Verify Vercel Postgres connection
vercel env pull .env.local

# Check if POSTGRES_URL is properly set
echo $POSTGRES_URL

# Test database migration
npm run db:push

# If still failing, check Vercel dashboard for database status
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

# Verify Vercel Postgres connection
vercel env pull .env.local
npm run db:push --dry-run

# Check Vercel deployment logs
vercel logs

# Check local development logs
tail -f .next/server.log
```

#### Cost Monitoring
- **Pinecone**: Monitor vector count in dashboard
- **HuggingFace**: Check request usage
- **Gemini**: Monitor API usage in Google Cloud Console
- **Vercel Postgres**: Monitor compute hours and storage in Vercel dashboard

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
3. **Vercel Postgres**: $20/month for Pro plan with more compute hours and storage

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