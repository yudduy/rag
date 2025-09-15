# Contributing to RAG-Enhanced Chatbot

Thank you for your interest in contributing to the RAG-Enhanced Chatbot! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm/pnpm
- PostgreSQL database
- OpenAI API key (for embeddings)
- Google AI API key (for chat)

### Local Development Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag
```

2. **Install dependencies**:
```bash
npm install
```

3. **Set up environment variables**:
```bash
cp .env.example .env.local
# Edit .env.local with your API keys and database URL
```

4. **Run database migrations**:
```bash
npm run db:migrate
```

5. **Start the development server**:
```bash
npm run dev
```

## 🏗️ Project Structure

```
├── app/                    # Next.js App Router pages and API routes
│   ├── (auth)/            # Authentication pages and API
│   └── (chat)/            # Chat interface and document APIs
├── components/            # React components
│   ├── custom/           # Custom application components
│   └── ui/               # Reusable UI components (shadcn/ui)
├── lib/                  # Core libraries and utilities
│   ├── rag-core.ts       # RAG functionality
│   └── document-processor.ts # Document processing
├── db/                   # Database schema and queries
│   ├── schema.ts         # Drizzle schema definitions
│   └── queries.ts        # Database query functions
└── public/               # Static assets
```

## 🎯 Development Guidelines

### Code Style
- **TypeScript**: Use strict TypeScript for all new code
- **ESLint**: Follow the configured ESLint rules
- **Prettier**: Use Prettier for code formatting
- **Import Order**: Follow the established import organization

### Component Guidelines
- **Server Components**: Use RSC by default, mark with "use client" only when needed
- **Type Safety**: All props and functions should be properly typed
- **Error Handling**: Implement proper error boundaries and error states
- **Accessibility**: Follow WCAG guidelines for UI components

### API Development
- **Authentication**: All API routes should check authentication
- **Validation**: Use Zod for input validation
- **Error Handling**: Return consistent error responses
- **Rate Limiting**: Consider rate limiting for resource-intensive endpoints

### RAG Development
- **Performance**: Optimize chunk sizes and similarity thresholds
- **User Isolation**: Ensure documents are properly isolated by user
- **Error Recovery**: Handle embedding and retrieval failures gracefully
- **Testing**: Test with various document formats and sizes

## 🧪 Testing

### Running Tests
```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Build test
npm run build
```

### Testing Guidelines
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test API endpoints and database operations
- **E2E Tests**: Test complete user workflows
- **Performance Tests**: Test with large documents and concurrent users

## 📦 Adding Dependencies

### Before Adding Dependencies
1. Check if functionality can be achieved with existing dependencies
2. Consider bundle size impact
3. Verify license compatibility
4. Check maintenance status and community support

### Dependency Categories
- **Core Dependencies**: Essential for application functionality
- **Dev Dependencies**: Development tools and testing utilities
- **Peer Dependencies**: Should be minimal and well-justified

## 🔄 Pull Request Process

### Before Submitting
1. **Branch Naming**: Use descriptive branch names (e.g., `feat/document-batch-upload`, `fix/memory-leak-vector-store`)
2. **Code Quality**: Ensure all linting and type checking passes
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update relevant documentation

### PR Guidelines
1. **Clear Title**: Use conventional commit format
2. **Detailed Description**: Explain what changes were made and why
3. **Screenshots**: Include screenshots for UI changes
4. **Breaking Changes**: Clearly document any breaking changes
5. **Migration Guide**: Provide migration instructions if needed

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
**Scopes**: `rag`, `ui`, `api`, `db`, `docs`, `config`

**Examples**:
- `feat(rag): add query expansion for better document retrieval`
- `fix(ui): resolve document upload progress indicator`
- `docs: update deployment guide with environment variables`

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues for duplicates
2. Verify the bug with the latest version
3. Test in different browsers/environments if applicable

### Bug Report Template
```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., macOS 14.0]
- Browser: [e.g., Chrome 120.0]
- Node.js: [e.g., 18.17.0]
- App Version: [e.g., 1.0.0]

## Additional Context
Screenshots, logs, or other relevant information.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
A clear description of the feature you'd like to see.

## Use Case
Describe the problem this feature would solve.

## Proposed Solution
Your ideas for how this could be implemented.

## Alternatives Considered
Other solutions you've considered.

## Additional Context
Any other relevant information.
```

## 🔒 Security

### Reporting Security Issues
- **Do NOT** create public issues for security vulnerabilities
- Email security issues to [security@yourproject.com]
- Include detailed steps to reproduce
- Allow reasonable time for response before disclosure

### Security Guidelines
- **Input Validation**: Always validate and sanitize user inputs
- **Authentication**: Verify user permissions for all operations
- **Data Isolation**: Ensure users can only access their own data
- **API Security**: Implement proper rate limiting and error handling

## 📚 Documentation

### Documentation Types
- **Code Comments**: Explain complex logic and business rules
- **API Documentation**: Document all endpoints with examples
- **User Guides**: Step-by-step instructions for end users
- **Architecture Docs**: High-level system design and decisions

### Writing Guidelines
- **Clarity**: Use clear, concise language
- **Examples**: Include practical examples
- **Up-to-date**: Keep documentation synchronized with code
- **Accessibility**: Use proper markdown formatting

## 🤝 Community Guidelines

### Code of Conduct
- **Be Respectful**: Treat all contributors with respect
- **Be Inclusive**: Welcome contributors from all backgrounds
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Remember that everyone is learning

### Getting Help
- **GitHub Discussions**: For general questions and ideas
- **Issues**: For bug reports and feature requests
- **Discord/Slack**: For real-time community chat (if available)

## 🏆 Recognition

Contributors who make significant contributions will be:
- Added to the contributors list in README.md
- Mentioned in release notes
- Invited to join the core contributor team (for ongoing contributors)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache License 2.0).

---

Thank you for contributing to making RAG-Enhanced Chatbot better! 🎉
