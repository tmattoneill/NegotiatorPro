# NegotiatorPro - React + FastAPI Proof of Concept

This POC demonstrates the migration from Gradio to React + FastAPI architecture.

## Architecture

```
Frontend (React + TypeScript)     Backend (FastAPI + Python)
Port 5173                         Port 8000
     │                                  │
     │  HTTP/REST API                   │
     └──────────────────────────────────┘
            /api/chat
            /api/auth/login
            /api/health
```

## Quick Start

### Terminal 1: Start Backend (FastAPI)

```bash
./run-api.sh
```

The API will be available at:
- API: http://localhost:8000
- Auto-generated docs: http://localhost:8000/api/docs
- Alternative docs: http://localhost:8000/api/redoc

### Terminal 2: Start Frontend (React)

```bash
./run-frontend.sh
```

The frontend will be available at:
- Frontend: http://localhost:5173

## Testing the POC

### 1. Test Backend Directly

Using curl:
```bash
# Health check
curl http://localhost:8000/api/health

# Login (admin password: admin123)
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password": "admin123"}'

# Chat (without auth for POC)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I negotiate a salary increase?",
    "use_premium_model": false,
    "use_preprocessing": true
  }'
```

Or visit the interactive API docs at http://localhost:8000/api/docs

### 2. Test Frontend

1. Open http://localhost:5173 in your browser
2. Type a negotiation question in the input box
3. Click "Send" or press Enter
4. Watch the AI response appear in real-time

### 3. Test Settings

- **Premium Model toggle**: Switch between default and premium LLM models
- **Text Preprocessing toggle**: Enable/disable text preprocessing
- **Partner Info**: Add context about your negotiation partner

## Features Implemented in POC

✅ **Backend (FastAPI)**
- REST API endpoints for chat and authentication
- Integration with existing RAG system
- JWT authentication middleware
- CORS configuration for React dev server
- Auto-generated API documentation
- Pydantic models for request/response validation

✅ **Frontend (React + TypeScript)**
- Modern React 18 with TypeScript
- Zustand state management
- Axios for API calls
- Clean, minimal chat UI
- Real-time message display
- Loading states
- Error handling
- Settings configuration (premium model, preprocessing, partner info)

## What's NOT in POC (Future Phases)

❌ WebSocket streaming (currently polling)
❌ Session management (save/load conversations)
❌ Admin panel UI
❌ Document upload UI
❌ Usage statistics UI
❌ Production build/deployment
❌ Authentication UI (login page)
❌ Mobile responsive design
❌ Dark mode
❌ Keyboard shortcuts

## File Structure

```
NegotiatorPro/
├── backend/
│   ├── api/                    # NEW: FastAPI application
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── routes/            # API endpoints
│   │   │   ├── auth.py        # Login endpoint
│   │   │   ├── chat.py        # Chat endpoint
│   │   │   └── health.py      # Health check
│   │   ├── models/            # Pydantic models
│   │   │   ├── requests.py    # Request schemas
│   │   │   └── responses.py   # Response schemas
│   │   └── middleware/        # Auth, CORS, etc.
│   │       └── auth.py        # JWT authentication
│   ├── rag_engine.py          # UNCHANGED: Core RAG logic
│   ├── llm_backend_config.py  # UNCHANGED: LLM management
│   └── ...                    # Other backend modules
│
├── frontend/                   # NEW: React application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── ChatContainer.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── services/          # API client
│   │   │   └── api.ts
│   │   ├── store/             # State management
│   │   │   └── chatStore.ts
│   │   ├── types/             # TypeScript types
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts         # Vite config with proxy
│
├── run-api.sh                  # Start FastAPI backend
├── run-frontend.sh             # Start React frontend
└── POC-README.md              # This file
```

## API Endpoints

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T12:00:00Z",
  "version": "1.0.0-poc"
}
```

### `POST /api/auth/login`
Admin login endpoint.

**Request:**
```json
{
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### `POST /api/chat`
Process a negotiation question.

**Request:**
```json
{
  "question": "How do I negotiate a salary increase?",
  "partner_info": "My manager is budget-conscious",
  "use_premium_model": false,
  "use_preprocessing": true
}
```

**Response:**
```json
{
  "answer": "Based on expert negotiation principles...",
  "model_used": "openai/gpt-4o-mini",
  "tokens_used": 1234,
  "processing_time": 2.5
}
```

## Technologies Used

### Backend
- **FastAPI** 0.121+ - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **python-jose** - JWT tokens
- **passlib** - Password hashing
- **LangChain** - RAG framework (existing)

### Frontend
- **React** 18 - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Zustand** - State management
- **Axios** - HTTP client
- **TanStack Query** - Data fetching (not yet used in POC)

## Environment Variables

Required in `.env`:
```bash
# Required for default setup
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for JWT auth (will use default if not set)
JWT_SECRET_KEY=your_secret_key_here

# Optional - for Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional - for Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## Next Steps

After validating the POC, the next phases would be:

1. **Phase 2**: Add session management (save/load conversations)
2. **Phase 3**: Implement admin panel UI
3. **Phase 4**: Add WebSocket streaming for real-time responses
4. **Phase 5**: Production hardening (security, performance)
5. **Phase 6**: Deploy and migrate from Gradio

## Troubleshooting

### Backend won't start
- Check that port 8000 is not in use: `lsof -i :8000`
- Verify dependencies are installed: `pip list | grep fastapi`
- Check logs for errors

### Frontend won't start
- Check that port 5173 is not in use: `lsof -i :5173`
- Verify dependencies are installed: `cd frontend && npm list`
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`

### CORS errors in browser
- Ensure backend is running on port 8000
- Check that Vite proxy is configured correctly in `vite.config.ts`
- Clear browser cache

### API calls fail
- Open browser DevTools → Network tab
- Check that requests are going to `/api/*`
- Verify backend is responding: `curl http://localhost:8000/api/health`

## Performance Notes

This POC is optimized for demonstration, not production. In production:
- Add response caching
- Implement connection pooling
- Use CDN for static assets
- Enable gzip compression
- Add rate limiting
- Implement request queuing for LLM calls

## Feedback

This POC demonstrates that the React + FastAPI architecture is:
- ✅ **Feasible**: Both systems work together seamlessly
- ✅ **Maintainable**: Clear separation of concerns
- ✅ **Scalable**: Can easily add features
- ✅ **Modern**: Uses current best practices

The migration from Gradio to React is well-architected and ready for full implementation!
