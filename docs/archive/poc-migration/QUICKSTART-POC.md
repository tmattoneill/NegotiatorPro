# 🚀 5-Minute POC Quickstart

Get the React + FastAPI POC running in 5 minutes!

## Prerequisites

```bash
# Check you have the required tools
python --version  # Should be 3.11+
node --version    # Should be 18+
npm --version     # Should be 9+
```

## Step 1: Install Dependencies (2 min)

```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

## Step 2: Set Environment Variables (1 min)

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Optional: Add JWT secret (or use default)
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

## Step 3: Start Backend (30 sec)

**Terminal 1:**
```bash
./run-api.sh
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
=== Starting NegotiatorPro FastAPI Backend ===
```

**Test it:**
Open http://localhost:8000/api/docs in your browser

## Step 4: Start Frontend (30 sec)

**Terminal 2:**
```bash
./run-frontend.sh
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

**Test it:**
Open http://localhost:5173 in your browser

## Step 5: Try It Out! (1 min)

### In the Browser (http://localhost:5173)

1. Type a question: "What is BATNA in negotiation?"
2. Click "Send" or press Enter
3. Watch the AI response appear!

### Try the Settings

- ✅ Toggle "Use Premium Model" (switches to o3-mini)
- ✅ Toggle "Text Preprocessing" (optimizes tokens)
- ✅ Add "Partner Info" (provides context)

## Testing the API Directly

### Health Check
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-15T12:00:00.000000",
  "version": "1.0.0-poc"
}
```

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}'
```

Expected response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I handle a lowball offer?",
    "use_premium_model": false,
    "use_preprocessing": true
  }'
```

Expected response:
```json
{
  "answer": "Based on expert negotiation principles...",
  "model_used": "openai/gpt-4o-mini",
  "tokens_used": null,
  "processing_time": 2.34
}
```

## Interactive API Documentation

Visit http://localhost:8000/api/docs for:
- 📚 Auto-generated API documentation
- 🧪 Interactive endpoint testing
- 📝 Request/response schemas
- 🔍 Try out all endpoints in your browser

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill it if needed
kill -9 $(lsof -t -i:8000)

# Try again
./run-api.sh
```

### Frontend won't start
```bash
# Check if port 5173 is in use
lsof -i :5173

# Kill it if needed
kill -9 $(lsof -t -i:5173)

# Clear node_modules if needed
cd frontend
rm -rf node_modules package-lock.json
npm install

# Try again
cd ..
./run-frontend.sh
```

### CORS errors in browser
- ✅ Ensure backend is running on port 8000
- ✅ Check browser console for errors
- ✅ Verify frontend proxy in `frontend/vite.config.ts`

### API returns authentication error
- ✅ Check `.env` has valid `OPENAI_API_KEY`
- ✅ Ensure no extra spaces in API key
- ✅ Verify key hasn't expired

### "Module not found" errors
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## What to Explore

### 1. Backend API
- 📖 Docs: http://localhost:8000/api/docs
- 🔍 Health: http://localhost:8000/api/health
- 🔐 Auth: POST http://localhost:8000/api/auth/login
- 💬 Chat: POST http://localhost:8000/api/chat

### 2. Frontend Code
```bash
# React components
frontend/src/components/ChatContainer.tsx
frontend/src/components/ChatMessage.tsx
frontend/src/components/ChatInput.tsx

# State management
frontend/src/store/chatStore.ts

# API client
frontend/src/services/api.ts

# TypeScript types
frontend/src/types/index.ts
```

### 3. Backend API Code
```bash
# FastAPI app
backend/api/main.py

# Routes
backend/api/routes/chat.py
backend/api/routes/auth.py

# Models
backend/api/models/requests.py
backend/api/models/responses.py

# Auth
backend/api/middleware/auth.py
```

## Next Steps

### ✅ POC Validated - Now What?

1. **Review Documentation**
   - Read `POC-SUMMARY.md` for validation results
   - Read `MIGRATION-COMPARISON.md` for detailed comparison
   - Read `POC-README.md` for complete guide

2. **Test Features**
   - Try different questions
   - Toggle premium model on/off
   - Add partner context
   - Test preprocessing

3. **Explore Code**
   - See how React components work
   - Check FastAPI route handlers
   - Review Pydantic models
   - Understand state management

4. **Make a Decision**
   - Approve full migration?
   - Set timeline (6-8 weeks)
   - Allocate resources
   - Begin Phase 3 (Feature Parity)

## POC Deliverables Checklist

- ✅ FastAPI backend with 3 endpoints
- ✅ React frontend with chat UI
- ✅ Integration with existing RAG system
- ✅ JWT authentication middleware
- ✅ State management (Zustand)
- ✅ API client (Axios)
- ✅ TypeScript types
- ✅ Auto-generated API docs
- ✅ Startup scripts
- ✅ Comprehensive documentation
- ✅ Working end-to-end flow

## Success Criteria

| Criteria | Status |
|----------|--------|
| Backend starts without errors | ✅ |
| Frontend starts without errors | ✅ |
| Can send chat message | ✅ |
| Receives AI response | ✅ |
| Settings toggles work | ✅ |
| API docs accessible | ✅ |
| Existing backend code unchanged | ✅ |
| Architecture validated | ✅ |

## Questions?

See the detailed documentation:
- `POC-README.md` - Complete usage guide
- `POC-SUMMARY.md` - Validation results
- `MIGRATION-COMPARISON.md` - Gradio vs React comparison
- `MIGRATION-TREE.txt` - File structure

Or check the original plan document at the top of this conversation!

---

**Estimated Time: 5 minutes**
**Difficulty: Easy**
**Status: ✅ READY TO RUN**
