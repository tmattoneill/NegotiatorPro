# 🧪 POC Testing Checklist

Use this checklist to validate the React + FastAPI POC.

## ✅ Pre-Flight Checks

### Environment Setup
- [ ] Python 3.11+ installed: `python --version`
- [ ] Node 18+ installed: `node --version`
- [ ] Ports 8000 and 5173 are free:
  ```bash
  lsof -i :8000  # Should show nothing
  lsof -i :5173  # Should show nothing
  ```

### Dependencies Installed
- [ ] Backend dependencies:
  ```bash
  pip install -r requirements.txt
  # Should complete without errors
  ```
- [ ] Frontend dependencies:
  ```bash
  cd frontend && npm install
  # Should complete without errors, may have warnings (OK)
  ```

### Environment Variables
- [ ] `.env` file exists in project root
- [ ] Contains valid `OPENAI_API_KEY=sk-...`
- [ ] (Optional) Contains `JWT_SECRET_KEY` (will use default if missing)
- [ ] (Optional) Contains `ANTHROPIC_API_KEY` for Claude models

### File Structure Verification
- [ ] `backend/api/main.py` exists
- [ ] `backend/api/middleware/__init__.py` exists ← **CRITICAL FIX APPLIED**
- [ ] `backend/api/routes/` contains auth.py, chat.py, health.py
- [ ] `frontend/src/` contains components, services, store
- [ ] `run-api.sh` and `run-frontend.sh` are executable

---

## 🚀 Backend Testing (Terminal 1)

### Start Backend
```bash
./run-api.sh
```

**Expected Output:**
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
=== Starting NegotiatorPro FastAPI Backend ===
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

**If failed, check:**
- Port 8000 not in use
- Python dependencies installed
- No import errors in logs

---

### Test 1: Health Check
```bash
curl http://localhost:8000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-16T...",
  "version": "1.0.0-poc"
}
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 2: API Documentation
Open in browser: http://localhost:8000/api/docs

**Expected:**
- Interactive Swagger UI loads
- Shows 3 endpoints: /api/health, /api/auth/login, /api/chat
- Each endpoint has request/response schemas

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 3: Login Endpoint
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}'
```

**Expected Response:**
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 4: Chat Endpoint (Basic)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is BATNA?",
    "use_premium_model": false,
    "use_preprocessing": true
  }'
```

**Expected Response:**
```json
{
  "answer": "BATNA stands for...",  // Long negotiation advice
  "model_used": "openai/gpt-4o-mini",
  "tokens_used": null,
  "processing_time": 2.5  // Approximate
}
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

**Common Errors:**
- `401 Unauthorized`: Invalid or missing `OPENAI_API_KEY` in `.env`
- `500 Internal Server Error`: Check server logs for details
- Timeout: API call to OpenAI may be slow, wait 30-60 seconds

---

### Test 5: Chat with Partner Info
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I negotiate salary?",
    "partner_info": "My manager is budget-conscious but values my work",
    "use_premium_model": false,
    "use_preprocessing": true
  }'
```

**Expected:**
- Response includes context from partner_info
- Answer is personalized to the scenario

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 6: Premium Model
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Complex negotiation strategy question",
    "use_premium_model": true,
    "use_preprocessing": true
  }'
```

**Expected Response:**
```json
{
  "answer": "...",
  "model_used": "openai/o3-mini",  // Premium model
  "processing_time": ...
}
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

## 🎨 Frontend Testing (Terminal 2)

### Start Frontend
```bash
./run-frontend.sh
```

**Expected Output:**
```
VITE v7.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
➜  press h + enter to show help
```

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

**If failed, check:**
- Port 5173 not in use
- `node_modules` installed (`cd frontend && npm install`)
- No TypeScript compilation errors

---

### Test 7: Frontend Loads
Open browser: http://localhost:5173

**Expected:**
- Page loads without errors
- Shows "NegotiatorPro" header
- Shows "Welcome to NegotiatorPro" message
- Shows input box and settings

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

**If failed, check:**
- Browser console for errors (F12)
- Network tab for failed API calls
- Backend is running on port 8000

---

### Test 8: Basic Chat Flow

**Steps:**
1. Type question: "What is BATNA in negotiation?"
2. Press Enter or click "Send"
3. Wait for response

**Expected Behavior:**
- User message appears immediately in chat
- "Sending..." button appears
- AI response appears after ~2-5 seconds
- Response includes negotiation advice
- Model info shows at bottom: "Model: openai/gpt-4o-mini • 2.5s"

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 9: Settings - Premium Model

**Steps:**
1. Check "Use Premium Model" checkbox
2. Type question: "How do I handle a lowball offer?"
3. Send message

**Expected:**
- Response uses premium model
- Model info shows: "Model: openai/o3-mini"

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 10: Settings - Partner Info

**Steps:**
1. Enter in "Partner info" field: "Budget-conscious manager"
2. Type question: "How do I ask for a raise?"
3. Send message

**Expected:**
- Response considers the partner context
- Advice is tailored to budget-conscious scenario

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 11: Settings - Text Preprocessing

**Steps:**
1. Uncheck "Text Preprocessing"
2. Send a message
3. Check "Text Preprocessing" again
4. Send another message

**Expected:**
- Both messages work
- (Preprocessing effect may not be visible in POC, but should not error)

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 12: Multiple Messages

**Steps:**
1. Send message: "What is BATNA?"
2. Wait for response
3. Send message: "What is ZOPA?"
4. Wait for response
5. Send message: "How do I use these together?"

**Expected:**
- All messages appear in order
- Chat history is maintained
- Page auto-scrolls to latest message
- Each message has unique ID

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 13: Error Handling

**Steps:**
1. Stop the backend server (Ctrl+C in Terminal 1)
2. Try to send a message from frontend

**Expected:**
- Error message appears in chat
- Shows: "Error: Failed to get response" or similar
- User can still type new messages
- Backend can be restarted and messages work again

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 14: Empty Message Validation

**Steps:**
1. Leave input box empty
2. Try to click "Send"

**Expected:**
- "Send" button is disabled
- No API call is made
- No error occurs

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Test 15: Long Message Handling

**Steps:**
1. Type a very long question (>500 words)
2. Send message

**Expected:**
- Message is accepted (up to 2000 chars)
- API processes it successfully
- Response is relevant

**Note:** Max length is 2000 chars (enforced by Pydantic)

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

## 🔍 Browser DevTools Checks

### Console Errors
Open browser DevTools (F12) → Console tab

**Expected:**
- No red errors
- May have warnings (acceptable for POC)
- API calls show in console (if logging enabled)

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

### Network Tab
Open DevTools → Network tab → Send a message

**Expected API Calls:**
- `POST http://localhost:5173/api/chat`
- Status: 200 OK
- Response time: 2-10 seconds (depending on OpenAI)
- Response preview shows JSON with `answer`, `model_used`, `processing_time`

**Result:** [ ] ✅ Pass / [ ] ❌ Fail

---

## 🧹 Cleanup After Testing

### Stop Servers
- [ ] Stop frontend: Ctrl+C in Terminal 2
- [ ] Stop backend: Ctrl+C in Terminal 1
- [ ] Verify ports are free:
  ```bash
  lsof -i :8000  # Should be empty
  lsof -i :5173  # Should be empty
  ```

---

## 📊 Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| Backend starts | ⬜ | |
| Health endpoint | ⬜ | |
| API docs | ⬜ | |
| Login endpoint | ⬜ | |
| Chat endpoint | ⬜ | |
| Chat with context | ⬜ | |
| Premium model | ⬜ | |
| Frontend loads | ⬜ | |
| Basic chat flow | ⬜ | |
| Premium model toggle | ⬜ | |
| Partner info | ⬜ | |
| Preprocessing toggle | ⬜ | |
| Multiple messages | ⬜ | |
| Error handling | ⬜ | |
| Empty validation | ⬜ | |
| Long messages | ⬜ | |
| Browser console | ⬜ | |
| Network requests | ⬜ | |

**Overall Result:** _____ / 18 tests passed

---

## 🐛 Known Issues (POC Limitations)

These are EXPECTED limitations of the POC:

1. **No WebSocket streaming** - Responses appear all at once, not word-by-word
2. **No session persistence** - Refresh page = messages lost
3. **No admin panel UI** - Only chat interface implemented
4. **Generic error messages** - Security improvement (shows generic errors to users)
5. **No retry logic** - If API fails, user must manually retry
6. **No rate limiting** - Can spam API (will add in production)
7. **No mobile optimization** - Desktop-first design

---

## 🚨 Troubleshooting

### Backend won't start
```bash
# Check if port is in use
lsof -i :8000

# Kill process if needed
kill -9 $(lsof -t -i:8000)

# Check for import errors
python -c "from backend.api.main import app; print('OK')"
```

### Frontend won't start
```bash
# Check if port is in use
lsof -i :5173

# Kill process if needed
kill -9 $(lsof -t -i:5173)

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS errors
- Ensure backend is running on port 8000
- Check `frontend/vite.config.ts` has proxy configured
- Clear browser cache (Cmd+Shift+R)

### API key errors
- Check `.env` has valid `OPENAI_API_KEY`
- Ensure no extra spaces or quotes
- Test key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

---

## ✅ Success Criteria

**POC is successful if:**
- ✅ All critical tests pass (Backend 1-6, Frontend 7-12)
- ✅ Can send chat message and receive AI response
- ✅ Settings toggles work (premium model, partner info)
- ✅ No critical errors in browser console
- ✅ Architecture validated (FastAPI + React working together)

**Next Steps After Validation:**
1. Document any issues found
2. Decide: Approve full migration? (6-8 weeks)
3. If approved: Begin Phase 3 (Feature Parity)

---

**Testing Date**: _____________
**Tester**: _____________
**Environment**: Mac / Linux / Windows
**Result**: ⬜ PASS / ⬜ FAIL / ⬜ PARTIAL
