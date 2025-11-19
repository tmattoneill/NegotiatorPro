# 🚀 NegotiatorPro - Quick Start Guide

## What You Need

1. **OpenAI API Key** - Get one from https://platform.openai.com/api-keys
2. **Python 3.8+ & Node.js 18+** OR **Docker** (choose one method below)

## Method 1: Run with Local Development (5 minutes)

```bash
# 1. Set up your environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Set up Python backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up React frontend
cd frontend
npm install
cd ..

# 4. Add your negotiation books (optional)
# Place PDF/TXT/DOCX files in the sources/ directory

# 5. Start the application (requires 2 terminals)
# Terminal 1 - Backend:
./run-api.sh

# Terminal 2 - Frontend:
./run-frontend.sh
```

**Access**:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Default Admin Password**: `admin123` (change this immediately!)

---

## Method 2: Run with Docker (Recommended for Production)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start with Docker Compose (both backend + frontend)
docker compose up -d

# 3. View logs (optional)
docker compose logs -f         # All services
docker compose logs -f backend  # Backend only
docker compose logs -f frontend # Frontend only
```

**Access**:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000

**Stop**: `docker compose down`

---

## First Steps After Starting

1. **Open the app** at http://localhost:5173
2. **Try a question** like: "How should I respond to a lowball offer?"
3. **Create conversations**: Start new chat sessions for different scenarios
4. **Toggle Premium Model**: Use advanced reasoning when needed
5. **Admin Features** (coming soon to React UI):
   - Access via FastAPI endpoints at http://localhost:8000/docs
   - Login with password: `admin123`
   - **IMPORTANT**: Change the admin password immediately!

---

## Features You Can Use

### Chat Interface (React Frontend)
- Ask negotiation questions in plain English
- Create multiple conversation sessions
- Toggle "Premium Model" for advanced reasoning
- View full conversation history
- Markdown-formatted AI responses

### Admin Features (FastAPI Backend)
- **Configuration**: Customize system prompts and AI behavior (API endpoints)
- **Documents**: Upload books, manage knowledge base (command line for now)
- **Analytics**: Track API usage and costs (via backend)
- **Security**: Change admin password (via backend API)

---

## Adding Your Own Negotiation Books

**Current Method** (command line):
1. Add PDF, TXT, DOC, or DOCX files to `sources/` directory
2. Run: `python scripts/rebuild_vectordb.py`
3. Restart the backend to load new documents

**Future**: React admin UI will support file uploads

**Example Sources** (not included, you need to add):
- "Getting to Yes" by Fisher & Ury
- "Never Split the Difference" by Chris Voss
- "Getting Past No" by William Ury
- Any negotiation content you want the AI to reference

---

## Troubleshooting

**"Module not found" errors**
- Backend: `pip install -r requirements.txt`
- Frontend: `cd frontend && npm install`

**"OpenAI API key not found"**
- Make sure `.env` file exists with `OPENAI_API_KEY=your_key_here`

**"No documents found"**
- Add PDF/TXT/DOCX files to `sources/` directory
- Run: `python scripts/rebuild_vectordb.py`

**"Port already in use"**
- Backend (8000): Change port in `run-api.sh` or use different port
- Frontend (5173): Change port in `vite.config.ts`
- Stop conflicting services: `lsof -ti:8000 | xargs kill -9`

**"Connection refused" in browser**
- Ensure both backend and frontend are running
- Check backend is at http://localhost:8000
- Check frontend is at http://localhost:5173

---

## Production Deployment

For production deployment with HTTPS, reverse proxy, and security hardening, see:
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete production deployment guide
- **[DOCKER-DEPLOY.md](DOCKER-DEPLOY.md)** - Docker-specific deployment

---

## Need Help?

- **Full Documentation**: See [README.md](README.md)
- **Testing**: See [TESTING.md](TESTING.md)
- **Admin Features**: See [ADMIN_FEATURES.md](ADMIN_FEATURES.md)
- **Issues**: Open an issue on GitHub

---

**Ready to negotiate like a pro!** 🤝
