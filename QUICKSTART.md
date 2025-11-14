# 🚀 NegotiatorPro - Quick Start Guide

## What You Need

1. **OpenAI API Key** - Get one from https://platform.openai.com/api-keys
2. **Python 3.8+** OR **Docker** (choose one method below)

## Method 1: Run with Python (5 minutes)

```bash
# 1. Set up your environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your negotiation books (optional)
# Place PDF/TXT/DOCX files in the sources/ directory

# 5. Run the application
python main.py
```

**Access**: Open http://localhost:7860 in your browser

**Default Admin Password**: `admin123` (change this immediately!)

---

## Method 2: Run with Docker (Recommended for Production)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start with Docker Compose
docker compose up -d

# 3. View logs (optional)
docker compose logs -f
```

**Access**: Open http://localhost:7860 in your browser

**Stop**: `docker compose down`

---

## First Steps After Starting

1. **Open the app** at http://localhost:7860
2. **Try a question** like: "How should I respond to a lowball offer?"
3. **Access Admin Panel**:
   - Click the "Admin" tab
   - Login with password: `admin123`
   - **IMPORTANT**: Change the admin password immediately!
4. **Upload your books** (Admin > Documents tab):
   - Upload PDF/DOCX/TXT files of negotiation books
   - Click "Rebuild Index" to process them

---

## Features You Can Use

### Chat Interface
- Ask negotiation questions in plain English
- Add context about your negotiation partner
- Toggle "Premium Model" for advanced reasoning
- Enable "Optimize Text" to reduce costs

### Admin Dashboard
- **Configuration**: Customize system prompts and AI behavior
- **Documents**: Upload books, manage knowledge base
- **Analytics**: Track API usage and costs
- **Security**: Change admin password

---

## Adding Your Own Negotiation Books

1. Go to Admin Panel > Documents tab
2. Upload PDF, TXT, DOC, or DOCX files
3. Click "Rebuild Index" to process new documents
4. Books are added to the AI's knowledge base

**Example Sources** (not included, you need to add):
- "Getting to Yes" by Fisher & Ury
- "Never Split the Difference" by Chris Voss
- "Getting Past No" by William Ury
- Any negotiation content you want the AI to reference

---

## Troubleshooting

**"No module named 'gradio'"**
- Run: `pip install -r requirements.txt`

**"OpenAI API key not found"**
- Make sure `.env` file exists with `OPENAI_API_KEY=your_key_here`

**"No documents found"**
- Add PDF/TXT/DOCX files to `sources/` directory
- Or upload via Admin Panel > Documents

**Port 7860 already in use**
- Change port in `.env`: `GRADIO_SERVER_PORT=8080`
- Or stop the other service: `lsof -ti:7860 | xargs kill -9`

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
