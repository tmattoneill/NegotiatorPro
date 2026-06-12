# Ollama Cloud Setup Guide

This guide explains how to configure NegotiatorPro to use Ollama cloud models (like `gpt-oss:120b-cloud`).

## Problem

When using cloud-hosted Ollama models, you may encounter a 404 error:
```
OllamaEndpointNotFoundError: Ollama call failed with status code 404
```

This happens because cloud models are hosted on `https://ollama.com`, not on your local Ollama instance (`http://localhost:11434`).

## Solution

### Step 1: Get Your Ollama API Key

1. Visit https://ollama.com/settings/keys
2. Create a new API key
3. Copy the key

### Step 2: Configure Environment Variables

Create or edit your `.env` file:

```bash
# Ollama Cloud Configuration
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_API_KEY=your_actual_api_key_here
```

**Note**: You can leave `OLLAMA_CLOUD_URL` unset - it defaults to `https://ollama.com`.

### Step 3: Configure the Backend in Admin Panel

1. Start the backend: `./run-api.sh`
2. Navigate to **Admin Panel** tab
3. Log in with your admin password
4. Go to **🤖 LLM Backends** tab
5. Select **"Ollama (Cloud)"** as the backend (not "Ollama (Local)")
6. Choose your model (e.g., `gpt-oss:120b-cloud`)
7. Click **Set Default Model** or **Set Premium Model**

### Step 4: Verify Configuration

The backend status should show:
- ✅ Ollama (Cloud): Enabled | 🔑 API Key: ✅

## Local vs Cloud Backends

### Ollama (Local)
- **Endpoint**: `http://localhost:11434`
- **Authentication**: None required
- **Models**: Models you've pulled locally with `ollama pull`
- **Usage**: `ollama serve` must be running

### Ollama (Cloud)
- **Endpoint**: `https://ollama.com`
- **Authentication**: API key required (from https://ollama.com/settings/keys)
- **Models**: Cloud-hosted models (e.g., `gpt-oss:120b-cloud`)
- **Usage**: No local server needed

## Troubleshooting

### Error: "API key not found"
- Ensure `OLLAMA_API_KEY` is set in your `.env` file
- Restart the application after updating `.env`

### Error: "404 Ollama endpoint not found"
- Verify you're using the **"Ollama (Cloud)"** backend, not "Ollama (Local)"
- Check that `OLLAMA_CLOUD_URL=https://ollama.com` (or leave unset for default)

### How to test if it's working
Check the logs when making a request - you should see:
```
Creating ChatOllama with kwargs: {'model': 'gpt-oss:120b-cloud', 'base_url': 'https://ollama.com', 'headers': {'Authorization': 'Bearer ...'}}
```

## Example Configuration

Complete `.env` file example:

```bash
# OpenAI (Required for default setup)
OPENAI_API_KEY=sk-...

# Ollama Cloud (for cloud models)
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_CLOUD_URL=https://ollama.com

# Ollama Local (optional, for local models)
# OLLAMA_BASE_URL=http://localhost:11434
```

## Model Selection Strategy

**For large cloud models (gpt-oss:120b-cloud)**:
- Use as **Premium Model** for complex negotiations
- Set as "ollama-cloud" backend in admin panel

**For local models (llama3.1:8b)**:
- Use as **Default Model** for faster, local responses
- Set as "ollama" backend in admin panel

## References

- Ollama Cloud Docs: https://docs.ollama.com/cloud
- API Keys: https://ollama.com/settings/keys
- Main Documentation: See CLAUDE.md
