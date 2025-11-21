# Configuration System Guide

## Overview

NegotiatorPro uses a centralized `config.json` file for easy management of system-wide settings, LLM models, UI features, and defaults.

## Location

```
/config.json
```

## Structure

The configuration file is organized into the following sections:

### 1. **App Info**
```json
{
  "app": {
    "name": "NegotiatorPro",
    "description": "AI-powered negotiation guidance",
    "version": "1.0.0",
    "environment": "development"
  }
}
```

### 2. **UI Configuration**
```json
{
  "ui": {
    "theme": {
      "primaryColor": "#3498db",
      "sidebarColor": "#2c3e50",
      "accentColor": "#667eea"
    },
    "features": {
      "enableUserProfiles": true,
      "enableModelSelection": true,
      "enableTextOptimization": true,
      "enablePremiumModels": true,
      "enableConversationHistory": true
    },
    "limits": {
      "maxMessageLength": 10000,
      "maxConversationsPerUser": 100,
      "sessionTimeoutMinutes": 60
    }
  }
}
```

### 3. **LLM Models** ⭐ MOST IMPORTANT

This section defines all available LLM providers and models:

```json
{
  "llm_models": [
    {
      "provider": "openai",
      "provider_prettyname": "OpenAI",
      "provider_description": "Industry-leading AI models from OpenAI",
      "enabled": true,
      "requires_api_key": true,
      "models": [
        {
          "id": "gpt-5-mini-2025-08-07",
          "name": "GPT-5 Mini",
          "description": "Latest compact GPT-5 model",
          "tier": "default",
          "context_window": 128000,
          "supports_streaming": true,
          "cost_per_1k_input_tokens": 0.00015,
          "cost_per_1k_output_tokens": 0.0006
        }
      ]
    }
  ]
}
```

**Model Properties:**
- `id`: Exact API model ID (e.g., `gpt-5-mini-2025-08-07`)
- `name`: Display name shown in UI
- `description`: User-friendly description
- `tier`: Either `"default"` or `"premium"`
- `context_window`: Token context limit
- `supports_streaming`: Enable streaming responses
- `cost_per_1k_input_tokens`: Cost per 1,000 input tokens (USD)
- `cost_per_1k_output_tokens`: Cost per 1,000 output tokens (USD)

### 4. **Defaults**

Default model and system settings:

```json
{
  "defaults": {
    "default_provider": "openai",
    "default_model": "gpt-5-mini-2025-08-07",
    "premium_provider": "anthropic",
    "premium_model": "claude-sonnet-4-5",
    "enable_text_preprocessing": true,
    "max_tokens": 4096,
    "temperature": 0.7,
    "streaming_enabled": true
  }
}
```

### 5. **Negotiation Framework**

PLEASE framework configuration:

```json
{
  "negotiation": {
    "framework": "PLEASE",
    "response_format": {
      "include_analysis": true,
      "include_questions": true,
      "include_draft_response": true,
      "include_scenarios": true,
      "include_self_assessment": true
    },
    "analysis_elements": [
      "Context",
      "Stakeholder Identification",
      "Current Position",
      "Potential Interests",
      "Emotional Tone",
      "Power Balance"
    ]
  }
}
```

### 6. **Security**

Authentication and security settings:

```json
{
  "security": {
    "jwt_expiration_minutes": 30,
    "session_timeout_minutes": 60,
    "password_min_length": 8,
    "require_email_verification": false,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 15
  }
}
```

## API Endpoints

Access configuration via REST API:

### Get Full Configuration
```bash
GET /api/config/
```

### Get App Info
```bash
GET /api/config/app
# Returns: {"name": "NegotiatorPro", "version": "1.0.0"}
```

### Get UI Configuration
```bash
GET /api/config/ui
# Returns theme, features, and limits
```

### Get LLM Models
```bash
# All providers
GET /api/config/llm-models

# Specific provider
GET /api/config/llm-models?provider=openai
GET /api/config/llm-models?provider=anthropic

# By tier
GET /api/config/llm-models?tier=premium
GET /api/config/llm-models?tier=default

# Specific model
GET /api/config/llm-models/gpt-5-mini-2025-08-07
```

### Get Providers
```bash
GET /api/config/providers
GET /api/config/providers?enabled_only=true
```

### Get Defaults
```bash
GET /api/config/defaults
GET /api/config/defaults/models
```

### Reload Configuration
```bash
POST /api/config/reload
# Reloads config.json without restarting backend
```

## Usage in Code

### Backend (Python)

```python
from backend.config_loader import config

# Get app info
app_name = config.get("app.name")

# Get LLM models for a provider
openai_models = config.get_llm_models("openai")

# Get default model
default_model = config.get_default_model()
# Returns: {"provider": "openai", "model": "gpt-5-mini-2025-08-07"}

# Get all premium models
premium_models = config.get_models_by_tier("premium")

# Get model details
model_info = config.get_model_by_id("claude-sonnet-4-5")
```

### Frontend (React)

```typescript
// Fetch config from API
const response = await api.get('/config/llm-models');
const providers = response.data.providers;

// Get default models
const defaults = await api.get('/config/defaults/models');
console.log(defaults.default); // { provider: "openai", model: "gpt-5-mini-2025-08-07" }
```

## Adding New Models

To add a new LLM model:

1. **Open `config.json`**

2. **Find the provider section** (or create a new one):
   ```json
   {
     "provider": "openai",
     "provider_prettyname": "OpenAI",
     "models": [...]
   }
   ```

3. **Add the model**:
   ```json
   {
     "id": "gpt-6-2026-01-01",
     "name": "GPT-6",
     "description": "Next generation GPT model",
     "tier": "premium",
     "context_window": 256000,
     "supports_streaming": true,
     "cost_per_1k_input_tokens": 0.005,
     "cost_per_1k_output_tokens": 0.02
   }
   ```

4. **Reload the configuration**:
   ```bash
   curl -X POST http://localhost:8000/api/config/reload
   ```

5. **Refresh the frontend** - The new model will appear in the dropdown!

## Current Models

### OpenAI
- **GPT-5 Mini** (`gpt-5-mini-2025-08-07`) - Default, fast and affordable
- **GPT-4.1** (`gpt-4.1-2025-04-14`) - Premium, most capable
- **O1 Mini** (`o1-mini-2024-09-12`) - Premium, advanced reasoning

### Anthropic (Claude)
- **Claude Sonnet 4.5** (`claude-sonnet-4-5`) - Premium, best reasoning
- **Claude Haiku 4.5** (`claude-haiku-4-5`) - Default, fast and efficient

### Ollama (Local)
- Llama 3.1 70B/8B
- Mistral 7B
- Mixtral 8x7B
- Qwen 2.5 72B/14B
- And more...

## Best Practices

1. **Always use valid model IDs** - Check OpenAI/Anthropic docs for exact IDs
2. **Set appropriate tiers** - `"default"` for fast/cheap, `"premium"` for powerful
3. **Update costs** - Keep pricing current for accurate usage tracking
4. **Test after changes** - Reload config and verify in UI
5. **Version control** - Commit config.json changes with descriptive messages

## Troubleshooting

**Models not showing up?**
1. Check `enabled: true` in provider config
2. Verify API keys are set in `.env`
3. Reload config: `POST /api/config/reload`
4. Restart backend if needed

**Wrong model selected?**
- Update `defaults.default_model` or `defaults.premium_model`
- Reload configuration

**Cost tracking incorrect?**
- Update `cost_per_1k_input_tokens` and `cost_per_1k_output_tokens`
- Check provider pricing pages for latest rates
