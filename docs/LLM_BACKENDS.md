# LLM backends

NegotiatorPro supports multiple LLM backends and lets you mix them — for example OpenAI GPT-4o Mini
as the default model and Claude 3.5 Sonnet as the premium model. Default and premium models can each
come from any backend.

The source of truth for available models is the `BACKENDS` dictionary in
`backend/llm_backend_config.py`. The lists below are illustrative; check the code for the current
set. Runtime selection state lives in `llm_backend_config.json` (see `CONFIGURATION.md`).

## Supported backends

1. **OpenAI** (default) — e.g. GPT-4o, GPT-4o Mini, O3 Mini, GPT-4 Turbo. Requires `OPENAI_API_KEY`
   (https://platform.openai.com/api-keys).
2. **Anthropic Claude** — e.g. Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku. Requires
   `ANTHROPIC_API_KEY` (https://console.anthropic.com/). Needs `langchain-anthropic` and `anthropic`
   (both in `requirements.txt`).
3. **DeepSeek** — DeepSeek chat/reasoner models. Requires `DEEPSEEK_API_KEY`.
4. **Ollama (local)** — e.g. Llama 3.1, Mistral, Mixtral, Qwen 2.5, Phi-3, Gemma 2. Requires a local
   Ollama install (https://ollama.ai/); default URL `http://localhost:11434`; no API key.
5. **Ollama (cloud)** — same models on a hosted instance. Requires `OLLAMA_API_KEY` and
   `OLLAMA_CLOUD_URL`. See `features/OLLAMA_CLOUD_SETUP.md` for the full setup.

## Setup

1. Add the API keys you need to `.env` (see `.env.example` for the full key list).
2. `pip install -r requirements.txt`.
3. In the admin panel → **LLM Backends** tab, view backend status and set the **Default** and
   **Premium** models.

For local Ollama: install Ollama, `ollama pull <model>` (e.g. `llama3.1:70b`), confirm with
`ollama list`, then select "Ollama (Local)" and the model in the admin panel.

## Model selection strategy

- **Default model**: fast and cost-effective for most queries (e.g. GPT-4o Mini, Claude 3 Haiku,
  Llama 3.1 8B).
- **Premium model**: stronger reasoning for complex negotiations (e.g. O3 Mini, Claude 3.5 Sonnet,
  Llama 3.1 70B).

Use the cheaper default for most traffic and the premium model only when needed. Approximate costs
are shown in the backend configuration; Ollama (local) is free.

## API format and compatibility

All backends use a unified chat message format:

```python
messages = [
    {"role": "system", "content": "You are a negotiation expert..."},
    {"role": "user", "content": "How do I negotiate salary?"},
]
response = llm.invoke(messages)
```

Backend-specific notes:

- **OpenAI** (`ChatOpenAI`): Chat Completion API (`/v1/chat/completions`); native role messages.
- **Anthropic** (`ChatAnthropic`): Messages API (`/v1/messages`); the system message is handled
  separately by the API; `api_key` maps to `anthropic_api_key`.
- **Ollama** (`ChatOllama`): Chat API (`/api/chat`), OpenAI-compatible format. Use the `ChatOllama`
  class, not base `Ollama`.

LangChain's chat-model abstractions handle provider-specific parameter mapping, endpoint routing,
request/response formatting, and retries, so the application code is the same regardless of backend.
The system also filters parameters a model does not support (e.g. `temperature` for o3-mini).

## Troubleshooting

- **"API key not found"**: ensure the key is in `.env` and restart the app.
- **Claude models missing**: `pip install langchain-anthropic anthropic`.
- **Ollama connection failed**: confirm Ollama is running (`ollama serve`) or check
  `OLLAMA_BASE_URL`.
- **Ollama 404 for cloud models** (e.g. `gpt-oss:120b-cloud`): use the `ollama-cloud` backend, set
  `OLLAMA_API_KEY` (https://ollama.com/settings/keys), and set `OLLAMA_CLOUD_URL=https://ollama.com`.
- **Model initialisation failed**: check logs for the specific error; the system falls back to the
  OpenAI default model.
