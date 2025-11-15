#!/usr/bin/env python3
"""
Fix both default and premium model configurations
Ensures proper model IDs (not display names)
"""

import json

CONFIG_FILE = "llm_backend_config.json"

# Read current config
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Fix default model - use Ollama with correct model ID
config["active_models"]["default"] = {
    "backend": "ollama",
    "model": "gpt-oss:120b-cloud"
}

# Fix premium model - use Ollama with correct model ID (or use same as default)
config["active_models"]["premium"] = {
    "backend": "ollama",
    "model": "gpt-oss:120b-cloud"
}

# Enable ollama backend
if "backend_settings" not in config:
    config["backend_settings"] = {}
config["backend_settings"]["ollama"] = {"enabled": True}

# Update active_backend
if "active_backend" in config:
    config["active_backend"] = "ollama"

# Save
with open(CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Fixed configuration:")
print(f"   Default model:  ollama/gpt-oss:120b-cloud")
print(f"   Premium model:  ollama/gpt-oss:120b-cloud")
print(f"   Backend:        ollama (local)")
print(f"   URL:            http://localhost:11434")
print(f"\n   Config file: {CONFIG_FILE}")
print("\n⚠️  Restart the app for changes to take effect")
