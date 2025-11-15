#!/usr/bin/env python3
"""
Fix the llm_backend_config.json file
Sets Ollama as default backend with correct model ID
"""

import json

CONFIG_FILE = "llm_backend_config.json"

# Read current config
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Use simple "ollama" backend (local Ollama handles cloud models automatically)
config["active_models"]["default"] = {
    "backend": "ollama",
    "model": "gpt-oss:120b-cloud"
}

# Enable ollama backend
if "backend_settings" not in config:
    config["backend_settings"] = {}
config["backend_settings"]["ollama"] = {"enabled": True}

# Remove or update the deprecated active_backend field
if "active_backend" in config:
    config["active_backend"] = "ollama"

# Save
with open(CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Fixed configuration:")
print(f"   Default model: ollama/gpt-oss:120b-cloud")
print(f"   Backend: ollama (local - handles cloud models automatically)")
print(f"   URL: http://localhost:11434")
print(f"\n   Config file: {CONFIG_FILE}")
print("\n⚠️  Restart the app for changes to take effect")
