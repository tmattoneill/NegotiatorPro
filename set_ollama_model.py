#!/usr/bin/env python3
"""
Quick script to set Ollama as the active model backend
"""

import json

CONFIG_FILE = "llm_backend_config.json"

# Read current config
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Update to use Ollama backend with gpt-oss:120b-cloud
config["active_models"]["default"] = {
    "backend": "ollama",
    "model": "gpt-oss:120b-cloud"
}

# Enable ollama backend
if "backend_settings" not in config:
    config["backend_settings"] = {}
config["backend_settings"]["ollama"] = {"enabled": True}

# Save
with open(CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Updated configuration to use Ollama backend")
print(f"   Default model: ollama/gpt-oss:120b-cloud")
print("\nRestart the app with: python main.py")
