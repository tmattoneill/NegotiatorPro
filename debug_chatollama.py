#!/usr/bin/env python3
"""
Debug script to test ChatOllama vs direct API
Run on your local Mac to see what ChatOllama is doing differently
"""

import sys

print("=" * 60)
print("ChatOllama Debug Script")
print("=" * 60)

# Test 1: Direct API call (your working code)
print("\n1. Testing direct API call...")
try:
    import requests
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "Say hi in 3 words"}],
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    content = data["message"]["content"]
    print(f"✅ Direct API works: {content[:50]}")
except Exception as e:
    print(f"❌ Direct API failed: {e}")
    sys.exit(1)

# Test 2: ChatOllama with minimal config
print("\n2. Testing ChatOllama with minimal config...")
try:
    from langchain_community.chat_models import ChatOllama

    llm = ChatOllama(
        model="gpt-oss:120b-cloud",
        base_url="http://localhost:11434"
    )

    messages = [{"role": "user", "content": "Say hi in 3 words"}]

    print(f"   ChatOllama instance: {llm}")
    print(f"   Model: {llm.model}")
    print(f"   Base URL: {llm.base_url}")

    response = llm.invoke(messages)
    print(f"✅ ChatOllama works: {response.content[:50]}")

except Exception as e:
    print(f"❌ ChatOllama failed: {type(e).__name__}: {e}")

    # Print more details
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

    # Check if it's a 404
    if "404" in str(e):
        print("\n⚠️  ChatOllama is getting a 404 error")
        print("   This suggests it's hitting a different endpoint than /api/chat")
        print("\n   Possible causes:")
        print("   - ChatOllama version incompatibility")
        print("   - ChatOllama using wrong endpoint path")
        print("   - Ollama version mismatch")

# Test 3: Check versions
print("\n3. Version information:")
try:
    import langchain_community
    print(f"   langchain-community: {langchain_community.__version__}")
except:
    print("   langchain-community: unknown version")

try:
    import subprocess
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    print(f"   ollama: {result.stdout.strip()}")
except:
    print("   ollama: version check failed")

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)
