#!/usr/bin/env bash
set -euo pipefail

MODEL=${LLM_MODEL:-llama3.1:8b-instruct-q5_K_M}

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Installing..."
  curl -fsSL https://ollama.ai/install.sh | sh
fi

echo "Pulling model $MODEL"
ollama pull "$MODEL"

echo "Done. Start the Ollama server with 'ollama serve' if not running."
