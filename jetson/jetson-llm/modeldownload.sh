#!/bin/bash
# Download recommended GGUF models for Jetson Orin Nano into ./models
#
# NOTE: With llama.cpp you usually DON'T need to pre-download — `llama-server`
# and `llama-cli` can pull straight from Hugging Face with the `-hf` flag, e.g.:
#     llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S --host 0.0.0.0 --port 8080 -ngl 99
# (this is exactly what `sjsujetsontool llama` runs). Use this script only if you
# want the files on local disk (e.g. for offline use or llama-cpp-python).

set -e
mkdir -p models && cd models

# Gemma 4 E2B (instruction-tuned) — default served by `sjsujetsontool llama`
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf --local-dir . --local-dir-use-symlinks False

# NVIDIA Nemotron-3 Nano 4B (reasoning/chat)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF \
  NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Qwen3 8B (larger, multilingual) — optional
# huggingface-cli download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

echo "✅ Models downloaded into $(pwd)"
