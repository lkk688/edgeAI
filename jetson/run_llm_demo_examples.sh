#!/bin/bash

# Unified LLM Demo Examples
# This script demonstrates various usage examples of the unified LLM demo

# Set this to your model path if using local models
LLAMA_MODEL_PATH="models/llama-2-7b-chat.q4_K_M.gguf"
TINY_LLAMA="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OLLAMA_MODEL="llama2:7b-chat"

# Colors for output
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

echo -e "${GREEN}===== Unified LLM Demo Examples =====${NC}\n"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Python not found. Please install Python 3.8+ to run this demo.${NC}"
    exit 1
fi

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Function to print commands before executing
run_command() {
    echo -e "${YELLOW}Running: $1${NC}\n"
    eval "$1"
    echo -e "\n${GREEN}Command completed.${NC}"
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read
}

# List available backends
print_header "Listing Available Backends and Models"
run_command "python unified_llm_demo.py --list"

# Basic examples
print_header "Basic Text Generation with Transformers"
run_command "python unified_llm_demo.py --backend transformers --model-name $TINY_LLAMA --prompt \"Explain edge AI in 2 sentences\" --max-tokens 50"

# Check if Ollama is available
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    print_header "Text Generation with Ollama"
    run_command "python unified_llm_demo.py --backend ollama --model-name $OLLAMA_MODEL --prompt \"What are the advantages of edge computing?\" --max-tokens 100"
fi

# Check if llama-cpp-python model exists
if [ -f "$LLAMA_MODEL_PATH" ]; then
    print_header "Text Generation with llama.cpp"
    run_command "python unified_llm_demo.py --backend llama_cpp --model-path $LLAMA_MODEL_PATH --prompt \"Compare cloud and edge computing\" --max-tokens 150"
fi

# Batch processing example
print_header "Batch Processing Example"
run_command "python unified_llm_demo.py --backend transformers --model-name $TINY_LLAMA --batch --prompts-file sample_prompts.txt --max-tokens 50"

# Benchmark example (simplified for demo)
print_header "Simple Benchmark Example"
run_command "python unified_llm_demo.py --benchmark --backends transformers --model-names $TINY_LLAMA --num-runs 2 --max-tokens 30"

# Full benchmark example (commented out as it requires multiple backends)
echo -e "\n${YELLOW}Full benchmark example (requires multiple backends):\n"
echo -e "python unified_llm_demo.py --benchmark \\\n  --backends transformers ollama llama_cpp \\\n  --model-names $TINY_LLAMA $OLLAMA_MODEL llama-2-7b-chat.q4_K_M.gguf \\\n  --model-paths null null $LLAMA_MODEL_PATH \\\n  --num-runs 3 --max-tokens 50${NC}\n"

print_header "All Examples Completed"
echo -e "${GREEN}For more information, see README_unified_llm_demo.md${NC}\n"