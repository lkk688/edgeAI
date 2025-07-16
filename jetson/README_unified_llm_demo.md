# Unified LLM Demo for Edge Devices

This tool provides a comprehensive demonstration of various LLM backends with device-specific optimizations for edge computing platforms. It allows you to benchmark and compare different LLM backends, models, and optimizations on your specific hardware.

## Features

- **Multiple LLM Backends**: Support for llama.cpp, Ollama, llama-cpp-python, ONNX Runtime, and Hugging Face Transformers
- **Device-Specific Optimizations**: Automatic detection and optimization for:
  - NVIDIA CUDA GPUs
  - NVIDIA Jetson devices
  - Apple Silicon (M1/M2/M3)
  - Intel/AMD CPUs
- **Model Selection**: Use models from Hugging Face, local GGUF files, or Ollama
- **Performance Benchmarking**: Compare inference speed and memory usage across backends
- **Visualization**: Generate charts comparing performance metrics
- **Memory Monitoring**: Track RAM and GPU memory usage during inference

## Requirements

Base requirements:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib (for visualization)

Optional dependencies (based on which backends you want to use):
- Hugging Face Transformers
- llama-cpp-python
- ONNX Runtime
- Ollama (server)

## Installation

```bash
# Base dependencies
pip install torch numpy matplotlib psutil

# For Hugging Face models
pip install transformers accelerate

# For llama.cpp Python bindings
pip install llama-cpp-python

# For ONNX Runtime
pip install onnxruntime

# For GPU monitoring
pip install gputil

# For async batch processing
pip install aiohttp
```

For Ollama, follow the installation instructions at [ollama.ai](https://ollama.ai).

## Usage

### List Available Backends

```bash
python unified_llm_demo.py --list
```

### Basic Text Generation

#### Using llama.cpp

```bash
python unified_llm_demo.py \
  --backend llama_cpp \
  --model-path models/llama-2-7b-chat.q4_K_M.gguf \
  --prompt "Explain the benefits of edge AI computing" \
  --max-tokens 200
```

#### Using Ollama

```bash
python unified_llm_demo.py \
  --backend ollama \
  --model-name llama2:7b-chat \
  --prompt "What are the advantages of running LLMs on edge devices?" \
  --max-tokens 150
```

#### Using Hugging Face Transformers

```bash
python unified_llm_demo.py \
  --backend transformers \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Compare cloud computing vs edge computing for AI applications" \
  --max-tokens 100 \
  --quantization int8
```

### Batch Processing

Process multiple prompts in batch mode:

```bash
python unified_llm_demo.py \
  --backend ollama \
  --model-name llama2:7b-chat \
  --batch \
  --prompts-file prompts.txt \
  --max-tokens 100
```

Where `prompts.txt` contains one prompt per line.

### Benchmarking

Compare multiple backends and models:

```bash
python unified_llm_demo.py \
  --benchmark \
  --backends llama_cpp ollama transformers \
  --model-names llama-2-7b-chat.q4_K_M.gguf llama2:7b-chat TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model-paths models/llama-2-7b-chat.q4_K_M.gguf null null \
  --num-runs 3 \
  --max-tokens 50 \
  --prompt "Explain the benefits of edge AI computing"
```

This will:
1. Run each backend with its corresponding model
2. Generate text for the same prompt 3 times per backend
3. Measure inference time and memory usage
4. Create a visualization comparing the results
5. Save the benchmark data to a JSON file

## Output

The benchmark results will be saved to:
- `benchmark_results/TIMESTAMP_llm_comparison_results.json` - Raw benchmark data
- `llm_backend_comparison.png` - Visualization of inference time and memory usage

## Device-Specific Optimizations

The script automatically detects your hardware and applies appropriate optimizations:

- **NVIDIA GPUs**: CUDA acceleration, half-precision, quantization
- **Jetson**: CUDA acceleration with Jetson-specific memory limits
- **Apple Silicon**: MPS acceleration when available
- **CPU**: Multi-threading optimizations

## Extending the Demo

To add support for additional backends:

1. Add the necessary imports and availability checks
2. Extend the `LLMBackendManager` class with methods for the new backend
3. Update the `get_optimal_settings` method in `DeviceManager` for the new backend

## Troubleshooting

- **Out of Memory**: Reduce model size, enable quantization, or reduce batch size
- **Slow Inference**: Check if hardware acceleration is properly enabled
- **Missing Dependencies**: Install the required packages for your chosen backend

## License

MIT