# 🚀 What Are LLMs?
**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)


LLMs (Large Language Models) are transformer-based models trained on vast datasets to understand and generate human-like text.

## 💬 Common Use Cases

* Chatbots and virtual assistants
* Code generation
* Summarization
* Translation



---

## 🛠️ Running LLMs on Jetson

Running LLMs on Jetson Orin Nano requires careful consideration of memory constraints, compute capabilities, and inference optimization. This section explores various LLM backends, their theoretical foundations, and practical implementations.

### 🎯 LLM Backend Comparison

| Backend | Memory Efficiency | Speed | Ease of Use | CUDA Support | Best For |
|---------|------------------|-------|-------------|--------------|----------|
| **llama.cpp** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Production inference |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Quick deployment |
| **llama-cpp-python** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Python integration |
| **TensorRT-LLM** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | Maximum performance |
| **ONNX Runtime** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | Cross-platform |
| **vLLM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Batch inference |

### 🧠 Theoretical Foundations

#### **Quantization Theory**
Quantization reduces model precision from FP32/FP16 to lower bit representations:

- **INT8 Quantization**: 8-bit integers, ~4x memory reduction
- **INT4 Quantization**: 4-bit integers, ~8x memory reduction  
- **GPTQ**: Post-training quantization preserving model quality
- **AWQ**: Activation-aware weight quantization

#### **Memory Optimization Strategies**
1. **KV-Cache Management**: Efficient attention cache storage
2. **Paged Attention**: Dynamic memory allocation for sequences
3. **Gradient Checkpointing**: Trade compute for memory during training
4. **Model Sharding**: Split large models across memory boundaries

#### **Inference Optimization**
- **Speculative Decoding**: Use smaller model to predict tokens
- **Continuous Batching**: Dynamic batching for variable sequence lengths
- **Flash Attention**: Memory-efficient attention computation
- **Kernel Fusion**: Combine operations to reduce memory transfers

### 🔧 LLM Backends for Edge Devices

<!-- The `unified_llm_demo.py` script provides a comprehensive framework for running various LLM backends on edge devices, with specific optimizations for different hardware platforms. Let's explore each backend, its device compatibility, and installation requirements. -->

#### **1. llama.cpp - High-Performance C++ Engine**

**Architecture**: Pure C++ implementation with CUDA acceleration
**Memory Model**: Efficient GGUF format with memory mapping
**Quantization**: K-quants (Q4_K_M, Q6_K) for optimal quality/speed trade-off
**Device Availability**:
- ✅ NVIDIA Jetson (CUDA-enabled)
- ✅ NVIDIA GPUs (CUDA)
- ✅ x86 CPUs
- ✅ Apple Silicon (Metal support via separate build)

<!-- **Installation**:
```bash
# Basic installation
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# For CUDA support (Jetson/NVIDIA GPUs)
make LLAMA_CUBLAS=1

# For CPU-only
make
``` -->
Local models are already downloaded under the `models` directory in `/Developer/models`, when inside the container, the `/Developer/models` folder has been mounted to `/models`:
```bash
$ sjsujetsontool shell
/models# ls
hf  mistral.gguf  qwen.gguf
#Download the model, if needed
/models$ wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral.gguf
```

llama.cpp requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. `llama-cli` is a CLI tool for accessing and experimenting with most of llama.cpp's functionality. Run in conversation mode: `llama-cli -m model.gguf` or add custom chat template: `llama-cli -m model.gguf -cnv --chat-template chatml`

Run a local downloaded model (`llama-cli` is already added in the path of the container):
```bash
root@sjsujetson-00:/workspace# llama-cli -m /models/mistral.gguf -p "Explain what is Nvidia jetson"
....
llama_perf_sampler_print:    sampling time =      11.06 ms /   185 runs   (    0.06 ms per token, 16731.48 tokens per second)
llama_perf_context_print:        load time =    1082.38 ms
llama_perf_context_print: prompt eval time =    2198.32 ms /    17 tokens (  129.31 ms per token,     7.73 tokens per second)
llama_perf_context_print:        eval time =   27024.20 ms /   167 runs   (  161.82 ms per token,     6.18 tokens per second)
llama_perf_context_print:       total time =   70364.22 ms /   184 tokens
```


`llama-server` is a lightweight, OpenAI API compatible, HTTP server for serving LLMs. Start a local HTTP server with default configuration on port 8080: `llama-server -m model.gguf --port 8080`, Basic web UI can be accessed via browser: `http://localhost:8080`. Chat completion endpoint: `http://localhost:8080/v1/chat/completions`
```bash
root@sjsujetson-00:/workspace# llama-server -m models/mistral.gguf --port 8080
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: Orin, compute capability 8.7, VMM: yes
build: 5752 (62af4642) with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for aarch64-linux-gnu
system info: n_threads = 6, n_threads_batch = 6, total_threads = 6

system_info: n_threads = 6 (n_threads_batch = 6) / 6 | CUDA : ARCHS = 870 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
.....
```

Send request via curl in another terminal (in the host machine or container):
```bash
sjsujetson@sjsujetson-01:~$ curl http://localhost:8080/completion -d '{
  "prompt": "Explain what is Nvidia jetson?",
  "n_predict": 100
}'
```

By default, llama-server listens only on 127.0.0.1 (localhost), which blocks external access. To enable external access, you need to bind to 0.0.0.0 (This tells it to accept connections from any IP address.):
```bash
llama-server -m ../models/mistral.gguf --port 8080 --host 0.0.0.0
```
If your Jetson device has ufw (Uncomplicated Firewall) or iptables enabled, open port 8080:
```bash
sudo ufw allow 8080/tcp
```
`llama-server` command is also integrated with `sjsujetsontool`, you can quickly start llama server via:
```bash
sjsujetsontool llama #it will launch llama server on port 8000
```

# llama cpp Python
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) is a Python library that provides bindings for llama.cpp. It provides 
- Low-level access to C API via ctypes interface.
- High-level Python API for text completion
    - OpenAI-like API
    - LangChain compatibility
    - LlamaIndex compatibility
- OpenAI compatible web server
    - Local Copilot replacement
    - Function Calling support
    - Vision API support
    - Multiple Models

All llama.cpp cmake build options can be set via the CMAKE_ARGS environment variable or via the --config-settings / -C cli flag during installation. llama-cpp-python cuda backend is already build and installed inside our container. 
```bash
root@sjsujetson-00:/workspace# python 
Python 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from llama_cpp import Llama
```

Run the test llama cpp python code:
```bash
root@sjsujetson-01:/Developer/edgeAI# python edgeLLM/llama_cpp_pythontest.py
....
Available chat formats from metadata: chat_template.default
Guessed chat format: mistral-instruct
llama_perf_context_print:        load time =    1874.08 ms
llama_perf_context_print: prompt eval time =    1873.02 ms /    11 tokens (  170.27 ms per token,     5.87 tokens per second)
llama_perf_context_print:        eval time =   25315.11 ms /   127 runs   (  199.33 ms per token,     5.02 tokens per second)
llama_perf_context_print:       total time =   27284.54 ms /   138 tokens
🕒 Inference time: 27.29 seconds
🔢 Tokens generated: 128
⚡ Tokens/sec: 4.69
```

**Optimal Settings by Device** (from unified_llm_demo.py):
```
# NVIDIA CUDA (Desktop GPUs)
n_gpu_layers=35, n_threads=8, n_batch=512, n_ctx=2048

# Jetson
n_gpu_layers=20, n_threads=6, n_batch=256, n_ctx=2048

# Apple Silicon
n_gpu_layers=0, n_threads=8, n_batch=512, n_ctx=2048

# CPU
n_gpu_layers=0, n_threads=8, n_batch=256, n_ctx=2048
```

#### **2. Ollama - Simplified LLM Deployment**

**Architecture**: Docker-based deployment with REST API
**Model Management**: Automatic model downloading and caching
**Concurrency**: Built-in request queuing and batching
**Device Availability**:
- ✅ NVIDIA Jetson (with Docker)
- ✅ NVIDIA GPUs
- ✅ x86 CPUs
- ✅ Apple Silicon (native ARM build)

<!-- **Installation**:
```bash
# macOS and Linux
curl -fsSL https://ollama.ai/install.sh | sh

# For Jetson, you may need to build from source
git clone https://github.com/ollama/ollama
cd ollama
go build
``` -->

**API Endpoint**: http://localhost:11434/api/generate

#### **3. Transformers - HuggingFace Library**

**Architecture**: Python-based with PyTorch/TensorFlow backend
**Memory Management**: Model parallelism and offloading options
**Optimization**: Supports quantization, caching, and JIT compilation
**Device Availability**:
- ✅ NVIDIA Jetson (with limitations on model size)
- ✅ NVIDIA GPUs
- ✅ x86 CPUs
- ✅ Apple Silicon (via MPS backend)

**Installation**:
```bash
# Basic installation
pip install transformers

# With PyTorch for GPU support
pip install torch transformers

# With quantization support
pip install transformers accelerate bitsandbytes
```

**Optimal Settings by Device** (from unified_llm_demo.py):
```
# NVIDIA CUDA (Desktop GPUs/Jetson)
device_map="auto", torch_dtype=torch.float16, load_in_8bit=True, use_cache=True

# Apple Silicon
device_map="mps", use_cache=True

# CPU
device_map="cpu", use_cache=True
```

#### **4. ONNX Runtime - Cross-Platform Optimization**

**Architecture**: Microsoft's cross-platform inference engine
**Optimization**: Graph optimization, operator fusion, memory planning
**Providers**: CUDA, TensorRT, CPU execution providers
**Device Availability**:
- ✅ NVIDIA Jetson (via CUDA provider)
- ✅ NVIDIA GPUs (via CUDA/TensorRT providers)
- ✅ x86 CPUs (via CPU provider)
- ✅ Apple Silicon (via CPU provider)

**Installation**:
```bash
# CPU-only version
pip install onnxruntime

# GPU-accelerated version
pip install onnxruntime-gpu

# For Jetson, you may need to build from source or use NVIDIA containers
```

**Optimal Settings by Device** (from unified_llm_demo.py):
```
# NVIDIA CUDA (Desktop GPUs/Jetson)
provider="CUDAExecutionProvider", optimization_level=99

# CPU or Apple Silicon
provider="CPUExecutionProvider", optimization_level=99
```

### 🔄 Device-Specific Optimizations

The `unified_llm_demo.py` script includes a `DeviceManager` class that automatically detects the hardware platform and applies optimal settings for each backend. Here's how it works:

#### **Device Detection Logic**:

```python
def _detect_device_type(self) -> str:
    # Check for NVIDIA GPU with CUDA
    if torch.cuda.is_available():
        # Check if it's a Jetson device
        if os.path.exists("/etc/nv_tegra_release") or \
           os.path.exists("/etc/nv_tegra_version"):
            return "jetson"
        else:
            return "cuda"
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple_silicon"
    
    # Default to CPU
    return "cpu"
```

#### **Available Optimizations by Device**:

| Optimization | Jetson | NVIDIA GPU | Apple Silicon | CPU |
|--------------|--------|------------|---------------|-----|
| ONNX         | ✅     | ✅         | ✅            | ✅  |
| Quantization | ✅     | ✅         | ❌            | ❌  |
| MPS          | ❌     | ❌         | ✅            | ❌  |
| CUDA         | ✅     | ✅         | ❌            | ❌  |
| Half Precision| ✅    | ✅         | ❌            | ❌  |
| INT8         | ✅     | ✅         | ❌            | ❌  |

### 🎯 Memory Optimization Techniques

Running LLMs on edge devices requires careful memory management. The `unified_llm_demo.py` script implements several techniques:

#### **1. Memory Optimization Function**

```python
def optimize_memory():
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache if using PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get memory info
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / (1024**3):.1f}GB")
    
    if torch.cuda.is_available():
        # Print GPU memory statistics
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_reserved = torch.cuda.memory_reserved(0)
        
        print(f"GPU Memory: {gpu_memory / (1024**3):.1f}GB total")
        print(f"GPU Allocated: {gpu_allocated / (1024**3):.1f}GB")
        print(f"GPU Reserved: {gpu_reserved / (1024**3):.1f}GB")
```

#### **2. Performance Monitoring**

The script includes a `performance_monitor` context manager that tracks:
- Execution time
- Memory usage (RAM and GPU)
- CPU usage
- GPU utilization and temperature (when available)

### 📊 Benchmarking Capabilities

The `unified_llm_demo.py` script includes a comprehensive benchmarking system through the `BenchmarkManager` class:

#### **1. Single Backend Benchmarking**

The `run_benchmark` method tests a specific backend and model with multiple prompts and runs, collecting:
- Inference times
- Memory usage
- Generated text quality

#### **2. Multi-Backend Comparison**

The `compare_backends` method allows comparing different backends and models on the same prompts, with visualization capabilities:

```python
# Example usage
benchmark_manager.compare_backends(
    prompts=sample_prompts,
    backends_models=[("llama_cpp", "llama-2-7b-chat.q4_K_M.gguf"), 
                    ("ollama", "llama2:7b-chat")],
    num_runs=3,
    max_tokens=50
)
```

#### **3. Visualization**

The `create_comparison_visualization` method generates bar charts comparing:
- Average inference time
- Memory usage
- Standard deviation

### 🚀 Running the Unified LLM Demo

The script provides a flexible command-line interface:

```bash
# List available backends
python unified_llm_demo.py --list

# Run with llama.cpp backend
python unified_llm_demo.py --backend llama_cpp \
    --model-path models/llama-2-7b-chat.q4_K_M.gguf \
    --prompt "Explain edge AI"

# Run with Ollama backend
python unified_llm_demo.py --backend ollama \
    --model-name llama2:7b-chat \
    --prompt "Explain edge AI"

# Run benchmark comparison
python unified_llm_demo.py --benchmark \
    --backends llama_cpp ollama \
    --model-names llama-2-7b-chat.q4_K_M.gguf llama2:7b-chat
```

### 🔄 GGUF Model Format

**GGUF (GPT-Generated Unified Format)** is the successor to GGML, designed for efficient LLM storage and inference:

#### **Format Advantages**:
- **Memory Mapping**: Direct file access without loading entire model into RAM
- **Metadata Storage**: Model configuration embedded in file
- **Quantization Support**: Multiple precision levels in single file
- **Cross-Platform**: Consistent format across architectures

#### **Quantization Levels for 7B Parameter Models**:

| Format | Size   | Quality | Speed    | Best For |
|--------|--------|---------|----------|----------|
| FP16   | 13.5GB | 100%    | Baseline | Maximum quality |
| Q8_0   | 7.2GB  | 99%     | 1.2x     | High quality, some speed |
| Q6_K   | 5.4GB  | 97%     | 1.5x     | Good balance |
| Q4_K_M | 4.1GB  | 95%     | 2.0x     | Recommended for most use |
| Q4_0   | 3.8GB  | 92%     | 2.2x     | Faster inference |
| Q3_K_M | 3.1GB  | 88%     | 2.5x     | Memory constrained |
| Q2_K   | 2.4GB  | 80%     | 3.0x     | Maximum speed |

For Jetson devices, the Q4_K_M format typically offers the best balance of quality, speed, and memory usage.



---

## ⚡ Jetson-Compatible Transformer Models

| Model            | Size    | Format | Notes                           |
| ---------------- | ------- | ------ | ------------------------------- |
| Mistral 7B       | 4–8GB   | GGUF   | Fast and widely supported       |
| Qwen 1.5/3 7B/8B | 5–9GB   | GGUF   | Open-source, multilingual       |
| LLaMA 2/3 7B     | 4–7GB   | GGUF   | General-purpose LLM             |
| DeepSeek 7B      | 4–8GB   | GGUF   | Math & reasoning focus          |
| DistilBERT       | \~250MB | HF     | Lightweight, good for NLP tasks |

---


### ⚠️ Common Issues and Solutions

#### Memory Issues
```python
# Problem: CUDA out of memory
# Solution: Implement memory management

import torch
import gc

def clear_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("🧹 Memory cleared")

# Use smaller batch sizes
BATCH_SIZE = 4  # Instead of 16 or 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs)
```

#### Model Loading Issues
```python
# Problem: Model fails to load
# Solution: Progressive fallback strategy

def load_model_with_fallback(model_name):
    strategies = [
        # Strategy 1: Full precision GPU
        lambda: AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto"
        ),
        # Strategy 2: Half precision GPU
        lambda: AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        ),
        # Strategy 3: 8-bit quantization
        lambda: AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto"
        ),
        # Strategy 4: CPU fallback
        lambda: AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"🔄 Trying loading strategy {i+1}...")
            model = strategy()
            print(f"✅ Model loaded with strategy {i+1}")
            return model
        except Exception as e:
            print(f"❌ Strategy {i+1} failed: {e}")
            clear_memory()
    
    raise RuntimeError("All loading strategies failed")

# Usage
model = load_model_with_fallback("gpt2-medium")
```

#### Performance Optimization
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cudnn.deterministic = False  # For better performance

# Use torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
    print("🚀 Model compiled for optimization")

# Optimize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,  # Use fast tokenizer
    padding_side="left"  # Better for generation
)
```

### 📊 Performance Monitoring Tools

```python
import psutil
import time
from contextlib import contextmanager

@contextmanager
def system_monitor():
    """Monitor system resources during inference"""
    # Initial readings
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    start_memory = psutil.virtual_memory().percent
    
    if torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    
    try:
        yield
    finally:
        # Final readings
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"\n📊 System Performance:")
        print(f"⏱️  Execution time: {end_time - start_time:.3f}s")
        print(f"💻 CPU usage: {end_cpu:.1f}%")
        print(f"🧠 RAM usage: {end_memory:.1f}%")
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_allocated() / 1024**2
            peak_gpu = torch.cuda.max_memory_allocated() / 1024**2
            print(f"🎮 GPU memory current: {current_gpu:.1f} MB")
            print(f"🔝 GPU memory peak: {peak_gpu:.1f} MB")

# Usage example
with system_monitor():
    result = model.generate(**inputs)
```

### 🎯 Jetson-Specific Optimizations

```python
# Check Jetson model and optimize accordingly
def get_jetson_config():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        
        if 'Orin Nano' in model:
            return {
                'max_memory_gb': 6,  # Leave 2GB for system
                'optimal_batch_size': 4,
                'use_fp16': True,
                'enable_flash_attention': False  # Not supported on older CUDA
            }
        elif 'Orin NX' in model:
            return {
                'max_memory_gb': 14,
                'optimal_batch_size': 8,
                'use_fp16': True,
                'enable_flash_attention': True
            }
        else:
            return {
                'max_memory_gb': 4,
                'optimal_batch_size': 2,
                'use_fp16': True,
                'enable_flash_attention': False
            }
    except:
        # Fallback for non-Jetson systems
        return {
            'max_memory_gb': 8,
            'optimal_batch_size': 8,
            'use_fp16': True,
            'enable_flash_attention': True
        }

# Apply Jetson-specific settings
config = get_jetson_config()
print(f"🤖 Detected configuration: {config}")

# Use configuration in model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if config['use_fp16'] else torch.float32,
    device_map="auto",
    max_memory={0: f"{config['max_memory_gb']}GB"}
)
```

### 📈 Benchmarking Framework

```python
class TransformerBenchmark:
    def __init__(self, model_name, device="auto"):
        self.model_name = model_name
        self.device = device
        self.results = []
    
    def benchmark_task(self, task_name, task_func, inputs, num_runs=5):
        """Benchmark a specific task"""
        print(f"\n🧪 Benchmarking {task_name}...")
        
        times = []
        for run in range(num_runs):
            start_time = time.time()
            result = task_func(inputs)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            
            if run == 0:  # Show first result
                print(f"📝 Sample output: {str(result)[:100]}...")
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        self.results.append({
            'task': task_name,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min(times),
            'max_time': max(times),
            'times': times
        })
        
        print(f"⏱️  Average: {avg_time:.3f}±{std_time:.3f}s")
        return avg_time
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n📊 BENCHMARK REPORT")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📈 Results:")
        
        for result in self.results:
            print(f"\n🎯 {result['task']}:")
            print(f"   Average: {result['avg_time']:.3f}s")
            print(f"   Std Dev: {result['std_time']:.3f}s")
            print(f"   Range: {result['min_time']:.3f}s - {result['max_time']:.3f}s")
        
        # Find best and worst performing tasks
        if self.results:
            best = min(self.results, key=lambda x: x['avg_time'])
            worst = max(self.results, key=lambda x: x['avg_time'])
            
            print(f"\n🏆 Fastest task: {best['task']} ({best['avg_time']:.3f}s)")
            print(f"🐌 Slowest task: {worst['task']} ({worst['avg_time']:.3f}s)")
            
            if len(self.results) > 1:
                speedup = worst['avg_time'] / best['avg_time']
                print(f"⚡ Performance ratio: {speedup:.2f}x")

# Example usage
benchmark = TransformerBenchmark("distilbert-base-uncased")

# Define benchmark tasks
def sentiment_task(texts):
    classifier = pipeline("sentiment-analysis")
    return [classifier(text) for text in texts]

def generation_task(prompts):
    generator = pipeline("text-generation", model="gpt2")
    return [generator(prompt, max_length=50) for prompt in prompts]

# Run benchmarks
test_texts = ["This is a test sentence."] * 5
test_prompts = ["The future of AI"] * 3

benchmark.benchmark_task("Sentiment Analysis", sentiment_task, test_texts)
benchmark.benchmark_task("Text Generation", generation_task, test_prompts)

# Generate report
benchmark.generate_report()
```
