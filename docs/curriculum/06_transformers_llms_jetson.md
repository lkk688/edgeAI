# 🤖 Transformers & LLMs on Jetson

## 🧠 What Are Transformers?

Transformers are a type of deep learning model designed to handle sequential data, such as text, audio, or even images. Introduced in the 2017 paper "Attention Is All You Need," transformers replaced recurrent neural networks in many NLP tasks.

### 🔑 Key Components

* **Self-Attention**: Each token attends to all other tokens in a sequence.
* **Positional Encoding**: Adds order information to input tokens.
* **Multi-head Attention**: Parallel attention mechanisms capture different relationships.
* **Feedforward Layers**: Apply transformations independently to each position.

### 📚 Popular Transformer Architectures

| Model      | Purpose                   | Examples                       |
| ---------- | ------------------------- | ------------------------------ |
| BERT       | Encoder (bi-directional)  | Question answering, embeddings |
| GPT        | Decoder (uni-directional) | Text generation                |
| T5         | Encoder-Decoder           | Translation, summarization     |
| LLaMA/Qwen | Open-source LLMs          | General language modeling      |

---

## 🚀 What Are LLMs?

LLMs (Large Language Models) are transformer-based models trained on vast datasets to understand and generate human-like text.

### 💬 Common Use Cases

* Chatbots and virtual assistants
* Code generation
* Summarization
* Translation

## 🤗 HuggingFace Transformers on Jetson

While large LLMs require quantization, many HuggingFace models (BERT, DistilBERT, TinyGPT) can run on Jetson using PyTorch + Transformers with ONNX export or quantized alternatives.

### ✅ Setup in PyTorch Container

```bash
docker run --rm -it --runtime nvidia \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.04-py3
```

Inside the container:

```bash
pip install transformers accelerate torch onnx optimum[onnxruntime-gpu]
```

### 🚀 Basic vs Accelerated Inference

| Approach | Speed | Memory | Complexity | Best For |
|----------|-------|--------|------------|----------|
| **Basic PyTorch** | Baseline | High | Low | Development, prototyping |
| **ONNX Runtime** | 2-3x faster | Medium | Medium | Production inference |
| **TensorRT** | 3-5x faster | Low | High | Optimized deployment |
| **Quantization** | 2-4x faster | 50% less | Medium | Resource-constrained |

---

## 🔧 Comprehensive HuggingFace Examples

### 📝 Example 1: Basic Text Classification (DistilBERT)

#### Basic Version
```python
from transformers import pipeline
import time

# Load pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Test inference
text = "Jetson Orin Nano delivers incredible AI performance at the edge!"
start_time = time.time()
result = classifier(text)
end_time = time.time()

print(f"Result: {result}")
print(f"Inference time: {end_time - start_time:.3f}s")
```

#### Accelerated Version (ONNX)
```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import time

# Load ONNX-optimized model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True,  # Convert to ONNX if not already
    provider="CUDAExecutionProvider"  # Use GPU
)

# Test inference
text = "Jetson Orin Nano delivers incredible AI performance at the edge!"
inputs = tokenizer(text, return_tensors="pt")

start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
end_time = time.time()

print(f"Predictions: {predictions}")
print(f"Inference time: {end_time - start_time:.3f}s")
```

### 🤖 Example 2: Text Generation (GPT-2)

#### Basic Version
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "Edge AI computing with Jetson"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

start_time = time.time()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
end_time = time.time()

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Generation time: {end_time - start_time:.3f}s")
```

#### Accelerated Version (Quantized + GPU)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load quantized model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer.pad_token = tokenizer.eos_token

# Generate with optimizations
prompt = "Edge AI computing with Jetson"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

start_time = time.time()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True  # Enable KV cache for faster generation
    )
end_time = time.time()

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Generation time: {end_time - start_time:.3f}s")
print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

### 🔍 Example 3: Question Answering (BERT)

#### Basic Version
```python
from transformers import pipeline

# Load QA pipeline
qa_pipeline = pipeline("question-answering", 
                      model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Define context and question
context = """
The NVIDIA Jetson Orin Nano is a powerful single-board computer designed for AI applications at the edge. 
It features an ARM Cortex-A78AE CPU and an integrated GPU with 1024 CUDA cores. 
The device supports up to 8GB of LPDDR5 memory and can deliver up to 40 TOPS of AI performance.
"""

question = "How many CUDA cores does the Jetson Orin Nano have?"

start_time = time.time()
result = qa_pipeline(question=question, context=context)
end_time = time.time()

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.3f}")
print(f"Inference time: {end_time - start_time:.3f}s")
```

#### Accelerated Version (Optimized + Batching)
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader

# Load model with optimizations
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Move to GPU and enable optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Enable torch.jit compilation for faster inference
model = torch.jit.script(model)

def answer_question_optimized(question, context):
    inputs = tokenizer.encode_plus(
        question, context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        confidence = (torch.max(start_scores) + torch.max(end_scores)).item() / 2
        
    return answer, confidence

# Test optimized version
context = """
The NVIDIA Jetson Orin Nano is a powerful single-board computer designed for AI applications at the edge. 
It features an ARM Cortex-A78AE CPU and an integrated GPU with 1024 CUDA cores. 
The device supports up to 8GB of LPDDR5 memory and can deliver up to 40 TOPS of AI performance.
"""

question = "How many CUDA cores does the Jetson Orin Nano have?"

start_time = time.time()
answer, confidence = answer_question_optimized(question, context)
end_time = time.time()

print(f"Answer: {answer}")
print(f"Confidence: {confidence:.3f}")
print(f"Inference time: {end_time - start_time:.3f}s")
```

### 🎯 Example 4: Named Entity Recognition (NER)

#### Basic Version
```python
from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("ner", 
                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                       aggregation_strategy="simple")

text = "NVIDIA Jetson Orin Nano was developed in Santa Clara, California by Jensen Huang's team."

start_time = time.time()
entities = ner_pipeline(text)
end_time = time.time()

print("Named Entities:")
for entity in entities:
    print(f"  {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.3f})")
print(f"Inference time: {end_time - start_time:.3f}s")
```

#### Accelerated Version (Batch Processing)
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Load model for batch processing
model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/distilbert-base-cased-finetuned-conll03-english"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def batch_ner(texts, batch_size=8):
    all_entities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Process each text in batch
        for j, text in enumerate(batch_texts):
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
            pred_labels = torch.argmax(predictions[j], dim=-1)
            
            entities = []
            for k, (token, label_id) in enumerate(zip(tokens, pred_labels)):
                if token not in ['[CLS]', '[SEP]', '[PAD]'] and label_id != 0:
                    label = model.config.id2label[label_id.item()]
                    entities.append((token, label))
            
            all_entities.append(entities)
    
    return all_entities

# Test batch processing
texts = [
    "NVIDIA Jetson Orin Nano was developed in Santa Clara, California.",
    "Jensen Huang founded NVIDIA Corporation in 1993.",
    "The device supports CUDA and TensorRT acceleration."
]

start_time = time.time()
batch_entities = batch_ner(texts)
end_time = time.time()

for i, entities in enumerate(batch_entities):
    print(f"Text {i+1} entities: {entities}")
print(f"Batch inference time: {end_time - start_time:.3f}s")
print(f"Average per text: {(end_time - start_time)/len(texts):.3f}s")
```

### 🚀 Example 5: Advanced Optimization Techniques

#### Memory-Efficient Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Memory-efficient model loading for large models
def load_model_efficiently(model_name, max_memory_gb=6):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate memory allocation
    max_memory = {0: f"{max_memory_gb}GB", "cpu": "2GB"}
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    
    # Load with memory management
    model = load_checkpoint_and_dispatch(
        model,
        model_name,
        device_map="auto",
        max_memory=max_memory,
        no_split_module_classes=["GPT2Block", "T5Block", "BertLayer"]
    )
    
    return model, tokenizer

# Usage
model, tokenizer = load_model_efficiently("gpt2-medium")
print(f"Model loaded successfully with memory optimization")
```

#### Dynamic Batching for Optimal Throughput
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import time

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }

def dynamic_batch_inference(texts, model, tokenizer, device):
    dataset = TextDataset(texts, tokenizer)
    
    # Find optimal batch size based on available memory
    optimal_batch_size = find_optimal_batch_size(model, tokenizer, device)
    
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False)
    
    results = []
    total_time = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            # Process predictions
            for i, pred in enumerate(predictions):
                results.append({
                    'text': batch['text'][i],
                    'prediction': pred.cpu().numpy(),
                    'confidence': torch.max(pred).item()
                })
    
    return results, total_time

def find_optimal_batch_size(model, tokenizer, device, max_batch_size=32):
    """Find the largest batch size that fits in memory"""
    for batch_size in range(max_batch_size, 0, -1):
        try:
            # Test with dummy data
            dummy_input = tokenizer(
                ["test text"] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                _ = model(**dummy_input)
            
            print(f"Optimal batch size found: {batch_size}")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return 1  # Fallback to batch size 1

# Example usage
texts = [
    "Jetson Orin Nano is perfect for edge AI applications.",
    "NVIDIA's edge computing platform enables real-time inference.",
    "Deep learning models run efficiently on Jetson devices."
] * 10  # Simulate larger dataset

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

results, inference_time = dynamic_batch_inference(texts, model, tokenizer, device)
print(f"Processed {len(texts)} texts in {inference_time:.3f}s")
print(f"Average time per text: {inference_time/len(texts):.4f}s")
```

#### Performance Monitoring and Profiling
```python
import torch
import time
import psutil
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager

@contextmanager
def performance_monitor():
    """Context manager for monitoring performance metrics"""
    # Initial measurements
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    start_cpu = psutil.cpu_percent()
    
    try:
        yield
    finally:
        # Final measurements
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        end_cpu = psutil.cpu_percent()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024**2  # MB
        cpu_usage = end_cpu - start_cpu
        
        print(f"\n📊 Performance Metrics:")
        print(f"⏱️  Execution time: {execution_time:.3f}s")
        print(f"🧠 GPU memory used: {memory_used:.1f} MB")
        print(f"💻 CPU usage change: {cpu_usage:.1f}%")
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            print(f"🎮 GPU utilization: {gpu.load*100:.1f}%")
            print(f"🌡️  GPU temperature: {gpu.temperature}°C")

def benchmark_model(model_name, prompts, num_runs=5):
    """Comprehensive model benchmarking"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n🧪 Benchmarking prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        run_times = []
        
        for run in range(num_runs):
            with performance_monitor():
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                end_time = time.time()
                
                run_time = end_time - start_time
                run_times.append(run_time)
                
                if run == 0:  # Show output for first run
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    print(f"Generated: {generated_text}")
        
        # Calculate statistics
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)
        
        results.append({
            'prompt': prompt,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'runs': run_times
        })
        
        print(f"📈 Average: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
    
    return results

# Example benchmarking
prompts = [
    "Edge AI computing with Jetson",
    "The future of artificial intelligence",
    "NVIDIA's contribution to deep learning"
]

benchmark_results = benchmark_model("gpt2", prompts, num_runs=3)

# Summary report
print("\n📋 Benchmark Summary:")
for i, result in enumerate(benchmark_results):
    print(f"Prompt {i+1}: {result['avg_time']:.3f}s average")
```

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

### 🔧 Backend Deep Dive

#### **1. llama.cpp - High-Performance C++ Engine**

**Architecture**: Pure C++ implementation with CUDA acceleration
**Memory Model**: Efficient GGUF format with mmap support
**Quantization**: K-quants (Q4_K_M, Q6_K) for optimal quality/speed

```bash
# Installation
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUBLAS=1  # Enable CUDA

# Download and convert model
wget https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin
python convert.py models/7B/ --outfile models/ggml-model-q4_0.gguf

# Run inference
./main -m models/ggml-model-q4_0.gguf -p "Hello, how are you?" -n 128
```

**Advanced Usage with Custom Parameters**:
```bash
# Optimized for Jetson Orin Nano (8GB)
./main -m models/llama-2-7b-chat.q4_K_M.gguf \
  -p "Explain quantum computing" \
  -n 256 \
  -c 2048 \
  --temp 0.7 \
  --top-p 0.9 \
  --threads 6 \
  --gpu-layers 35
```

#### **2. Ollama - Simplified LLM Deployment**

**Architecture**: Docker-based deployment with REST API
**Model Management**: Automatic model downloading and caching
**Concurrency**: Built-in request queuing and batching

```bash
# Installation
curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run model
ollama pull llama2:7b-chat-q4_K_M
ollama run llama2:7b-chat-q4_K_M
```

**Python Integration**:
```python
import requests
import json

def chat_with_ollama(prompt, model="llama2:7b-chat-q4_K_M"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 2048
        }
    }
    
    response = requests.post(url, json=data)
    return response.json()["response"]

# Usage
response = chat_with_ollama("Explain the benefits of edge AI")
print(response)
```

**Batch Processing with Ollama**:
```python
import asyncio
import aiohttp

async def batch_inference(prompts, model="llama2:7b-chat-q4_K_M"):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            task = asyncio.create_task(
                generate_async(session, prompt, model)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

async def generate_async(session, prompt, model):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    async with session.post(url, json=data) as response:
        result = await response.json()
        return result["response"]

# Usage
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "Define artificial intelligence"
]

results = asyncio.run(batch_inference(prompts))
for i, result in enumerate(results):
    print(f"Response {i+1}: {result}")
```

#### **3. llama-cpp-python - Python Bindings**

**Architecture**: Python wrapper around llama.cpp with ctypes
**Memory Management**: Automatic garbage collection and memory mapping
**Streaming**: Real-time token generation support

```bash
# Installation with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

**Basic Usage**:
```python
from llama_cpp import Llama

# Initialize model with GPU acceleration
llm = Llama(
    model_path="./models/llama-2-7b-chat.q4_K_M.gguf",
    n_gpu_layers=35,  # Offload layers to GPU
    n_ctx=2048,       # Context window
    n_threads=6,      # CPU threads
    verbose=False
)

# Generate response
output = llm(
    "Explain the importance of edge computing:",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    echo=True
)

print(output['choices'][0]['text'])
```

**Streaming Generation**:
```python
def stream_response(prompt, max_tokens=256):
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7
    )
    
    for output in stream:
        token = output['choices'][0]['text']
        print(token, end='', flush=True)
    print()  # New line at end

# Usage
stream_response("The future of AI on edge devices is")
```

**Chat Interface with Memory**:
```python
class ChatBot:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=35,
            n_ctx=4096,
            chat_format="llama-2"
        )
        self.conversation_history = []
    
    def chat(self, user_input):
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response = self.llm.create_chat_completion(
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=256
        )
        
        assistant_message = response['choices'][0]['message']['content']
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset_conversation(self):
        self.conversation_history = []

# Usage
chatbot = ChatBot("./models/llama-2-7b-chat.q4_K_M.gguf")
response = chatbot.chat("Hello! Can you help me understand transformers?")
print(response)
```

#### **4. TensorRT-LLM - Maximum Performance**

**Architecture**: NVIDIA's optimized inference engine
**Optimization**: Kernel fusion, precision calibration, memory optimization
**Deployment**: Containerized deployment with Triton Inference Server

```bash
# Installation (requires NVIDIA Container Toolkit)
docker pull nvcr.io/nvidia/tensorrt:23.08-py3

# Build optimized engine
python build.py --model_dir ./llama-2-7b-hf \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 1024 \
                --max_output_len 512 \
                --output_dir ./trt_engines/llama-2-7b-fp16
```

**Python Inference**:
```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

class TensorRTLLM:
    def __init__(self, engine_dir):
        self.runner = ModelRunner.from_dir(engine_dir)
    
    def generate(self, input_text, max_output_len=256):
        inputs = self.runner.tokenizer.encode(input_text)
        
        outputs = self.runner.generate(
            batch_input_ids=[inputs],
            max_new_tokens=max_output_len,
            temperature=0.7,
            top_p=0.9
        )
        
        output_text = self.runner.tokenizer.decode(outputs[0][0])
        return output_text

# Usage
llm = TensorRTLLM("./trt_engines/llama-2-7b-fp16")
response = llm.generate("Explain edge AI applications:")
print(response)
```

#### **5. ONNX Runtime - Cross-Platform Optimization**

**Architecture**: Microsoft's cross-platform inference engine
**Optimization**: Graph optimization, operator fusion, memory planning
**Providers**: CUDA, TensorRT, CPU execution providers

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

class ONNXLLMInference:
    def __init__(self, model_path, tokenizer_name):
        # Configure ONNX Runtime with CUDA
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def generate(self, prompt, max_length=256):
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="np", 
            padding=True, 
            truncation=True
        )
        
        # Run inference
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        outputs = self.session.run(None, ort_inputs)
        
        # Decode output
        generated_ids = outputs[0]
        response = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        return response

# Usage
llm = ONNXLLMInference(
    "./models/llama-2-7b-chat.onnx",
    "meta-llama/Llama-2-7b-chat-hf"
)
response = llm.generate("What are the advantages of ONNX Runtime?")
print(response)
```

#### **6. vLLM - High-Throughput Serving**

**Architecture**: PagedAttention for memory efficiency
**Batching**: Continuous batching with dynamic sequence lengths
**Serving**: OpenAI-compatible API server

```bash
# Installation
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048
```

**Client Usage**:
```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # Dummy key for local server
)

def chat_completion(messages, model="meta-llama/Llama-2-7b-chat-hf"):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=256
    )
    return completion.choices[0].message.content

# Usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the benefits of vLLM"}
]

response = chat_completion(messages)
print(response)
```

### 🔄 GGUF Model Format Deep Dive

**GGUF (GPT-Generated Unified Format)** is the successor to GGML, designed for efficient LLM storage and inference:

#### **Format Advantages**:
- **Memory Mapping**: Direct file access without loading into RAM
- **Metadata Storage**: Model configuration embedded in file
- **Quantization Support**: Multiple precision levels in single file
- **Cross-Platform**: Consistent format across architectures

#### **Quantization Levels**:
```python
# Quantization comparison for 7B parameter model
quantization_info = {
    "fp16": {"size": "13.5GB", "quality": "100%", "speed": "baseline"},
    "q8_0": {"size": "7.2GB", "quality": "99%", "speed": "1.2x"},
    "q6_K": {"size": "5.4GB", "quality": "97%", "speed": "1.5x"},
    "q4_K_M": {"size": "4.1GB", "quality": "95%", "speed": "2.0x"},
    "q4_0": {"size": "3.8GB", "quality": "92%", "speed": "2.2x"},
    "q3_K_M": {"size": "3.1GB", "quality": "88%", "speed": "2.5x"},
    "q2_K": {"size": "2.4GB", "quality": "80%", "speed": "3.0x"}
}

for quant, info in quantization_info.items():
    print(f"{quant}: {info['size']} | Quality: {info['quality']} | Speed: {info['speed']}")
```

### 🎯 Jetson-Specific Optimizations

#### **Memory Management**:
```python
import psutil
import gc
import torch

def optimize_jetson_memory():
    """Optimize memory usage for Jetson devices"""
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache if using PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get memory info
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / 1024**3:.1f}GB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)
        
        print(f"GPU Memory: {gpu_memory / 1024**3:.1f}GB total")
        print(f"GPU Allocated: {gpu_allocated / 1024**3:.1f}GB")
        print(f"GPU Cached: {gpu_cached / 1024**3:.1f}GB")

# Call before loading models
optimize_jetson_memory()
```

#### **Performance Monitoring**:
```python
import time
import threading
from collections import deque

class JetsonMonitor:
    def __init__(self):
        self.metrics = {
            'inference_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100)
        }
        self.monitoring = False
    
    def start_monitoring(self):
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _monitor_loop(self):
        while self.monitoring:
            # Collect metrics
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            
            if torch.cuda.is_available():
                gpu_util = torch.cuda.utilization()
                self.metrics['gpu_utilization'].append(gpu_util)
            
            time.sleep(1)
    
    def log_inference_time(self, inference_time):
        self.metrics['inference_times'].append(inference_time)
    
    def get_stats(self):
        if not self.metrics['inference_times']:
            return "No inference data available"
        
        avg_time = sum(self.metrics['inference_times']) / len(self.metrics['inference_times'])
        avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
        
        stats = f"""
        Average Inference Time: {avg_time:.2f}s
        Average Memory Usage: {avg_memory:.1f}%
        Tokens per Second: {1/avg_time:.1f}
        """
        
        if self.metrics['gpu_utilization']:
            avg_gpu = sum(self.metrics['gpu_utilization']) / len(self.metrics['gpu_utilization'])
            stats += f"Average GPU Utilization: {avg_gpu:.1f}%"
        
        return stats

# Usage
monitor = JetsonMonitor()
monitor.start_monitoring()

# During inference
start_time = time.time()
# ... run inference ...
inference_time = time.time() - start_time
monitor.log_inference_time(inference_time)

print(monitor.get_stats())
```

### 📊 Performance Benchmarking

```python
import time
import statistics
from typing import List, Dict, Any

class LLMBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_backend(self, backend_name: str, model, prompts: List[str], 
                         num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark a specific LLM backend"""
        print(f"Benchmarking {backend_name}...")
        
        inference_times = []
        token_counts = []
        
        for run in range(num_runs):
            for prompt in prompts:
                start_time = time.time()
                
                # Generate response (implementation depends on backend)
                if backend_name == "llama-cpp-python":
                    response = model(prompt, max_tokens=128)
                    tokens = len(response['choices'][0]['text'].split())
                elif backend_name == "ollama":
                    response = self._ollama_generate(prompt)
                    tokens = len(response.split())
                
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                token_counts.append(tokens)
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        avg_tokens = statistics.mean(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        results = {
            'backend': backend_name,
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'avg_tokens_generated': avg_tokens,
            'tokens_per_second': tokens_per_second,
            'total_runs': len(inference_times)
        }
        
        self.results[backend_name] = results
        return results
    
    def _ollama_generate(self, prompt: str) -> str:
        """Helper method for Ollama API calls"""
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:7b-chat-q4_K_M",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    
    def compare_backends(self) -> str:
        """Generate comparison report"""
        if not self.results:
            return "No benchmark results available"
        
        report = "\n=== LLM Backend Comparison ===\n"
        report += f"{'Backend':<20} {'Avg Time (s)':<15} {'Tokens/sec':<12} {'Std Dev':<10}\n"
        report += "-" * 60 + "\n"
        
        for backend, results in self.results.items():
            report += f"{backend:<20} {results['avg_inference_time']:<15.2f} "
            report += f"{results['tokens_per_second']:<12.1f} {results['std_inference_time']:<10.2f}\n"
        
        return report

# Usage example
benchmark = LLMBenchmark()

# Test prompts
test_prompts = [
    "Explain machine learning in simple terms.",
    "What are the benefits of edge computing?",
    "Describe the transformer architecture."
]

# Benchmark different backends (pseudo-code)
# benchmark.benchmark_backend("llama-cpp-python", llama_cpp_model, test_prompts)
# benchmark.benchmark_backend("ollama", None, test_prompts)

print(benchmark.compare_backends())
```

---



## 🧪 Comprehensive Lab: HuggingFace Transformers Optimization

### 🎯 Objective

Implement and compare basic vs accelerated transformer inference on Jetson, measuring performance improvements.

### ✅ Task 1: Basic vs Accelerated Sentiment Analysis

#### Part A: Basic Implementation
```python
from transformers import pipeline
import time
import torch

# Basic pipeline
print("🔄 Loading basic sentiment analysis pipeline...")
classifier = pipeline("sentiment-analysis")

# Test texts
texts = [
    "Jetson Orin Nano is an awesome edge AI device!",
    "The performance of this transformer model is disappointing.",
    "NVIDIA's edge computing solutions are revolutionary.",
    "This AI inference is too slow for real-time applications.",
    "The optimization techniques significantly improved speed."
]

# Benchmark basic version
print("\n📊 Basic Pipeline Performance:")
start_time = time.time()
basic_results = []
for text in texts:
    result = classifier(text)
    basic_results.append(result[0])
basic_time = time.time() - start_time

print(f"Total time: {basic_time:.3f}s")
print(f"Average per text: {basic_time/len(texts):.3f}s")
for i, (text, result) in enumerate(zip(texts, basic_results)):
    print(f"Text {i+1}: {result['label']} ({result['score']:.3f})")
```

#### Part B: Accelerated Implementation
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch
import time

# Load ONNX-optimized model
print("\n🚀 Loading ONNX-optimized model...")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    # Try to load ONNX model
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
        provider="CUDAExecutionProvider"
    )
    print("✅ ONNX model loaded successfully")
except Exception as e:
    print(f"⚠️ ONNX loading failed: {e}")
    print("📦 Falling back to PyTorch with optimizations...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

# Batch processing for acceleration
def batch_classify(texts, model, tokenizer, batch_size=8):
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device if using PyTorch
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process results
        for j, pred in enumerate(predictions):
            label_id = torch.argmax(pred).item()
            score = torch.max(pred).item()
            label = "POSITIVE" if label_id == 1 else "NEGATIVE"
            results.append({"label": label, "score": score})
    
    return results

# Benchmark accelerated version
print("\n⚡ Accelerated Pipeline Performance:")
start_time = time.time()
accelerated_results = batch_classify(texts, model, tokenizer)
accelerated_time = time.time() - start_time

print(f"Total time: {accelerated_time:.3f}s")
print(f"Average per text: {accelerated_time/len(texts):.3f}s")
print(f"🚀 Speedup: {basic_time/accelerated_time:.2f}x")

for i, (text, result) in enumerate(zip(texts, accelerated_results)):
    print(f"Text {i+1}: {result['label']} ({result['score']:.3f})")
```

### ✅ Task 2: Memory-Efficient Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

# Configure quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

print("🧠 Loading memory-efficient text generation model...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✅ Quantized model loaded")
except Exception as e:
    print(f"⚠️ Quantization failed: {e}")
    print("📦 Loading standard model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

tokenizer.pad_token = tokenizer.eos_token

# Generation function with monitoring
def generate_with_monitoring(prompt, max_length=50):
    print(f"\n🎯 Generating for: '{prompt}'")
    
    # Monitor memory before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**2
        print(f"💾 GPU memory before: {memory_before:.1f} MB")
    
    # Generate
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if hasattr(model, 'device'):
        input_ids = input_ids.to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    generation_time = time.time() - start_time
    
    # Monitor memory after
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1024**2
        print(f"💾 GPU memory after: {memory_after:.1f} MB")
        print(f"📈 Memory increase: {memory_after - memory_before:.1f} MB")
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    tokens_generated = len(output[0]) - len(input_ids[0])
    
    print(f"⏱️ Generation time: {generation_time:.3f}s")
    print(f"🔤 Tokens generated: {tokens_generated}")
    print(f"⚡ Tokens/second: {tokens_generated/generation_time:.1f}")
    print(f"📝 Generated text: {generated_text}")
    
    return generated_text, generation_time

# Test generation with different prompts
prompts = [
    "Edge AI computing with Jetson",
    "The future of artificial intelligence",
    "NVIDIA's contribution to deep learning"
]

total_time = 0
for prompt in prompts:
    _, gen_time = generate_with_monitoring(prompt)
    total_time += gen_time

print(f"\n📊 Summary:")
print(f"Total generation time: {total_time:.3f}s")
print(f"Average per prompt: {total_time/len(prompts):.3f}s")
```

### ✅ Task 3: Performance Comparison Dashboard

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Collect performance data
performance_data = defaultdict(list)

# Add your timing results here
performance_data['Basic Pipeline'].extend([basic_time/len(texts)])  # From Task 1
performance_data['Accelerated Pipeline'].extend([accelerated_time/len(texts)])  # From Task 1
performance_data['Text Generation'].extend([total_time/len(prompts)])  # From Task 2

# Create performance visualization
def create_performance_dashboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance comparison bar chart
    methods = list(performance_data.keys())
    times = [np.mean(performance_data[method]) for method in methods]
    
    bars = ax1.bar(methods, times, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_ylabel('Average Time (seconds)')
    ax1.set_title('Performance Comparison')
    ax1.set_ylim(0, max(times) * 1.2)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Memory usage simulation (replace with actual measurements)
    memory_usage = [100, 60, 80]  # Example values in MB
    ax2.bar(methods, memory_usage, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Efficiency')
    
    plt.tight_layout()
    plt.savefig('jetson_transformer_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Performance dashboard saved as 'jetson_transformer_performance.png'")

# Generate dashboard
create_performance_dashboard()
```

### 📋 Deliverables

1. **Performance Comparison Report**:
   - Basic vs accelerated inference times
   - Memory usage measurements
   - Speedup calculations

2. **Code Implementation**:
   - Working basic and accelerated versions
   - Memory monitoring integration
   - Error handling for different scenarios

3. **Visualization**:
   - Performance dashboard chart
   - Memory efficiency comparison

4. **Analysis Document**:
   - Identify bottlenecks
   - Recommend optimization strategies
   - Discuss trade-offs between speed and accuracy

### 🎯 Bonus Challenges

1. **ONNX Export**: Convert a model to ONNX format and compare performance
2. **Custom Optimization**: Implement gradient checkpointing for memory savings
3. **Multi-Model Pipeline**: Chain multiple models (e.g., NER → Classification)
4. **Real-time Inference**: Create a streaming inference pipeline

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

## 🧪 Lab: Run a Transformer Model with `llama.cpp`

### 🎯 Objective

Run a quantized transformer (e.g., Mistral) using `llama.cpp` with GPU acceleration.

### ✅ Setup

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUDA=on
make -j
```

### 📥 Download GGUF Model

```bash
curl -L -o models/mistral.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 🚀 Run Inference

```bash
./main -m models/mistral.gguf -p "What is Jetson Orin Nano?"
```

---

## 🧪 Lab: Use Ollama for REST API-based Inference

### ✅ Setup Ollama (inside Docker or local)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral
```

### 🛠️ REST API Call (Python)

```python
import requests
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "mistral",
    "prompt": "What is edge AI?",
    "stream": False
})
print(response.json()["response"])
```

---

## 🧪 Lab: Use llama-cpp-python for Local GPU Inference

### ✅ Install Python bindings

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

### 🧠 Inference with Python

```python
from llama_cpp import Llama
llm = Llama(model_path="models/mistral.gguf", n_gpu_layers=100)
print(llm("Explain transformers in 3 sentences."))
```

---

## 🔧 Troubleshooting & Best Practices

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

---

## 📋 Lab Deliverables

### 🎯 Required Deliverables

1. **Performance Analysis Report**:
   - Comparison of basic vs accelerated inference
   - Memory usage analysis
   - Throughput measurements (tokens/second)
   - Latency analysis for different batch sizes

2. **Implementation Portfolio**:
   - Working code for all optimization techniques
   - Error handling and fallback strategies
   - Memory monitoring integration
   - Performance visualization dashboard

3. **Optimization Documentation**:
   - Identified bottlenecks and solutions
   - Jetson-specific configuration recommendations
   - Trade-off analysis (speed vs accuracy vs memory)

4. **Benchmark Results**:
   - Systematic performance comparison
   - Resource utilization metrics
   - Scalability analysis

### 🏆 Bonus Achievements

- **ONNX Master**: Successfully convert and optimize models using ONNX Runtime
- **Memory Wizard**: Implement advanced memory management techniques
- **Speed Demon**: Achieve >3x speedup over baseline implementation
- **Multi-Model Maestro**: Create efficient pipeline with multiple models
- **Real-time Ranger**: Build streaming inference system

---

## 📌 Summary

### 🎓 What You've Learned

* **Transformer Fundamentals**: Understanding of attention mechanisms and model architectures
* **Optimization Techniques**: ONNX conversion, quantization, batching, and memory management
* **Jetson-Specific Optimizations**: Platform-aware configuration and resource management
* **Performance Analysis**: Systematic benchmarking and bottleneck identification
* **Production Readiness**: Error handling, monitoring, and deployment strategies

### 🚀 Key Takeaways

1. **Basic vs Accelerated**: Proper optimization can achieve 2-5x performance improvements
2. **Memory Management**: Critical for running larger models on resource-constrained devices
3. **Batch Processing**: Significantly improves throughput for multiple inputs
4. **Platform Awareness**: Jetson-specific optimizations are essential for best performance
5. **Monitoring**: Real-time performance tracking enables proactive optimization

### 🔄 Optimization Hierarchy

```
🏆 Best Performance
├── TensorRT (3-5x speedup)
├── ONNX Runtime (2-3x speedup)
├── Quantization (2-4x speedup, 50% memory)
├── Batch Processing (2-3x throughput)
├── Mixed Precision (1.5-2x speedup)
└── Basic PyTorch (baseline)
```

### 🎯 Next Steps

- Explore TensorRT optimization for maximum performance
- Implement custom CUDA kernels for specialized operations
- Build production-ready inference servers
- Integrate with edge deployment frameworks
- Develop real-time applications with optimized models

→ Next: [NLP + Inference Optimization](07_nlp_applications_llm_optimization.md)
