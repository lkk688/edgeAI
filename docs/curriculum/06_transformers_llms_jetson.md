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

---

## 🛠️ Running Transformers on Jetson

Running LLMs on Jetson Orin Nano is feasible with quantized models and optimized backends like:

* `llama.cpp` (C++-based inference engine with CUDA)
* `ollama` (Easy LLM deployment with REST API)
* `llama-cpp-python` (Python bindings for llama.cpp)

For inference on Jetson:

* Use **GGUF models**: quantized transformer weights in optimized format
* Choose **Q4\_K\_M** or **Q6\_K** quantization for best performance vs. quality

### 🔄 GGUF Inference Stack

1. Download GGUF model (e.g., from HuggingFace)
2. Run using `llama.cpp` or `ollama` CLI
3. Integrate with Python/REST API for apps

---

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
pip install transformers accelerate torch onnx
```

---

## 🧪 Lab: Run HuggingFace Transformer Model in PyTorch Container

### 🎯 Objective

Download and run a HuggingFace transformer model for text classification or generation.

### ✅ Task 1: Sentiment Classification with DistilBERT

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("Jetson Nano is an awesome edge AI device!")
print(result)
```

### ✅ Task 2: Text Generation with TinyGPT2

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

input_ids = tokenizer("Jetson devices are", return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=30)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 📋 Deliverables

* Output from both tasks
* Screenshot of Python session
* Optional: Export to ONNX using `torch.onnx.export`

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

## 📋 Lab Deliverables

* Output of model inference using CLI and Python
* Screenshot of GPU stats (e.g., `tegrastats`)
* Bonus: compare speed of llama.cpp vs ollama vs HuggingFace PyTorch

---

## 📌 Summary

* Transformers use self-attention for sequence modeling
* LLMs can run on Jetson using quantized GGUF models or lightweight HF models
* `llama.cpp` + CUDA enables local inference
* Labs use CLI, REST API, HuggingFace and Python tools

→ Next: [NLP + Inference Optimization](07_nlp_applications_llm_optimization.md)
