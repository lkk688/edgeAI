# 🧠 NLP Applications & LLM Optimization on Jetson

## ✨ What is NLP?

Natural Language Processing (NLP) is a subfield of AI that enables machines to read, understand, and generate human language.

### 💬 Common NLP Tasks

* Text classification (e.g., sentiment analysis)
* Named Entity Recognition (NER)
* Machine translation
* Question answering
* Summarization
* Chatbots & agents

---

## 🤖 Why Optimize LLMs on Jetson?

Jetson Orin Nano has limited power and memory (e.g., 8GB), so optimizing models for:

* 💾 Lower memory usage
* ⚡ Faster inference latency
* 🔌 Better energy efficiency

Enables real-time NLP applications at the edge.

---

## 🚀 Optimization Strategies

### ✅ 1. Model Quantization

Quantization reduces the precision of model weights (e.g., FP32 → INT8 or Q4) to shrink size and improve inference speed.

#### 🔍 What is Q4\_K\_M?

* Q4 = 4-bit quantization (16x smaller than FP32)
* K = Grouped quantization for accuracy preservation
* M = Variant with optimized metadata handling

Q4\_K\_M is commonly used in `llama.cpp` for **best quality/speed tradeoff** on Jetson.

### ✅ 2. Use Smaller or Distilled Models

Distillation creates smaller models (e.g., DistilBERT) by mimicking larger models while reducing parameters.

* Faster and lighter than full LLMs

### ✅ 3. Use TensorRT or ONNX for Inference

Export HuggingFace or PyTorch models to ONNX and use:

* `onnxruntime-gpu`
* `TensorRT` engines (for low latency and reduced memory use)

### ✅ 4. Offload Selected Layers

For large models, tools like `llama-cpp-python` allow setting `n_gpu_layers` to control how many transformer layers use GPU vs CPU.

---

## 🧪 Lab: Compare LLM Inference in Containers

### 🎯 Objective

Evaluate inference speed and memory usage for different LLM deployment methods on Jetson inside Docker containers.

### 🔧 Setup Container for Each Method

#### HuggingFace Container:

```bash
docker run --rm -it --runtime nvidia \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
```

Inside container:

```bash
pip install transformers accelerate torch
```

#### llama.cpp Container:

```bash
docker run --rm -it --runtime nvidia \
  -v $(pwd)/models:/models \
  jetson-llama-cpp /bin/bash
```

(Assumes container has CUDA + llama.cpp compiled)

#### Ollama Container:

```bash
docker run --rm -it --network host \
  -v ollama:/root/.ollama ollama/ollama
```

---

### 🔁 Inference Tasks (Same Prompt)

Prompt: "Explain the future of AI in education."

#### ✅ HuggingFace (inside PyTorch container)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

start = time.time()
output = model.generate(tokenizer("Explain the future of AI in education.", return_tensors="pt").input_ids)
print("Latency:", time.time() - start)
```

#### ✅ llama-cpp-python (inside custom container)

```python
from llama_cpp import Llama
llm = Llama(model_path="/models/mistral.gguf", n_gpu_layers=80)
print(llm("Explain the future of AI in education."))
```

#### ✅ Ollama API (outside or in container)

```python
import requests
r = requests.post("http://localhost:11434/api/generate", json={
  "model": "mistral",
  "prompt": "Explain the future of AI in education.",
  "stream": False
})
print(r.json()["response"])
```

---

## 📊 Record Results

| Method              | Latency (s) | Tokens/sec | GPU Mem (MB) |
| ------------------- | ----------- | ---------- | ------------ |
| HuggingFace PyTorch |             |            |              |
| llama-cpp-python    |             |            |              |
| Ollama REST API     |             |            |              |

Use `tegrastats` or `jtop` to observe GPU memory and CPU usage during inference.

---

## 📋 Lab Deliverables

* Completed benchmark table
* Screenshots of `tegrastats` during inference
* Analysis: Which approach is fastest, lightest, and most accurate for Jetson?

---

## 🧪 Bonus Lab: Export HuggingFace → ONNX → TensorRT

1. Export:

```python
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
dummy = torch.randint(0, 100, (1, 64))
torch.onnx.export(model, (dummy,), "model.onnx", input_names=["input_ids"])
```

2. Convert:

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt
```

3. Run using TensorRT Python bindings or `onnxruntime-gpu`

---

## 📌 Summary

* Quantization (e.g., Q4\_K\_M) is critical for deploying LLMs on Jetson
* PyTorch, llama.cpp, and Ollama offer tradeoffs in speed and usability
* Docker-based environments isolate dependencies for testing
* Hands-on comparison helps students choose optimal deployment method

→ Next: [Prompt Engineering](08_prompt_engineering_langchain_jetson.md)
