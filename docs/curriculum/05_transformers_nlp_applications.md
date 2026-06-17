# 🤖 Transformers on Jetson
**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)


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

## 🚀 Getting Started: the Jetson container

Run everything inside the `jetson-dev` container, which ships with PyTorch (CUDA) and the HuggingFace `transformers`, `datasets`, `accelerate`, `sentencepiece`, and `sacremoses` libraries preinstalled:

```bash
sjsujetsontool shell                 # /Developer is mounted into the container
cd /Developer/edgeAI/jetson          # toolkits live here
python3 -c "import transformers, torch; print(transformers.__version__, torch.cuda.is_available())"
```

The course repo is at **`/Developer/edgeAI`** (mounted from the host). Models download from HuggingFace on first use and are cached. Pass `device=0` to a pipeline to run on the GPU.

> [!TIP]
> Missing a package? `pip install <pkg>` inside the container; an instructor can then commit/publish the image so all nodes get it via `sjsujetsontool update`.

## 🤗 HuggingFace Transformers on Jetson

While large LLMs require quantization, many HuggingFace models (BERT, DistilBERT, TinyGPT) can run on Jetson using PyTorch + Transformers with ONNX export or quantized alternatives.


### 🚀 Basic vs Accelerated Inference

| Approach | Speed | Memory | Complexity | Best For |
|----------|-------|--------|------------|----------|
| **Basic PyTorch** | Baseline | High | Low | Development, prototyping |
| **ONNX Runtime** | 2-3x faster | Medium | Medium | Production inference |
| **TensorRT** | 3-5x faster | Low | High | Optimized deployment |
| **Quantization** | 2-4x faster | 50% less | Medium | Resource-constrained |

## ✨ What is NLP?

Natural Language Processing (NLP) is a subfield of AI that enables machines to read, understand, and generate human language.

### 💬 Common NLP Tasks

* **Text Classification** (e.g., sentiment analysis, spam detection)
* **Named Entity Recognition (NER)** (extracting entities like names, locations)
* **Machine Translation** (translating between languages)
* **Question Answering** (extracting answers from context)
* **Text Summarization** (generating concise summaries)
* **Chatbots & Conversational AI** (interactive dialogue systems)
* **Text Generation** (creating human-like text)
* **Information Extraction** (structured data from unstructured text)

---

## 🧪 Hands-on: HuggingFace `pipeline` examples

The `pipeline` API is the fastest way to run a task — it loads a suitable model + tokenizer and handles pre/post-processing. Run these inside the container (start `python3` and paste, or save to a file). Each downloads a small model on first run; add `device=0` to use the GPU.

### 1. Sentiment analysis (text classification)
```python
from transformers import pipeline
clf = pipeline("sentiment-analysis", device=0)
print(clf("Jetson is amazing for edge AI!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. Zero-shot classification (no training, your own labels)
```python
clf = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0)
print(clf("The new GPU drivers improved frame rates a lot.",
          candidate_labels=["hardware", "food", "politics"]))
# highest score on 'hardware'
```

### 3. Named Entity Recognition (NER)
```python
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=0)
print(ner("Dr. Liu teaches at San Jose State University in California."))
# entities: PER 'Dr. Liu', ORG 'San Jose State University', LOC 'California'
```

### 4. Question answering (extractive)
```python
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)
print(qa(question="How many CUDA cores does it have?",
         context="The Jetson Orin Nano has 1024 CUDA cores and 32 Tensor cores."))
# {'answer': '1024', 'score': ...}
```

### 5. Summarization
```python
summarizer = pipeline("summarization", model="t5-small", device=0)
text = ("The NVIDIA Jetson Orin Nano is a compact edge AI computer with an "
        "Ampere GPU, ideal for running CNNs and small language models locally.")
print(summarizer(text, max_length=40, min_length=10)[0]["summary_text"])
```

### 6. Translation (English → French)
```python
tr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=0)
print(tr("Edge AI runs models directly on the device.")[0]["translation_text"])
# requires sentencepiece + sacremoses (preinstalled)
```

### 7. Fill-mask (masked language modeling)
```python
fill = pipeline("fill-mask", model="distilbert-base-uncased", device=0)
print([r["token_str"] for r in fill("The Jetson is great for [MASK] AI.")][:3])
```

### 8. Text generation
```python
gen = pipeline("text-generation", model="distilgpt2", device=0)
print(gen("Edge AI on the Jetson lets you", max_new_tokens=30)[0]["generated_text"])
```

### 9. Sentence embeddings (semantic similarity)
```python
import numpy as np
emb = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=0)
def vec(s):
    a = np.array(emb(s)[0]).mean(axis=0)      # mean-pool token vectors
    return a / np.linalg.norm(a)
a, b = vec("How fast is the Jetson?"), vec("What is the speed of the Orin Nano?")
print("cosine similarity:", float(a @ b))     # ~0.5 for related sentences, lower for unrelated
```

> [!TIP]
> **Loading a model directly** (more control than `pipeline`): use `AutoTokenizer`/`AutoModelForSequenceClassification`, move it to `cuda`, and call it with `torch.no_grad()`. For Jetson, add `torch_dtype=torch.float16` to halve memory.

---

## 🔧 Comprehensive HuggingFace Examples with `transformers_llm_demo.py`

Instead of implementing individual examples, we've created a comprehensive demonstration script called `transformers_llm_demo.py` that showcases various NLP applications using HuggingFace transformers with optimization techniques specifically designed for Jetson devices.

This script provides a modular, command-line driven interface for exploring different NLP tasks and acceleration methods. Let's explore the key features and optimization techniques implemented in this demo.

### 📋 Available Applications

The `transformers_llm_demo.py` script supports seven different NLP applications:

1. **Text Classification (Sentiment Analysis)**
   - Analyzes text sentiment using DistilBERT models
   - Provides both basic and ONNX-optimized implementations

2. **Text Generation (GPT-2)**
   - Generates text continuations from prompts using GPT-2
   - Implements both basic and quantized+GPU accelerated versions

3. **Question Answering (BERT)**
   - Extracts answers from context passages using BERT models
   - Offers basic pipeline and optimized JIT-compiled implementations

4. **Named Entity Recognition (NER)**
   - Identifies entities (people, organizations, locations) in text
   - Provides both basic and batch-optimized implementations

5. **Batch Processing Demo**
   - Demonstrates efficient processing of multiple texts
   - Automatically determines optimal batch sizes for your hardware

6. **Model Benchmarking**
   - Measures performance metrics across multiple runs
   - Reports detailed statistics on inference time and resource usage

7. **Performance Comparison**
   - Directly compares basic vs. optimized implementations
   - Calculates speedup factors and memory efficiency gains

### ⚡ Optimization Techniques

The demo implements several optimization techniques that are particularly valuable for edge devices like the Jetson:

#### 1. ONNX Runtime and TensorRT Acceleration

**What it does:** Provides hardware-optimized inference using ONNX Runtime with GPU acceleration and TensorRT integration for maximum performance on Jetson devices.

**Implementation details:**
- Uses `onnxruntime` directly with CUDA execution provider for GPU acceleration
- Integrates `tensorrt` for additional optimization on NVIDIA hardware
- Automatically selects appropriate execution provider (CUDA, TensorRT, or CPU)
- Handles fallback to basic implementation if acceleration libraries are unavailable

**What you need to add:**
- Install ONNX Runtime GPU: `pip install onnxruntime-gpu`
- Install TensorRT: Follow NVIDIA's installation guide for your Jetson device
- For optimal performance, ensure both libraries are properly configured for your hardware

#### 2. 8-bit Quantization

**What it does:** Reduces model precision from 32-bit to 8-bit, decreasing memory usage and increasing inference speed.

**Implementation details:**
- Uses `BitsAndBytesConfig` for configuring quantization parameters
- Enables FP32 CPU offloading for handling operations not supported in INT8
- Combines with FP16 (half-precision) for operations that benefit from it

**What you need to add:**
- Install bitsandbytes: `pip install bitsandbytes`
- May require Jetson-specific compilation for optimal performance

#### 3. JIT Compilation

**What it does:** Compiles model operations into optimized machine code at runtime.

**Implementation details:**
- Uses `torch.jit.script()` to compile models
- Implements graceful fallback if compilation fails
- Applied to question answering models for faster inference

**What you need to add:**
- No additional packages required (built into PyTorch)
- Ensure you're using a recent PyTorch version with good JIT support

#### 4. Batch Processing

**What it does:** Processes multiple inputs simultaneously for higher throughput.

**Implementation details:**
- Custom `TextDataset` class for efficient batch handling
- Dynamic batch size determination based on available memory
- Particularly effective for NER and classification tasks

**What you need to add:**
- No additional packages required
- Consider adjusting batch sizes based on your specific Jetson model

#### 5. GPU Memory Optimization

**What it does:** Carefully manages GPU memory to prevent out-of-memory errors on memory-constrained devices.

**Implementation details:**
- Implements `find_optimal_batch_size()` to automatically determine the largest workable batch size
- Uses `torch.cuda.empty_cache()` to free memory between operations
- Monitors memory usage with the `performance_monitor()` context manager

**What you need to add:**
- Optional: Install GPUtil for enhanced GPU monitoring: `pip install gputil`

#### 6. KV Caching for Text Generation

**What it does:** Caches key-value pairs in transformer attention layers to avoid redundant computations during text generation.

**Implementation details:**
- Enables `use_cache=True` in the model generation parameters
- Particularly effective for autoregressive generation tasks
- Combined with quantization for maximum efficiency

**What you need to add:**
- No additional packages required (built into transformers library)

### 🚀 Using the Demo

The demo can be run from the command line with various options:

```bash
# List available applications
python transformers_llm_demo.py --list

# Run text classification with optimization
python transformers_llm_demo.py --app 1 --text "Jetson is amazing for edge AI!" --optimize

# Run text generation with custom parameters
python transformers_llm_demo.py --app 2 --prompt "Edge AI computing with Jetson" --max-length 100

# Run question answering
python transformers_llm_demo.py --app 3 --question "How many CUDA cores?" --context "The Jetson has 1024 CUDA cores"
```

The script provides detailed performance metrics for each run, including:
- Inference time
- Memory usage
- CPU/GPU utilization
- Temperature monitoring (when available)

### 📊 Performance Monitoring

The demo includes a comprehensive performance monitoring system that tracks:

- Execution time for each operation
- GPU memory allocation and usage
- CPU utilization changes
- GPU load and temperature (when available)

This monitoring helps identify bottlenecks and optimize your models for the specific constraints of Jetson devices.
