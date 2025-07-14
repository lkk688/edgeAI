# 🧠 NLP Applications & LLM Optimization on Jetson

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

## 🎯 Popular NLP Applications & Jetson Optimization

### 1. 😊 Sentiment Analysis

**Application**: Analyzing emotions and opinions in text (social media, reviews, customer feedback)

**Popular Datasets**:
* **IMDB Movie Reviews** (50K reviews, binary sentiment)
* **Stanford Sentiment Treebank (SST)** (fine-grained sentiment)
* **Amazon Product Reviews** (multi-domain sentiment)
* **Twitter Sentiment140** (1.6M tweets)

**Jetson Optimization**:
```python
# Optimized DistilBERT for sentiment analysis
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Use quantized model for faster inference
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    torch_dtype=torch.float16  # Half precision for memory efficiency
)

# Optimize for Jetson
model.eval()
model = torch.jit.script(model)  # TorchScript compilation
```

**Performance Metrics**:
* Accuracy: 91-93% on SST-2
* Inference time: ~15ms on Jetson Orin Nano
* Memory usage: ~400MB

---

### 2. 🏷️ Named Entity Recognition (NER)

**Application**: Extracting entities like persons, organizations, locations from text

**Popular Datasets**:
* **CoNLL-2003** (English NER benchmark)
* **OntoNotes 5.0** (18 entity types)
* **WikiNER** (multilingual NER)
* **MIT Restaurant/Movie** (domain-specific NER)

**Jetson Optimization**:
```python
# Efficient NER with spaCy on Jetson
import spacy
from spacy import displacy

# Load optimized small model
nlp = spacy.load("en_core_web_sm")

# Batch processing for efficiency
def process_ner_batch(texts, batch_size=32):
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                   for ent in doc.ents]
        results.append(entities)
    return results
```

**Performance Metrics**:
* F1-Score: 88-92% on CoNLL-2003
* Processing speed: ~1000 tokens/sec on Jetson
* Memory usage: ~200MB

---

### 3. ❓ Question Answering

**Application**: Extracting answers from context passages

**Popular Datasets**:
* **SQuAD 1.1/2.0** (Stanford Question Answering Dataset)
* **Natural Questions** (real Google search queries)
* **MS MARCO** (machine reading comprehension)
* **QuAC** (conversational question answering)

**Jetson Optimization**:
```python
# Optimized BERT-based QA system
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class OptimizedQASystem:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-cased-distilled-squad",
            torch_dtype=torch.float16
        )
        self.model.eval()
    
    def answer_question(self, question, context, max_length=384):
        inputs = self.tokenizer.encode_plus(
            question, context,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
        # Extract answer
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer
```

**Performance Metrics**:
* EM Score: 78-82% on SQuAD 1.1
* F1 Score: 85-88% on SQuAD 1.1
* Inference time: ~50ms per question on Jetson

---

### 4. 📝 Text Summarization

**Application**: Generating concise summaries of long documents

**Popular Datasets**:
* **CNN/DailyMail** (news article summarization)
* **XSum** (BBC articles with single-sentence summaries)
* **Reddit TIFU** (informal text summarization)
* **PubMed** (scientific paper abstracts)

**Jetson Optimization**:
```python
# Efficient summarization with T5-small
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class JetsonSummarizer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-small",
            torch_dtype=torch.float16
        )
        self.model.eval()
    
    def summarize(self, text, max_length=150, min_length=30):
        input_text = f"summarize: {text}"
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=2,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
```

**Performance Metrics**:
* ROUGE-1: 28-32% on CNN/DailyMail
* ROUGE-L: 25-28% on CNN/DailyMail
* Generation time: ~200ms per summary on Jetson

---

### 5. 🌐 Machine Translation

**Application**: Translating text between different languages

**Popular Datasets**:
* **WMT (Workshop on Machine Translation)** (annual translation tasks)
* **OPUS** (collection of translated texts)
* **Multi30K** (multilingual image descriptions)
* **FLORES** (low-resource language evaluation)

**Jetson Optimization**:
```python
# Efficient translation with MarianMT
from transformers import MarianMTModel, MarianTokenizer
import torch

class JetsonTranslator:
    def __init__(self, src_lang="en", tgt_lang="fr"):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.model.eval()
    
    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=128)
        
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# Batch translation for efficiency
def translate_batch(texts, translator, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = [translator.translate(text) for text in batch]
        results.extend(batch_results)
    return results
```

**Performance Metrics**:
* BLEU Score: 25-35% (depending on language pair)
* Translation speed: ~100 words/sec on Jetson
* Memory usage: ~600MB

---

### 6. 🤖 Conversational AI & Chatbots

**Application**: Interactive dialogue systems and virtual assistants

**Popular Datasets**:
* **PersonaChat** (personality-based conversations)
* **Empathetic Dialogues** (emotion-aware conversations)
* **MultiWOZ** (task-oriented dialogues)
* **BlendedSkillTalk** (open-domain conversations)

**Jetson Optimization**:
```python
# Efficient chatbot with DialoGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class JetsonChatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float16
        )
        self.model.eval()
        self.chat_history_ids = None
    
    def chat(self, user_input):
        # Encode user input
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )
        
        # Append to chat history
        bot_input_ids = torch.cat([
            self.chat_history_ids, new_user_input_ids
        ], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
        
        # Generate response
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=1000,
                num_beams=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        return response
    
    def reset_conversation(self):
        self.chat_history_ids = None
```

**Performance Metrics**:
* Perplexity: 15-20 on PersonaChat
* Response time: ~300ms per turn on Jetson
* Context length: Up to 512 tokens

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

## 📊 NLP Application Evaluation Labs

### 🧪 Lab 1: Multi-Application NLP Benchmark Suite

**Objective**: Evaluate and compare different NLP applications on Jetson using standardized datasets

#### Setup Evaluation Environment

```bash
# Create evaluation container
docker run --rm -it --runtime nvidia \
  -v $(pwd)/nlp_eval:/workspace \
  -v $(pwd)/datasets:/datasets \
  nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash

# Install evaluation dependencies
pip install transformers datasets evaluate rouge-score sacrebleu spacy
python -m spacy download en_core_web_sm
```

#### Comprehensive Evaluation Script

```python
# nlp_evaluation_suite.py
import time
import torch
import psutil
import json
from datasets import load_dataset
from transformers import pipeline
from evaluate import load
import numpy as np

class JetsonNLPEvaluator:
    def __init__(self):
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage"""
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        return {
            'result': result,
            'latency': end_time - start_time,
            'memory_mb': (peak_memory - start_memory) / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent()
        }
    
    def evaluate_sentiment_analysis(self):
        """Evaluate sentiment analysis on IMDB dataset"""
        print("🔍 Evaluating Sentiment Analysis...")
        
        # Load model and dataset
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        dataset = load_dataset("imdb", split="test[:100]")  # Sample for quick eval
        
        def run_sentiment_batch(texts):
            return classifier(texts, batch_size=8)
        
        # Measure performance
        texts = [sample['text'][:512] for sample in dataset]  # Truncate for speed
        perf = self.measure_performance(run_sentiment_batch, texts)
        
        # Calculate accuracy
        predictions = perf['result']
        labels = [sample['label'] for sample in dataset]
        
        correct = sum(1 for pred, label in zip(predictions, labels)
                     if (pred['label'] == 'POSITIVE' and label == 1) or 
                        (pred['label'] == 'NEGATIVE' and label == 0))
        
        accuracy = correct / len(labels)
        
        self.results['sentiment_analysis'] = {
            'accuracy': accuracy,
            'avg_latency_ms': (perf['latency'] / len(texts)) * 1000,
            'throughput_samples_sec': len(texts) / perf['latency'],
            'memory_mb': perf['memory_mb'],
            'dataset': 'IMDB (100 samples)'
        }
    
    def evaluate_question_answering(self):
        """Evaluate QA on SQuAD dataset"""
        print("🔍 Evaluating Question Answering...")
        
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        dataset = load_dataset("squad", split="validation[:50]")
        
        def run_qa_batch(qa_pairs):
            results = []
            for qa in qa_pairs:
                result = qa_pipeline(
                    question=qa['question'],
                    context=qa['context']
                )
                results.append(result)
            return results
        
        qa_pairs = [{
            'question': sample['question'],
            'context': sample['context']
        } for sample in dataset]
        
        perf = self.measure_performance(run_qa_batch, qa_pairs)
        
        # Calculate F1 and EM scores
        predictions = [pred['answer'] for pred in perf['result']]
        references = [[sample['answers']['text'][0]] for sample in dataset]
        
        # Simple F1 calculation (token overlap)
        f1_scores = []
        for pred, ref_list in zip(predictions, references):
            ref = ref_list[0]
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0:
                f1_scores.append(0.0)
                continue
                
            precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
            recall = len(pred_tokens & ref_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        
        self.results['question_answering'] = {
            'f1_score': avg_f1,
            'avg_latency_ms': (perf['latency'] / len(qa_pairs)) * 1000,
            'throughput_qa_sec': len(qa_pairs) / perf['latency'],
            'memory_mb': perf['memory_mb'],
            'dataset': 'SQuAD (50 samples)'
        }
    
    def evaluate_summarization(self):
        """Evaluate text summarization"""
        print("🔍 Evaluating Text Summarization...")
        
        summarizer = pipeline(
            "summarization",
            model="t5-small",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        # Use CNN/DailyMail dataset
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:20]")
        
        def run_summarization_batch(articles):
            return summarizer(
                articles,
                max_length=150,
                min_length=30,
                batch_size=2
            )
        
        articles = [sample['article'][:1024] for sample in dataset]  # Truncate
        perf = self.measure_performance(run_summarization_batch, articles)
        
        # Calculate ROUGE scores
        rouge = load("rouge")
        predictions = [pred['summary_text'] for pred in perf['result']]
        references = [sample['highlights'] for sample in dataset]
        
        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references
        )
        
        self.results['summarization'] = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'avg_latency_ms': (perf['latency'] / len(articles)) * 1000,
            'throughput_articles_sec': len(articles) / perf['latency'],
            'memory_mb': perf['memory_mb'],
            'dataset': 'CNN/DailyMail (20 samples)'
        }
    
    def evaluate_ner(self):
        """Evaluate Named Entity Recognition"""
        print("🔍 Evaluating Named Entity Recognition...")
        
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        dataset = load_dataset("conll2003", split="test[:100]")
        
        def run_ner_batch(texts):
            return ner_pipeline(texts, batch_size=8)
        
        texts = [" ".join(sample['tokens']) for sample in dataset]
        perf = self.measure_performance(run_ner_batch, texts)
        
        self.results['named_entity_recognition'] = {
            'avg_latency_ms': (perf['latency'] / len(texts)) * 1000,
            'throughput_texts_sec': len(texts) / perf['latency'],
            'memory_mb': perf['memory_mb'],
            'dataset': 'CoNLL-2003 (100 samples)'
        }
    
    def run_full_evaluation(self):
        """Run complete NLP evaluation suite"""
        print("🚀 Starting Jetson NLP Evaluation Suite...")
        
        self.evaluate_sentiment_analysis()
        self.evaluate_question_answering()
        self.evaluate_summarization()
        self.evaluate_ner()
        
        return self.results
    
    def save_results(self, filename="jetson_nlp_results.json"):
        """Save evaluation results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Results saved to {filename}")

# Run evaluation
if __name__ == "__main__":
    evaluator = JetsonNLPEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.save_results()
    
    # Print summary
    print("\n📈 Evaluation Summary:")
    for task, metrics in results.items():
        print(f"\n{task.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
```

---

### 🧪 Lab 2: Advanced Optimization Techniques

**Objective**: Implement and compare advanced optimization strategies for NLP models on Jetson

#### Dynamic Quantization Comparison

```python
# quantization_comparison.py
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.quantization as quantization

class QuantizationBenchmark:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def benchmark_model(self, model, test_texts, num_runs=10):
        """Benchmark model performance"""
        model.eval()
        
        # Warmup
        inputs = self.tokenizer(test_texts[0], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            _ = model(**inputs)
        
        # Measure performance
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for _ in range(num_runs):
            for text in test_texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        avg_latency = (end_time - start_time) / (num_runs * len(test_texts))
        memory_usage = (peak_memory - start_memory) / 1024 / 1024  # MB
        
        return {
            'avg_latency_ms': avg_latency * 1000,
            'memory_mb': memory_usage,
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
    
    def compare_quantization_methods(self):
        """Compare different quantization approaches"""
        test_texts = [
            "This movie is absolutely fantastic!",
            "I hate this boring film.",
            "The plot was okay, nothing special.",
            "Amazing cinematography and great acting.",
            "Worst movie I've ever seen."
        ]
        
        results = {}
        
        # 1. Original FP32 model
        print("📊 Testing FP32 model...")
        model_fp32 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        results['fp32'] = self.benchmark_model(model_fp32, test_texts)
        
        # 2. FP16 model
        print("📊 Testing FP16 model...")
        model_fp16 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            model_fp16 = model_fp16.cuda()
        results['fp16'] = self.benchmark_model(model_fp16, test_texts)
        
        # 3. Dynamic quantization (INT8)
        print("📊 Testing Dynamic Quantization (INT8)...")
        model_int8 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model_int8 = torch.quantization.quantize_dynamic(
            model_int8, {torch.nn.Linear}, dtype=torch.qint8
        )
        results['int8_dynamic'] = self.benchmark_model(model_int8, test_texts)
        
        # 4. TorchScript optimization
        print("📊 Testing TorchScript optimization...")
        model_script = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model_script.eval()
        example_input = self.tokenizer(
            test_texts[0], return_tensors="pt", padding=True, truncation=True
        )
        model_script = torch.jit.trace(model_script, (example_input['input_ids'],))
        
        # Custom benchmark for TorchScript
        def benchmark_torchscript(model, texts):
            start_time = time.time()
            for text in texts * 10:  # 10 runs
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    _ = model(inputs['input_ids'])
            end_time = time.time()
            return {
                'avg_latency_ms': ((end_time - start_time) / (len(texts) * 10)) * 1000,
                'memory_mb': 0,  # Simplified
                'model_size_mb': 0  # Simplified
            }
        
        results['torchscript'] = benchmark_torchscript(model_script, test_texts)
        
        return results
    
    def print_comparison(self, results):
        """Print comparison results"""
        print("\n🔍 Quantization Comparison Results:")
        print(f"{'Method':<15} {'Latency (ms)':<12} {'Memory (MB)':<12} {'Model Size (MB)':<15}")
        print("-" * 60)
        
        for method, metrics in results.items():
            print(f"{method:<15} {metrics['avg_latency_ms']:<12.2f} "
                  f"{metrics['memory_mb']:<12.1f} {metrics['model_size_mb']:<15.1f}")
        
        # Calculate speedup and compression ratios
        fp32_latency = results['fp32']['avg_latency_ms']
        fp32_size = results['fp32']['model_size_mb']
        
        print("\n📈 Optimization Gains:")
        for method, metrics in results.items():
            if method != 'fp32':
                speedup = fp32_latency / metrics['avg_latency_ms']
                compression = fp32_size / metrics['model_size_mb'] if metrics['model_size_mb'] > 0 else 0
                print(f"{method}: {speedup:.2f}x speedup, {compression:.2f}x compression")

# Run quantization benchmark
if __name__ == "__main__":
    benchmark = QuantizationBenchmark()
    results = benchmark.compare_quantization_methods()
    benchmark.print_comparison(results)
```

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

### For Lab 1 (Multi-Application Benchmark):
* Completed evaluation results JSON file
* Performance comparison charts for all NLP tasks
* Analysis report identifying best models for each task on Jetson
* Resource utilization graphs (`tegrastats` screenshots)

### For Lab 2 (Optimization Techniques):
* Quantization comparison table
* Memory usage analysis
* Speedup and compression ratio calculations
* Recommendations for production deployment

### For Lab 3 (LLM Container Comparison):
* Completed benchmark table
* Screenshots of `tegrastats` during inference
* Analysis: Which approach is fastest, lightest, and most accurate for Jetson?

---

## 🎯 Advanced NLP Optimization Strategies

### 1. 🔧 Model Pruning for Jetson

```python
# model_pruning.py
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification

def prune_model(model, pruning_ratio=0.2):
    """Apply structured pruning to transformer model"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    return model

# Example usage
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
pruned_model = prune_model(model, pruning_ratio=0.3)
```

### 2. 🚀 Knowledge Distillation

```python
# knowledge_distillation.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculate distillation loss"""
        # Soft targets from teacher
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss
        distill_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        return total_loss
```

### 3. 🔄 Dynamic Batching for Real-time Inference

```python
# dynamic_batching.py
import asyncio
import time
from collections import deque
from typing import List, Tuple

class DynamicBatcher:
    def __init__(self, model, tokenizer, max_batch_size=8, max_wait_time=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = deque()
        self.processing = False
    
    async def add_request(self, text: str) -> str:
        """Add inference request to queue"""
        future = asyncio.Future()
        self.request_queue.append((text, future))
        
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future
    
    async def process_batch(self):
        """Process requests in batches"""
        self.processing = True
        
        while self.request_queue:
            batch = []
            futures = []
            start_time = time.time()
            
            # Collect batch
            while (len(batch) < self.max_batch_size and 
                   self.request_queue and 
                   (time.time() - start_time) < self.max_wait_time):
                
                text, future = self.request_queue.popleft()
                batch.append(text)
                futures.append(future)
                
                if not self.request_queue:
                    await asyncio.sleep(0.01)  # Small wait for more requests
            
            if batch:
                # Process batch
                results = await self.inference_batch(batch)
                
                # Return results
                for future, result in zip(futures, results):
                    future.set_result(result)
        
        self.processing = False
    
    async def inference_batch(self, texts: List[str]) -> List[str]:
        """Run inference on batch"""
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return [f"Prediction: {pred.item()}" for pred in predictions]
```

### 4. 📊 Real-time Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import torch
from collections import deque
import matplotlib.pyplot as plt
from threading import Thread

class JetsonNLPMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'throughput': deque(maxlen=window_size),
            'gpu_memory': deque(maxlen=window_size),
            'cpu_usage': deque(maxlen=window_size),
            'timestamps': deque(maxlen=window_size)
        }
        self.monitoring = False
    
    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        monitor_thread = Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            timestamp = time.time()
            
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                gpu_memory = 0
            
            # CPU usage
            cpu_usage = psutil.cpu_percent()
            
            self.metrics['gpu_memory'].append(gpu_memory)
            self.metrics['cpu_usage'].append(cpu_usage)
            self.metrics['timestamps'].append(timestamp)
            
            time.sleep(0.1)  # Monitor every 100ms
    
    def log_inference(self, latency, batch_size=1):
        """Log inference metrics"""
        self.metrics['latency'].append(latency * 1000)  # Convert to ms
        self.metrics['throughput'].append(batch_size / latency)  # samples/sec
    
    def get_stats(self):
        """Get current statistics"""
        if not self.metrics['latency']:
            return {}
        
        return {
            'avg_latency_ms': sum(self.metrics['latency']) / len(self.metrics['latency']),
            'avg_throughput': sum(self.metrics['throughput']) / len(self.metrics['throughput']),
            'avg_gpu_memory_mb': sum(self.metrics['gpu_memory']) / len(self.metrics['gpu_memory']),
            'avg_cpu_usage': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']),
            'total_inferences': len(self.metrics['latency'])
        }
    
    def plot_metrics(self, save_path="nlp_performance.png"):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Latency
        axes[0, 0].plot(list(self.metrics['latency']))
        axes[0, 0].set_title('Inference Latency (ms)')
        axes[0, 0].set_ylabel('Latency (ms)')
        
        # Throughput
        axes[0, 1].plot(list(self.metrics['throughput']))
        axes[0, 1].set_title('Throughput (samples/sec)')
        axes[0, 1].set_ylabel('Samples/sec')
        
        # GPU Memory
        axes[1, 0].plot(list(self.metrics['gpu_memory']))
        axes[1, 0].set_title('GPU Memory Usage (MB)')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # CPU Usage
        axes[1, 1].plot(list(self.metrics['cpu_usage']))
        axes[1, 1].set_title('CPU Usage (%)')
        axes[1, 1].set_ylabel('CPU %')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        print(f"📊 Performance plots saved to {save_path}")
```

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

## 🚀 Production Deployment Strategies

### 1. 🐳 Multi-Stage Docker Optimization

```dockerfile
# Dockerfile.nlp-production
# Multi-stage build for optimized NLP deployment
FROM nvcr.io/nvidia/pytorch:24.04-py3 as builder

# Install build dependencies
RUN pip install transformers torch-audio torchaudio torchvision
RUN pip install onnx onnxruntime-gpu tensorrt

# Copy and optimize models
COPY models/ /tmp/models/
COPY scripts/optimize_models.py /tmp/
RUN python /tmp/optimize_models.py

# Production stage
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Install only runtime dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.1.0 \
    onnxruntime-gpu==1.16.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0

# Copy optimized models
COPY --from=builder /tmp/optimized_models/ /app/models/
COPY src/ /app/src/

WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. 🌐 FastAPI Production Server

```python
# src/main.py - Production NLP API Server
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import torch
from transformers import pipeline
import time
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("🚀 Loading NLP models...")
    
    # Load optimized models
    models["sentiment"] = pipeline(
        "sentiment-analysis",
        model="/app/models/sentiment_optimized",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16
    )
    
    models["qa"] = pipeline(
        "question-answering",
        model="/app/models/qa_optimized",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16
    )
    
    models["summarization"] = pipeline(
        "summarization",
        model="/app/models/summarization_optimized",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16
    )
    
    logger.info("✅ Models loaded successfully")
    yield
    
    # Cleanup
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Jetson NLP API",
    description="Optimized NLP services for NVIDIA Jetson",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class SentimentRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 8

class SentimentResponse(BaseModel):
    results: List[dict]
    processing_time_ms: float

class QARequest(BaseModel):
    questions: List[str]
    contexts: List[str]

class QAResponse(BaseModel):
    answers: List[str]
    processing_time_ms: float

class SummarizationRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 150
    min_length: Optional[int] = 30

class SummarizationResponse(BaseModel):
    summaries: List[str]
    processing_time_ms: float

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }

# Sentiment Analysis endpoint
@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if "sentiment" not in models:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")
    
    start_time = time.time()
    
    try:
        # Process in batches for efficiency
        results = []
        for i in range(0, len(request.texts), request.batch_size):
            batch = request.texts[i:i + request.batch_size]
            batch_results = models["sentiment"](batch)
            results.extend(batch_results)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            results=results,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Question Answering endpoint
@app.post("/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    if "qa" not in models:
        raise HTTPException(status_code=503, detail="QA model not loaded")
    
    if len(request.questions) != len(request.contexts):
        raise HTTPException(status_code=400, detail="Questions and contexts must have same length")
    
    start_time = time.time()
    
    try:
        answers = []
        for question, context in zip(request.questions, request.contexts):
            result = models["qa"](question=question, context=context)
            answers.append(result["answer"])
        
        processing_time = (time.time() - start_time) * 1000
        
        return QAResponse(
            answers=answers,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text Summarization endpoint
@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    if "summarization" not in models:
        raise HTTPException(status_code=503, detail="Summarization model not loaded")
    
    start_time = time.time()
    
    try:
        summaries = []
        for text in request.texts:
            result = models["summarization"](
                text,
                max_length=request.max_length,
                min_length=request.min_length,
                do_sample=False
            )
            summaries.append(result[0]["summary_text"])
        
        processing_time = (time.time() - start_time) * 1000
        
        return SummarizationResponse(
            summaries=summaries,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoint
@app.post("/batch")
async def batch_process(background_tasks: BackgroundTasks):
    """Handle large batch processing jobs"""
    # Implementation for handling large batch jobs
    # This would typically use a job queue like Celery
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. 📊 Load Testing & Performance Validation

```python
# load_test.py - Performance validation script
import asyncio
import aiohttp
import time
import json
from typing import List
import statistics

class NLPLoadTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_sentiment_endpoint(self, session, texts: List[str]):
        """Test sentiment analysis endpoint"""
        start_time = time.time()
        
        async with session.post(
            f"{self.base_url}/sentiment",
            json={"texts": texts, "batch_size": 8}
        ) as response:
            result = await response.json()
            latency = time.time() - start_time
            
            return {
                "endpoint": "sentiment",
                "latency": latency,
                "status": response.status,
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
    
    async def test_qa_endpoint(self, session, questions: List[str], contexts: List[str]):
        """Test question answering endpoint"""
        start_time = time.time()
        
        async with session.post(
            f"{self.base_url}/qa",
            json={"questions": questions, "contexts": contexts}
        ) as response:
            result = await response.json()
            latency = time.time() - start_time
            
            return {
                "endpoint": "qa",
                "latency": latency,
                "status": response.status,
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
    
    async def run_concurrent_tests(self, num_concurrent=10, num_requests=100):
        """Run concurrent load tests"""
        print(f"🚀 Starting load test: {num_concurrent} concurrent users, {num_requests} requests")
        
        # Test data
        test_texts = [
            "This is an amazing product!",
            "I'm not satisfied with this service.",
            "The quality is okay, nothing special."
        ]
        
        test_questions = ["What is the main topic?"] * 3
        test_contexts = [
            "This article discusses artificial intelligence and machine learning.",
            "The weather today is sunny with a chance of rain.",
            "Cooking pasta requires boiling water and adding salt."
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Create concurrent tasks
            for i in range(num_requests):
                if i % 2 == 0:
                    task = self.test_sentiment_endpoint(session, test_texts)
                else:
                    task = self.test_qa_endpoint(session, test_questions, test_contexts)
                
                tasks.append(task)
                
                # Limit concurrency
                if len(tasks) >= num_concurrent:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    self.results.extend([r for r in results if not isinstance(r, Exception)])
                    tasks = []
            
            # Process remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                self.results.extend([r for r in results if not isinstance(r, Exception)])
    
    def analyze_results(self):
        """Analyze load test results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Group by endpoint
        sentiment_results = [r for r in self.results if r["endpoint"] == "sentiment"]
        qa_results = [r for r in self.results if r["endpoint"] == "qa"]
        
        print("\n📊 Load Test Results:")
        print("=" * 50)
        
        for endpoint_name, results in [("Sentiment", sentiment_results), ("QA", qa_results)]:
            if not results:
                continue
                
            latencies = [r["latency"] for r in results]
            processing_times = [r["processing_time_ms"] for r in results]
            success_rate = len([r for r in results if r["status"] == 200]) / len(results)
            
            print(f"\n{endpoint_name} Analysis:")
            print(f"  Total requests: {len(results)}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Avg latency: {statistics.mean(latencies):.3f}s")
            print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")
            print(f"  P99 latency: {statistics.quantiles(latencies, n=100)[98]:.3f}s")
            print(f"  Avg processing time: {statistics.mean(processing_times):.1f}ms")
            print(f"  Throughput: {len(results) / sum(latencies):.1f} req/s")

# Run load test
async def main():
    tester = NLPLoadTester()
    await tester.run_concurrent_tests(num_concurrent=5, num_requests=50)
    tester.analyze_results()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Final Challenge: Complete NLP Pipeline on Jetson

### 🏆 Challenge Objective

Build a complete, production-ready NLP pipeline that processes real-world data and demonstrates all optimization techniques learned in this tutorial.

### 📋 Challenge Requirements

#### 1. **Multi-Modal NLP System**
Implement a system that handles:
- **Text Classification** (sentiment analysis on product reviews)
- **Information Extraction** (NER on news articles)
- **Question Answering** (FAQ system for customer support)
- **Text Summarization** (news article summarization)
- **Real-time Chat** (customer service chatbot)

#### 2. **Optimization Implementation**
- Apply **quantization** (FP16 minimum, INT8 preferred)
- Implement **dynamic batching** for throughput optimization
- Use **model pruning** to reduce memory footprint
- Deploy with **TensorRT** optimization where possible
- Implement **caching** for frequently requested content

#### 3. **Production Deployment**
- **Containerized deployment** with multi-stage Docker builds
- **REST API** with proper error handling and logging
- **Load balancing** for high availability
- **Monitoring and metrics** collection
- **Auto-scaling** based on resource utilization

#### 4. **Performance Benchmarking**
- **Latency analysis** (P50, P95, P99 percentiles)
- **Throughput measurement** (requests per second)
- **Resource utilization** (GPU/CPU/memory usage)
- **Accuracy validation** on standard datasets
- **Cost analysis** (inference cost per request)

### 🛠️ Implementation Guide

```python
# challenge_solution.py - Complete NLP Pipeline
import asyncio
import torch
from fastapi import FastAPI, WebSocket, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import redis
import json
from typing import Dict, List
import time
import logging
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    SENTIMENT = "sentiment"
    NER = "ner"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CHAT = "chat"

@dataclass
class ProcessingResult:
    task_type: TaskType
    result: Dict
    processing_time_ms: float
    model_used: str
    cached: bool = False

class JetsonNLPPipeline:
    def __init__(self):
        self.models = {}
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.performance_monitor = JetsonNLPMonitor()
        
    async def initialize_models(self):
        """Initialize all optimized models"""
        logging.info("🚀 Initializing NLP pipeline...")
        
        # Load optimized models with quantization
        self.models[TaskType.SENTIMENT] = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        self.models[TaskType.NER] = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        self.models[TaskType.QA] = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        self.models[TaskType.SUMMARIZATION] = pipeline(
            "summarization",
            model="t5-small",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
        # Chat model with optimization
        self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.models[TaskType.CHAT] = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float16
        )
        
        logging.info("✅ All models initialized successfully")
    
    async def process_request(self, task_type: TaskType, data: Dict) -> ProcessingResult:
        """Process NLP request with caching and optimization"""
        # Check cache first
        cache_key = f"{task_type.value}:{hash(str(data))}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return ProcessingResult(
                task_type=task_type,
                result=json.loads(cached_result),
                processing_time_ms=0,
                model_used=f"{task_type.value}_cached",
                cached=True
            )
        
        # Process with appropriate model
        start_time = time.time()
        
        if task_type == TaskType.SENTIMENT:
            result = self.models[task_type](data["texts"], batch_size=8)
        elif task_type == TaskType.NER:
            result = self.models[task_type](data["texts"], batch_size=8)
        elif task_type == TaskType.QA:
            result = self.models[task_type](
                question=data["question"],
                context=data["context"]
            )
        elif task_type == TaskType.SUMMARIZATION:
            result = self.models[task_type](
                data["text"],
                max_length=data.get("max_length", 150),
                min_length=data.get("min_length", 30)
            )
        elif task_type == TaskType.CHAT:
            # Implement chat logic
            result = await self._process_chat(data["message"], data.get("history", []))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result
        self.cache.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
        
        # Log performance
        self.performance_monitor.log_inference(processing_time / 1000)
        
        return ProcessingResult(
            task_type=task_type,
            result=result,
            processing_time_ms=processing_time,
            model_used=task_type.value
        )
    
    async def _process_chat(self, message: str, history: List[str]) -> Dict:
        """Process chat message with context"""
        # Implement chat processing logic
        # This is a simplified version
        return {"response": f"Echo: {message}"}

# FastAPI application
app = FastAPI(title="Jetson NLP Challenge Solution")
nlp_pipeline = JetsonNLPPipeline()

@app.on_event("startup")
async def startup_event():
    await nlp_pipeline.initialize_models()

@app.post("/process")
async def process_nlp_request(task_type: str, data: Dict):
    try:
        task = TaskType(task_type)
        result = await nlp_pipeline.process_request(task, data)
        return result.__dict__
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    
    try:
        while True:
            message = await websocket.receive_text()
            
            result = await nlp_pipeline.process_request(
                TaskType.CHAT,
                {"message": message, "history": chat_history}
            )
            
            chat_history.append(message)
            chat_history.append(result.result["response"])
            
            await websocket.send_json(result.__dict__)
    
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/metrics")
async def get_metrics():
    return nlp_pipeline.performance_monitor.get_stats()
```

### 📊 Evaluation Criteria

| Criterion | Weight | Excellent (90-100%) | Good (70-89%) | Satisfactory (50-69%) |
|-----------|--------|-------------------|---------------|----------------------|
| **Functionality** | 25% | All 5 NLP tasks working perfectly | 4/5 tasks working | 3/5 tasks working |
| **Optimization** | 25% | All optimization techniques applied | Most optimizations applied | Basic optimizations |
| **Performance** | 20% | <50ms P95 latency, >100 req/s | <100ms P95, >50 req/s | <200ms P95, >20 req/s |
| **Production Ready** | 15% | Full deployment with monitoring | Basic deployment | Local deployment only |
| **Code Quality** | 10% | Clean, documented, tested | Well-structured | Basic implementation |
| **Innovation** | 5% | Novel optimizations/features | Creative solutions | Standard implementation |

### 🎯 Bonus Challenges

1. **Multi-Language Support**: Extend the pipeline to handle multiple languages
2. **Edge Deployment**: Deploy on actual Jetson hardware with resource constraints
3. **Federated Learning**: Implement model updates without centralized data
4. **Real-time Streaming**: Process continuous data streams with low latency
5. **Custom Models**: Train and deploy domain-specific models

---

## 📌 Summary

* **Comprehensive NLP Applications**: Covered 6 major NLP tasks with Jetson-specific optimizations
* **Advanced Optimization Techniques**: Quantization, pruning, distillation, and dynamic batching
* **Production Deployment**: Multi-stage Docker builds, FastAPI servers, and load testing
* **Performance Monitoring**: Real-time metrics collection and analysis
* **Practical Evaluation**: Standardized benchmarking on popular datasets
* **Complete Pipeline**: End-to-end solution from development to production

This tutorial provides a comprehensive foundation for deploying production-ready NLP applications on Jetson devices, balancing performance, accuracy, and resource efficiency.

→ Next: [Prompt Engineering](08_prompt_engineering_langchain_jetson.md)
