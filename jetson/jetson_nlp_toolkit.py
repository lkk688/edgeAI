#!/usr/bin/env python3
# jetson_nlp_toolkit.py - All-in-one NLP toolkit for Jetson devices

import argparse
import asyncio
import json
import logging
import os
import psutil
import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization

# Optional imports - will be loaded on demand
try:
    from datasets import load_dataset
    from evaluate import load
    from transformers import (
        AutoModelForCausalLM, 
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== NLP Evaluation Suite =====

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
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers datasets evaluate'")
            return
            
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
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers datasets evaluate'")
            return
            
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
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers datasets evaluate rouge-score'")
            return
            
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
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers datasets evaluate'")
            return
            
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
        
    def print_summary(self):
        """Print evaluation summary"""
        print("\n📈 Evaluation Summary:")
        for task, metrics in self.results.items():
            print(f"\n{task.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")

# ===== Quantization Benchmark =====

class QuantizationBenchmark:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers'")
            return
            
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

# ===== LLM Inference Comparison =====

class LLMInferenceComparison:
    def __init__(self):
        self.results = {}
        self.prompt = "Explain the future of AI in education."
    
    def run_huggingface(self, model_name="sshleifer/tiny-gpt2"):
        """Run inference using HuggingFace Transformers"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it with 'pip install transformers'")
            return False
            
        print(f"🔍 Testing HuggingFace inference with {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            # Measure GPU memory before
            start_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Run inference
            start = time.time()
            input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                
            output = model.generate(input_ids, max_length=100)
            latency = time.time() - start
            
            # Measure GPU memory after
            end_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Decode output
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            num_tokens = len(output[0])
            tokens_per_sec = num_tokens / latency
            
            self.results['huggingface'] = {
                'latency_sec': latency,
                'tokens_per_sec': tokens_per_sec,
                'gpu_memory_mb': end_memory - start_memory,
                'output_text': output_text,
                'num_tokens': num_tokens
            }
            
            print(f"✅ HuggingFace inference completed in {latency:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error running HuggingFace inference: {e}")
            return False
    
    def run_llamacpp(self, model_path="/models/mistral.gguf", n_gpu_layers=80):
        """Run inference using llama.cpp"""
        if not LLAMACPP_AVAILABLE:
            logger.error("llama-cpp-python library not available. Please install it with 'pip install llama-cpp-python'")
            return False
            
        print(f"🔍 Testing llama.cpp inference with {model_path}...")
        
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            # Measure GPU memory before
            start_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Initialize model
            llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers)
            
            # Run inference
            start = time.time()
            output = llm(self.prompt)
            latency = time.time() - start
            
            # Measure GPU memory after
            end_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Extract metrics
            output_text = output['choices'][0]['text'] if isinstance(output, dict) else output
            num_tokens = len(output_text.split())
            tokens_per_sec = num_tokens / latency
            
            self.results['llamacpp'] = {
                'latency_sec': latency,
                'tokens_per_sec': tokens_per_sec,
                'gpu_memory_mb': end_memory - start_memory,
                'output_text': output_text,
                'num_tokens': num_tokens
            }
            
            print(f"✅ llama.cpp inference completed in {latency:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error running llama.cpp inference: {e}")
            return False
    
    def run_ollama(self, model_name="mistral", api_url="http://localhost:11434/api/generate"):
        """Run inference using Ollama API"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available. Please install it with 'pip install requests'")
            return False
            
        print(f"🔍 Testing Ollama inference with {model_name}...")
        
        try:
            # Measure GPU memory before (if possible)
            start_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Run inference
            start = time.time()
            r = requests.post(api_url, json={
                "model": model_name,
                "prompt": self.prompt,
                "stream": False
            })
            latency = time.time() - start
            
            # Measure GPU memory after (if possible)
            end_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Extract metrics
            response = r.json()
            output_text = response.get("response", "")
            num_tokens = len(output_text.split())
            tokens_per_sec = num_tokens / latency
            
            self.results['ollama'] = {
                'latency_sec': latency,
                'tokens_per_sec': tokens_per_sec,
                'gpu_memory_mb': end_memory - start_memory,
                'output_text': output_text,
                'num_tokens': num_tokens
            }
            
            print(f"✅ Ollama inference completed in {latency:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error running Ollama inference: {e}")
            return False
    
    def print_comparison(self):
        """Print comparison results"""
        if not self.results:
            print("❌ No inference results to compare")
            return
            
        print("\n📊 LLM Inference Comparison:")
        print(f"{'Method':<20} {'Latency (s)':<12} {'Tokens/sec':<12} {'GPU Mem (MB)':<15}")
        print("-" * 60)
        
        for method, metrics in self.results.items():
            print(f"{method:<20} {metrics['latency_sec']:<12.2f} "
                  f"{metrics['tokens_per_sec']:<12.1f} {metrics['gpu_memory_mb']:<15.1f}")
            
        # Print output samples
        print("\n📝 Output Samples:")
        for method, metrics in self.results.items():
            print(f"\n{method.upper()} OUTPUT:")
            print(f"Prompt: {self.prompt}")
            print(f"Response: {metrics['output_text'][:200]}...")
    
    def save_results(self, filename="llm_inference_results.json"):
        """Save results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Results saved to {filename}")

# ===== Model Pruning =====

def prune_model(model, pruning_ratio=0.2):
    """Apply structured pruning to transformer model"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    return model

# ===== Knowledge Distillation =====

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

# ===== Dynamic Batching =====

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

# ===== Performance Monitoring =====

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
        plt.close()
        
        print(f"📊 Performance plots saved to {save_path}")

# ===== FastAPI Server =====

class NLPServer:
    def __init__(self, host="0.0.0.0", port=8000):
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available. Please install it with 'pip install fastapi uvicorn'")
            return
            
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Jetson NLP API",
            description="Optimized NLP services for NVIDIA Jetson",
            version="1.0.0"
        )
        self.setup_routes()
        
    def setup_routes(self):
        """Set up API routes"""
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            }
        
        @self.app.post("/sentiment")
        async def analyze_sentiment(texts: List[str]):
            if not TRANSFORMERS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Transformers library not available")
                
            try:
                # Initialize sentiment model
                classifier = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16
                )
                
                # Process texts
                start_time = time.time()
                results = classifier(texts, batch_size=8)
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "results": results,
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/qa")
        async def question_answering(questions: List[str], contexts: List[str]):
            if not TRANSFORMERS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Transformers library not available")
                
            if len(questions) != len(contexts):
                raise HTTPException(status_code=400, detail="Questions and contexts must have same length")
                
            try:
                # Initialize QA model
                qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16
                )
                
                # Process QA pairs
                start_time = time.time()
                answers = []
                for question, context in zip(questions, contexts):
                    result = qa_pipeline(question=question, context=context)
                    answers.append(result["answer"])
                    
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "answers": answers,
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"QA error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/summarize")
        async def summarize_text(texts: List[str], max_length: int = 150, min_length: int = 30):
            if not TRANSFORMERS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Transformers library not available")
                
            try:
                # Initialize summarization model
                summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16
                )
                
                # Process texts
                start_time = time.time()
                summaries = []
                for text in texts:
                    result = summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    summaries.append(result[0]["summary_text"])
                    
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "summaries": summaries,
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """Run the FastAPI server"""
        import uvicorn
        print(f"🚀 Starting NLP server at http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

# ===== Load Testing =====

class NLPLoadTester:
    def __init__(self, base_url="http://localhost:8000"):
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available. Please install it with 'pip install aiohttp'")
            return
            
        self.base_url = base_url
        self.results = []
    
    async def test_sentiment_endpoint(self, session, texts: List[str]):
        """Test sentiment analysis endpoint"""
        start_time = time.time()
        
        async with session.post(
            f"{self.base_url}/sentiment",
            json={"texts": texts}
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

# ===== Main Function =====

def main():
    parser = argparse.ArgumentParser(description="Jetson NLP Toolkit - All-in-one NLP applications and optimizations")
    
    # Main command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Run NLP evaluation suite")
    eval_parser.add_argument("--task", choices=["sentiment", "qa", "summarization", "ner", "all"], 
                           default="all", help="NLP task to evaluate")
    eval_parser.add_argument("--output", type=str, default="jetson_nlp_results.json", 
                           help="Output file for results")
    
    # Optimization command
    opt_parser = subparsers.add_parser("optimize", help="Run optimization benchmarks")
    opt_parser.add_argument("--method", choices=["quantization", "pruning", "distillation", "all"], 
                          default="quantization", help="Optimization method to benchmark")
    opt_parser.add_argument("--model", type=str, 
                          default="distilbert-base-uncased-finetuned-sst-2-english", 
                          help="Model to optimize")
    opt_parser.add_argument("--ratio", type=float, default=0.2, 
                          help="Pruning ratio (for pruning method)")
    
    # LLM inference command
    llm_parser = subparsers.add_parser("llm", help="Compare LLM inference methods")
    llm_parser.add_argument("--method", choices=["huggingface", "llamacpp", "ollama", "all"], 
                          default="all", help="LLM inference method to test")
    llm_parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", 
                          help="Model name for HuggingFace")
    llm_parser.add_argument("--model-path", type=str, default="/models/mistral.gguf", 
                          help="Model path for llama.cpp")
    llm_parser.add_argument("--ollama-model", type=str, default="mistral", 
                          help="Model name for Ollama")
    llm_parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", 
                          help="Ollama API URL")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run NLP server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # Load test command
    loadtest_parser = subparsers.add_parser("loadtest", help="Run load tests against NLP server")
    loadtest_parser.add_argument("--url", type=str, default="http://localhost:8000", 
                               help="Server URL to test")
    loadtest_parser.add_argument("--concurrent", type=int, default=10, 
                               help="Number of concurrent users")
    loadtest_parser.add_argument("--requests", type=int, default=100, 
                               help="Total number of requests")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "evaluate":
        evaluator = JetsonNLPEvaluator()
        
        if args.task == "all":
            results = evaluator.run_full_evaluation()
        elif args.task == "sentiment":
            evaluator.evaluate_sentiment_analysis()
        elif args.task == "qa":
            evaluator.evaluate_question_answering()
        elif args.task == "summarization":
            evaluator.evaluate_summarization()
        elif args.task == "ner":
            evaluator.evaluate_ner()
        
        evaluator.save_results(args.output)
        evaluator.print_summary()
    
    elif args.command == "optimize":
        if args.method == "quantization" or args.method == "all":
            benchmark = QuantizationBenchmark(model_name=args.model)
            results = benchmark.compare_quantization_methods()
            benchmark.print_comparison(results)
        
        if args.method == "pruning" or args.method == "all":
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers library not available. Please install it with 'pip install transformers'")
                return
                
            print(f"\n🔍 Testing model pruning with ratio {args.ratio}...")
            model = AutoModelForSequenceClassification.from_pretrained(args.model)
            pruned_model = prune_model(model, pruning_ratio=args.ratio)
            
            # Compare model sizes
            original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            pruned_size = sum(p.numel() * p.element_size() for p in pruned_model.parameters()) / 1024 / 1024
            
            print(f"Original model size: {original_size:.2f} MB")
            print(f"Pruned model size: {pruned_size:.2f} MB")
            print(f"Size reduction: {(original_size - pruned_size) / original_size:.2%}")
    
    elif args.command == "llm":
        llm_comparison = LLMInferenceComparison()
        
        if args.method == "huggingface" or args.method == "all":
            llm_comparison.run_huggingface(model_name=args.model)
        
        if args.method == "llamacpp" or args.method == "all":
            llm_comparison.run_llamacpp(model_path=args.model_path)
        
        if args.method == "ollama" or args.method == "all":
            llm_comparison.run_ollama(model_name=args.ollama_model, api_url=args.ollama_url)
        
        llm_comparison.print_comparison()
        llm_comparison.save_results()
    
    elif args.command == "server":
        server = NLPServer(host=args.host, port=args.port)
        server.run()
    
    elif args.command == "loadtest":
        async def run_loadtest():
            tester = NLPLoadTester(base_url=args.url)
            await tester.run_concurrent_tests(num_concurrent=args.concurrent, num_requests=args.requests)
            tester.analyze_results()
        
        asyncio.run(run_loadtest())

if __name__ == "__main__":
    main()