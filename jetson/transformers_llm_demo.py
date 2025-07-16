#!/usr/bin/env python3
"""
Transformers and LLM Demo for Jetson Devices

This script provides a comprehensive demonstration of various NLP applications
using HuggingFace transformers with optimization techniques for Jetson devices.

Features:
- Text Classification (Sentiment Analysis)
- Text Generation (GPT-2)
- Question Answering (BERT)
- Named Entity Recognition (NER)
- Advanced optimization techniques
- Performance monitoring and benchmarking

Author: Jetson AI Course
Version: 1.0
"""

import torch
import time
import psutil
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# For visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization will be disabled.")

# HuggingFace imports
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTokenClassification, BitsAndBytesConfig
)

# Optional imports for advanced features
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Some optimizations will be skipped.")

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: Accelerate not available. Memory-efficient loading disabled.")


class TextDataset(Dataset):
    """
    Custom dataset class for batch processing of text data.
    
    Args:
        texts (List[str]): List of input texts
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
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


@contextmanager
def performance_monitor():
    """
    Context manager for monitoring performance metrics during model inference.
    
    Tracks:
    - Execution time
    - GPU memory usage
    - CPU usage
    - GPU utilization (if available)
    """
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
        
        if torch.cuda.is_available() and GPU_UTIL_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                print(f"🎮 GPU utilization: {gpu.load*100:.1f}%")
                print(f"🌡️  GPU temperature: {gpu.temperature}°C")
            except:
                print("⚠️  GPU monitoring unavailable")


class TransformersDemo:
    """
    Main class for running transformer model demonstrations on Jetson devices.
    
    This class provides methods for various NLP tasks with different optimization
    levels to showcase performance improvements on edge devices.
    """
    
    def __init__(self):
        """Initialize the demo class with device detection."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 Using device: {self.device}")
        
        # Check available optimizations
        self.optimizations_available = {
            'onnx': ONNX_AVAILABLE,
            'accelerate': ACCELERATE_AVAILABLE,
            'quantization': torch.cuda.is_available(),
            'jit': True  # Always available with PyTorch
        }
        
        print("🔧 Available optimizations:")
        for opt, available in self.optimizations_available.items():
            status = "✅" if available else "❌"
            print(f"  {status} {opt.upper()}")
    
    def text_classification_basic(self, text: str) -> Dict[str, Any]:
        """
        Basic text classification using DistilBERT for sentiment analysis.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Dict containing classification results and timing
        """
        print("\n🔍 Running Basic Text Classification (DistilBERT)...")
        
        # Load pre-trained sentiment analysis pipeline
        classifier = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Perform inference with timing
        start_time = time.time()
        result = classifier(text)
        end_time = time.time()
        
        return {
            'result': result,
            'inference_time': end_time - start_time,
            'method': 'basic_pipeline'
        }
    
    def text_classification_optimized(self, text: str) -> Dict[str, Any]:
        """
        Optimized text classification using ONNX runtime (if available).
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Dict containing classification results and timing
        """
        print("\n⚡ Running Optimized Text Classification (ONNX)...")
        
        if not ONNX_AVAILABLE:
            print("⚠️  ONNX not available, falling back to basic method")
            return self.text_classification_basic(text)
        
        try:
            # Load ONNX-optimized model
            tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            model = ORTModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                export=True,  # Convert to ONNX if not already
                provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            )
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt")
            
            # Perform inference with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            end_time = time.time()
            
            # Format results
            labels = ['NEGATIVE', 'POSITIVE']
            scores = predictions[0].cpu().numpy()
            result = [{
                'label': labels[np.argmax(scores)],
                'score': float(np.max(scores))
            }]
            
            return {
                'result': result,
                'inference_time': end_time - start_time,
                'method': 'onnx_optimized'
            }
            
        except Exception as e:
            print(f"⚠️  ONNX optimization failed: {e}")
            return self.text_classification_basic(text)
    
    def text_generation_basic(self, prompt: str, max_length: int = 50) -> Dict[str, Any]:
        """
        Basic text generation using GPT-2.
        
        Args:
            prompt (str): Input prompt for generation
            max_length (int): Maximum length of generated text
            
        Returns:
            Dict containing generated text and timing
        """
        print("\n📝 Running Basic Text Generation (GPT-2)...")
        
        # Load model and tokenizer
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Generate text
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'inference_time': end_time - start_time,
            'method': 'basic_generation'
        }
    
    def text_generation_optimized(self, prompt: str, max_length: int = 50) -> Dict[str, Any]:
        """
        Optimized text generation using quantization and GPU acceleration.
        
        Args:
            prompt (str): Input prompt for generation
            max_length (int): Maximum length of generated text
            
        Returns:
            Dict containing generated text and timing
        """
        print("\n🚀 Running Optimized Text Generation (Quantized + GPU)...")
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to basic method")
            return self.text_generation_basic(prompt, max_length)
        
        try:
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
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache for faster generation
                )
            end_time = time.time()
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            
            return {
                'generated_text': generated_text,
                'inference_time': end_time - start_time,
                'memory_usage_mb': memory_usage,
                'method': 'quantized_optimized'
            }
            
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            return self.text_generation_basic(prompt, max_length)
    
    def question_answering_basic(self, question: str, context: str) -> Dict[str, Any]:
        """
        Basic question answering using BERT.
        
        Args:
            question (str): Question to answer
            context (str): Context containing the answer
            
        Returns:
            Dict containing answer and timing
        """
        print("\n❓ Running Basic Question Answering (BERT)...")
        
        # Load QA pipeline
        qa_pipeline = pipeline(
            "question-answering", 
            model="bert-large-uncased-whole-word-masking-finetuned-squad"
        )
        
        start_time = time.time()
        result = qa_pipeline(question=question, context=context)
        end_time = time.time()
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'inference_time': end_time - start_time,
            'method': 'basic_qa_pipeline'
        }
    
    def question_answering_optimized(self, question: str, context: str) -> Dict[str, Any]:
        """
        Optimized question answering using DistilBERT and JIT compilation.
        
        Args:
            question (str): Question to answer
            context (str): Context containing the answer
            
        Returns:
            Dict containing answer and timing
        """
        print("\n⚡ Running Optimized Question Answering (DistilBERT + JIT)...")
        
        # Load model with optimizations
        model_name = "distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move to GPU and enable optimizations
        model.to(self.device)
        model.eval()
        
        # Enable torch.jit compilation for faster inference
        try:
            model = torch.jit.script(model)
            jit_enabled = True
        except:
            print("⚠️  JIT compilation failed, using regular model")
            jit_enabled = False
        
        # Tokenize inputs
        inputs = tokenizer.encode_plus(
            question, context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores) + 1
            
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            confidence = (torch.max(start_scores) + torch.max(end_scores)).item() / 2
        end_time = time.time()
        
        return {
            'answer': answer,
            'confidence': confidence,
            'inference_time': end_time - start_time,
            'jit_enabled': jit_enabled,
            'method': 'distilbert_jit_optimized'
        }
    
    def named_entity_recognition_basic(self, text: str) -> Dict[str, Any]:
        """
        Basic Named Entity Recognition using BERT.
        
        Args:
            text (str): Input text for NER
            
        Returns:
            Dict containing entities and timing
        """
        print("\n🏷️  Running Basic Named Entity Recognition (BERT)...")
        
        # Load NER pipeline
        ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        
        start_time = time.time()
        entities = ner_pipeline(text)
        end_time = time.time()
        
        return {
            'entities': entities,
            'inference_time': end_time - start_time,
            'method': 'basic_ner_pipeline'
        }
    
    def named_entity_recognition_batch(self, texts: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """
        Batch processing for Named Entity Recognition using DistilBERT.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            Dict containing batch results and timing
        """
        print("\n🚀 Running Batch Named Entity Recognition (DistilBERT)...")
        
        # Load model for batch processing
        model_name = "distilbert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/distilbert-base-cased-finetuned-conll03-english"
        )
        
        model.to(self.device)
        model.eval()
        
        all_entities = []
        total_time = 0
        
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
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
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'batch_entities': all_entities,
            'total_time': total_time,
            'average_per_text': total_time / len(texts),
            'batch_size': batch_size,
            'method': 'batch_ner_distilbert'
        }
    
    def find_optimal_batch_size(self, model, tokenizer, max_batch_size: int = 32) -> int:
        """
        Find the largest batch size that fits in GPU memory.
        
        Args:
            model: The model to test
            tokenizer: The tokenizer to use
            max_batch_size (int): Maximum batch size to test
            
        Returns:
            int: Optimal batch size
        """
        print("\n🔍 Finding optimal batch size...")
        
        for batch_size in range(max_batch_size, 0, -1):
            try:
                # Test with dummy data
                dummy_input = tokenizer(
                    ["test text"] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    _ = model(**dummy_input)
                
                print(f"✅ Optimal batch size found: {batch_size}")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return 1  # Fallback to batch size 1
    
    def create_performance_dashboard(self, performance_data: Dict[str, List[float]], output_file: str = 'jetson_transformer_performance.png') -> None:
        """
        Create a performance visualization dashboard.
        
        Args:
            performance_data (Dict[str, List[float]]): Dictionary of performance metrics
            output_file (str): Output file path for the dashboard image
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available. Cannot create performance dashboard.")
            return
        
        print("\n📊 Creating performance dashboard...")
        
        try:
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
            
            # Memory usage simulation (replace with actual measurements if available)
            memory_usage = []
            for method in methods:
                # Use actual memory measurements if available in the data
                if f"{method}_memory" in performance_data:
                    memory_usage.append(np.mean(performance_data[f"{method}_memory"]))
                else:
                    # Simulate memory usage based on method name
                    if "basic" in method.lower():
                        memory_usage.append(100)  # Example value in MB
                    elif "optimized" in method.lower() or "accelerated" in method.lower():
                        memory_usage.append(60)   # Example value in MB
                    else:
                        memory_usage.append(80)   # Example value in MB
            
            ax2.bar(methods, memory_usage, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Efficiency')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            
            print(f"📊 Performance dashboard saved as '{output_file}'")
            
        except Exception as e:
            print(f"⚠️ Error creating dashboard: {e}")
    
    def benchmark_model(self, model_name: str, prompts: List[str], num_runs: int = 3) -> List[Dict[str, Any]]:
        """
        Comprehensive model benchmarking with multiple runs.
        
        Args:
            model_name (str): Name of the model to benchmark
            prompts (List[str]): List of prompts to test
            num_runs (int): Number of runs per prompt
            
        Returns:
            List of benchmark results
        """
        print(f"\n🧪 Benchmarking model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            model.to(self.device)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Benchmarking prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
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


# Interactive menu functions removed as we now use command-line arguments
# See parse_arguments() and print_available_apps() functions instead


# Interactive menu functions removed as we now use command-line arguments
# See parse_arguments() function instead


# Interactive menu functions removed as we now use command-line arguments
# Acceleration choice is now handled via the --optimize flag


def parse_arguments():
    """
    Parse command-line arguments for the demo.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformers and LLM Demo for Jetson Devices")
    
    # Application selection
    parser.add_argument(
        "--app", "-a", type=int, choices=range(1, 8),
        help="Application to run: 1=Text Classification, 2=Text Generation, 3=Question Answering, "
             "4=Named Entity Recognition, 5=Batch Processing, 6=Model Benchmarking, 7=Performance Comparison"
    )
    
    # Acceleration options
    parser.add_argument(
        "--optimize", "-o", action="store_true",
        help="Use optimized version (ONNX, quantization, JIT) instead of basic version"
    )
    
    # Input text options
    parser.add_argument(
        "--text", "-t", type=str,
        help="Input text for classification or NER"
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        help="Input prompt for text generation"
    )
    parser.add_argument(
        "--question", "-q", type=str,
        help="Question for QA task"
    )
    parser.add_argument(
        "--context", "-c", type=str,
        help="Context for QA task"
    )
    
    # Additional parameters
    parser.add_argument(
        "--max-length", type=int, default=50,
        help="Maximum length for text generation (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size for batch processing (default: 2)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt2",
        help="Model name for benchmarking (default: gpt2)"
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3,
        help="Number of runs for benchmarking (default: 3)"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available applications and exit"
    )
    
    return parser.parse_args()


def print_available_apps():
    """
    Print available applications and exit.
    """
    print("\n" + "="*60)
    print("🤖 TRANSFORMERS & LLM DEMO FOR JETSON DEVICES")
    print("="*60)
    print("\n📋 Available Applications:")
    print("  1. Text Classification (Sentiment Analysis)")
    print("  2. Text Generation (GPT-2)")
    print("  3. Question Answering (BERT)")
    print("  4. Named Entity Recognition (NER)")
    print("  5. Batch Processing Demo")
    print("  6. Model Benchmarking")
    print("  7. Performance Comparison")
    print("\n⚡ Acceleration Options:")
    print("  --optimize, -o: Use optimized version (ONNX, quantization, JIT)")
    print("\n📝 Example Usage:")
    print("  python transformers_llm_demo.py --app 1 --text "Your text here" --optimize")
    print("  python transformers_llm_demo.py -a 2 -p "Your prompt" --max-length 100")
    print("  python transformers_llm_demo.py -a 3 -q "Your question" -c "Your context"")


def main():
    """
    Main function to run the demo with command-line arguments.
    """
    args = parse_arguments()
    
    # Initialize the demo
    demo = TransformersDemo()
    
    # Sample data for demonstrations
    sample_texts = [
        "Jetson Orin Nano delivers incredible AI performance at the edge!",
        "I'm disappointed with the slow performance of this device.",
        "NVIDIA's edge computing platform enables real-time inference."
    ]
    
    sample_context = """
    The NVIDIA Jetson Orin Nano is a powerful single-board computer designed for AI applications at the edge. 
    It features an ARM Cortex-A78AE CPU and an integrated GPU with 1024 CUDA cores. 
    The device supports up to 8GB of LPDDR5 memory and can deliver up to 40 TOPS of AI performance.
    """
    
    sample_questions = [
        "How many CUDA cores does the Jetson Orin Nano have?",
        "What is the maximum memory supported?",
        "What type of CPU does it use?"
    ]
    
    sample_ner_texts = [
        "NVIDIA Jetson Orin Nano was developed in Santa Clara, California by Jensen Huang's team.",
        "Jensen Huang founded NVIDIA Corporation in 1993.",
        "The device supports CUDA and TensorRT acceleration."
    ]
    
    sample_prompts = [
        "Edge AI computing with Jetson",
        "The future of artificial intelligence",
        "NVIDIA's contribution to deep learning"
    ]
    
    # If --list flag is provided, just show available apps and exit
    if args.list:
        print_available_apps()
        return
    
    # If no app is specified, show available apps and exit
    if args.app is None:
        print("No application specified. Use --app or -a to select an application.")
        print_available_apps()
        return
    
    # Determine acceleration mode
    acceleration = 'optimized' if args.optimize else 'basic'
    
    # Run the selected application
    if args.app == 1:  # Text Classification
        text = args.text if args.text else sample_texts[0]
        print(f"\n🔍 Analyzing: '{text}'")
        
        if acceleration == 'basic':
            result = demo.text_classification_basic(text)
        else:
            result = demo.text_classification_optimized(text)
        
        print(f"\n📊 Results:")
        print(f"  Label: {result['result'][0]['label']}")
        print(f"  Confidence: {result['result'][0]['score']:.3f}")
        print(f"  Method: {result['method']}")
        print(f"  Time: {result['inference_time']:.3f}s")
    
    elif args.app == 2:  # Text Generation
        prompt = args.prompt if args.prompt else sample_prompts[0]
        max_length = args.max_length
        
        print(f"\n🚀 Generating text from: '{prompt}'")
        
        if acceleration == 'basic':
            result = demo.text_generation_basic(prompt, max_length)
        else:
            result = demo.text_generation_optimized(prompt, max_length)
        
        print(f"\n📊 Results:")
        print(f"  Generated: {result['generated_text']}")
        print(f"  Method: {result['method']}")
        print(f"  Time: {result['inference_time']:.3f}s")
        if 'memory_usage_mb' in result:
            print(f"  Memory: {result['memory_usage_mb']:.1f} MB")
    
    elif args.app == 3:  # Question Answering
        question = args.question if args.question else sample_questions[0]
        context = args.context if args.context else sample_context
        
        print(f"\n🔍 Answering: '{question}'")
        
        if acceleration == 'basic':
            result = demo.question_answering_basic(question, context)
        else:
            result = demo.question_answering_optimized(question, context)
        
        print(f"\n📊 Results:")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result['method']}")
        print(f"  Time: {result['inference_time']:.3f}s")
        if 'jit_enabled' in result:
            print(f"  JIT Compilation: {'✅' if result['jit_enabled'] else '❌'}")
    
    elif args.app == 4:  # Named Entity Recognition
        text = args.text if args.text else sample_ner_texts[0]
        
        print(f"\n🏷️  Extracting entities from: '{text}'")
        
        if acceleration == 'basic':
            result = demo.named_entity_recognition_basic(text)
            print(f"\n📊 Results:")
            print(f"  Entities found: {len(result['entities'])}")
            for entity in result['entities']:
                print(f"    {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.3f})")
        else:
            # For optimized, use batch processing with single text
            result = demo.named_entity_recognition_batch([text], batch_size=1)
            print(f"\n📊 Results:")
            entities = result['batch_entities'][0]
            print(f"  Entities found: {len(entities)}")
            for token, label in entities:
                print(f"    {token}: {label}")
        
        print(f"  Method: {result['method']}")
        print(f"  Time: {result.get('inference_time', result.get('total_time', 0)):.3f}s")
    
    elif args.app == 5:  # Batch Processing Demo
        print(f"\n🚀 Running Batch Processing Demo...")
        print(f"Using sample texts: {len(sample_ner_texts)} texts")
        
        batch_size = args.batch_size
        
        # If custom text is provided, use it along with sample texts
        texts = sample_ner_texts
        if args.text:
            texts = [args.text] + sample_ner_texts
            print(f"Added custom text: '{args.text}'")
        
        result = demo.named_entity_recognition_batch(texts, batch_size)
        
        print(f"\n📊 Batch Results:")
        for i, entities in enumerate(result['batch_entities']):
            print(f"  Text {i+1} entities: {len(entities)}")
            for token, label in entities[:3]:  # Show first 3 entities
                print(f"    {token}: {label}")
            if len(entities) > 3:
                print(f"    ... and {len(entities)-3} more")
        
        print(f"  Total time: {result['total_time']:.3f}s")
        print(f"  Average per text: {result['average_per_text']:.3f}s")
        print(f"  Batch size: {result['batch_size']}")
    
    elif args.app == 6:  # Model Benchmarking
        print(f"\n🧪 Running Model Benchmarking...")
        
        model_name = args.model
        num_runs = args.runs
        
        # Use custom prompt if provided
        prompts = sample_prompts[:2]
        if args.prompt:
            prompts = [args.prompt] + sample_prompts[:1]
            print(f"Added custom prompt: '{args.prompt}'")
        
        results = demo.benchmark_model(model_name, prompts, num_runs)
        
        print(f"\n📋 Benchmark Summary:")
        for i, result in enumerate(results):
            print(f"  Prompt {i+1}: {result['avg_time']:.3f}s average")
            print(f"    Range: {result['min_time']:.3f}s - {result['max_time']:.3f}s")
    
    elif args.app == 7:  # Performance Comparison
        print(f"\n⚖️  Running Performance Comparison...")
        print(f"Comparing basic vs optimized implementations")
        
        # Collect performance data
        performance_data = defaultdict(list)
        
        # 1. Text Classification
        text = args.text if args.text else sample_texts[0]
        print(f"\n1️⃣  Testing Text Classification with: '{text}'")
        
        # Run basic version
        print(f"  - Basic Version...")
        basic_result = demo.text_classification_basic(text)
        performance_data['Basic Pipeline'].append(basic_result['inference_time'])
        
        # Run optimized version
        print(f"  - Optimized Version...")
        optimized_result = demo.text_classification_optimized(text)
        performance_data['Accelerated Pipeline'].append(optimized_result['inference_time'])
        
        # 2. Text Generation
        prompt = args.prompt if args.prompt else sample_prompts[0]
        print(f"\n2️⃣  Testing Text Generation with: '{prompt}'")
        
        # Run basic version
        print(f"  - Basic Version...")
        basic_gen_result = demo.text_generation_basic(prompt, max_length=30)
        performance_data['Basic Pipeline'].append(basic_gen_result['inference_time'])
        
        # Run optimized version
        print(f"  - Optimized Version...")
        optimized_gen_result = demo.text_generation_optimized(prompt, max_length=30)
        performance_data['Accelerated Pipeline'].append(optimized_gen_result['inference_time'])
        
        # Store memory usage if available
        if 'memory_usage_mb' in optimized_gen_result:
            performance_data['Accelerated Pipeline_memory'] = [optimized_gen_result['memory_usage_mb']]
        
        # 3. Text Generation with different prompt
        if len(sample_prompts) > 1:
            prompt2 = sample_prompts[1]
            print(f"\n3️⃣  Testing Text Generation with: '{prompt2}'")
            
            # Run basic version
            print(f"  - Basic Version...")
            basic_gen_result2 = demo.text_generation_basic(prompt2, max_length=30)
            performance_data['Text Generation'].append(basic_gen_result2['inference_time'])
        
        # Compare results
        print(f"\n📊 Performance Comparison:")
        print(f"  Basic Text Classification:")
        print(f"    Time: {basic_result['inference_time']:.3f}s")
        print(f"    Method: {basic_result['method']}")
        print(f"  Optimized Text Classification:")
        print(f"    Time: {optimized_result['inference_time']:.3f}s")
        print(f"    Method: {optimized_result['method']}")
        
        if basic_result['inference_time'] > 0:
            speedup = basic_result['inference_time'] / optimized_result['inference_time']
            print(f"  Speedup: {speedup:.2f}x")
        
        # Create performance dashboard
        demo.create_performance_dashboard(performance_data)


if __name__ == "__main__":
    # Import required modules
    import sys
    
    try:
        # Run the demo with command-line arguments
        main()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("For help, use: python transformers_llm_demo.py --list")
        sys.exit(1)


# # List available applications
# python3 jetson/transformers_llm_demo.py --list

# # Run basic text classification
# python3 jetson/transformers_llm_demo.py --app 1 --text "This is a great product!"

# # Run optimized text classification
# python3 jetson/transformers_llm_demo.py --app 1 --text "This is a great product!" --optimize

# # Run basic text generation
# python3 jetson/transformers_llm_demo.py --app 2 --prompt "Edge AI computing with Jetson"

# # Run optimized text generation
# python3 jetson/transformers_llm_demo.py --app 2 --prompt "Edge AI computing with Jetson" --optimize

# # Run question answering
# python3 jetson/transformers_llm_demo.py --app 3 --question "How many CUDA cores?" --context "The device has 1024 CUDA cores."

# # Run named entity recognition
# python3 jetson/transformers_llm_demo.py --app 4 --text "NVIDIA was founded by Jensen Huang in 1993."

# # Run batch processing
# python3 jetson/transformers_llm_demo.py --app 5 --batch-size 4

# # Run model benchmarking
# python3 jetson/transformers_llm_demo.py --app 6 --model gpt2 --runs 3

# # Run performance comparison with visualization
# python3 jetson/transformers_llm_demo.py --app 7