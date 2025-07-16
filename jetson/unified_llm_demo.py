#!/usr/bin/env python3
"""
Unified LLM Demo for Edge Devices

This script provides a comprehensive demonstration of various LLM backends
with device-specific optimizations for edge computing platforms.

Features:
- Multiple LLM backends (llama.cpp, Ollama, llama-cpp-python, ONNX Runtime, etc.)
- Device-specific optimizations (NVIDIA CUDA, Jetson, Apple Silicon, Intel/AMD CPU)
- Model selection and configuration
- Performance benchmarking and visualization
- Memory usage monitoring

Author: Edge AI Course
Version: 1.0
"""

import os
import sys
import time
import json
import psutil
import argparse
import platform
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from contextlib import contextmanager
from collections import defaultdict
import gc
import threading
from pathlib import Path
from datetime import datetime

# For visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization will be disabled.")

# Optional imports for transformers
try:
    from transformers import (
        pipeline, AutoTokenizer, AutoModelForCausalLM, 
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. HuggingFace models will be disabled.")

# Optional imports for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. ONNX optimizations will be disabled.")

# Optional imports for llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    LLAMA_CPP_PYTHON_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Llama.cpp Python bindings will be disabled.")

# Optional imports for Ollama
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: Requests not available. Ollama API will be disabled.")

# Optional imports for async batch processing
try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("Warning: Asyncio/Aiohttp not available. Async batch processing will be disabled.")

# Optional imports for GPU monitoring
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")


@contextmanager
def performance_monitor(device_type="unknown"):
    """
    Context manager for monitoring performance metrics during model inference.
    
    Args:
        device_type (str): Type of device being monitored (cuda, cpu, apple_silicon, jetson)
    
    Tracks:
    - Execution time
    - Memory usage (RAM and GPU if available)
    - CPU usage
    - GPU utilization (if available)
    """
    # Initial measurements
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    start_ram = psutil.virtual_memory().used / (1024 ** 3)  # GB
    
    # GPU-specific measurements
    start_gpu_mem = 0
    if device_type == "cuda" and torch.cuda.is_available():
        start_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    
    try:
        yield
    finally:
        # Final measurements
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_ram = psutil.virtual_memory().used / (1024 ** 3)  # GB
        
        # Calculate metrics
        execution_time = end_time - start_time
        ram_used = end_ram - start_ram  # GB
        cpu_usage = end_cpu - start_cpu
        
        print(f"\n📊 Performance Metrics:")
        print(f"⏱️  Execution time: {execution_time:.3f}s")
        print(f"💻 CPU usage change: {cpu_usage:.1f}%")
        print(f"🧠 RAM used: {ram_used:.3f} GB")
        
        # GPU-specific reporting
        if device_type == "cuda" and torch.cuda.is_available():
            end_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_mem_used = end_gpu_mem - start_gpu_mem
            print(f"🎮 GPU memory used: {gpu_mem_used:.3f} GB")
            
            if GPU_UTIL_AVAILABLE:
                try:
                    gpu = GPUtil.getGPUs()[0]
                    print(f"🎮 GPU utilization: {gpu.load*100:.1f}%")
                    print(f"🌡️  GPU temperature: {gpu.temperature}°C")
                except:
                    print("⚠️  GPU monitoring unavailable")
        
        # Return metrics as a dictionary
        return {
            "execution_time": execution_time,
            "ram_used_gb": ram_used,
            "cpu_usage_percent": cpu_usage
        }


class DeviceManager:
    """
    Manages device detection and optimization settings for different hardware platforms.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.device_type = self._detect_device_type()
        self.optimizations = self._get_available_optimizations()
        
        # Print device information
        self._print_device_info()
    
    def _detect_device_type(self) -> str:
        """
        Detect the type of device and hardware platform.
        
        Returns:
            str: Device type (cuda, cpu, apple_silicon, jetson)
        """
        # Check for NVIDIA GPU with CUDA
        if torch.cuda.is_available():
            # Check if it's a Jetson device
            if os.path.exists("/etc/nv_tegra_release") or \
               os.path.exists("/etc/nv_tegra_version"):
                return "jetson"
            else:
                return "cuda"
        
        # Check for Apple Silicon
        if self.system == "Darwin" and self.machine == "arm64":
            return "apple_silicon"
        
        # Default to CPU
        return "cpu"
    
    def _get_available_optimizations(self) -> Dict[str, bool]:
        """
        Determine which optimizations are available for the current device.
        
        Returns:
            Dict[str, bool]: Dictionary of available optimizations
        """
        optimizations = {
            "onnx": ONNX_AVAILABLE,
            "quantization": self.device_type in ["cuda", "jetson"],
            "mps": self.device_type == "apple_silicon" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "cuda": self.device_type == "cuda",
            "llama_cpp": LLAMA_CPP_PYTHON_AVAILABLE,
            "ollama": OLLAMA_AVAILABLE,
            "async_batch": ASYNC_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE,
            "half_precision": self.device_type in ["cuda", "jetson"],
            "int8": self.device_type in ["cuda", "jetson"] and TRANSFORMERS_AVAILABLE
        }
        
        return optimizations
    
    def _print_device_info(self):
        """
        Print information about the detected device and available optimizations.
        """
        print(f"\n🖥️  Device Information:")
        print(f"  System: {self.system}")
        print(f"  Architecture: {self.machine}")
        print(f"  Device Type: {self.device_type}")
        
        if self.device_type == "cuda":
            print(f"  CUDA Version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n🔧 Available Optimizations:")
        for opt, available in self.optimizations.items():
            status = "✅" if available else "❌"
            print(f"  {status} {opt.upper()}")
    
    def get_optimal_settings(self, backend: str) -> Dict[str, Any]:
        """
        Get optimal settings for a specific backend based on the current device.
        
        Args:
            backend (str): The LLM backend (llama_cpp, ollama, transformers, onnx)
            
        Returns:
            Dict[str, Any]: Dictionary of optimal settings
        """
        settings = {}
        
        if backend == "llama_cpp" and self.optimizations["llama_cpp"]:
            if self.device_type == "cuda":
                settings = {
                    "n_gpu_layers": 35,  # Offload most layers to GPU
                    "n_threads": min(8, os.cpu_count() or 4),
                    "n_batch": 512,
                    "n_ctx": 2048
                }
            elif self.device_type == "jetson":
                settings = {
                    "n_gpu_layers": 20,  # Fewer layers for Jetson
                    "n_threads": min(6, os.cpu_count() or 2),
                    "n_batch": 256,
                    "n_ctx": 2048
                }
            elif self.device_type == "apple_silicon":
                settings = {
                    "n_gpu_layers": 0,  # CPU only for now
                    "n_threads": min(8, os.cpu_count() or 4),
                    "n_batch": 512,
                    "n_ctx": 2048
                }
            else:  # CPU
                settings = {
                    "n_gpu_layers": 0,
                    "n_threads": min(8, os.cpu_count() or 4),
                    "n_batch": 256,
                    "n_ctx": 2048
                }
        
        elif backend == "transformers" and self.optimizations["transformers"]:
            if self.device_type == "cuda":
                settings = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16 if self.optimizations["half_precision"] else torch.float32,
                    "load_in_8bit": self.optimizations["int8"],
                    "use_cache": True
                }
            elif self.device_type == "jetson":
                settings = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16 if self.optimizations["half_precision"] else torch.float32,
                    "load_in_8bit": self.optimizations["int8"],
                    "use_cache": True
                }
            elif self.device_type == "apple_silicon" and self.optimizations["mps"]:
                settings = {
                    "device_map": "mps",
                    "use_cache": True
                }
            else:  # CPU
                settings = {
                    "device_map": "cpu",
                    "use_cache": True
                }
        
        elif backend == "onnx" and self.optimizations["onnx"]:
            if self.device_type == "cuda":
                settings = {
                    "provider": "CUDAExecutionProvider",
                    "optimization_level": 99
                }
            elif self.device_type == "jetson":
                settings = {
                    "provider": "CUDAExecutionProvider",
                    "optimization_level": 99
                }
            else:  # CPU or Apple Silicon
                settings = {
                    "provider": "CPUExecutionProvider",
                    "optimization_level": 99
                }
        
        return settings


class LLMBackendManager:
    """
    Manages different LLM backends and provides a unified interface for inference.
    """
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.available_backends = self._get_available_backends()
        self.loaded_models = {}
    
    def _get_available_backends(self) -> Dict[str, bool]:
        """
        Determine which LLM backends are available.
        
        Returns:
            Dict[str, bool]: Dictionary of available backends
        """
        backends = {
            "llama_cpp": self.device_manager.optimizations["llama_cpp"],
            "ollama": self.device_manager.optimizations["ollama"],
            "transformers": self.device_manager.optimizations["transformers"],
            "onnx": self.device_manager.optimizations["onnx"]
        }
        
        return backends
    
    def load_model(self, backend: str, model_name: str, model_path: Optional[str] = None, 
                  quantization: Optional[str] = None) -> bool:
        """
        Load a model using the specified backend.
        
        Args:
            backend (str): The LLM backend to use
            model_name (str): Name or identifier of the model
            model_path (Optional[str]): Path to model file (for local models)
            quantization (Optional[str]): Quantization level (e.g., q4_0, q4_K_M, q8_0)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self.available_backends.get(backend, False):
            print(f"⚠️  Backend {backend} is not available")
            return False
        
        model_key = f"{backend}_{model_name}"
        
        try:
            if backend == "llama_cpp":
                if not model_path:
                    print("⚠️  Model path is required for llama_cpp backend")
                    return False
                
                # Get optimal settings for the device
                settings = self.device_manager.get_optimal_settings("llama_cpp")
                
                # Load the model
                model = Llama(
                    model_path=model_path,
                    n_gpu_layers=settings["n_gpu_layers"],
                    n_ctx=settings["n_ctx"],
                    n_batch=settings["n_batch"],
                    n_threads=settings["n_threads"],
                    verbose=False
                )
                
                self.loaded_models[model_key] = {
                    "model": model,
                    "backend": backend,
                    "name": model_name,
                    "settings": settings
                }
                
                print(f"✅ Loaded {model_name} with llama_cpp backend")
                return True
            
            elif backend == "ollama":
                # For Ollama, we don't actually load the model here,
                # but we check if the Ollama server is running
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        # Store connection info
                        self.loaded_models[model_key] = {
                            "model": None,  # No actual model object
                            "backend": backend,
                            "name": model_name,
                            "settings": {"url": "http://localhost:11434/api/generate"}
                        }
                        print(f"✅ Connected to Ollama server, model {model_name} will be used")
                        return True
                    else:
                        print("⚠️  Ollama server is not responding correctly")
                        return False
                except requests.exceptions.ConnectionError:
                    print("⚠️  Ollama server is not running")
                    return False
            
            elif backend == "transformers":
                # Get optimal settings for the device
                settings = self.device_manager.get_optimal_settings("transformers")
                
                # Configure quantization if requested
                if quantization == "int8" and self.device_manager.optimizations["int8"]:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    settings["quantization_config"] = quantization_config
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **settings
                )
                
                # Set pad token if needed
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.loaded_models[model_key] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "backend": backend,
                    "name": model_name,
                    "settings": settings
                }
                
                print(f"✅ Loaded {model_name} with transformers backend")
                return True
            
            elif backend == "onnx":
                if not model_path:
                    print("⚠️  Model path is required for ONNX backend")
                    return False
                
                # Get optimal settings for the device
                settings = self.device_manager.get_optimal_settings("onnx")
                
                # Configure session options
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Set up providers
                providers = [
                    (settings["provider"], {}),
                    'CPUExecutionProvider'
                ]
                
                # Load ONNX model
                session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=providers
                )
                
                # Load tokenizer if model_name is provided
                tokenizer = None
                if model_name:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self.loaded_models[model_key] = {
                    "model": session,
                    "tokenizer": tokenizer,
                    "backend": backend,
                    "name": model_name,
                    "settings": settings
                }
                
                print(f"✅ Loaded {model_name} with ONNX backend")
                return True
            
            else:
                print(f"⚠️  Unknown backend: {backend}")
                return False
        
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def generate_text(self, backend: str, model_name: str, prompt: str, 
                     max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text using the specified backend and model.
        
        Args:
            backend (str): The LLM backend to use
            model_name (str): Name or identifier of the model
            prompt (str): Input prompt for generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            
        Returns:
            Dict[str, Any]: Dictionary containing generated text and metrics
        """
        model_key = f"{backend}_{model_name}"
        
        if model_key not in self.loaded_models:
            print(f"⚠️  Model {model_name} with backend {backend} is not loaded")
            return {"error": "Model not loaded"}
        
        model_info = self.loaded_models[model_key]
        
        try:
            start_time = time.time()
            
            if backend == "llama_cpp":
                model = model_info["model"]
                output = model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    echo=False
                )
                generated_text = output["choices"][0]["text"]
            
            elif backend == "ollama":
                url = model_info["settings"]["url"]
                data = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    generated_text = response.json()["response"]
                else:
                    return {"error": f"Ollama API error: {response.text}"}
            
            elif backend == "transformers":
                model = model_info["model"]
                tokenizer = model_info["tokenizer"]
                
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                
                # Move to the appropriate device
                if hasattr(model, "device"):
                    input_ids = input_ids.to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            elif backend == "onnx":
                session = model_info["model"]
                tokenizer = model_info["tokenizer"]
                
                if not tokenizer:
                    return {"error": "Tokenizer not available for ONNX model"}
                
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="np")
                
                # Get input names and run inference
                input_names = [input.name for input in session.get_inputs()]
                ort_inputs = {name: inputs[name.split('.')[-1]].astype(np.int64) 
                             for name in input_names if name.split('.')[-1] in inputs}
                
                # Run inference
                outputs = session.run(None, ort_inputs)
                
                # Decode output
                output_ids = outputs[0]
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            else:
                return {"error": f"Unknown backend: {backend}"}
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Get memory usage
            memory_usage = None
            if self.device_manager.device_type == "cuda" and torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "inference_time": inference_time,
                "memory_usage_gb": memory_usage,
                "backend": backend,
                "model": model_name
            }
        
        except Exception as e:
            print(f"❌ Error generating text: {str(e)}")
            return {"error": str(e)}
    
    def batch_generate(self, backend: str, model_name: str, prompts: List[str], 
                      max_tokens: int = 256, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batch mode.
        
        Args:
            backend (str): The LLM backend to use
            model_name (str): Name or identifier of the model
            prompts (List[str]): List of input prompts
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing generated text and metrics
        """
        results = []
        
        # For Ollama, use async batch processing if available
        if backend == "ollama" and ASYNC_AVAILABLE:
            return self._batch_generate_ollama_async(model_name, prompts, max_tokens, temperature)
        
        # For other backends, process sequentially
        for prompt in prompts:
            result = self.generate_text(backend, model_name, prompt, max_tokens, temperature)
            results.append(result)
        
        return results
    
    async def _generate_async(self, session, url, model_name, prompt, max_tokens, temperature):
        """
        Helper method for async generation with Ollama.
        """
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        start_time = time.time()
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "generated_text": result["response"],
                    "prompt": prompt,
                    "inference_time": end_time - start_time,
                    "backend": "ollama",
                    "model": model_name
                }
            else:
                return {"error": f"Ollama API error: {await response.text()}"}
    
    def _batch_generate_ollama_async(self, model_name: str, prompts: List[str], 
                                   max_tokens: int, temperature: float) -> List[Dict[str, Any]]:
        """
        Batch generation using Ollama with async processing.
        """
        model_key = f"ollama_{model_name}"
        
        if model_key not in self.loaded_models:
            print(f"⚠️  Model {model_name} with Ollama backend is not loaded")
            return [{"error": "Model not loaded"}] * len(prompts)
        
        model_info = self.loaded_models[model_key]
        url = model_info["settings"]["url"]
        
        async def run_batch():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for prompt in prompts:
                    task = asyncio.create_task(
                        self._generate_async(session, url, model_name, prompt, max_tokens, temperature)
                    )
                    tasks.append(task)
                
                return await asyncio.gather(*tasks)
        
        # Run the async batch
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(run_batch())
        return results


class BenchmarkManager:
    """
    Manages benchmarking of LLM backends and models.
    """
    
    def __init__(self, device_manager: DeviceManager, llm_manager: LLMBackendManager):
        self.device_manager = device_manager
        self.llm_manager = llm_manager
        self.results = []
    
    def run_benchmark(self, backend: str, model_name: str, prompts: List[str], 
                     num_runs: int = 3, max_tokens: int = 50, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Run a benchmark for a specific backend and model.
        
        Args:
            backend (str): The LLM backend to use
            model_name (str): Name or identifier of the model
            prompts (List[str]): List of prompts to test
            num_runs (int): Number of runs per prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            
        Returns:
            Dict[str, Any]: Dictionary containing benchmark results
        """
        print(f"\n🧪 Benchmarking {backend} with model {model_name}...")
        
        model_key = f"{backend}_{model_name}"
        if model_key not in self.llm_manager.loaded_models:
            print(f"⚠️  Model {model_name} with backend {backend} is not loaded")
            return {"error": "Model not loaded"}
        
        benchmark_results = {
            "backend": backend,
            "model": model_name,
            "device": self.device_manager.device_type,
            "num_runs": num_runs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "prompt_results": []
        }
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            prompt_result = {
                "prompt": prompt,
                "runs": []
            }
            
            run_times = []
            memory_usages = []
            
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}...")
                
                with performance_monitor(self.device_manager.device_type) as metrics:
                    result = self.llm_manager.generate_text(
                        backend, model_name, prompt, max_tokens, temperature
                    )
                
                if "error" in result:
                    print(f"❌ Error in run {run+1}: {result['error']}")
                    prompt_result["runs"].append({
                        "error": result["error"],
                        "run": run + 1
                    })
                    continue
                
                run_times.append(result["inference_time"])
                if "memory_usage_gb" in result and result["memory_usage_gb"] is not None:
                    memory_usages.append(result["memory_usage_gb"])
                
                prompt_result["runs"].append({
                    "inference_time": result["inference_time"],
                    "memory_usage_gb": result.get("memory_usage_gb"),
                    "generated_text": result["generated_text"],
                    "run": run + 1
                })
                
                # Show output for first run
                if run == 0:
                    print(f"  Generated: {result['generated_text'][:100]}...")
            
            # Calculate statistics
            if run_times:
                avg_time = sum(run_times) / len(run_times)
                min_time = min(run_times)
                max_time = max(run_times)
                
                prompt_result["stats"] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "std_dev": np.std(run_times) if len(run_times) > 1 else 0
                }
                
                if memory_usages:
                    prompt_result["stats"]["avg_memory_gb"] = sum(memory_usages) / len(memory_usages)
                
                print(f"📈 Average: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
            
            benchmark_results["prompt_results"].append(prompt_result)
        
        # Calculate overall statistics
        all_times = [run["inference_time"] for pr in benchmark_results["prompt_results"] 
                    for run in pr["runs"] if "inference_time" in run]
        
        if all_times:
            benchmark_results["overall_stats"] = {
                "avg_time": sum(all_times) / len(all_times),
                "min_time": min(all_times),
                "max_time": max(all_times),
                "std_dev": np.std(all_times) if len(all_times) > 1 else 0,
                "total_runs_completed": len(all_times)
            }
        
        # Store results
        self.results.append(benchmark_results)
        
        return benchmark_results
    
    def compare_backends(self, prompts: List[str], backends_models: List[Tuple[str, str]], 
                        num_runs: int = 3, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Compare multiple backends and models on the same prompts.
        
        Args:
            prompts (List[str]): List of prompts to test
            backends_models (List[Tuple[str, str]]): List of (backend, model) pairs to compare
            num_runs (int): Number of runs per prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: Dictionary containing comparison results
        """
        print(f"\n⚖️  Running Performance Comparison...")
        
        comparison_results = {
            "device": self.device_manager.device_type,
            "num_runs": num_runs,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
            "backend_results": []
        }
        
        for backend, model_name in backends_models:
            print(f"\n🔍 Testing {backend} with model {model_name}...")
            
            benchmark_result = self.run_benchmark(
                backend, model_name, prompts, num_runs, max_tokens
            )
            
            if "error" not in benchmark_result:
                comparison_results["backend_results"].append({
                    "backend": backend,
                    "model": model_name,
                    "overall_stats": benchmark_result.get("overall_stats", {})
                })
        
        # Create visualization
        self.create_comparison_visualization(comparison_results)
        
        # Save results to file
        self.save_results_to_file(comparison_results, "llm_comparison_results.json")
        
        return comparison_results
    
    def create_comparison_visualization(self, comparison_results: Dict[str, Any], 
                                       output_file: str = "llm_backend_comparison.png") -> None:
        """
        Create a visualization of the comparison results.
        
        Args:
            comparison_results (Dict[str, Any]): Comparison results
            output_file (str): Output file path for the visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available. Cannot create visualization.")
            return
        
        print("\n📊 Creating comparison visualization...")
        
        try:
            # Extract data for plotting
            backends = []
            avg_times = []
            std_devs = []
            memory_usages = []
            
            for result in comparison_results["backend_results"]:
                backend_name = f"{result['backend']}\n{result['model'].split('/')[-1]}"
                backends.append(backend_name)
                
                stats = result["overall_stats"]
                avg_times.append(stats.get("avg_time", 0))
                std_devs.append(stats.get("std_dev", 0))
                memory_usages.append(stats.get("avg_memory_gb", 0))
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot inference times
            bars = ax1.bar(backends, avg_times, yerr=std_devs, capsize=5, 
                         color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#a1c181'][:len(backends)])
            ax1.set_ylabel('Average Inference Time (seconds)')
            ax1.set_title('Inference Time Comparison')
            ax1.set_ylim(0, max(avg_times) * 1.2 if avg_times else 1)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, avg_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom')
            
            # Plot memory usage if available
            if any(memory_usages):
                ax2.bar(backends, memory_usages, 
                       color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#a1c181'][:len(backends)])
                ax2.set_ylabel('Memory Usage (GB)')
                ax2.set_title('Memory Efficiency')
                ax2.set_ylim(0, max(memory_usages) * 1.2 if memory_usages else 1)
            else:
                ax2.text(0.5, 0.5, 'Memory usage data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            # Add title and metadata
            plt.suptitle(f'LLM Backend Comparison on {comparison_results["device"]} Device', fontsize=16)
            plt.figtext(0.5, 0.01, 
                       f'Benchmark: {comparison_results["num_runs"]} runs, {comparison_results["max_tokens"]} tokens, '
                       f'{len(comparison_results["backend_results"])} backends', 
                       ha='center')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            
            print(f"📊 Comparison visualization saved as '{output_file}'")
            
        except Exception as e:
            print(f"⚠️ Error creating visualization: {str(e)}")
    
    def save_results_to_file(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Results to save
            filename (str): Output filename
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("benchmark_results")
            output_dir.mkdir(exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_with_timestamp = f"{timestamp}_{filename}"
            output_path = output_dir / filename_with_timestamp
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving results: {str(e)}")


def optimize_memory():
    """
    Optimize memory usage before loading models.
    """
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
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)
            
            print(f"GPU Memory: {gpu_memory / (1024**3):.1f}GB total")
            print(f"GPU Allocated: {gpu_allocated / (1024**3):.1f}GB")
            print(f"GPU Reserved: {gpu_reserved / (1024**3):.1f}GB")
        except Exception as e:
            print(f"Error getting GPU memory info: {str(e)}")


def print_available_backends_and_models():
    """
    Print available backends and example models.
    """
    print("\n" + "="*60)
    print("🤖 UNIFIED LLM DEMO FOR EDGE DEVICES")
    print("="*60)
    
    print("\n📋 Available Backends:")
    
    if LLAMA_CPP_PYTHON_AVAILABLE:
        print("  1. llama_cpp - High-Performance C++ Engine with Python bindings")
        print("     Example models: llama-2-7b-chat.q4_K_M.gguf, mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    else:
        print("  ❌ llama_cpp - Not available (install with: pip install llama-cpp-python)")
    
    if OLLAMA_AVAILABLE:
        print("  2. ollama - Simplified LLM Deployment with REST API")
        print("     Example models: llama2:7b-chat, mistral:7b-instruct")
    else:
        print("  ❌ ollama - Not available (install Ollama from: https://ollama.ai)")
    
    if TRANSFORMERS_AVAILABLE:
        print("  3. transformers - HuggingFace Transformers Library")
        print("     Example models: gpt2, facebook/opt-350m, TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        print("  ❌ transformers - Not available (install with: pip install transformers)")
    
    if ONNX_AVAILABLE:
        print("  4. onnx - ONNX Runtime for Cross-Platform Optimization")
        print("     Requires ONNX model files (.onnx)")
    else:
        print("  ❌ onnx - Not available (install with: pip install onnxruntime)")
    
    print("\n📝 Example Usage:")
    print("  python unified_llm_demo.py --backend llama_cpp --model-path models/llama-2-7b-chat.q4_K_M.gguf --prompt \"Explain edge AI\"")
    print("  python unified_llm_demo.py --backend ollama --model-name llama2:7b-chat --prompt \"Explain edge AI\"")
    print("  python unified_llm_demo.py --backend transformers --model-name gpt2 --prompt \"Explain edge AI\"")
    print("  python unified_llm_demo.py --benchmark --backends llama_cpp ollama --model-names llama-2-7b-chat.q4_K_M.gguf llama2:7b-chat")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Unified LLM Demo for Edge Devices")
    
    # Backend and model selection
    parser.add_argument(
        "--backend", type=str, choices=["llama_cpp", "ollama", "transformers", "onnx"],
        help="LLM backend to use"
    )
    parser.add_argument(
        "--model-name", type=str,
        help="Name or identifier of the model"
    )
    parser.add_argument(
        "--model-path", type=str,
        help="Path to model file (for local models)"
    )
    
    # Input options
    parser.add_argument(
        "--prompt", type=str,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    
    # Optimization options
    parser.add_argument(
        "--quantization", type=str, choices=["int8", "none"],
        help="Quantization level for transformers models"
    )
    
    # Batch processing
    parser.add_argument(
        "--batch", action="store_true",
        help="Enable batch processing mode"
    )
    parser.add_argument(
        "--prompts-file", type=str,
        help="Path to file containing prompts for batch processing (one per line)"
    )
    
    # Benchmarking
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark mode"
    )
    parser.add_argument(
        "--backends", type=str, nargs="+",
        help="List of backends to benchmark"
    )
    parser.add_argument(
        "--model-names", type=str, nargs="+",
        help="List of model names to benchmark (must match number of backends)"
    )
    parser.add_argument(
        "--model-paths", type=str, nargs="+",
        help="List of model paths to benchmark (for local models)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Number of runs for benchmarking (default: 3)"
    )
    
    # Utility options
    parser.add_argument(
        "--list", action="store_true",
        help="List available backends and example models"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the demo.
    """
    args = parse_arguments()
    
    # If --list flag is provided, just show available backends and exit
    if args.list:
        print_available_backends_and_models()
        return
    
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Initialize LLM backend manager
    llm_manager = LLMBackendManager(device_manager)
    
    # Initialize benchmark manager
    benchmark_manager = BenchmarkManager(device_manager, llm_manager)
    
    # Optimize memory before loading models
    optimize_memory()
    
    # Sample prompts for demonstrations
    sample_prompts = [
        "Explain the benefits of edge AI computing",
        "What are the advantages of running LLMs on edge devices?",
        "Compare cloud computing vs edge computing for AI applications"
    ]
    
    # Run in benchmark mode
    if args.benchmark:
        if not args.backends or not args.model_names or len(args.backends) != len(args.model_names):
            print("⚠️  For benchmarking, you must provide equal numbers of backends and model names")
            return
        
        # Load models for benchmarking
        backends_models = []
        for i, (backend, model_name) in enumerate(zip(args.backends, args.model_names)):
            model_path = args.model_paths[i] if args.model_paths and i < len(args.model_paths) else None
            
            if llm_manager.load_model(backend, model_name, model_path, args.quantization):
                backends_models.append((backend, model_name))
        
        if not backends_models:
            print("❌ No models loaded for benchmarking")
            return
        
        # Use provided prompts or sample prompts
        prompts = []
        if args.prompts_file:
            try:
                with open(args.prompts_file, 'r') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"⚠️  Error reading prompts file: {str(e)}")
        
        if not prompts:
            if args.prompt:
                prompts = [args.prompt] + sample_prompts[:1]
            else:
                prompts = sample_prompts[:2]  # Use first two sample prompts
        
        # Run comparison benchmark
        benchmark_manager.compare_backends(
            prompts, backends_models, args.num_runs, args.max_tokens
        )
        
        return
    
    # Run in normal mode (single backend)
    if not args.backend or not (args.model_name or args.model_path):
        print("⚠️  You must specify a backend and either a model name or model path")
        print_available_backends_and_models()
        return
    
    # Load the model
    if not llm_manager.load_model(args.backend, args.model_name, args.model_path, args.quantization):
        print("❌ Failed to load model")
        return
    
    # Run in batch mode
    if args.batch:
        prompts = []
        if args.prompts_file:
            try:
                with open(args.prompts_file, 'r') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"⚠️  Error reading prompts file: {str(e)}")
                return
        elif args.prompt:
            prompts = [args.prompt] + sample_prompts
        else:
            prompts = sample_prompts
        
        print(f"\n🚀 Running batch processing with {len(prompts)} prompts...")
        results = llm_manager.batch_generate(
            args.backend, args.model_name, prompts, args.max_tokens, args.temperature
        )
        
        for i, result in enumerate(results):
            print(f"\n📝 Result {i+1}/{len(results)}:")
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"Prompt: {result['prompt']}")
                print(f"Generated: {result['generated_text']}")
                print(f"Time: {result['inference_time']:.3f}s")
        
        return
    
    # Run in single prompt mode
    prompt = args.prompt if args.prompt else sample_prompts[0]
    
    print(f"\n🚀 Generating text with {args.backend} backend...")
    print(f"Prompt: {prompt}")
    
    with performance_monitor(device_manager.device_type):
        result = llm_manager.generate_text(
            args.backend, args.model_name, prompt, args.max_tokens, args.temperature
        )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"\n📝 Generated Text:")
        print(result["generated_text"])
        print(f"\n⏱️  Inference Time: {result['inference_time']:.3f}s")
        if "memory_usage_gb" in result and result["memory_usage_gb"] is not None:
            print(f"🧠 Memory Usage: {result['memory_usage_gb']:.3f} GB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("For help, use: python unified_llm_demo.py --list")
        sys.exit(1)