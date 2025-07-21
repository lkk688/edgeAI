#!/usr/bin/env python3

"""Jetson Prompt Engineering Toolkit

A unified toolkit for prompt engineering techniques, LangChain integration, and advanced
LLM capabilities on NVIDIA Jetson platforms.

This toolkit supports multiple LLM backends:
1. OpenAI API
2. Local Ollama
3. Local llama-cpp-python

Features:
- Core prompt engineering techniques
- LangChain integration
- Structured output generation
- Tool calling capabilities
- Function calling
- Model Context Protocol (MCP) integration
- Advanced lab exercises

Usage:
    python jetson_prompt_toolkit.py --mode [mode] [options]

Examples:
    # Test basic prompt engineering techniques
    python jetson_prompt_toolkit.py --mode basic --technique cot
    
    # Compare different LLM backends
    python jetson_prompt_toolkit.py --mode compare --backends openai,ollama
    
    # Generate structured output
    python jetson_prompt_toolkit.py --mode structured --output optimization_plan
    
    # Use tool calling capabilities
    python jetson_prompt_toolkit.py --mode tools --request "Benchmark my YOLOv8 model"
    
    # Run advanced lab exercises
    python jetson_prompt_toolkit.py --mode lab
"""

import argparse
import json
import os
import random
import time
import inspect
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import wraps

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# LangChain imports (optional)
try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False

try:
    from langchain_community.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_LLAMACPP_AVAILABLE = True
except ImportError:
    LANGCHAIN_LLAMACPP_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ===== Core Prompter Classes =====

class OpenAIPrompter:
    """OpenAI API-based prompter"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def prompt(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send prompt to OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return response.choices[0].message.content
    
    def simple_prompt(self, prompt: str, system_msg: str = None) -> str:
        """Simple prompt interface"""
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})
        return self.prompt(messages)


class OllamaPrompter:
    """Local Ollama-based prompter"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests package not installed. Install with: pip install requests")
        
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def prompt(self, prompt: str, system_msg: str = None, **kwargs) -> str:
        """Send prompt to Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "num_predict": kwargs.get('max_tokens', 1000)
            }
        }
        
        if system_msg:
            payload["system"] = system_msg
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface for conversation"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7)
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"Ollama error: {response.text}")


class LlamaCppPrompter:
    """Local llama-cpp-python based prompter"""
    
    def __init__(self, model_path: str, **kwargs):
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=kwargs.get('n_gpu_layers', -1),  # Use all GPU layers
            n_ctx=kwargs.get('n_ctx', 4096),  # Context window
            n_batch=kwargs.get('n_batch', 512),  # Batch size
            verbose=kwargs.get('verbose', False)
        )
        print(f"✅ Loaded model: {model_path}")
    
    def prompt(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        start_time = time.time()
        
        response = self.llm(
            prompt,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            stop=kwargs.get('stop', ["\n\n"]),
            echo=False
        )
        
        inference_time = time.time() - start_time
        print(f"⏱️ Inference time: {inference_time:.2f}s")
        
        return response['choices'][0]['text'].strip()
    
    def chat_prompt(self, system_msg: str, user_msg: str) -> str:
        """Format as chat conversation"""
        prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        return self.prompt(prompt, stop=["<|im_end|>"])


# ===== LangChain Integration =====

class LangChainOpenAIPrompter:
    """LangChain with OpenAI integration"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        if not LANGCHAIN_AVAILABLE or not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError("LangChain or LangChain OpenAI not installed. Install with: pip install langchain langchain-openai")
        
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.7
        )
    
    def simple_prompt(self, prompt: str, system_message: str = None) -> str:
        """Simple prompting with optional system message"""
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def template_prompt(self, template: str, **kwargs) -> str:
        """Use prompt templates for reusable prompts"""
        prompt_template = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt_template.format(**kwargs)
        
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return response.content
    
    def chain_prompts(self, prompts: list) -> list:
        """Chain multiple prompts together"""
        results = []
        context = ""
        
        for i, prompt in enumerate(prompts):
            if i > 0:
                full_prompt = f"Previous context: {context}\n\nNew task: {prompt}"
            else:
                full_prompt = prompt
            
            response = self.simple_prompt(full_prompt)
            results.append(response)
            context = response[:200]  # Keep context manageable
        
        return results


class LangChainOllamaPrompter:
    """LangChain with Ollama integration"""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        if not LANGCHAIN_AVAILABLE or not LANGCHAIN_OLLAMA_AVAILABLE:
            raise ImportError("LangChain or LangChain Ollama not installed. Install with: pip install langchain langchain-community")
        
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.7
        )
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            response = requests.get(f"{self.llm.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def simple_prompt(self, prompt: str, system_message: str = None) -> str:
        """Simple prompting with optional system message"""
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def streaming_prompt(self, prompt: str):
        """Stream response for long outputs"""
        for chunk in self.llm.stream([HumanMessage(content=prompt)]):
            yield chunk.content
    
    def batch_prompts(self, prompts: list) -> list:
        """Process multiple prompts efficiently"""
        messages_list = [[HumanMessage(content=prompt)] for prompt in prompts]
        responses = self.llm.batch(messages_list)
        return [response.content for response in responses]


class LangChainLlamaCppPrompter:
    """LangChain with llama-cpp-python integration"""
    
    def __init__(self, model_path: str, n_gpu_layers: int = 80):
        if not LANGCHAIN_AVAILABLE or not LANGCHAIN_LLAMACPP_AVAILABLE:
            raise ImportError("LangChain or LangChain LlamaCpp not installed. Install with: pip install langchain langchain-community")
        
        # Callback for streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=False,
        )
    
    def simple_prompt(self, prompt: str, system_message: str = None) -> str:
        """Simple prompting with optional system message"""
        if system_message:
            full_prompt = f"System: {system_message}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"
        
        response = self.llm.invoke(full_prompt)
        return response
    
    def template_prompt(self, template: str, **kwargs) -> str:
        """Use prompt templates"""
        prompt_template = PromptTemplate.from_template(template)
        formatted_prompt = prompt_template.format(**kwargs)
        
        response = self.llm.invoke(formatted_prompt)
        return response
    
    def conversation_prompt(self, messages: list) -> str:
        """Handle conversation format"""
        conversation = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                conversation += f"System: {content}\n\n"
            elif role == "user":
                conversation += f"Human: {content}\n\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n\n"
        
        conversation += "Assistant:"
        response = self.llm.invoke(conversation)
        return response


# ===== Prompt Engineering Techniques =====

class PromptType(Enum):
    BASIC = "basic"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    ROLE_BASED = "role_based"
    IN_CONTEXT_LEARNING = "in_context_learning"


@dataclass
class PromptResult:
    prompt_type: PromptType
    response: str
    response_time: float
    token_count: int
    quality_score: float


class PromptEngineeringLab:
    """Comprehensive lab for testing prompt engineering techniques"""
    
    def __init__(self, llm_clients: Dict[str, Any]):
        self.llm_clients = llm_clients
        self.results = []
    
    def create_prompts(self, task: str, context: str = "") -> Dict[PromptType, str]:
        """Create different prompt variations for the same task"""
        
        prompts = {
            PromptType.BASIC: f"{task}",
            
            PromptType.CHAIN_OF_THOUGHT: f"""
{task}

Let's think step by step:
1. First, I need to understand the problem
2. Then, I'll analyze the requirements
3. Next, I'll consider the constraints
4. Finally, I'll provide a comprehensive solution

Step-by-step analysis:
""",
            
            PromptType.FEW_SHOT: f"""
Here are some examples of similar tasks:

Example 1:
Task: Optimize a CNN model for mobile deployment
Solution: Use quantization, pruning, and knowledge distillation

Example 2:
Task: Reduce inference latency for object detection
Solution: Use TensorRT, optimize batch size, and implement async processing

Now solve this task:
{task}

Solution:
""",
            
            PromptType.ROLE_BASED: f"""
You are a senior NVIDIA Jetson optimization engineer with 10+ years of experience in edge AI deployment. You specialize in maximizing performance while minimizing power consumption.

Task: {task}

As an expert, provide your professional recommendation:
""",
            
            PromptType.IN_CONTEXT_LEARNING: f"""
I'll teach you a new format for Jetson optimization reports:

Format:
```
OPTIMIZATION REPORT
==================
Model: [model_name]
Current Performance: [fps] FPS, [latency]ms latency
Target Performance: [target_fps] FPS

Optimization Strategy:
- Technique 1: [description] -> Expected gain: [X]%
- Technique 2: [description] -> Expected gain: [Y]%

Implementation Priority:
1. [High priority item]
2. [Medium priority item]
3. [Low priority item]

Risk Assessment: [Low/Medium/High]
Estimated Timeline: [X] days
```

Now use this format to solve:
{task}
"""
        }
        
        return prompts
    
    def evaluate_response(self, response: str, expected_keywords: List[str]) -> float:
        """Simple quality evaluation based on keyword presence"""
        score = 0.0
        response_lower = response.lower()
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                score += 1.0
        
        return min(score / len(expected_keywords), 1.0) if expected_keywords else 0.5
    
    def run_prompt_comparison(self, task: str, expected_keywords: List[str]) -> List[PromptResult]:
        """Run the same task with different prompting techniques"""
        prompts = self.create_prompts(task)
        results = []
        
        for prompt_type, prompt_text in prompts.items():
            print(f"\nTesting {prompt_type.value} prompting...")
            
            for llm_name, llm in self.llm_clients.items():
                start_time = time.time()
                
                try:
                    response = llm.simple_prompt(prompt_text)
                    response_time = time.time() - start_time
                    
                    # Estimate token count (rough approximation)
                    token_count = len(response.split()) * 1.3
                    
                    # Evaluate quality
                    quality_score = self.evaluate_response(response, expected_keywords)
                    
                    result = PromptResult(
                        prompt_type=prompt_type,
                        response=response,
                        response_time=response_time,
                        token_count=int(token_count),
                        quality_score=quality_score
                    )
                    
                    results.append(result)
                    print(f"  {llm_name}: {quality_score:.2f} quality, {response_time:.2f}s")
                    
                except Exception as e:
                    print(f"  {llm_name}: Error - {e}")
        
        return results
    
    def generate_comparison_report(self, results: List[PromptResult]) -> str:
        """Generate a comprehensive comparison report"""
        report = "\n" + "=" * 60
        report += "\nPROMPT ENGINEERING COMPARISON REPORT\n"
        report += "=" * 60 + "\n"
        
        # Group by prompt type
        by_type = {}
        for result in results:
            if result.prompt_type not in by_type:
                by_type[result.prompt_type] = []
            by_type[result.prompt_type].append(result)
        
        # Calculate averages
        for prompt_type, type_results in by_type.items():
            avg_quality = sum(r.quality_score for r in type_results) / len(type_results)
            avg_time = sum(r.response_time for r in type_results) / len(type_results)
            avg_tokens = sum(r.token_count for r in type_results) / len(type_results)
            
            report += f"\n{prompt_type.value.upper()}:\n"
            report += f"  Average Quality: {avg_quality:.3f}\n"
            report += f"  Average Time: {avg_time:.2f}s\n"
            report += f"  Average Tokens: {avg_tokens:.0f}\n"
        
        # Find best performing technique
        best_quality = max(results, key=lambda r: r.quality_score)
        fastest = min(results, key=lambda r: r.response_time)
        
        report += f"\nBEST PERFORMERS:\n"
        report += f"  Highest Quality: {best_quality.prompt_type.value} ({best_quality.quality_score:.3f})\n"
        report += f"  Fastest Response: {fastest.prompt_type.value} ({fastest.response_time:.2f}s)\n"
        
        return report


# ===== Structured Output =====

if PYDANTIC_AVAILABLE:
    class JetsonOptimizationPlan(BaseModel):
        """Structured plan for Jetson model optimization"""
        model_name: str = Field(description="Name of the AI model")
        current_performance: dict = Field(description="Current FPS and latency metrics")
        optimization_steps: List[str] = Field(description="List of optimization techniques to apply")
        expected_improvement: dict = Field(description="Expected performance gains")
        estimated_time: str = Field(description="Time required for optimization")
        difficulty_level: str = Field(description="Beginner, Intermediate, or Advanced")
        required_tools: List[str] = Field(description="Tools and libraries needed")

    class JetsonDeviceComparison(BaseModel):
        """Structured comparison of Jetson devices"""
        device_name: str = Field(description="Jetson device name")
        gpu_cores: int = Field(description="Number of GPU cores")
        cpu_cores: int = Field(description="Number of CPU cores")
        ram_gb: int = Field(description="RAM in GB")
        power_consumption: str = Field(description="Power consumption")
        best_use_cases: List[str] = Field(description="Recommended use cases")
        price_range: str = Field(description="Approximate price range")

    class ToolCall(BaseModel):
        """Represents a tool call request"""
        tool_name: str = Field(description="Name of the tool to call")
        parameters: dict = Field(description="Parameters for the tool")
        reasoning: str = Field(description="Why this tool is needed")

    class JetsonSystemInfo(BaseModel):
        """System information tool result"""
        gpu_memory_used: float = Field(description="GPU memory usage in MB")
        cpu_usage: float = Field(description="CPU usage percentage")
        temperature: float = Field(description="System temperature in Celsius")
        available_models: List[str] = Field(description="Available AI models")

    class ModelBenchmark(BaseModel):
        """Model benchmarking tool result"""
        model_name: str = Field(description="Name of the benchmarked model")
        fps: float = Field(description="Frames per second")
        latency_ms: float = Field(description="Latency in milliseconds")
        memory_usage_mb: float = Field(description="Memory usage in MB")
        accuracy: Optional[float] = Field(description="Model accuracy if available")


class StructuredPrompter:
    """Prompter that returns structured data"""
    
    def __init__(self, llm):
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic not installed. Install with: pip install pydantic")
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Install with: pip install langchain")
        
        self.llm = llm
    
    def get_optimization_plan(self, model_description: str) -> Any:
        """Get a structured optimization plan"""
        parser = PydanticOutputParser(pydantic_object=JetsonOptimizationPlan)
        
        prompt = PromptTemplate(
            template="""
You are a Jetson optimization expert. Create a detailed optimization plan for the following model:

Model Description: {model_description}

Provide a comprehensive optimization strategy.

{format_instructions}
""",
            input_variables=["model_description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        formatted_prompt = prompt.format(model_description=model_description)
        response = self.llm.invoke(formatted_prompt)
        
        try:
            return parser.parse(response)
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Raw response: {response}")
            return None
    
    def compare_jetson_devices(self, devices: List[str]) -> List[Any]:
        """Get structured comparison of Jetson devices"""
        parser = PydanticOutputParser(pydantic_object=JetsonDeviceComparison)
        
        results = []
        for device in devices:
            prompt = PromptTemplate(
                template="""
Provide detailed specifications and information about the {device}.

Include technical specs, performance characteristics, and recommended use cases.

{format_instructions}
""",
                input_variables=["device"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            formatted_prompt = prompt.format(device=device)
            response = self.llm.invoke(formatted_prompt)
            
            try:
                parsed_result = parser.parse(response)
                results.append(parsed_result)
            except Exception as e:
                print(f"Parsing error for {device}: {e}")
                continue
        
        return results


# ===== Tool Calling =====

class JetsonToolkit:
    """Collection of Jetson-specific tools"""
    
    @staticmethod
    def get_system_info() -> Any:
        """Get current Jetson system information"""
        try:
            # Simulate getting GPU memory (would use actual NVIDIA tools)
            gpu_memory = 2048.5  # MB
            cpu_usage = psutil.cpu_percent() if PSUTIL_AVAILABLE else 45.2
            temperature = 45.2  # Would read from thermal sensors
            
            # Simulate available models
            available_models = ["yolov8n.engine", "resnet50.onnx", "mobilenet.trt"]
            
            if PYDANTIC_AVAILABLE:
                return JetsonSystemInfo(
                    gpu_memory_used=gpu_memory,
                    cpu_usage=cpu_usage,
                    temperature=temperature,
                    available_models=available_models
                )
            else:
                return {
                    "gpu_memory_used": gpu_memory,
                    "cpu_usage": cpu_usage,
                    "temperature": temperature,
                    "available_models": available_models
                }
        except Exception as e:
            print(f"Error getting system info: {e}")
            return None
    
    @staticmethod
    def benchmark_model(model_path: str, input_size: tuple = (640, 640)) -> Any:
        """Benchmark a model on Jetson"""
        try:
            # Simulate benchmarking (would run actual inference)
            model_name = model_path.split('/')[-1]
            
            # Simulate results based on model type
            if "yolo" in model_name.lower():
                fps = 28.5
                latency = 35.1
                memory = 1024.0
            elif "resnet" in model_name.lower():
                fps = 45.2
                latency = 22.1
                memory = 512.0
            else:
                fps = 60.0
                latency = 16.7
                memory = 256.0
            
            if PYDANTIC_AVAILABLE:
                return ModelBenchmark(
                    model_name=model_name,
                    fps=fps,
                    latency_ms=latency,
                    memory_usage_mb=memory,
                    accuracy=0.85
                )
            else:
                return {
                    "model_name": model_name,
                    "fps": fps,
                    "latency_ms": latency,
                    "memory_usage_mb": memory,
                    "accuracy": 0.85
                }
        except Exception as e:
            print(f"Error benchmarking model: {e}")
            return None
    
    @staticmethod
    def optimize_model(model_path: str, target_precision: str = "fp16") -> dict:
        """Optimize a model for Jetson deployment"""
        try:
            # Simulate optimization process
            optimization_result = {
                "status": "success",
                "original_size_mb": 245.6,
                "optimized_size_mb": 123.2,
                "speedup_factor": 2.3,
                "output_path": f"/opt/models/optimized_{model_path.split('/')[-1]}",
                "optimization_time_seconds": 45.2
            }
            return optimization_result
        except Exception as e:
            return {"status": "error", "message": str(e)}


class ToolCallingPrompter:
    """LLM prompter with tool calling capabilities"""
    
    def __init__(self, llm):
        self.llm = llm
        self.toolkit = JetsonToolkit()
        
        # Define available tools
        self.available_tools = {
            "get_system_info": {
                "description": "Get current Jetson system information including GPU memory, CPU usage, and temperature",
                "parameters": {},
                "returns": "JetsonSystemInfo object"
            },
            "benchmark_model": {
                "description": "Benchmark an AI model on Jetson to measure performance",
                "parameters": {
                    "model_path": "Path to the model file",
                    "input_size": "Input size as (height, width) tuple, default (640, 640)"
                },
                "returns": "ModelBenchmark object with FPS, latency, and memory usage"
            },
            "optimize_model": {
                "description": "Optimize a model for Jetson deployment using TensorRT",
                "parameters": {
                    "model_path": "Path to the model file",
                    "target_precision": "Target precision: fp32, fp16, or int8"
                },
                "returns": "Dictionary with optimization results"
            }
        }
    
    def create_tool_calling_prompt(self, user_request: str) -> str:
        """Create a prompt that enables tool calling"""
        tools_description = json.dumps(self.available_tools, indent=2)
        
        prompt = f"""
You are a Jetson AI assistant with access to specialized tools. Analyze the user's request and determine if you need to call any tools to provide a complete answer.

Available Tools:
{tools_description}

User Request: {user_request}

If you need to call tools, respond with a JSON object containing:
{{
    "needs_tools": true,
    "tool_calls": [
        {{
            "tool_name": "tool_name",
            "parameters": {{"param1": "value1"}},
            "reasoning": "Why this tool is needed"
        }}
    ],
    "response_plan": "How you'll use the tool results to answer the user"
}}

If no tools are needed, respond with:
{{
    "needs_tools": false,
    "direct_response": "Your direct answer to the user"
}}

Analyze the request and respond:
"""
        return prompt
    
    def execute_tool_call(self, tool_call: dict) -> any:
        """Execute a tool call and return the result"""
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        if tool_name == "get_system_info":
            return self.toolkit.get_system_info()
        elif tool_name == "benchmark_model":
            return self.toolkit.benchmark_model(**parameters)
        elif tool_name == "optimize_model":
            return self.toolkit.optimize_model(**parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def process_request_with_tools(self, user_request: str) -> str:
        """Process a user request, calling tools if needed"""
        # Step 1: Determine if tools are needed
        tool_prompt = self.create_tool_calling_prompt(user_request)
        response = self.llm.invoke(tool_prompt)
        
        try:
            # Parse the response
            response_data = json.loads(response)
            
            if not response_data.get("needs_tools", False):
                return response_data.get("direct_response", "No response provided")
            
            # Step 2: Execute tool calls
            tool_results = []
            for tool_call in response_data.get("tool_calls", []):
                result = self.execute_tool_call(tool_call)
                tool_results.append({
                    "tool_call": tool_call,
                    "result": result
                })
            
            # Step 3: Generate final response with tool results
            final_prompt = f"""
User Request: {user_request}

Tool Results:
{json.dumps(tool_results, indent=2, default=str)}

Based on the tool results above, provide a comprehensive answer to the user's request. Include specific data from the tool results and practical recommendations.

Response:
"""
            
            final_response = self.llm.invoke(final_prompt)
            return final_response
            
        except json.JSONDecodeError:
            return f"Error parsing tool response: {response}"
        except Exception as e:
            return f"Error processing request: {e}"


# ===== Function Calling =====

class FunctionRegistry:
    """Registry for functions that can be called by LLMs"""
    
    def __init__(self):
        self.functions = {}
    
    def register(self, func: Callable) -> Callable:
        """Register a function with the registry"""
        self.functions[func.__name__] = func
        return func
    
    def call_function(self, func_name: str, **kwargs) -> Any:
        """Call a registered function"""
        if func_name not in self.functions:
            raise ValueError(f"Function '{func_name}' not registered")
        
        return self.functions[func_name](**kwargs)
    
    def get_function_schema(self) -> str:
        """Get schema for all registered functions"""
        schema = ""
        
        for name, func in self.functions.items():
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No description available"
            
            schema += f"Function: {name}\n"
            schema += f"Description: {doc}\n"
            schema += "Parameters:\n"
            
            for param_name, param in sig.parameters.items():
                param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
                default = "" if param.default == inspect.Parameter.empty else f" (default: {param.default})"
                schema += f"  - {param_name}: {param_type}{default}\n"
            
            schema += "\n"
        
        return schema


class FunctionCallingPrompter:
    """LLM prompter with function calling capabilities"""
    
    def __init__(self, llm, function_registry: FunctionRegistry):
        self.llm = llm
        self.registry = function_registry
    
    def create_function_calling_prompt(self, user_request: str) -> str:
        """Create prompt for function calling"""
        function_schema = self.registry.get_function_schema()
        
        prompt = f"""
You are a Jetson AI assistant with access to specialized functions. Analyze the user's request and determine which functions to call.

Available Functions:
{function_schema}

User Request: {user_request}

If you need to call functions, respond with a JSON object:
{{
    "needs_functions": true,
    "function_calls": [
        {{
            "function_name": "function_name",
            "parameters": {{"param1": "value1"}},
            "reasoning": "Why this function is needed"
        }}
    ],
    "execution_plan": "How you'll use the function results"
}}

If no functions are needed, respond with:
{{
    "needs_functions": false,
    "direct_response": "Your direct answer"
}}

Analyze and respond:
"""
        return prompt
    
    def execute_function_calls(self, function_calls: list) -> list:
        """Execute multiple function calls"""
        results = []
        for call in function_calls:
            func_name = call["function_name"]
            parameters = call["parameters"]
            
            try:
                result = self.registry.call_function(func_name, **parameters)
                results.append({
                    "function_call": call,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "function_call": call,
                    "error": str(e),
                    "status": "error"
                })
        
        return results
    
    def process_with_functions(self, user_request: str) -> str:
        """Process request with function calling"""
        # Step 1: Determine function calls
        function_prompt = self.create_function_calling_prompt(user_request)
        response = self.llm.invoke(function_prompt)
        
        try:
            response_data = json.loads(response)
            
            if not response_data.get("needs_functions", False):
                return response_data.get("direct_response", "No response provided")
            
            # Step 2: Execute functions
            function_results = self.execute_function_calls(
                response_data.get("function_calls", [])
            )
            
            # Step 3: Generate final response
            final_prompt = f"""
User Request: {user_request}

Function Execution Results:
{json.dumps(function_results, indent=2, default=str)}

Based on the function results, provide a comprehensive answer to the user's request. Include specific data, insights, and actionable recommendations.

Response:
"""
            
            return self.llm.invoke(final_prompt)
            
        except json.JSONDecodeError:
            return f"Error parsing function response: {response}"
        except Exception as e:
            return f"Error processing request: {e}"


# ===== Model Context Protocol (MCP) =====

class MCPServer:
    """Simplified MCP server implementation for Jetson tools"""
    
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.prompts = {}
    
    def register_tool(self, name: str, description: str, handler: Callable):
        """Register a tool with MCP server"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "inputSchema": self._extract_schema(handler)
        }
    
    def register_resource(self, uri: str, name: str, description: str, content_provider: Callable):
        """Register a resource (data source)"""
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description,
            "provider": content_provider
        }
    
    def register_prompt(self, name: str, description: str, template: str):
        """Register a prompt template"""
        self.prompts[name] = {
            "name": name,
            "description": description,
            "template": template
        }
    
    def _extract_schema(self, handler: Callable) -> dict:
        """Extract JSON schema from function signature"""
        sig = inspect.signature(handler)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
                properties[param_name] = {"type": param_type}
                
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def list_tools(self) -> dict:
        """List available tools"""
        return {"tools": list(self.tools.values())}
    
    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool with arguments"""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            result = self.tools[name]["handler"](**arguments)
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as e:
            return {"error": str(e)}
    
    def get_prompt(self, name: str, arguments: dict = None) -> dict:
        """Get a prompt template with arguments"""
        if name not in self.prompts:
            return {"error": f"Prompt '{name}' not found"}
        
        template = self.prompts[name]["template"]
        if arguments:
            try:
                formatted = template.format(**arguments)
                return {"messages": [{"role": "user", "content": {"type": "text", "text": formatted}}]}
            except KeyError as e:
                return {"error": f"Missing argument: {e}"}
        
        return {"messages": [{"role": "user", "content": {"type": "text", "text": template}}]}


class MCPPrompter:
    """LLM prompter with MCP integration"""
    
    def __init__(self, llm, mcp_server: MCPServer):
        self.llm = llm
        self.mcp = mcp_server
    
    def process_with_mcp(self, user_request: str) -> str:
        """Process request using MCP tools and prompts"""
        # Get available tools
        tools = self.mcp.list_tools()
        
        # Create MCP-aware prompt
        mcp_prompt = f"""
You have access to MCP tools and prompts. Analyze the user request and determine the best approach.

Available Tools:
{json.dumps(tools, indent=2)}

User Request: {user_request}

If you need to use MCP tools, respond with:
{{
    "action": "use_tools",
    "tools_needed": [
        {{"tool_name": "tool_name", "arguments": {{"arg1": "value1"}}}}
    ]
}}

If you need a specific prompt template, respond with:
{{
    "action": "use_prompt",
    "prompt_name": "prompt_name",
    "arguments": {{"arg1": "value1"}}
}}

If you can answer directly, respond with:
{{
    "action": "direct_response",
    "response": "Your answer"
}}

Analyze and respond:
"""
        
        response = self.llm.invoke(mcp_prompt)
        
        try:
            action_data = json.loads(response)
            action = action_data.get("action")
            
            if action == "use_tools":
                # Execute MCP tools
                tool_results = []
                for tool_call in action_data.get("tools_needed", []):
                    result = self.mcp.call_tool(
                        tool_call["tool_name"],
                        tool_call["arguments"]
                    )
                    tool_results.append(result)
                
                # Generate response with tool results
                final_prompt = f"""
User Request: {user_request}

MCP Tool Results:
{json.dumps(tool_results, indent=2)}

Provide a comprehensive response based on the tool results:
"""
                return self.llm.invoke(final_prompt)
            
            elif action == "use_prompt":
                # Use MCP prompt template
                prompt_result = self.mcp.get_prompt(
                    action_data["prompt_name"],
                    action_data.get("arguments", {})
                )
                
                if "error" in prompt_result:
                    return f"Error using prompt: {prompt_result['error']}"
                
                # Execute the prompt
                prompt_text = prompt_result["messages"][0]["content"]["text"]
                return self.llm.invoke(prompt_text)
            
            elif action == "direct_response":
                return action_data.get("response", "No response provided")
            
        except json.JSONDecodeError:
            return f"Error parsing MCP response: {response}"
        except Exception as e:
            return f"Error processing MCP request: {e}"


# ===== Advanced Jetson AI Assistant =====

class JetsonAIAssistant:
    """Advanced AI assistant for Jetson development"""
    
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.tool_registry = self._setup_tools()
        self.mcp_server = self._setup_mcp()
    
    def _setup_tools(self) -> Dict[str, Callable]:
        """Setup comprehensive tool registry"""
        tools = {
            "system_monitor": self._monitor_system,
            "model_benchmark": self._benchmark_model,
            "optimization_advisor": self._get_optimization_advice,
            "deployment_validator": self._validate_deployment,
            "performance_predictor": self._predict_performance,
            "resource_planner": self._plan_resources
        }
        return tools
    
    def _setup_mcp(self) -> MCPServer:
        """Setup MCP server with Jetson-specific capabilities"""
        mcp = MCPServer()
        
        # Register advanced prompt templates
        mcp.register_prompt(
            "performance_analysis",
            "Analyze model performance and suggest improvements",
            """
Perform a comprehensive performance analysis for:

Model: {model_name}
Current Metrics:
- FPS: {current_fps}
- Latency: {current_latency}ms
- Memory Usage: {memory_usage}MB
- Power Draw: {power_draw}W

Target Requirements:
- Minimum FPS: {target_fps}
- Maximum Latency: {max_latency}ms
- Memory Budget: {memory_budget}MB
- Power Budget: {power_budget}W

Platform: {platform}
Use Case: {use_case}

Provide:
1. Performance gap analysis
2. Bottleneck identification
3. Optimization roadmap with priorities
4. Risk assessment for each optimization
5. Expected timeline and resource requirements

Analysis:
"""
        )
        
        mcp.register_prompt(
            "deployment_strategy",
            "Create deployment strategy for production",
            """
Create a production deployment strategy for:

Application: {app_name}
Models: {models}
Target Devices: {devices}
Scale: {scale}
SLA Requirements: {sla}

Consider:
- Model versioning and updates
- A/B testing capabilities
- Monitoring and alerting
- Rollback procedures
- Performance optimization
- Security considerations

Deployment Strategy:
"""
        )
        
        return mcp
    
    def _monitor_system(self, duration: int = 60) -> Dict[str, Any]:
        """Advanced system monitoring"""
        import random
        import time
        
        # Simulate comprehensive monitoring
        metrics = {
            "monitoring_duration": duration,
            "system_health": {
                "cpu_usage_avg": round(random.uniform(20, 80), 2),
                "cpu_usage_peak": round(random.uniform(80, 95), 2),
                "gpu_utilization_avg": round(random.uniform(30, 90), 2),
                "memory_usage_gb": round(random.uniform(4, 14), 2),
                "temperature_avg": round(random.uniform(40, 70), 2),
                "temperature_peak": round(random.uniform(70, 85), 2),
                "power_consumption_avg": round(random.uniform(10, 25), 2),
                "thermal_throttling_events": random.randint(0, 5)
            },
            "performance_metrics": {
                "inference_fps": round(random.uniform(15, 60), 2),
                "latency_p50": round(random.uniform(10, 50), 2),
                "latency_p95": round(random.uniform(20, 100), 2),
                "memory_efficiency": round(random.uniform(0.6, 0.9), 3),
                "gpu_memory_usage_mb": round(random.uniform(1000, 4000), 2)
            },
            "alerts": [
                "High temperature detected at 14:32",
                "Memory usage approaching limit"
            ] if random.random() > 0.7 else []
        }
        
        return metrics
    
    def _benchmark_model(self, model_path: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model benchmarking"""
        import random
        
        model_name = model_path.split('/')[-1]
        
        # Simulate detailed benchmarking
        results = {
            "model_info": {
                "name": model_name,
                "path": model_path,
                "size_mb": round(random.uniform(50, 500), 2),
                "precision": test_config.get("precision", "fp16")
            },
            "performance": {
                "fps_avg": round(random.uniform(20, 80), 2),
                "fps_min": round(random.uniform(10, 30), 2),
                "fps_max": round(random.uniform(50, 100), 2),
                "latency_avg": round(random.uniform(15, 80), 2),
                "latency_p95": round(random.uniform(30, 120), 2),
                "throughput_imgs_sec": round(random.uniform(15, 75), 2)
            },
            "resource_usage": {
                "gpu_memory_mb": round(random.uniform(500, 3000), 2),
                "cpu_usage_percent": round(random.uniform(20, 60), 2),
                "power_draw_w": round(random.uniform(8, 22), 2),
                "temperature_c": round(random.uniform(45, 75), 2)
            },
            "accuracy_metrics": {
                "map_50": round(random.uniform(0.7, 0.95), 3),
                "map_75": round(random.uniform(0.5, 0.8), 3),
                "precision": round(random.uniform(0.8, 0.95), 3),
                "recall": round(random.uniform(0.75, 0.9), 3)
            },
            "optimization_suggestions": [
                "Consider TensorRT FP16 optimization for 2x speedup",
                "Batch processing could improve throughput by 30%",
                "Dynamic shapes optimization available"
            ]
        }
        
        return results
    
    def _get_optimization_advice(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization advice for a model"""
        # Simulated implementation
        return {"advice": "Optimization advice would be provided here"}
    
    def _validate_deployment(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a deployment configuration"""
        # Simulated implementation
        return {"validation": "Deployment validation would be performed here"}
    
    def _predict_performance(self, model_info: Dict[str, Any], target_device: str) -> Dict[str, Any]:
        """Predict model performance on a target device"""
        # Simulated implementation
        return {"prediction": "Performance prediction would be provided here"}
    
    def _plan_resources(self, application_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Plan resources for an application"""
        # Simulated implementation
        return {"plan": "Resource planning would be performed here"}
    
    def process_complex_request(self, user_request: str) -> str:
        """Process complex requests using multiple tools and MCP"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_request})
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
You are an advanced Jetson AI assistant with access to comprehensive tools and MCP capabilities.

Conversation History:
{json.dumps(self.conversation_history[-3:], indent=2)}

Current Request: {user_request}

Available Tools:
{json.dumps(list(self.tool_registry.keys()), indent=2)}

Analyze the request and create an execution plan. Respond with:
{{
    "analysis": "Your understanding of the request",
    "execution_plan": [
        {{
            "step": 1,
            "action": "tool_call|mcp_prompt|direct_response",
            "details": "Specific action details",
            "reasoning": "Why this step is needed"
        }}
    ],
    "expected_outcome": "What the user should expect"
}}

Analyze and plan:
"""
        
        try:
            plan_response = self.llm.invoke(analysis_prompt)
            plan = json.loads(plan_response)
            
            # Execute the plan
            execution_results = []
            for step in plan.get("execution_plan", []):
                if step["action"] == "tool_call":
                    # Execute tool call
                    tool_name = step["details"].get("tool_name")
                    if tool_name in self.tool_registry:
                        result = self.tool_registry[tool_name](**step["details"].get("parameters", {}))
                        execution_results.append({"step": step["step"], "result": result})
                
                elif step["action"] == "mcp_prompt":
                    # Use MCP prompt
                    prompt_name = step["details"].get("prompt_name")
                    prompt_args = step["details"].get("arguments", {})
                    prompt_result = self.mcp_server.get_prompt(prompt_name, prompt_args)
                    execution_results.append({"step": step["step"], "result": prompt_result})
                
                elif step["action"] == "direct_response":
                    # Direct response
                    execution_results.append({"step": step["step"], "result": step["details"].get("response")})
            
            # Generate final response
            final_prompt = f"""
User Request: {user_request}

Execution Results:
{json.dumps(execution_results, indent=2, default=str)}

Based on the execution results, provide a comprehensive, actionable response to the user's request.
Include specific data points, insights, and recommendations.

Response:
"""
            
            final_response = self.llm.invoke(final_prompt)
            self.conversation_history.append({"role": "assistant", "content": final_response})
            
            return final_response
            
        except json.JSONDecodeError as e:
            error_response = f"Error parsing execution plan: {e}. Please try a more specific request."
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
        except Exception as e:
            error_response = f"Error processing request: {e}. Please try again."
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response


# ===== Production MCP Application =====

class ProductionMCPApp:
    """Production-ready MCP application for Jetson AI development"""
    
    def __init__(self, llm):
        self.llm = llm
        self.mcp_server = self._setup_mcp_server()
    
    def _setup_mcp_server(self) -> MCPServer:
        """Setup production MCP server with tools and prompts"""
        mcp = MCPServer()
        
        # Register production tools
        mcp.register_tool(
            "jetson_diagnostics",
            "Run comprehensive diagnostics on Jetson device",
            self._run_diagnostics
        )
        
        mcp.register_tool(
            "model_optimizer",
            "Optimize model for production deployment",
            self._optimize_model
        )
        
        mcp.register_tool(
            "deployment_manager",
            "Manage model deployment to production",
            self._deploy_model
        )
        
        mcp.register_tool(
            "performance_analyzer",
            "Analyze production performance metrics",
            self._analyze_performance
        )
        
        # Register production prompt templates
        mcp.register_prompt(
            "production_readiness_check",
            "Assess production readiness of a model",
            self._get_production_readiness_template()
        )
        
        mcp.register_prompt(
            "incident_response",
            "Generate incident response plan",
            self._get_incident_response_template()
        )
        
        return mcp
    
    def _get_production_readiness_template(self) -> str:
        """Template for production readiness assessment"""
        return """
PRODUCTION READINESS ASSESSMENT
==============================

Model: {model_name}
Version: {version}
Target Deployment: {deployment_target}

Please assess the production readiness of this model by evaluating the following criteria:

1. Performance Metrics
   - Accuracy: {accuracy}
   - Latency: {latency}ms
   - Throughput: {throughput} inferences/second
   - Memory Usage: {memory_usage}MB

2. Robustness
   - Tested on {test_dataset_size} samples
   - Edge cases covered: {edge_cases_covered}
   - Adversarial testing: {adversarial_testing}

3. Deployment Infrastructure
   - Target hardware: {target_hardware}
   - Containerization: {containerization}
   - Monitoring: {monitoring}
   - Logging: {logging}

4. Operational Considerations
   - Update strategy: {update_strategy}
   - Rollback procedure: {rollback_procedure}
   - Performance degradation detection: {degradation_detection}

Based on the above information, provide:

1. Production Readiness Score (0-100)
2. Critical gaps that must be addressed before deployment
3. Recommendations for improving production stability
4. Risk assessment for immediate deployment
5. Suggested timeline for addressing identified issues

Assessment:
"""
    
    def _get_incident_response_template(self) -> str:
        """Template for incident response planning"""
        return """
INCIDENT RESPONSE PLAN
=====================

System: {system_name}
Critical Component: {component_name}
Incident Type: {incident_type}
Severity: {severity}

Please generate a comprehensive incident response plan for the following scenario:

{incident_description}

The plan should include:

1. Immediate Response Actions
   - Detection mechanisms
   - Triage steps
   - Containment procedures

2. Investigation Process
   - Data collection requirements
   - Analysis methodology
   - Root cause identification approach

3. Mitigation Strategy
   - Short-term fixes
   - Long-term solutions
   - Verification procedures

4. Communication Plan
   - Internal stakeholders notification
   - External communications (if applicable)
   - Escalation paths

5. Recovery Process
   - Service restoration steps
   - Data integrity verification
   - Performance validation

6. Post-Incident Activities
   - Documentation requirements
   - Lessons learned process
   - Preventive measures

Incident Response Plan:
"""
    
    def _run_diagnostics(self, device_id: str, diagnostic_level: str = "comprehensive") -> dict:
        """Run diagnostics on Jetson device"""
        # Simulated implementation
        return {"status": "Diagnostics completed", "results": "Detailed results would be here"}
    
    def _optimize_model(self, model_path: str, target_device: str, precision: str = "fp16") -> dict:
        """Optimize model for production"""
        # Simulated implementation
        return {"status": "Optimization completed", "optimized_model": f"{model_path}_optimized"}
    
    def _deploy_model(self, model_path: str, deployment_config: dict) -> dict:
        """Deploy model to production"""
        # Simulated implementation
        return {"status": "Deployment completed", "endpoint": "https://jetson-inference-api/v1/predict"}
    
    def _analyze_performance(self, model_id: str, time_period: str = "24h") -> dict:
        """Analyze production performance"""
        # Simulated implementation
        return {"status": "Analysis completed", "metrics": "Performance metrics would be here"}
    
    def run_production_workflow(self, workflow_type: str, workflow_params: dict) -> str:
        """Run a production workflow"""
        if workflow_type == "model_deployment":
            # Model deployment workflow
            model_path = workflow_params.get("model_path")
            target_device = workflow_params.get("target_device")
            
            # 1. Run diagnostics
            diagnostics = self._run_diagnostics(target_device)
            
            # 2. Optimize model
            optimization = self._optimize_model(model_path, target_device)
            
            # 3. Deploy model
            deployment = self._deploy_model(
                optimization["optimized_model"],
                {"device": target_device, "replicas": 1}
            )
            
            # 4. Generate report
            report_prompt = self.mcp_server.get_prompt(
                "production_readiness_check",
                {
                    "model_name": model_path.split("/")[-1],
                    "version": "1.0",
                    "deployment_target": target_device,
                    "accuracy": "95%",
                    "latency": "25",
                    "throughput": "40",
                    "memory_usage": "1024",
                    "test_dataset_size": "10,000",
                    "edge_cases_covered": "Yes",
                    "adversarial_testing": "Limited",
                    "target_hardware": target_device,
                    "containerization": "Docker",
                    "monitoring": "Prometheus",
                    "logging": "ELK Stack",
                    "update_strategy": "Blue-Green",
                    "rollback_procedure": "Automated",
                    "degradation_detection": "Implemented"
                }
            )
            
            prompt_text = report_prompt["messages"][0]["content"]["text"]
            report = self.llm.invoke(prompt_text)
            
            return f"Deployment workflow completed. Endpoint: {deployment['endpoint']}\n\nReadiness Report:\n{report}"
            
        elif workflow_type == "performance_optimization":
            # Performance optimization workflow
            # Simulated implementation
            return "Performance optimization workflow would be executed here"
            
        elif workflow_type == "production_validation":
            # Production validation workflow
            # Simulated implementation
            return "Production validation workflow would be executed here"
            
        else:
            return f"Unknown workflow type: {workflow_type}"


# ===== Main Function =====

def main():
    """Main function for the Jetson Prompt Engineering Toolkit"""
    parser = argparse.ArgumentParser(description="Jetson Prompt Engineering Toolkit")
    
    # Main mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["basic", "compare", "structured", "tools", "functions", "mcp", "assistant", "production", "lab"],
        required=True,
        help="Operation mode"
    )
    
    # Basic prompt engineering options
    parser.add_argument(
        "--technique", 
        type=str, 
        choices=["cot", "few_shot", "role", "icl", "all"],
        help="Prompt engineering technique to demonstrate"
    )
    
    # Backend selection
    parser.add_argument(
        "--backends", 
        type=str, 
        default="openai",
        help="Comma-separated list of backends to use (openai,ollama,llamacpp)"
    )
    
    # Model paths/names
    parser.add_argument(
        "--openai-model", 
        type=str, 
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--ollama-model", 
        type=str, 
        default="llama3.1:8b",
        help="Ollama model to use"
    )
    parser.add_argument(
        "--llamacpp-model", 
        type=str, 
        help="Path to llama-cpp-python model"
    )
    
    # Structured output options
    parser.add_argument(
        "--output", 
        type=str, 
        choices=["optimization_plan", "device_comparison"],
        help="Type of structured output to generate"
    )
    
    # Tool/function calling options
    parser.add_argument(
        "--request", 
        type=str, 
        help="User request for tool/function calling demos"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize backends
    backends = {}
    backend_list = args.backends.split(",")
    
    if "openai" in backend_list:
        try:
            backends["openai"] = OpenAIPrompter(model=args.openai_model)
            print(f"✅ Initialized OpenAI backend with model: {args.openai_model}")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI backend: {e}")
    
    if "ollama" in backend_list:
        try:
            backends["ollama"] = OllamaPrompter(model=args.ollama_model)
            if backends["ollama"].is_available():
                print(f"✅ Initialized Ollama backend with model: {args.ollama_model}")
            else:
                print("❌ Ollama server not available. Is it running?")
                backends.pop("ollama")
        except Exception as e:
            print(f"❌ Failed to initialize Ollama backend: {e}")
    
    if "llamacpp" in backend_list and args.llamacpp_model:
        try:
            backends["llamacpp"] = LlamaCppPrompter(model_path=args.llamacpp_model)
            print(f"✅ Initialized llama-cpp-python backend with model: {args.llamacpp_model}")
        except Exception as e:
            print(f"❌ Failed to initialize llama-cpp-python backend: {e}")
    
    if not backends:
        print("❌ No backends initialized. Exiting.")
        return
    
    # Select primary backend for operations that need only one
    primary_backend = next(iter(backends.values()))
    
    # Execute requested mode
    if args.mode == "basic":
        run_basic_prompt_demo(backends, args.technique)
    elif args.mode == "compare":
        run_backend_comparison(backends)
    elif args.mode == "structured":
        run_structured_output_demo(primary_backend, args.output)
    elif args.mode == "tools":
        run_tool_calling_demo(primary_backend, args.request)
    elif args.mode == "functions":
        run_function_calling_demo(primary_backend, args.request)
    elif args.mode == "mcp":
        run_mcp_demo(primary_backend, args.request)
    elif args.mode == "assistant":
        run_assistant_demo(primary_backend, args.request)
    elif args.mode == "production":
        run_production_demo(primary_backend, args.request)
    elif args.mode == "lab":
        run_lab_demo(primary_backend, args.request)


def run_basic_prompt_demo(backends, technique):
    """Run basic prompt engineering technique demonstrations"""
    if technique == "cot" or technique == "all":
        print("\n===== Chain-of-Thought Demonstration =====\n")
        task = "Optimize YOLOv8 for real-time object detection on Jetson Orin Nano with 4GB RAM"
        
        basic_prompt = f"{task}"
        cot_prompt = f"{task}\n\nLet's think step by step:"
        advanced_cot_prompt = f"{task}\n\nLet's analyze this systematically:\n1. Understand the hardware constraints\n2. Identify YOLOv8 bottlenecks\n3. Apply optimization techniques\n4. Measure and validate results"
        
        for name, backend in backends.items():
            print(f"\n{name.upper()} Backend Results:")
            print("\nBasic Prompt:")
            print(backend.simple_prompt(basic_prompt)[:300] + "...")
            
            print("\nChain-of-Thought Prompt:")
            print(backend.simple_prompt(cot_prompt)[:300] + "...")
            
            print("\nAdvanced Chain-of-Thought Prompt:")
            print(backend.simple_prompt(advanced_cot_prompt)[:300] + "...")
    
    if technique == "few_shot" or technique == "all":
        print("\n===== Few-Shot Learning Demonstration =====\n")
        # Implementation here
    
    if technique == "role" or technique == "all":
        print("\n===== Role-Based Prompting Demonstration =====\n")
        # Implementation here
    
    if technique == "icl" or technique == "all":
        print("\n===== In-Context Learning Demonstration =====\n")
        # Implementation here


def run_backend_comparison(backends):
    """Compare different LLM backends"""
    print("\n===== Backend Comparison =====\n")
    
    prompt = """Explain how to optimize deep learning models for NVIDIA Jetson platforms. 
    Include specific techniques for:
    1. Model quantization
    2. TensorRT conversion
    3. Memory optimization
    4. Throughput maximization
    
    Provide concrete examples where possible.
    """
    
    for name, backend in backends.items():
        print(f"\n{name.upper()} Response:")
        try:
            start_time = time.time()
            response = backend.simple_prompt(prompt)
            elapsed = time.time() - start_time
            
            print(f"[Response time: {elapsed:.2f}s]")
            print(response)
        except Exception as e:
            print(f"Error: {e}")


def run_structured_output_demo(backend, output_type):
    """Demonstrate structured output generation"""
    if not LANGCHAIN_AVAILABLE or not PYDANTIC_AVAILABLE:
        print("❌ LangChain and/or Pydantic not available. Install with: pip install langchain pydantic")
        return
    
    print("\n===== Structured Output Demonstration =====\n")
    
    # Create LangChain wrapper if needed
    if isinstance(backend, OpenAIPrompter):
        if not LANGCHAIN_OPENAI_AVAILABLE:
            print("❌ LangChain OpenAI not available. Install with: pip install langchain-openai")
            return
        langchain_backend = LangChainOpenAIPrompter(model=backend.model)
    elif isinstance(backend, OllamaPrompter):
        if not LANGCHAIN_OLLAMA_AVAILABLE:
            print("❌ LangChain Ollama not available. Install with: pip install langchain-community")
            return
        langchain_backend = LangChainOllamaPrompter(model=backend.model)
    elif isinstance(backend, LlamaCppPrompter):
        if not LANGCHAIN_LLAMACPP_AVAILABLE:
            print("❌ LangChain LlamaCpp not available. Install with: pip install langchain-community")
            return
        langchain_backend = LangChainLlamaCppPrompter(model_path=backend.model_path)
    else:
        print("❌ Unsupported backend for structured output")
        return
    
    structured_prompter = StructuredPrompter(langchain_backend)
    
    if output_type == "optimization_plan":
        print("Generating YOLOv8 optimization plan...\n")
        plan = structured_prompter.get_optimization_plan(
            "YOLOv8-nano model for person detection on Jetson Orin Nano with 4GB RAM, targeting 30 FPS"
        )
        print(json.dumps(plan.dict() if hasattr(plan, "dict") else plan, indent=2))
    
    elif output_type == "device_comparison":
        print("Generating Jetson device comparison...\n")
        devices = ["Jetson Nano", "Jetson Xavier NX", "Jetson AGX Orin"]
        comparisons = structured_prompter.compare_jetson_devices(devices)
        
        for comparison in comparisons:
            print(f"\n{comparison.device_name}:")
            print(json.dumps(comparison.dict() if hasattr(comparison, "dict") else comparison, indent=2))


def run_tool_calling_demo(backend, request):
    """Demonstrate tool calling capabilities"""
    print("\n===== Tool Calling Demonstration =====\n")
    
    if not request:
        request = "What's the current system status of my Jetson device?"
    
    print(f"User Request: {request}\n")
    
    tool_prompter = ToolCallingPrompter(backend)
    response = tool_prompter.process_request_with_tools(request)
    
    print("Response:")
    print(response)


def run_function_calling_demo(backend, request):
    """Demonstrate function calling capabilities"""
    print("\n===== Function Calling Demonstration =====\n")
    
    if not request:
        request = "I need to optimize my YOLOv8 model for Jetson Nano"
    
    print(f"User Request: {request}\n")
    
    # Setup function registry
    registry = FunctionRegistry()
    
    @registry.register
    def get_jetson_specs(device_type: str) -> dict:
        """Get specifications for a Jetson device"""
        specs = {
            "jetson_nano": {
                "gpu": "128-core Maxwell",
                "cpu": "Quad-core ARM A57",
                "memory": "4GB",
                "storage": "16GB eMMC",
                "power": "5-10W"
            },
            "jetson_xavier_nx": {
                "gpu": "384-core Volta",
                "cpu": "6-core Carmel ARM",
                "memory": "8GB",
                "storage": "16GB eMMC",
                "power": "10-15W"
            },
            "jetson_orin_nano": {
                "gpu": "1024-core Ampere",
                "cpu": "6-core Arm Cortex-A78AE",
                "memory": "8GB",
                "storage": "64GB eMMC",
                "power": "7-15W"
            },
            "jetson_agx_orin": {
                "gpu": "2048-core Ampere",
                "cpu": "12-core Arm Cortex-A78AE",
                "memory": "32GB",
                "storage": "64GB eMMC",
                "power": "15-60W"
            }
        }
        
        device_type = device_type.lower().replace(" ", "_")
        return specs.get(device_type, {"error": "Unknown device type"})
    
    @registry.register
    def recommend_optimization_techniques(model_type: str, target_fps: int, device_type: str) -> list:
        """Recommend optimization techniques for a model on Jetson"""
        techniques = [
            "TensorRT conversion",
            "FP16 precision",
            "Layer fusion",
            "Batch size optimization"
        ]
        
        if "yolo" in model_type.lower():
            techniques.extend(["Pruning", "Input resolution reduction"])
        
        if target_fps > 30:
            techniques.extend(["INT8 quantization", "Model distillation"])
        
        if "nano" in device_type.lower():
            techniques.append("Memory optimization")
        
        return techniques
    
    function_prompter = FunctionCallingPrompter(backend, registry)
    response = function_prompter.process_with_functions(request)
    
    print("Response:")
    print(response)


def run_mcp_demo(backend, request):
    """Demonstrate MCP capabilities"""
    print("\n===== MCP Demonstration =====\n")
    
    if not request:
        request = "Create a deployment checklist for my YOLOv8 model on Jetson Xavier NX"
    
    print(f"User Request: {request}\n")
    
    # Setup MCP server
    mcp_server = MCPServer()
    
    # Register tools
    mcp_server.register_tool(
        "get_device_info",
        "Get information about a Jetson device",
        lambda device_type: {
            "name": device_type,
            "specs": "Device specifications would be here"
        }
    )
    
    mcp_server.register_tool(
        "check_model_compatibility",
        "Check if a model is compatible with a device",
        lambda model_name, device_type: {
            "compatible": True,
            "notes": "Compatibility notes would be here"
        }
    )
    
    # Register prompts
    mcp_server.register_prompt(
        "deployment_checklist",
        "Generate a deployment checklist",
        """
DEPLOYMENT CHECKLIST
===================

Model: {model_name}
Device: {device_type}
Use Case: {use_case}

Please create a comprehensive deployment checklist covering:

1. Pre-deployment requirements
2. Optimization steps
3. Testing procedures
4. Deployment process
5. Post-deployment monitoring

Checklist:
"""
    )
    
    mcp_prompter = MCPPrompter(backend, mcp_server)
    response = mcp_prompter.process_with_mcp(request)
    
    print("Response:")
    print(response)


def run_assistant_demo(backend, request):
    """Demonstrate advanced assistant capabilities"""
    print("\n===== Advanced Assistant Demonstration =====\n")
    
    if not request:
        request = "I need to deploy multiple AI models on my Jetson AGX Orin for a smart retail application"
    
    print(f"User Request: {request}\n")
    
    assistant = JetsonAIAssistant(backend)
    response = assistant.process_complex_request(request)
    
    print("Response:")
    print(response)


def run_production_demo(backend, request):
    """Demonstrate production MCP application"""
    print("\n===== Production MCP Application Demonstration =====\n")
    
    app = ProductionMCPApp(backend)
    
    # Run model deployment workflow
    result = app.run_production_workflow(
        "model_deployment",
        {
            "model_path": "/opt/models/yolov8n.pt",
            "target_device": "Jetson AGX Orin"
        }
    )
    
    print("Workflow Result:")
    print(result)


def run_lab_demo(backend, request):
    """Run the advanced prompt engineering lab exercises"""
    print("\n===== Advanced Prompt Engineering Lab =====\n")
    
    # Exercise 1: Prompt Engineering Comparison
    print("\n--- Exercise 1: Prompt Engineering Comparison ---\n")
    
    test_scenarios = [
        "Optimize YOLOv8 for Jetson Nano",
        "Compare Jetson Orin Nano vs Xavier NX for computer vision",
        "Explain how to deploy a model with TensorRT on Jetson"
    ]
    
    lab = PromptEngineeringLab(backend)
    results = lab.run_prompt_comparison(test_scenarios)
    report = lab.generate_comparison_report(results)
    
    print("Prompt Engineering Comparison Report:")
    print(report)
    
    # Exercise 2: Advanced Tool Integration
    print("\n--- Exercise 2: Advanced Tool Integration ---\n")
    
    complex_scenarios = [
        "I need to optimize my YOLOv8 model for real-time performance on Jetson Orin",
        "What's the best deployment strategy for multiple models on Jetson AGX Orin?",
        "Can you help me analyze why my model is running slowly on Jetson Xavier NX?"
    ]
    
    assistant = JetsonAIAssistant(backend)
    
    for scenario in complex_scenarios:
        print(f"\nScenario: {scenario}")
        response = assistant.process_complex_request(scenario)
        print(f"Response: {response[:300]}...")
    
    # Exercise 3: Production-Ready MCP Application
    print("\n--- Exercise 3: Production-Ready MCP Application ---\n")
    
    app = ProductionMCPApp(backend)
    
    workflows = [
        ("model_deployment", {
            "model_path": "/opt/models/yolov8n.pt",
            "target_device": "Jetson AGX Orin"
        }),
        ("performance_optimization", {
            "model_name": "YOLOv8",
            "target_fps": 30,
            "device": "Jetson Orin Nano"
        }),
        ("production_validation", {
            "model_path": "/opt/models/yolov8n.onnx",
            "target_device": "Jetson Xavier NX",
            "requirements": {
                "min_fps": 25,
                "max_memory": "4GB",
                "accuracy_threshold": 0.85
            }
        })
    ]
    
    for workflow_type, parameters in workflows:
        print(f"\nRunning workflow: {workflow_type}")
        result = app.run_production_workflow(workflow_type, parameters)
        print(f"Result status: {result['status']}")
        print(f"Duration: {result.get('duration', 'N/A')} seconds")


if __name__ == "__main__":
    main()