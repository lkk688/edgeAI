# 🧠 Advanced Prompt Engineering with LangChain on Jetson

> **Note:** All code examples in this tutorial have been consolidated into a unified Python script called `jetson_prompt_toolkit.py`. See the [Unified Python Script](#-unified-python-script) section at the end of this document for installation and usage instructions.

## 🎯 What is Prompt Engineering?

Prompt engineering is the art and science of crafting effective inputs to guide large language models (LLMs) toward desired outputs. It's the bridge between human intent and AI understanding.

**Why it matters on Jetson:**
- 🚀 **Efficiency**: Better prompts = fewer tokens = faster inference on edge devices
- 🎯 **Accuracy**: Precise instructions lead to more reliable outputs
- 💰 **Cost-effective**: Reduces API calls and computational overhead
- 🔒 **Safety**: Proper prompting prevents harmful or biased responses

---

## 🏗️ Prompt Engineering Fundamentals

### 📋 Core Principles

1. **Clarity**: Be specific and unambiguous
2. **Context**: Provide relevant background information
3. **Structure**: Use consistent formatting and organization
4. **Examples**: Show the model what you want (few-shot learning)
5. **Constraints**: Set clear boundaries and expectations

### 🎨 Prompt Anatomy

```
[SYSTEM MESSAGE] - Sets the AI's role and behavior
[CONTEXT] - Background information
[INSTRUCTION] - What you want the AI to do
[FORMAT] - How you want the output structured
[EXAMPLES] - Sample inputs and outputs
[CONSTRAINTS] - Limitations and requirements
```

---

## 💡 Platform Setup: Three Inference Pathways

### 🔹 1. OpenAI API Setup

```python
import openai
from openai import OpenAI
import os
from typing import List, Dict

class OpenAIPrompter:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model  # Cost-effective for learning
    
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

# Usage
openai_prompter = OpenAIPrompter()
response = openai_prompter.simple_prompt(
    "Explain edge computing in simple terms",
    "You are a helpful AI assistant specializing in technology."
)
print(response)
```

> **Note:** This class is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --backends openai
> ```
>
> The LangChain integration is also available:
> ```bash
> python jetson_prompt_toolkit.py --mode compare --backends openai
> ```

### 🔹 2. Local Ollama Setup

```python
import requests
import json
from typing import List, Dict, Optional

class OllamaPrompter:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
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

# Setup and usage
ollama_prompter = OllamaPrompter()

if ollama_prompter.is_available():
    response = ollama_prompter.prompt(
        "Explain the benefits of running AI models locally on Jetson devices",
        "You are an expert in edge AI and NVIDIA Jetson platforms."
    )
    print(response)
else:
    print("Ollama not available. Start with: ollama serve")
```

> **Note:** This class is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --backends ollama
> ```

### 🔹 3. Local llama-cpp-python Setup

```python
from llama_cpp import Llama
import time
from typing import Optional

class LlamaCppPrompter:
    def __init__(self, model_path: str, **kwargs):
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

# Usage (requires downloaded model)
# llama_prompter = LlamaCppPrompter("/models/qwen2.5-3b-instruct-q4_k_m.gguf")
# response = llama_prompter.chat_prompt(
#     "You are an AI assistant specialized in edge computing.",
#     "What are the advantages of running LLMs on Jetson devices?"
# )
# print(response)
```

> **Note:** This class is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --backends llamacpp --model_path /path/to/your/model.gguf
> ```

---

## 🎨 Core Prompt Engineering Techniques

### 🧠 1. Chain-of-Thought (CoT) Reasoning

Guide the model to think step-by-step for complex problems.

```python
def demonstrate_chain_of_thought():
    """Compare basic vs chain-of-thought prompting"""
    
    # Basic prompt
    basic_prompt = "What is 15% of 240?"
    
    # Chain-of-thought prompt
    cot_prompt = """
Solve this step by step:
What is 15% of 240?

Think through this carefully:
1. First, convert the percentage to a decimal
2. Then multiply by the number
3. Show your work
"""
    
    # Advanced CoT with reasoning
    advanced_cot = """
You are a math tutor. Solve this problem step-by-step, explaining your reasoning:

Problem: What is 15% of 240?

Please:
1. Explain what the problem is asking
2. Show the mathematical steps
3. Verify your answer makes sense
4. Provide the final answer
"""
    
    return basic_prompt, cot_prompt, advanced_cot

# Test with different models
def test_cot_across_models():
    basic, cot, advanced = demonstrate_chain_of_thought()
    
    models = {
        "OpenAI": openai_prompter,
        "Ollama": ollama_prompter if ollama_prompter.is_available() else None,
        # "LlamaCpp": llama_prompter  # Uncomment if available
    }
    
    for model_name, prompter in models.items():
        if prompter is None:
            continue
            
        print(f"\n🔍 {model_name} - Chain of Thought Comparison")
        print("=" * 50)
        
        # Basic prompt
        if hasattr(prompter, 'simple_prompt'):
            basic_response = prompter.simple_prompt(basic)
        else:
            basic_response = prompter.prompt(basic)
        print(f"Basic: {basic_response[:100]}...")
        
        # CoT prompt
        if hasattr(prompter, 'simple_prompt'):
            cot_response = prompter.simple_prompt(advanced)
        else:
            cot_response = prompter.prompt(advanced)
        print(f"CoT: {cot_response[:100]}...")

# Run the test
# test_cot_across_models()
```

> **Note:** These Chain-of-Thought examples are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --technique cot
> ```

### 🎯 2. Few-Shot Learning

Provide examples to guide the model's behavior.

```python
def demonstrate_few_shot_learning():
    """Show the power of examples in prompting"""
    
    # Zero-shot prompt
    zero_shot = "Classify the sentiment of this text: 'The Jetson Orin is amazing for AI development!'"
    
    # Few-shot prompt with examples
    few_shot = """
Classify the sentiment of the following texts as Positive, Negative, or Neutral:

Examples:
Text: "I love using CUDA for parallel computing!"
Sentiment: Positive

Text: "The installation process was frustrating and took hours."
Sentiment: Negative

Text: "The device has 8GB of RAM."
Sentiment: Neutral

Text: "This AI model runs incredibly fast on the Jetson!"
Sentiment: Positive

Now classify this:
Text: "The Jetson Orin is amazing for AI development!"
Sentiment:
"""
    
    # Structured few-shot with format specification
    structured_few_shot = """
You are a sentiment analysis expert. Classify text sentiment and provide confidence.

Format your response as: Sentiment: [POSITIVE/NEGATIVE/NEUTRAL] (Confidence: X%)

Examples:
Text: "CUDA programming is so powerful!"
Response: Sentiment: POSITIVE (Confidence: 95%)

Text: "The setup documentation is unclear."
Response: Sentiment: NEGATIVE (Confidence: 85%)

Text: "The device weighs 500 grams."
Response: Sentiment: NEUTRAL (Confidence: 90%)

Now analyze:
Text: "The Jetson Orin is amazing for AI development!"
Response:
"""
    
    return zero_shot, few_shot, structured_few_shot

# Test few-shot learning
def test_few_shot_learning():
    zero, few, structured = demonstrate_few_shot_learning()
    
    print("🎯 Few-Shot Learning Demonstration")
    print("=" * 40)
    
    # Test with OpenAI
    print("\n📝 Zero-shot:")
    print(openai_prompter.simple_prompt(zero))
    
    print("\n📚 Few-shot:")
    print(openai_prompter.simple_prompt(few))
    
    print("\n🏗️ Structured few-shot:")
    print(openai_prompter.simple_prompt(structured))

# test_few_shot_learning()
```

> **Note:** These Few-Shot Learning examples are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --technique few_shot
> ```

### 🔄 3. Think Step by Step vs Think Hard

Compare different reasoning triggers.

```python
def compare_reasoning_triggers():
    """Compare different ways to trigger reasoning"""
    
    problem = "A Jetson Orin Nano has 8GB RAM. If an AI model uses 60% of available RAM, and the system reserves 1GB for OS, how much RAM is the model actually using?"
    
    prompts = {
        "Direct": problem,
        
        "Think Step by Step": f"{problem}\n\nThink step by step.",
        
        "Think Hard": f"{problem}\n\nThink hard about this problem.",
        
        "Let's Work Through This": f"{problem}\n\nLet's work through this systematically:",
        
        "Detailed Analysis": f"""
Problem: {problem}

Please provide a detailed analysis:
1. Identify what information we have
2. Determine what we need to calculate
3. Show your mathematical work
4. Verify your answer makes sense
""",
        
        "Expert Mode": f"""
You are a computer systems expert. Analyze this memory allocation problem:

{problem}

Provide:
- Clear breakdown of available vs. used memory
- Step-by-step calculation
- Practical implications for AI development
"""
    }
    
    return prompts

def test_reasoning_triggers():
    """Test different reasoning approaches"""
    prompts = compare_reasoning_triggers()
    
    print("🧠 Reasoning Trigger Comparison")
    print("=" * 50)
    
    for trigger_name, prompt in prompts.items():
        print(f"\n🔍 {trigger_name}:")
        print("-" * 30)
        
        try:
            response = openai_prompter.simple_prompt(prompt)
            # Show first 200 characters
            print(f"{response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()  # Add spacing

# test_reasoning_triggers()
```

> **Note:** These Reasoning Trigger examples are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --technique reasoning
> ```

### 🎭 4. Role-Based Prompting

Assign specific roles to get specialized responses.

```python
def demonstrate_role_based_prompting():
    """Show how different roles affect responses"""
    
    question = "How should I optimize my deep learning model for deployment on Jetson Orin?"
    
    roles = {
        "Generic AI": {
            "system": "You are a helpful AI assistant.",
            "prompt": question
        },
        
        "ML Engineer": {
            "system": "You are a senior machine learning engineer with 10 years of experience in model optimization and edge deployment.",
            "prompt": question
        },
        
        "NVIDIA Expert": {
            "system": "You are an NVIDIA developer advocate specializing in Jetson platforms, TensorRT optimization, and edge AI deployment.",
            "prompt": question
        },
        
        "Performance Specialist": {
            "system": "You are a performance optimization specialist focused on real-time AI inference on resource-constrained devices.",
            "prompt": f"""
{question}

Please provide:
1. Specific optimization techniques
2. Performance benchmarking approaches
3. Trade-offs between accuracy and speed
4. Practical implementation steps
"""
        },
        
        "Beginner-Friendly Tutor": {
            "system": "You are a patient AI tutor who explains complex concepts in simple terms with practical examples.",
            "prompt": f"""
{question}

Please explain this in beginner-friendly terms with:
- Simple explanations of technical concepts
- Step-by-step guidance
- Common pitfalls to avoid
- Practical examples
"""
        }
    }
    
    return roles

def test_role_based_prompting():
    """Test different role assignments"""
    roles = demonstrate_role_based_prompting()
    
    print("🎭 Role-Based Prompting Comparison")
    print("=" * 50)
    
    for role_name, role_config in roles.items():
        print(f"\n👤 {role_name}:")
        print("-" * 30)
        
        try:
            response = openai_prompter.simple_prompt(
                role_config["prompt"],
                role_config["system"]
            )
            # Show first 300 characters
            print(f"{response[:300]}{'...' if len(response) > 300 else ''}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()  # Add spacing

# test_role_based_prompting()
```

> **Note:** These Role-Based Prompting examples are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --technique roles
> ```

### 🔄 5. In-Context Learning

Teach the model new tasks within the conversation.

```python
def demonstrate_in_context_learning():
    """Show how to teach models new formats/tasks"""
    
    # Teaching a custom format for Jetson specs
    in_context_prompt = """
I'll teach you a format for describing Jetson device specifications:

Format: [Device] | [GPU] | [CPU] | [RAM] | [Storage] | [Power] | [Use Case]

Examples:
Jetson Nano | 128-core Maxwell GPU | Quad-core ARM A57 | 4GB LPDDR4 | microSD | 5W | IoT/Education
Jetson Xavier NX | 384-core Volta GPU | 6-core Carmel ARM | 8GB LPDDR4x | microSD | 10W | Edge AI
Jetson AGX Orin | 2048-core Ampere GPU | 12-core Cortex-A78AE | 64GB LPDDR5 | NVMe SSD | 60W | Autonomous Vehicles

Now format this device:
Jetson Orin Nano: 1024-core Ampere GPU, 6-core Cortex-A78AE CPU, 8GB LPDDR5 RAM, microSD storage, 15W power consumption, suitable for robotics and edge AI applications.

Formatted specification:
"""
    
    # Teaching code documentation style
    code_doc_prompt = """
I'll teach you how to document Jetson AI code with our team's style:

Style: Brief description + Parameters + Example + Performance note

Example 1:
```python
def load_tensorrt_model(engine_path: str) -> trt.ICudaEngine:
    """Load TensorRT engine for optimized inference.
    
    Args:
        engine_path: Path to .engine file
    
    Returns:
        TensorRT engine ready for inference
    
    Example:
        engine = load_tensorrt_model("yolo.engine")
    
    Performance: ~3x faster than ONNX on Jetson Orin
    """
```

Example 2:
```python
def preprocess_image(image: np.ndarray, target_size: tuple) -> torch.Tensor:
    """Preprocess image for model inference.
    
    Args:
        image: Input image as numpy array
        target_size: (height, width) for resizing
    
    Returns:
        Preprocessed tensor ready for model
    
    Example:
        tensor = preprocess_image(img, (640, 640))
    
    Performance: GPU preprocessing saves 15ms per frame
    """
```

Now document this function using our style:
```python
def optimize_model_for_jetson(model_path: str, precision: str = "fp16") -> str:
    # This function converts PyTorch models to TensorRT for Jetson deployment
    # It takes a model path and precision setting
    # Returns path to optimized engine file
    # Typical speedup is 2-4x on Jetson devices
```

Documented function:
"""
    
    return in_context_prompt, code_doc_prompt

def test_in_context_learning():
    """Test in-context learning capabilities"""
    spec_prompt, code_prompt = demonstrate_in_context_learning()
    
    print("🔄 In-Context Learning Demonstration")
    print("=" * 50)
    
    print("\n📋 Learning Custom Specification Format:")
    print("-" * 40)
    response1 = openai_prompter.simple_prompt(spec_prompt)
    print(response1)
    
    print("\n💻 Learning Code Documentation Style:")
    print("-" * 40)
    response2 = openai_prompter.simple_prompt(code_prompt)
    print(response2)

# test_in_context_learning()
```

> **Note:** These In-Context Learning examples are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode basic --technique in_context
> ```

---

## 🔗 Introduction to LangChain

LangChain is a powerful framework that simplifies building applications with Large Language Models (LLMs). It provides abstractions for prompt management, model integration, and complex workflows.

### 🌟 Why LangChain for Prompt Engineering?

1. **Model Agnostic**: Switch between OpenAI, Ollama, and local models seamlessly
2. **Prompt Templates**: Reusable, parameterized prompts
3. **Chain Composition**: Combine multiple prompts and models
4. **Memory Management**: Maintain conversation context
5. **Output Parsing**: Structure model responses automatically

### 📦 Installation for Jetson

```bash
# Core LangChain
pip install langchain langchain-community

# For OpenAI integration
pip install langchain-openai

# For local model support
pip install langchain-ollama

# For structured output
pip install pydantic
```

### 🔹 LangChain with OpenAI

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class LangChainOpenAIPrompter:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            api_key=api_key,
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

# Example usage
langchain_openai = LangChainOpenAIPrompter("your-api-key")

# Template example
jetson_template = """
You are a Jetson AI expert. Explain {concept} for {audience} level.
Focus on {platform} platform specifics.
Provide {num_examples} practical examples.
"""

response = langchain_openai.template_prompt(
    jetson_template,
    concept="TensorRT optimization",
    audience="intermediate",
    platform="Jetson Orin Nano",
    num_examples="3"
)
print(response)
```

### 🔹 LangChain with Local Ollama

```python
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class LangChainOllamaPrompter:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
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

# Example usage
langchain_ollama = LangChainOllamaPrompter()

if langchain_ollama.is_available():
    # Streaming example
    print("🔄 Streaming response:")
    for chunk in langchain_ollama.streaming_prompt(
        "Explain how to optimize YOLO models for Jetson Orin in detail"
    ):
        print(chunk, end="", flush=True)
    
    # Batch processing
    jetson_questions = [
        "What is the difference between Jetson Nano and Orin?",
        "How to install PyTorch on Jetson?",
        "Best practices for TensorRT optimization?"
    ]
    
    batch_responses = langchain_ollama.batch_prompts(jetson_questions)
    for q, a in zip(jetson_questions, batch_responses):
        print(f"Q: {q}")
        print(f"A: {a[:100]}...\n")
```

> **Note:** This LangChain Ollama integration is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode compare --backends ollama
> ```

### 🔹 LangChain with llama-cpp-python

```python
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LangChainLlamaCppPrompter:
    def __init__(self, model_path: str, n_gpu_layers: int = 80):
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

# Example usage (uncomment when model is available)
# langchain_llamacpp = LangChainLlamaCppPrompter("/path/to/model.gguf")

# Conversation example
# conversation = [
#     {"role": "system", "content": "You are a Jetson AI expert."},
#     {"role": "user", "content": "How do I optimize models for Jetson?"},
#     {"role": "assistant", "content": "There are several key approaches..."},
#     {"role": "user", "content": "Tell me more about TensorRT specifically."}
# ]
# 
# response = langchain_llamacpp.conversation_prompt(conversation)
# print(response)
```

> **Note:** This LangChain LlamaCpp integration is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode compare --backends llamacpp --model_path /path/to/your/model.gguf
> ```

### 🔄 Comparing LangChain Approaches

```python
def compare_langchain_backends():
    """Compare the same prompt across different LangChain backends"""
    
    prompt = """
Explain the key differences between these Jetson optimization techniques:
1. TensorRT optimization
2. CUDA kernel optimization
3. Memory management optimization

Provide practical examples for each.
"""
    
    backends = {
        "OpenAI (GPT-3.5)": langchain_openai,
        "Ollama (Local)": langchain_ollama if langchain_ollama.is_available() else None,
        # "LlamaCpp (Local)": langchain_llamacpp  # Uncomment if available
    }
    
    print("🔄 LangChain Backend Comparison")
    print("=" * 50)
    
    for backend_name, backend in backends.items():
        if backend is None:
            print(f"\n❌ {backend_name}: Not available")
            continue
        
        print(f"\n🔍 {backend_name}:")
        print("-" * 30)
        
        try:
            import time
            start_time = time.time()
            response = backend.simple_prompt(prompt)
            end_time = time.time()
            
            print(f"Response time: {end_time - start_time:.2f}s")
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        print()

# compare_langchain_backends()
```

> **Note:** This LangChain backend comparison is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode compare
> ```

---

## 🏗️ Structured Output with LangChain

LangChain excels at converting unstructured LLM responses into structured data using Pydantic models.

### 📋 Basic Structured Output

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

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

def create_structured_prompter():
    """Create a prompter that returns structured data"""
    
    class StructuredPrompter:
        def __init__(self, llm):
            self.llm = llm
        
        def get_optimization_plan(self, model_description: str) -> JetsonOptimizationPlan:
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
        
        def compare_jetson_devices(self, devices: List[str]) -> List[JetsonDeviceComparison]:
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
    
    return StructuredPrompter

# Example usage
StructuredPrompter = create_structured_prompter()
structured_prompter = StructuredPrompter(langchain_openai.llm)

# Get optimization plan
model_desc = "YOLOv8 object detection model, currently running at 15 FPS on Jetson Orin Nano, need to achieve 30 FPS for real-time application"
optimization_plan = structured_prompter.get_optimization_plan(model_desc)

if optimization_plan:
    print("🏗️ Structured Optimization Plan:")
    print(f"Model: {optimization_plan.model_name}")
    print(f"Current Performance: {optimization_plan.current_performance}")
    print(f"Steps: {optimization_plan.optimization_steps}")
    print(f"Expected Improvement: {optimization_plan.expected_improvement}")
    print(f"Difficulty: {optimization_plan.difficulty_level}")

# Compare devices
jetson_devices = ["Jetson Nano", "Jetson Orin Nano", "Jetson AGX Orin"]
device_comparisons = structured_prompter.compare_jetson_devices(jetson_devices)

print("\n📊 Device Comparison:")
for device in device_comparisons:
    print(f"\n{device.device_name}:")
    print(f"  GPU Cores: {device.gpu_cores}")
    print(f"  RAM: {device.ram_gb}GB")
    print(f"  Power: {device.power_consumption}")
    print(f"  Use Cases: {', '.join(device.best_use_cases[:2])}")
```

> **Note:** This Structured Output example is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode structured
> ```

### 🛠️ Tool Calling with Structured Output

Enable LLMs to call external tools and functions through structured prompting.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json
import subprocess
import psutil

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

class JetsonToolkit:
    """Collection of Jetson-specific tools"""
    
    @staticmethod
    def get_system_info() -> JetsonSystemInfo:
        """Get current Jetson system information"""
        try:
            # Simulate getting GPU memory (would use actual NVIDIA tools)
            gpu_memory = 2048.5  # MB
            cpu_usage = psutil.cpu_percent()
            temperature = 45.2  # Would read from thermal sensors
            
            # Simulate available models
            available_models = ["yolov8n.engine", "resnet50.onnx", "mobilenet.trt"]
            
            return JetsonSystemInfo(
                gpu_memory_used=gpu_memory,
                cpu_usage=cpu_usage,
                temperature=temperature,
                available_models=available_models
            )
        except Exception as e:
            print(f"Error getting system info: {e}")
            return None
    
    @staticmethod
    def benchmark_model(model_path: str, input_size: tuple = (640, 640)) -> ModelBenchmark:
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
            
            return ModelBenchmark(
                model_name=model_name,
                fps=fps,
                latency_ms=latency,
                memory_usage_mb=memory,
                accuracy=0.85
            )
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

# Example usage
tool_prompter = ToolCallingPrompter(langchain_openai.llm)

# Test tool calling
test_requests = [
    "What's the current system status of my Jetson?",
    "I want to benchmark my YOLOv8 model at /models/yolov8n.onnx",
    "How can I optimize my ResNet model for better performance?",
    "Compare the performance of YOLOv8 vs MobileNet on Jetson"
]

print("🛠️ Tool Calling Examples:")
print("=" * 50)

for i, request in enumerate(test_requests[:2], 1):  # Test first 2 requests
    print(f"\n{i}. Request: {request}")
    print("-" * 40)
    response = tool_prompter.process_request_with_tools(request)
    print(f"Response: {response[:300]}...")
```

> **Note:** This Tool Calling example is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode tools
> ```

---

## 🔧 Function Calling and MCP Protocol

### 🎯 Function Calling with LangChain

Function calling allows LLMs to execute specific functions based on natural language requests.

```python
from typing import Dict, Any, Callable
import inspect
from functools import wraps

class FunctionRegistry:
    """Registry for callable functions with automatic documentation"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_docs: Dict[str, Dict] = {}
    
    def register(self, name: str = None, description: str = None):
        """Decorator to register functions"""
        def decorator(func: Callable):
            func_name = name or func.__name__
            
            # Extract function signature and docstring
            sig = inspect.signature(func)
            doc = description or func.__doc__ or "No description available"
            
            # Build parameter documentation
            params = {}
            for param_name, param in sig.parameters.items():
                params[param_name] = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty
                }
            
            self.functions[func_name] = func
            self.function_docs[func_name] = {
                "description": doc,
                "parameters": params,
                "return_type": str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else "Any"
            }
            
            return func
        return decorator
    
    def get_function_schema(self) -> str:
        """Get JSON schema of all registered functions"""
        return json.dumps(self.function_docs, indent=2)
    
    def call_function(self, name: str, **kwargs) -> Any:
        """Call a registered function with parameters"""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found")
        
        try:
            return self.functions[name](**kwargs)
        except Exception as e:
            return {"error": f"Function call failed: {str(e)}"}

# Create function registry
jetson_functions = FunctionRegistry()

# Register Jetson-specific functions
@jetson_functions.register(
    description="Monitor Jetson system resources and performance metrics"
)
def monitor_jetson_resources(duration_seconds: int = 10) -> Dict[str, Any]:
    """Monitor Jetson system resources for specified duration"""
    import time
    import random
    
    # Simulate monitoring (would use actual system calls)
    metrics = {
        "monitoring_duration": duration_seconds,
        "average_cpu_usage": round(random.uniform(20, 80), 2),
        "peak_gpu_memory_mb": round(random.uniform(1000, 3000), 2),
        "average_temperature_c": round(random.uniform(35, 65), 2),
        "power_consumption_w": round(random.uniform(8, 25), 2),
        "thermal_throttling_events": random.randint(0, 3)
    }
    
    return metrics

@jetson_functions.register(
    description="Deploy and test an AI model on Jetson with performance analysis"
)
def deploy_model(model_path: str, test_images: int = 100, precision: str = "fp16") -> Dict[str, Any]:
    """Deploy model and run performance tests"""
    import time
    import random
    
    # Simulate deployment process
    time.sleep(1)  # Simulate deployment time
    
    results = {
        "model_path": model_path,
        "deployment_status": "success",
        "precision": precision,
        "test_images_processed": test_images,
        "average_fps": round(random.uniform(15, 60), 2),
        "average_latency_ms": round(random.uniform(10, 100), 2),
        "memory_usage_mb": round(random.uniform(500, 2000), 2),
        "accuracy_score": round(random.uniform(0.8, 0.95), 3),
        "deployment_time_seconds": round(random.uniform(30, 120), 2)
    }
    
    return results

@jetson_functions.register(
    description="Compare multiple AI models on Jetson platform"
)
def compare_models(model_paths: list, test_dataset: str = "coco_val") -> Dict[str, Any]:
    """Compare performance of multiple models"""
    import random
    
    comparisons = []
    for model_path in model_paths:
        model_name = model_path.split('/')[-1]
        comparison = {
            "model_name": model_name,
            "fps": round(random.uniform(10, 50), 2),
            "latency_ms": round(random.uniform(20, 100), 2),
            "memory_mb": round(random.uniform(400, 1500), 2),
            "accuracy": round(random.uniform(0.75, 0.92), 3),
            "power_efficiency": round(random.uniform(1.5, 4.0), 2)  # FPS per Watt
        }
        comparisons.append(comparison)
    
    # Find best model for each metric
    best_fps = max(comparisons, key=lambda x: x['fps'])
    best_accuracy = max(comparisons, key=lambda x: x['accuracy'])
    best_efficiency = max(comparisons, key=lambda x: x['power_efficiency'])
    
    return {
        "test_dataset": test_dataset,
        "models_compared": len(model_paths),
        "detailed_results": comparisons,
        "recommendations": {
            "best_for_speed": best_fps['model_name'],
            "best_for_accuracy": best_accuracy['model_name'],
            "best_for_efficiency": best_efficiency['model_name']
        }
    }

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

# Example usage
function_prompter = FunctionCallingPrompter(langchain_openai.llm, jetson_functions)

# Test function calling
function_test_requests = [
    "Monitor my Jetson system for 30 seconds and tell me about performance",
    "Deploy my YOLOv8 model at /models/yolov8s.onnx and test it with 50 images",
    "Compare these models: ['/models/yolo.onnx', '/models/resnet.trt', '/models/mobilenet.engine']"
]

print("\n🔧 Function Calling Examples:")
print("=" * 50)

for i, request in enumerate(function_test_requests[:2], 1):
    print(f"\n{i}. Request: {request}")
    print("-" * 40)
    response = function_prompter.process_with_functions(request)
    print(f"Response: {response[:300]}...")
```

> **Note:** This Function Calling example is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode functions
> ```

### 🌐 Model Context Protocol (MCP)

MCP is a new standard for connecting LLMs to external tools and data sources. Here's how it works from a prompt engineering perspective:

```python
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

# Create MCP server for Jetson
jetson_mcp = MCPServer()

# Register tools
def jetson_system_status() -> dict:
    """Get comprehensive Jetson system status"""
    return {
        "gpu_utilization": 75.2,
        "memory_usage_gb": 6.4,
        "temperature_c": 52.1,
        "power_draw_w": 18.5,
        "cpu_frequency_mhz": 1900,
        "gpu_frequency_mhz": 1300
    }

def optimize_inference_pipeline(model_type: str, target_fps: int) -> dict:
    """Optimize inference pipeline for target performance"""
    optimizations = {
        "yolo": ["TensorRT FP16", "Dynamic batching", "CUDA streams"],
        "resnet": ["TensorRT INT8", "Layer fusion", "Memory pooling"],
        "transformer": ["Flash attention", "KV cache optimization", "Quantization"]
    }
    
    return {
        "model_type": model_type,
        "target_fps": target_fps,
        "recommended_optimizations": optimizations.get(model_type, ["General TensorRT optimization"]),
        "estimated_speedup": "2.5x",
        "implementation_complexity": "Medium"
    }

jetson_mcp.register_tool("jetson_status", "Get current Jetson system status", jetson_system_status)
jetson_mcp.register_tool("optimize_pipeline", "Optimize inference pipeline", optimize_inference_pipeline)

# Register prompt templates
jetson_mcp.register_prompt(
    "model_optimization",
    "Generate optimization plan for AI model",
    """
You are a Jetson optimization expert. Create a detailed optimization plan for:

Model: {model_name}
Current Performance: {current_fps} FPS
Target Performance: {target_fps} FPS
Platform: {jetson_model}

Provide:
1. Specific optimization techniques
2. Expected performance gains
3. Implementation steps
4. Potential trade-offs

Optimization Plan:
"""
)

jetson_mcp.register_prompt(
    "deployment_checklist",
    "Generate deployment checklist for Jetson AI application",
    """
Create a comprehensive deployment checklist for:

Application: {app_name}
Model: {model_type}
Target Device: {device_type}
Performance Requirements: {requirements}

Include:
- Pre-deployment testing
- Performance validation
- Monitoring setup
- Troubleshooting steps

Deployment Checklist:
"""
)

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

# Example usage
mcp_prompter = MCPPrompter(langchain_openai.llm, jetson_mcp)

# Test MCP integration
mcp_test_requests = [
    "What's my current Jetson system status?",
    "Create an optimization plan for my YOLOv8 model running at 15 FPS, targeting 30 FPS on Jetson Orin Nano",
    "Generate a deployment checklist for my object detection application"
]

print("\n🌐 MCP Integration Examples:")
print("=" * 50)

for i, request in enumerate(mcp_test_requests[:2], 1):
    print(f"\n{i}. Request: {request}")
    print("-" * 40)
    response = mcp_prompter.process_with_mcp(request)
    print(f"Response: {response[:300]}...")
```

> **Note:** This MCP integration example is included in the unified `jetson_prompt_toolkit.py` script. You can run it with:
> ```bash
> python jetson_prompt_toolkit.py --mode mcp
> ```

---

## 🧪 Lab: Advanced Prompt Engineering on Jetson with LangChain

### 🎯 Lab Objectives

1. **Master Core Techniques**: Implement and compare different prompt engineering approaches
2. **Build Tool Integration**: Create LLM systems that can call external tools and functions
3. **Develop MCP Applications**: Build applications using the Model Context Protocol
4. **Optimize for Jetson**: Apply prompt engineering specifically for edge AI scenarios

### 🛠️ Lab Setup

```bash
# Install required packages
pip install langchain langchain-openai langchain-community
pip install ollama llama-cpp-python
pip install pydantic psutil

# Start Ollama (if using local models)
ollama serve
ollama pull llama3.2:3b
```

### 📋 Exercise 1: Prompt Engineering Comparison

Create a comprehensive comparison system for different prompting techniques:

```python
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

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
                    response = llm.invoke(prompt_text)
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

# Initialize lab
lab = PromptEngineeringLab({
    "openai": langchain_openai.llm,
    "ollama": langchain_ollama.llm,
    "llamacpp": langchain_llamacpp.llm
})

# Test scenarios
test_scenarios = [
    {
        "task": "How can I optimize a YOLOv8 model running at 15 FPS to achieve 30 FPS on Jetson Orin Nano?",
        "keywords": ["tensorrt", "quantization", "batch", "optimization", "fp16", "int8", "engine"]
    },
    {
        "task": "What's the best approach to deploy multiple AI models simultaneously on Jetson while maintaining real-time performance?",
        "keywords": ["pipeline", "scheduling", "memory", "concurrent", "optimization", "resource"]
    }
]

print("🧪 Starting Prompt Engineering Lab...")
print("=" * 50)

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n📋 Scenario {i}: {scenario['task'][:50]}...")
    results = lab.run_prompt_comparison(scenario["task"], scenario["keywords"])
    report = lab.generate_comparison_report(results)
    print(report)
```

### 📋 Exercise 2: Advanced Tool Integration

Build a comprehensive Jetson AI assistant with multiple tool capabilities:

```python
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
                    if "error" not in prompt_result:
                        prompt_text = prompt_result["messages"][0]["content"]["text"]
                        result = self.llm.invoke(prompt_text)
                        execution_results.append({"step": step["step"], "result": result})
            
            # Generate final response
            final_prompt = f"""
User Request: {user_request}

Execution Results:
{json.dumps(execution_results, indent=2, default=str)}

Based on the execution results, provide a comprehensive, actionable response to the user. Include:
1. Direct answer to their question
2. Specific data and insights from the tools
3. Actionable recommendations
4. Next steps they should consider

Response:
"""
            
            final_response = self.llm.invoke(final_prompt)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": final_response})
            
            return final_response
            
        except Exception as e:
            return f"Error processing complex request: {e}"

# Initialize advanced assistant
assistant = JetsonAIAssistant(langchain_openai.llm)

# Test complex scenarios
complex_scenarios = [
    "I need to deploy 3 different AI models on my Jetson Orin for a production surveillance system. The models are YOLOv8 for detection, a classification model for verification, and a tracking model. I need 30 FPS minimum with less than 20W power consumption. Can you help me create an optimization and deployment plan?",
    
    "My current object detection pipeline is running at 18 FPS but I need 25 FPS. The model is using 3.2GB GPU memory and the system temperature reaches 78°C under load. What's the best optimization strategy?",
    
    "I want to compare the performance of YOLOv8, YOLOv10, and RT-DETR on Jetson Orin Nano for real-time person detection in retail environments. Can you help me set up benchmarks and provide recommendations?"
]

print("\n🤖 Advanced AI Assistant Testing:")
print("=" * 60)

for i, scenario in enumerate(complex_scenarios[:2], 1):
    print(f"\n📋 Complex Scenario {i}:")
    print(f"Request: {scenario[:100]}...")
    print("-" * 50)
    response = assistant.process_complex_request(scenario)
    print(f"Response: {response[:400]}...")
```

### 📋 Exercise 3: Production-Ready MCP Application

Create a production-ready application using MCP for Jetson AI development:

```python
class ProductionMCPApp:
    """Production-ready MCP application for Jetson AI development"""
    
    def __init__(self):
        self.mcp_server = self._initialize_mcp_server()
        self.llm_clients = self._initialize_llm_clients()
        self.session_manager = SessionManager()
        self.performance_tracker = PerformanceTracker()
    
    def _initialize_mcp_server(self) -> MCPServer:
        """Initialize comprehensive MCP server"""
        server = MCPServer()
        
        # Register production tools
        server.register_tool("jetson_diagnostics", "Run comprehensive Jetson diagnostics", self._run_diagnostics)
        server.register_tool("model_optimizer", "Optimize models for production deployment", self._optimize_model)
        server.register_tool("deployment_manager", "Manage model deployments", self._manage_deployment)
        server.register_tool("performance_analyzer", "Analyze system performance", self._analyze_performance)
        
        # Register production prompt templates
        server.register_prompt(
            "production_readiness_check",
            "Comprehensive production readiness assessment",
            self._get_production_readiness_template()
        )
        
        server.register_prompt(
            "incident_response",
            "Generate incident response plan",
            self._get_incident_response_template()
        )
        
        return server
    
    def _get_production_readiness_template(self) -> str:
        return """
CONDUCT PRODUCTION READINESS ASSESSMENT
=====================================

Application: {app_name}
Environment: {environment}
Models: {models}
Expected Load: {expected_load}
SLA Requirements: {sla_requirements}

Assessment Areas:

1. PERFORMANCE VALIDATION
   - Benchmark results under expected load
   - Latency and throughput analysis
   - Resource utilization assessment
   - Stress testing results

2. RELIABILITY & STABILITY
   - Error handling mechanisms
   - Failover procedures
   - Recovery strategies
   - Memory leak detection

3. MONITORING & OBSERVABILITY
   - Metrics collection setup
   - Alerting configuration
   - Logging implementation
   - Dashboard availability

4. SECURITY CONSIDERATIONS
   - Model security validation
   - Data privacy compliance
   - Access control implementation
   - Vulnerability assessment

5. OPERATIONAL READINESS
   - Deployment automation
   - Rollback procedures
   - Documentation completeness
   - Team training status

Provide a comprehensive assessment with:
- Go/No-Go recommendation
- Critical issues to address
- Risk mitigation strategies
- Timeline for production deployment

ASSESSMENT REPORT:
"""
    
    def _get_incident_response_template(self) -> str:
        return """
INCIDENT RESPONSE PLAN
====================

Incident Type: {incident_type}
Severity: {severity}
Affected Systems: {affected_systems}
Impact: {impact}

IMMEDIATE ACTIONS (0-15 minutes):
1. Assess incident scope and impact
2. Implement immediate containment
3. Notify stakeholders
4. Begin incident logging

SHORT-TERM ACTIONS (15-60 minutes):
1. Detailed investigation
2. Implement workarounds
3. Escalate if necessary
4. Communicate status updates

LONG-TERM ACTIONS (1+ hours):
1. Root cause analysis
2. Permanent fix implementation
3. System validation
4. Post-incident review

SPECIFIC PROCEDURES:
- Performance degradation: {performance_procedures}
- Model accuracy issues: {accuracy_procedures}
- System failures: {failure_procedures}
- Security incidents: {security_procedures}

CONTACT INFORMATION:
- On-call engineer: {oncall_contact}
- Escalation manager: {escalation_contact}
- Vendor support: {vendor_contact}

Generate detailed incident response plan:
"""
    
    def run_production_workflow(self, workflow_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run production workflows using MCP"""
        
        workflow_results = {
            "workflow_type": workflow_type,
            "start_time": time.time(),
            "parameters": parameters,
            "steps": [],
            "status": "running"
        }
        
        try:
            if workflow_type == "model_deployment":
                workflow_results["steps"] = self._execute_deployment_workflow(parameters)
            elif workflow_type == "performance_optimization":
                workflow_results["steps"] = self._execute_optimization_workflow(parameters)
            elif workflow_type == "production_validation":
                workflow_results["steps"] = self._execute_validation_workflow(parameters)
            
            workflow_results["status"] = "completed"
            workflow_results["end_time"] = time.time()
            workflow_results["duration"] = workflow_results["end_time"] - workflow_results["start_time"]
            
        except Exception as e:
            workflow_results["status"] = "failed"
            workflow_results["error"] = str(e)
            workflow_results["end_time"] = time.time()
        
        return workflow_results

# Initialize production app
production_app = ProductionMCPApp()

print("\n🏭 Production MCP Application Ready")
print("Available workflows: model_deployment, performance_optimization, production_validation")
```

> **Note:** These lab exercises are included in the unified `jetson_prompt_toolkit.py` script. You can run them with:
> ```bash
> python jetson_prompt_toolkit.py --mode lab
> ```

### 📊 Lab Results and Analysis

After completing all exercises, analyze your results:

1. **Prompt Engineering Effectiveness**: Which techniques worked best for different types of tasks?
2. **Tool Integration Performance**: How did tool calling improve response quality and accuracy?
3. **MCP Protocol Benefits**: What advantages did MCP provide over direct tool calling?
4. **Jetson-Specific Optimizations**: Which prompt engineering approaches were most effective for edge AI scenarios?

### 🎯 Next Steps

1. **Experiment with Custom Models**: Try the techniques with different local models
2. **Build Domain-Specific Tools**: Create tools specific to your AI application domain
3. **Implement Production Monitoring**: Add comprehensive logging and monitoring to your MCP applications
4. **Optimize for Edge Deployment**: Focus on minimizing latency and resource usage for production edge AI systems

---

## 🎉 Conclusion

This tutorial covered comprehensive prompt engineering techniques for Jetson AI development, from basic prompting strategies to advanced tool calling and MCP protocol implementation. You've learned how to:

- **Master Core Techniques**: Chain-of-thought, few-shot learning, role-based prompting, and in-context learning
- **Integrate LangChain**: Leverage LangChain for model-agnostic prompt engineering and structured output
- **Implement Tool Calling**: Enable LLMs to interact with external tools and functions
- **Use MCP Protocol**: Build scalable applications using the Model Context Protocol
- **Optimize for Jetson**: Apply prompt engineering specifically for edge AI scenarios

These techniques enable you to build sophisticated AI applications that can reason, plan, and execute complex tasks on Jetson platforms, making your edge AI systems more intelligent and capable.

### 🎯 Goal

Test and compare prompt engineering on three backends:

1. llama-cpp-python
2. Ollama
3. OpenAI API

### 🔁 Prompt Types to Try

* Instructional prompt
* Few-shot learning
* Chain-of-thought reasoning
* Rewriting and follow-ups

---

## 📋 Deliverables

* Code + PromptTemplate examples
* Comparison table of responses from three backends
* Bonus: Add memory support for follow-up context

---

## 🧠 Summary

* Start with local inference and basic prompting
* Move to API-based or structured LangChain interfaces
* Use LangChain to modularize prompt types and switch LLM backends (OpenAI, Ollama, or llama-cpp)
* Jetson Orin Nano supports local inference with quantized models using llama.cpp or Ollama

---

## 🚀 Unified Python Script

All the Python code examples from this tutorial have been consolidated into a single, unified script called `jetson_prompt_toolkit.py`. This script provides a command-line interface to experiment with different prompt engineering techniques, backends, and advanced features.

### 📥 Installation

```bash
# Clone the repository if you haven't already
git clone https://github.com/yourusername/edgeAI.git
cd edgeAI

# Install dependencies
pip install openai langchain langchain-openai langchain-community pydantic

# Optional: Install Ollama for local inference
# Follow instructions at https://ollama.ai/

# Optional: Install llama-cpp-python for local inference
pip install llama-cpp-python
```

### 🔧 Usage Examples

#### Basic Prompt Engineering Techniques

```bash
# Test Chain-of-Thought reasoning with OpenAI
python jetson_prompt_toolkit.py --mode basic --technique cot --backends openai

# Compare all techniques across multiple backends
python jetson_prompt_toolkit.py --mode basic --technique all --backends openai,ollama
```

#### Compare Different Backends

```bash
# Compare responses from different backends
python jetson_prompt_toolkit.py --mode compare --backends openai,ollama,llamacpp --llamacpp-model /path/to/model.gguf
```

#### Structured Output Generation

```bash
# Generate a YOLOv8 optimization plan
python jetson_prompt_toolkit.py --mode structured --output optimization_plan --backends openai

# Compare Jetson devices
python jetson_prompt_toolkit.py --mode structured --output device_comparison --backends openai
```

#### Tool Calling Demonstrations

```bash
# Process a request using tool calling
python jetson_prompt_toolkit.py --mode tools --request "What's the current system status of my Jetson device?"
```

#### Function Calling Demonstrations

```bash
# Process a request using function calling
python jetson_prompt_toolkit.py --mode functions --request "I need to optimize my YOLOv8 model for Jetson Nano"
```

#### MCP Protocol Demonstrations

```bash
# Process a request using MCP
python jetson_prompt_toolkit.py --mode mcp --request "Create a deployment checklist for my YOLOv8 model on Jetson Xavier NX"
```

#### Advanced Assistant Demonstrations

```bash
# Process a complex request with the Jetson AI Assistant
python jetson_prompt_toolkit.py --mode assistant --request "I need to deploy multiple AI models on my Jetson AGX Orin for a smart retail application"
```

#### Production MCP Application

```bash
# Run a production workflow
python jetson_prompt_toolkit.py --mode production
```

### 🔄 Switching Models

You can specify which models to use with each backend:

```bash
# Use a specific OpenAI model
python jetson_prompt_toolkit.py --mode basic --technique cot --backends openai --openai-model gpt-4o

# Use a specific Ollama model
python jetson_prompt_toolkit.py --mode basic --technique cot --backends ollama --ollama-model llama3.1:8b

# Use a specific llama-cpp-python model
python jetson_prompt_toolkit.py --mode basic --technique cot --backends llamacpp --llamacpp-model /path/to/model.gguf
```

