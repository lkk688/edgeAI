# 🧠 Prompt Engineering with LangChain on Jetson

## 🎯 What is Prompt Engineering?

Prompt engineering is the practice of crafting effective inputs to guide the behavior of large language models (LLMs).

It's essential for improving accuracy, safety, and reliability of LLM responses — especially on edge devices like Jetson.

---

## 💡 Inference Pathways on Jetson and Cloud

### 🔹 1. Local Model Inference with Prompt (No LangChain)

Run a model like Mistral or LLaMA via `llama-cpp-python` directly:

```python
from llama_cpp import Llama
llm = Llama(model_path="/models/mistral.gguf", n_gpu_layers=80)
print(llm("What is edge computing?"))
```

You can modify the prompt directly to test prompt effectiveness.

---

### 🔹 2. OpenAI API Inference

You can run prompts through a cloud-hosted LLM via OpenAI's API:

```python
import openai
openai.api_key = "YOUR_API_KEY"
response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[{"role": "user", "content": "Explain Jetson Orin Nano."}]
)
print(response.choices[0].message.content)
```

---

## 🔗 LangChain Integrations for Prompt Engineering

LangChain provides a modular way to structure and compare prompts across different model backends.

### 🔹 3. LangChain with OpenAI

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(api_key="YOUR_API_KEY")
template = "Explain this in two sentences: {topic}"
prompt = PromptTemplate.from_template(template)
print((prompt | llm).invoke({"topic": "LLMs on edge devices"}))
```

---

### 🔹 4. LangChain with Local Ollama

Ollama provides a REST API similar to OpenAI. LangChain supports this natively:

```python
from langchain.llms import OpenAI
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

Use the same prompt templates. Make sure `ollama serve` is running with the model loaded.

---

### 🔹 5. LangChain with llama-cpp-python

```python
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

llm = LlamaCpp(model_path="/models/mistral.gguf", n_gpu_layers=80, temperature=0.7)
template = "You are a helpful assistant. Question: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["question"], template=template)
chain = prompt | llm
print(chain.invoke({"question": "Explain what is edge computing."}))
```

---

## 🧪 Lab: Prompt Engineering on Jetson with LangChain

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

→ Next: [RAG Apps](09_rag_app_langchain_jetson.md)
