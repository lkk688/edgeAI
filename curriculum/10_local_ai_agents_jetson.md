# 🤖 Local AI Agents on Jetson

## 🧠 What are AI Agents?

AI Agents are autonomous programs that use LLMs to reason, decide, and act based on goals, memory, and tools.

With LangChain and Jetson, we can create local agents that:

* Run without internet
* Use llama.cpp/Ollama for language reasoning
* Perform file searches, tool use, and code execution

---

## 🔧 Agent Architecture

1. **User Input**
2. **Planner (LLM)**: Understands and plans
3. **Tool Executor**: Runs tasks (search, read, run code)
4. **Memory/Context**
5. **Response Generator**

---

## 🤝 Function Calling Support (Tool Use via JSON)

Function calling enables structured tool invocation from LLMs using JSON schema.

### Why it's useful:

* Enables precision in tool execution
* Easy to parse and extend with APIs
* Supported by OpenAI, Ollama (some models), and LangChain

LangChain integrates this using the ReActAgent + Tool definitions.

---

## 🧰 What is MCP (Multi-Component Prompting)?

MCP involves splitting prompts into parts to:

* Improve performance on edge LLMs
* Isolate tool planning from reasoning
* Modularize prompt templates

Useful when chaining reasoning across multiple tools or responses.

---

## 🔌 Agent Tools

LangChain provides many tools for agents:

* `Python REPL` (math, logic, data)
* `File system access`
* `Search/QA`
* `Terminal commands` (⚠️ safe usage required)

---

## 🚀 Agent Types (LangChain)

* **ZeroShotAgent**: Selects tools via LLM
* **ReActAgent**: Combines reasoning and acting
* **Toolkits**: Bundled tools for DevOps, CSV, web

---

## 🧪 Lab: Local LangChain Agent with llama.cpp or Ollama

### 🧰 Setup

```bash
pip install langchain llama-cpp-python
```

---

### ✅ llama.cpp Backend

```python
from langchain.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import LlamaCpp

llm = LlamaCpp(model_path="/models/mistral.gguf", n_gpu_layers=80)
tools = [PythonREPLTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is 12 squared minus 3?")
```

---

### ✅ Ollama Backend (OpenAI-compatible API)

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import PythonREPLTool

ollama_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
tools = [PythonREPLTool()]
agent = initialize_agent(tools, ollama_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Solve 2^6 and explain.")
```

---

### 🧪 Bonus: Add Filesystem Tool

```python
from langchain.tools import Tool
import os

def list_files():
    return "\n".join(os.listdir("./"))

fs_tool = Tool(name="List Files", func=list_files, description="List files in current directory")
tools.append(fs_tool)
```

---

## 📋 Lab Deliverables

* Implement local agent using llama.cpp and/or Ollama
* Add at least 2 tools (Python + custom tool)
* Demonstrate tool use, function call-like behavior, and multi-turn reasoning

---

## 🧠 Summary

* Agents = LLM + memory + tools
* LangChain agents support both llama-cpp-python and Ollama with GGUF models
* Use structured tool calling and MCP to improve modularity and performance
* Jetson Orin Nano can run lightweight agents entirely offline

→ Next: [Final Project: Hackathon & Challenges](11_final_challenges_hackathon.md)
