# 📘 Jetson + AI Curriculum Guide

Welcome to the official documentation for the **SJSU Cyber-AI Curriculum**. This guide is designed for high school and university students to learn about embedded systems, Linux, AI, and cybersecurity through hands-on labs on the NVIDIA Jetson Orin Nano platform.

---

## 🧭 Getting Started

### 🔧 How to Build This Documentation

You can build this curriculum as a local HTML site or generate a PDF using [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

#### 1. Install MkDocs and dependencies

```bash
% conda activate mypy311
pip install mkdocs mkdocs-material
pip install mkdocs-revealjs
```

#### 2. Build and serve the site locally

```bash
mkdocs serve
```

Open your browser to: [http://localhost:8000](http://localhost:8000)

#### 3. Build for static HTML site

```bash
mkdocs build
```

HTML files will be generated in the `site/` directory.

#### 4. Export to PDF (Optional)

Install `weasyprint` or use browser print-to-PDF from `localhost:8000`.

---

## 📚 Curriculum Structure

### 🔰 Introduction

* [✅ sjsujetsontool Guide](curriculum/00_sjsujetsontool_guide.md)
* [🔧 Introduction to NVIDIA Jetson](curriculum/01a_nvidia_jetson.md)
* [🐧 Linux + Networking Tools](curriculum/01b_linux_networking_tools.md)
* [💡 Linux OS Basics](curriculum/01b_linux_basics.md)
* [🔍 Packet Sniffing & Monitoring](curriculum/01c_packet_sniffing_monitoring.md)
* [🛡️ Cyber Defense Tools](curriculum/01d_linux_cyber_defense_basics.md)
* [⚔️ Cyber Attack Simulation](curriculum/01e_linux_cyber_attack_simulation.md)

### 🧰 Core Systems

* [🧠 Programming Environment (Python/C++/CUDA)](curriculum/02_programming_env_python_cpp_cuda.md)
* [🚀 Accelerated Python + CUDA](curriculum/03_accelerated_computing_python_cuda.md)
* [🔢 Numpy for AI](curriculum/04a_numpy_pytorch_intro.md)
* [🔥 PyTorch Basics](curriculum/04b_pytorch_intro.md)

### 🛡️ Cyber Systems

* [📡 Network Tools](curriculum/01b_linux_networking_tools.md)
* [🧪 Packet Analysis Lab](curriculum/01c_packet_sniffing_monitoring.md)
* [🧰 Defense Tools Lab](curriculum/01d_linux_cyber_defense_basics.md)
* [💣 Simulated Attacks & Detection](curriculum/01e_linux_cyber_attack_simulation.md)

### 🤖 AI & LLM

* [🖼️ CNN + Image Processing on Jetson](curriculum/05_cnn_image_processing_jetson.md)
* [🔍 YOLO + Vision-Language Models](curriculum/05b_yolo_vlm_object_detection.md)
* [🧠 Transformers & LLMs](curriculum/06_transformers_llms_jetson.md)
* [📚 NLP + LLM Optimization](curriculum/07_nlp_applications_llm_optimization.md)
* [✍️ Prompt Engineering + LangChain](curriculum/08_prompt_engineering_langchain_jetson.md)
* [🔎 RAG with LangChain](curriculum/09_rag_app_langchain_jetson.md)
* [🧑‍💻 Local AI Agents](curriculum/10_local_ai_agents_jetson.md)

### 🧪 Final Project

* [🏆 Hackathon & Project Challenges](curriculum/11_final_challenges_hackathon.md)

---

## 👨‍🏫 Author

**Kaikai Liu**
Email: [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)
San Jose State University

---

> Learn. Build. Defend. Empower with Edge AI on Jetson.

---
