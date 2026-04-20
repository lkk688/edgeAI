# 📘 Jetson + AI Curriculum Guide

Welcome to the official documentation for the **SJSU Cyber-AI Curriculum**. This guide is designed for students to learn about embedded systems, Linux, AI, and cybersecurity through hands-on labs on the NVIDIA Jetson Orin Nano platform.

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

#### 5. Push to Github Pages (Optional)

```bash
mkdocs gh-deploy
```

<!-- To https://github.com/lkk688/edgeAI.git
 * [new branch]      gh-pages -> gh-pages
INFO    -  Your documentation should shortly be available at:
           https://lkk688.github.io/edgeAI/ -->
---

## 📚 Curriculum Structure

### 🔰 Getting Started with Jetson

* [✅ sjsujetsontool Guide](curriculum/00_sjsujetsontool_guide.md)
* [📋 sjsujetsontool Cheatsheet](curriculum/00b_sjsujetsontool_cheatsheet.md)
* [🔧 Introduction to NVIDIA Jetson](curriculum/01a_nvidia_jetson.md)
* [🚀 CUDA Programming on Jetson](curriculum/01b_jetson_cuda.md)

### 🐧 Linux Fundamentals

* [💡 Linux OS Basics](curriculum/02a_linux_basics.md)
* [🌐 Linux Networking Tools](curriculum/03a_linux_networking_tools.md)
* [🌐 Packet Sniffing & Monitoring](curriculum/03b_packet_sniffing_monitoring.md)

### 🤖 AI & LLM

* [🧠 Deep Learning & CNN](curriculum/04_deeplearning_cnn.md)
* [🖼️ CNN Image Processing on Jetson](curriculum/04b_cnn_image_processing_jetson.md)
* [🧠 Transformers & NLP Applications](curriculum/05_transformers_nlp_applications.md)
* [🚀 Large Language Models on Jetson](curriculum/06_llms_jetson.md)
* [📚 NLP Applications & LLM Optimization](curriculum/07_nlp_applications_llm_optimization.md)
* [✍️ Prompt Engineering & LangChain](curriculum/08_prompt_engineering_langchain_jetson.md)
* [🔎 RAG Applications with LangChain](curriculum/09_rag_app_langchain_jetson.md)
* [🤖 Local AI Agents on Jetson](curriculum/10_local_ai_agents_jetson.md)
* [🎙️ Voice Assistant on Jetson](curriculum/10b_voice_assistant_jetson.md)

<!-- ### 🧪 Final Project

* [🏆 Hackathon & Project Challenges](curriculum/11_final_challenges_hackathon.md) -->

---

## 👨‍🏫 Author

**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

---

> Learn. Build. Defend. Empower with Edge AI on Jetson.

---
