# 🧠 NVIDIA Jetson Orin Nano Student Guide

**Author:** Kaikai Liu
**Email:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

---

## 📌 Overview

This guide introduces the **NVIDIA Jetson Orin Nano**, explains how to install and use our custom Jetson utility script `sjsujetsontool`, and provides step-by-step instructions for development tasks such as launching servers, running AI models, setting up Jupyter, and managing devices.

---

## 🧠 What Is NVIDIA Jetson Orin Nano?

The **Jetson Orin Nano** is a powerful, energy-efficient AI edge computing board by NVIDIA. Key features:

* ✅ 6-core ARM Cortex CPU
* ✅ Ampere GPU with up to 1024 CUDA cores
* ✅ Ideal for robotics, vision, AI model serving, and cyber experiments
* ✅ Supports JetPack SDK with Ubuntu, CUDA, cuDNN, TensorRT

---

## 🌐 Connecting to Jetson via `.local` Hostname

Jetson devices with mDNS enabled can be accessed using the `.local` hostname from macOS or Linux:

```bash
ssh username@jetson-hostname.local
```

For example:

```bash
ssh sjsujetson@jetson-01.local
```

> If this doesn't work, make sure `avahi-daemon` is running on Jetson and that your network supports mDNS.

---

## ⚙️ Installing `sjsujetsontool`

A command-line tool for Jetson-based workflows: container management, model serving, RAG apps, and more.

### ✅ One-line install (no sudo required)

```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
```

Then reload shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

Verify:

```bash
sjsujetsontool list
```

---

## 🧪 Common Usage Examples

### 🟢 `sjsujetsontool jupyter`

Launches JupyterLab on port 8888 from inside the Jetson's Docker container. It allows interactive Python notebooks for AI model testing, data exploration, and debugging.

### 🧠 `sjsujetsontool ollama`

Starts the Ollama LLM server inside the container, exposing an OpenAI-compatible API on port 11434. Useful for LangChain and AI chat integration.

### 🔬 `sjsujetsontool llama`

Starts the `llama.cpp` server (C++ GGUF LLM inference engine) on port 8000. Loads a `.gguf` model and serves an HTTP API for tokenized prompt completion.

### 🐍 `sjsujetsontool run <script.py>`

Runs any Python script inside the preconfigured container. Ensures all ML/AI libraries and GPU drivers are properly set up.

### 🚀 `sjsujetsontool fastapi`

Launches a FastAPI backend on port 8001, useful for serving AI endpoints for real-time apps.

### 📚 `sjsujetsontool rag`

Runs a LangChain Retrieval-Augmented Generation demo server with local documents and vector search.

### 📦 `sjsujetsontool status`

Displays:

* Docker container state
* GPU stats from `tegrastats`
* Port listening status for key services

### 🔧 `sjsujetsontool set-hostname <name> [github_user]`

Changes device hostname, regenerates system identity, writes `/etc/device-id`, and adds GitHub SSH keys.

### 🔐 `sjsujetsontool setup-ssh <github_user>`

Fetches the GitHub user's public SSH keys and adds them to `~/.ssh/authorized_keys` for passwordless SSH login.

### 📁 `sjsujetsontool mount-nfs <host> <remote_path> <mount_point>`

Mounts a shared folder from a remote NFS server. Useful for accessing centralized data or logging remotely.

### 🛠️ `sjsujetsontool build`

Rebuilds the base development Docker image with CUDA, Python, PyTorch, and other libraries pre-installed.

### 🛑 `sjsujetsontool stop`

Stops the running Docker container started by previous commands.

### 🧾 `sjsujetsontool update`

Downloads the latest version of `sjsujetsontool` from GitHub and replaces the local version, keeping a backup.

### 📋 `sjsujetsontool list`

Displays all available commands with usage examples.

---

## ⚠️ Safety Guidelines

* 🔌 **Power Supply**: Use a 5A USB-C adapter or official barrel jack for stability.
* 💾 **SSD Cloning**: Change the hostname and machine-id after cloning to prevent network conflicts.
* 🔐 **SSH Security**: Only install SSH keys from trusted GitHub accounts.
* 🧼 **Disk Cleanup**: Remove cache and large datasets before creating system images.
* 📦 **Containers**: Always stop containers with `sjsujetsontool stop` before unplugging.

---

## 🧭 Ready to Learn and Build

You're now equipped to:

* Run AI models (LLaMA, Mistral, DeepSeek, etc.)
* Build and test FastAPI/LLM applications
* Access Jetson remotely with SSH or VS Code
* Run real-time cyber/AI experiments on the edge!

---

Made with 💻 by [Kaikai Liu](mailto:kaikai.liu@sjsu.edu) — [GitHub Repo](https://github.com/lkk688/edgeAI)
