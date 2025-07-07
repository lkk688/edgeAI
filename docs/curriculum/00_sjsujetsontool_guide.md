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
ssh sjsujetson@sjsujetson-01.local
```

> If this doesn't work, make sure `avahi-daemon` is running on Jetson and that your network supports mDNS.

---

## ⚙️ Installing `sjsujetsontool`

A command-line tool for Jetson-based workflows: container management, model serving, AI apps, and more.

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

The `sjsujetsontool` wraps python apps running via container and makes running code inside the container easy to use. `docker` without sudo is already setup in the jetson device. Check existing containers available in the Jetson:
```bash
sjsujetson@sjsujetson-01:~$ docker images
REPOSITORY                TAG              IMAGE ID       CREATED         SIZE
jetson-llm-v1             latest           8236678f7ef1   6 days ago      9.89GB
jetson-pytorch-v1         latest           da28af1b9eed   9 days ago      9.71GB
hello-world               latest           f1f77a0f96b7   5 months ago    5.2kB
nvcr.io/nvidia/pytorch    24.12-py3-igpu   ee796da7f569   6 months ago    9.63GB
nvcr.io/nvidia/l4t-base   r36.2.0          46b8e6a6a6a7   19 months ago   750MB
sjsujetson@sjsujetson-01:~$ sjsujetsontool shell #enter into the container
root@sjsujetson-01:/workspace#
```

The `\Developer` and `\Developer\models` folders in the jetson host are mounted to the container in the path of `\Developer` and `\models`

### ✅ Exter the Container Shell
Run the `sjsujetsontool shell` command line to enter into the shell of the container
```bash
sjsujetson@sjsujetson-01:/Developer/edgeAI$ sjsujetsontool shell
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
root@sjsujetson-01:/workspace#
```

---

## 🧪 Common Usage Examples


### 🧾 `sjsujetsontool update`

Downloads the latest version of `sjsujetsontool` from GitHub and replaces the local version, keeping a backup.

### 📋 `sjsujetsontool list`

Displays all available commands with usage examples.

### 🟢 `sjsujetsontool jupyter`

Launches JupyterLab on port 8888 from inside the Jetson's Docker container. It allows interactive Python notebooks for AI model testing, data exploration, and debugging.

```bash
sjsujetson@sjsujetson-01:~$ sjsujetsontool jupyter
....
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://hostname:8888/lab?token=3bbbf2fbea22e917bdbace45cb414bbaeb52f1251163adcf
        http://127.0.0.1:8888/lab?token=3bbbf2fbea22e917bdbace45cb414bbaeb52f1251163adcf
```
you can access the jupyter server via the the provided url. If you want to remote access the jupyter server from another computer, you can replace the hostname with the IP address of the device. 

### 🧠 `sjsujetsontool ollama`

This section introduces its integrated `ollama` command group, which allows you to manage, run, and query large language models inside a Docker container on your Jetson.

`sjsujetsontool ollama <subcommand>` enables local management and interaction with Ollama models from inside a persistent Jetson container.

Supported subcommands:

| Subcommand       | Description                                  |
|------------------|----------------------------------------------|
| `serve`          | Start Ollama REST API server (port 11434)    |
| `run <model>`    | Run specified model in interactive CLI       |
| `list`           | List all installed Ollama models             |
| `pull <model>`   | Download a new model                         |
| `delete <model>` | Remove a model from disk                     |
| `status`         | Check if Ollama server is running            |
| `ask`            | Ask model a prompt via REST API              |


🚀 Commands and Usage

1. Start the Ollama Server
```bash
sjsujetsontool ollama serve
```

Starts the Ollama REST server inside the container, listening on http://localhost:11434.


2. Run a Model in CLI Mode
```bash
$ sjsujetsontool ollama run mistral
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
💬 Launching model 'mistral' in CLI...
pulling manifest 
pulling ff82381e2bea: 100% ▕██████████████████▏ 4.1 GB                         
pulling 43070e2d4e53: 100% ▕██████████████████▏  11 KB                         
pulling 1ff5b64b61b9: 100% ▕██████████████████▏  799 B                         
pulling ed11eda7790d: 100% ▕██████████████████▏   30 B                         
pulling 42347cd80dc8: 100% ▕██████████████████▏  485 B                         
verifying sha256 digest 
writing manifest 
success 
>>> Send a message (/? for help)
```
Launches interactive terminal mode using the mistral model. Enter `\exit` to exit.

3. List Installed Models
```bash
$ sjsujetsontool ollama list
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
📃 Installed models:
NAME               ID              SIZE      MODIFIED           
mistral:latest     3944fe81ec14    4.1 GB    About a minute ago    
llama3.2:latest    a80c4f17acd5    2.0 GB    2 hours ago           
qwen2:latest       dd314f039b9d    4.4 GB    9 days ago            
llama3.2:3b        a80c4f17acd5    2.0 GB    9 days ago 
```
Shows a table of downloaded models and their sizes.

4. Download a New Model
```bash
sjsujetsontool ollama pull llama3
```
Pulls the specified model into the container. Examples include:
	•	phi3
	•	mistral
	•	llama3
	•	qwen:7b

5. Delete a Model
```bash
sjsujetsontool ollama delete mistral
```
Frees up disk space by removing the model.


6. Check Server Status
```bash
sjsujetsontool ollama status
```
Checks if the REST API is running on port 11434.

7. Ask a Prompt (with auto-pull + caching)
```bash
sjsujetsontool ollama ask "What is nvidia jetson orin?"
```
Uses the last used model, or you can specify one:
```bash
sjsujetsontool ollama ask --model mistral "Explain transformers in simple terms."
```
	•	Automatically pulls model if not available
	•	Remembers last used model in .last_ollama_model under workspace/


🧪 Example: Simple Chat Session

Pull and run mistral model
```bash
sjsujetsontool ollama pull mistral
sjsujetsontool ollama run mistral
```
Ask directly via REST
```bash
sjsujetsontool ollama ask --model mistral "Give me a Jetson-themed poem."
```

⸻

🧰 Troubleshooting
	•	Port already in use: Run sudo lsof -i :11434 and kill the process if needed.
	•	Model not found: Use sjsujetsontool ollama pull <model> manually before ask or run.
	•	Server not running: Start with sjsujetsontool ollama serve before using REST API.



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
