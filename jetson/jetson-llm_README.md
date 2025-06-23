#🎯 Ollama on Jetson with GPU Support

Ollama is a popular open-source tool that allows users to easily run a large language models (LLMs) locally on their own computer, serving as an accessible entry point to LLMs for many.

It now offers out-of-the-box support for the Jetson platform with CUDA support, enabling Jetson users to seamlessly install Ollama with a single command and start using it immediately.

Ollama uses llama.cpp for inference, which various API benchmarks and comparisons are provided for on the Llava page. It gets roughly half of peak performance versus the faster APIs like NanoLLM , but is generally considered fast enough for text chat.

Supported hardware: Orin Nano (8GB), Orin NX (16GB), AGX Orin (32/64GB)
Supported JetPack versions: 5.x (L4T r35.x) and 6.x (L4T r36.x)

🧩 Installation Methods

✅ 1. Native Install (Recommended)
```bash
sjsujetson@sjsujetson-01:~/Developer$ curl -fsSL https://ollama.com/install.sh | sh
>>> Installing ollama to /usr/local
[sudo] password for sjsujetson: 
>>> Downloading Linux arm64 bundle
######################################################################## 100.0%
>>> Downloading JetPack 6 components
######################################################################## 100.0%
>>> Creating ollama user...
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
>>> NVIDIA JetPack ready.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
```

Run llama model:
```bash
sjsujetson@sjsujetson-01:~/Developer$ ollama run llama3.2:3b
```
End a chat session, just type: `/exit`
Reset the chat context (clear memory): `/reset`

Pull a model (download from Ollama Hub):
```bash
ollama pull phi3
ollama pull deepseek-coder
sjsujetson@sjsujetson-01:~/Developer$ ollama pull qwen2
ollama run qwen2
sjsujetson@sjsujetson-01:~/Developer$ ollama show qwen2
  Model
    architecture        qwen2    
    parameters          7.6B     
    context length      32768    
    embedding length    3584     
    quantization        Q4_0
```

List all available local models:
```bash
ollama list
```

Docker image:
```bash
sudo docker build --network=host -t jetson-llm .
```