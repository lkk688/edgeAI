# 🚀 Large Language Models on Jetson
**Author:** Dr. Kaikai Liu, Ph.D.
**Position:** Associate Professor, Computer Engineering
**Institution:** San Jose State University
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

LLMs (Large Language Models) are transformer-based models trained on vast datasets to understand and generate human-like text. This chapter shows how to **run, serve, and chat with modern LLMs on the Jetson Orin Nano** using our `sjsujetsontool` utility, how to call **NVIDIA's cloud API**, and how to reach a **shared GPU server** over our Headscale network.

> 🧩 **Hands-on code lives in [`jetson/jetson-llm/`](../../jetson/jetson-llm/)** — the tutorial keeps commands short and asks you to *run* those scripts rather than paste long code.

## 💬 Common Use Cases
* Chatbots and virtual assistants · Code generation · Summarization · Translation · On-device RAG and agents

---

## 🛠️ Running LLMs on Jetson

Running LLMs on the Orin Nano (8 GB shared CPU/GPU memory) means caring about **memory, quantization, and inference speed**. We use three complementary options:

| Where the model runs | Command | Good for |
|---|---|---|
| **On this Jetson** (llama.cpp, GPU) | `sjsujetsontool llama` / `chat --local` | private, offline, small–mid models |
| **NVIDIA Build API** (cloud) | `sjsujetsontool setup-nvapi` → `chat --nvidia` | large models, no local VRAM cost |
| **Shared GPU server** (RTX board over Headscale) | `chat --server` | a big model shared by the whole class |

### 🎯 Backend comparison

| Backend | Memory Efficiency | Speed | Ease of Use | CUDA | Best For |
|---------|------------------|-------|-------------|------|----------|
| **llama.cpp** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Production inference (what we use) |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Quick deployment |
| **llama-cpp-python** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Python integration |
| **TensorRT-LLM** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | Maximum performance |
| **vLLM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Batch inference |

### 🧠 Theoretical foundations

**Quantization** reduces weight precision to fit larger models in less memory:
- **INT8** ≈ 4× smaller · **INT4** ≈ 8× smaller · **GPTQ/AWQ** preserve quality post-training.

**Memory strategies:** KV-cache management, paged attention, model sharding.
**Inference tricks:** speculative decoding, continuous batching, flash attention, kernel fusion.

> The `Q4_K_M` / `Q4_K_S` GGUF quantization is the sweet spot on Orin Nano — near-FP16 quality at ~4× smaller size.

### 📦 Recommended models (2025+)

We switched away from older 7B models (Mistral, LLaMA-2) to newer, smaller, stronger models — see the catalog at [jetson-ai-lab.com/models](https://www.jetson-ai-lab.com/models/):

| Model | HF GGUF repo (`-hf` ref) | Size (Q4) | Notes |
|---|---|---|---|
| **Gemma 4 E2B** (default) | `unsloth/gemma-4-E2B-it-GGUF:Q4_K_S` | ~2 GB | Fast, strong instruction-following |
| **Nemotron-3 Nano 4B** | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M` | ~2.5 GB | NVIDIA reasoning/chat |
| **Qwen3 8B** | `unsloth/Qwen3-8B-GGUF:Q4_K_M` | ~5 GB | Multilingual, larger |

`llama.cpp` can stream these straight from Hugging Face with `-hf` — **no manual download needed**. For offline use, [`jetson/jetson-llm/modeldownload.sh`](../../jetson/jetson-llm/modeldownload.sh) fetches them to local disk.

---

## ① Serve & query a model with llama.cpp

`sjsujetsontool` builds CUDA-accelerated `llama-server`/`llama-cli` inside the container and serves **Gemma 4 E2B** on port **8080**:

```bash
sjsujetsontool llama          # serves unsloth/gemma-4-E2B-it-GGUF on 0.0.0.0:8080 (-ngl 99, all layers on GPU)
```
`llama-server` is an **OpenAI-compatible** HTTP server. A basic web UI is at `http://localhost:8080`; the chat endpoint is `http://localhost:8080/v1/chat/completions`.

**Quick one-shot CLI (no server):**
```bash
sjsujetsontool llama-cli -p "Explain what an NVIDIA Jetson Orin Nano is."
```

**Serve a different model** (e.g. NVIDIA Nemotron-3 Nano 4B) — drop into the container shell and point `llama-server` at any `-hf` ref:
```bash
sjsujetsontool shell
# inside the container:
llama-server -hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 -ngl 99
```

### Query the server
With `curl`:
```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain what is NVIDIA Jetson?"}]}'
```
Or run the client script (prints the reply + tokens/sec; pure stdlib, works anywhere):
```bash
python3 jetson/jetson-llm/llama_client.py -p "Explain what is NVIDIA Jetson?"
# reasoning models (Qwen3, Nemotron): add --no-think for a direct answer
```
See [`llama_client.py`](../../jetson/jetson-llm/llama_client.py) — it targets any OpenAI-compatible URL (`--url`, `--api-key`, `--model`).

> **External access:** `sjsujetsontool llama` already binds `0.0.0.0`, so other machines can reach `http://<jetson-ip>:8080`. If a firewall is on: `sudo ufw allow 8080/tcp`.

### 🖼️ Vision: ask about an image

Both **Gemma 4 E2B** and **Qwen3.5** are *multimodal* — they accept images. `llama-server` does this when it also loads the model's **mmproj** (multimodal projector):

- **On the Jetson:** `sjsujetsontool llama` serves Gemma 4 E2B via `-hf`, which **auto-downloads and loads the mmproj** — vision is on by default, no extra step.
- **On a `gputool` server (Qwen3.5):** download the projector once and it's auto-detected — see [gputool guide → vision](00c_gputool_guide.md).

Send an image with the OpenAI vision format (a base64 `image_url`). Use the helper script:
```bash
# uses a generated test image if you omit --image
python3 jetson/jetson-llm/vision_test.py --image my_photo.jpg -p "What is in this image?"
# → "A yellow circle on a blue square background."   (built-in test image)
```
[`vision_test.py`](../../jetson/jetson-llm/vision_test.py) targets any vision endpoint (`--url`, `--api-key`). The same works against the shared server: `--url https://llm.forgengi.org/node05/v1 --api-key <token>`.

> The CUDA build uses `LLAMA_CURL=OFF`, so pass images as **base64 data URIs** (the script does this) rather than remote URLs.

---

## ② Interactive chat — one client, three backends

`sjsujetsontool chat` is a streaming terminal client (Rich UI) that lets you pick **where the model runs**:

```bash
sjsujetsontool chat            # menu: 1) local llama.cpp  2) NVIDIA API  3) our LLM server
sjsujetsontool chat --local    # this Jetson (:8080)
```
In-chat commands: `/help /exit /server /save /reset /system /think /temp /set /preset /config`. Full reference: **[sjsujetsontool guide → `chat`](00_sjsujetsontool_guide.md#-sjsujetsontool-chat--one-chat-client-three-backends)**.

**Experiment with generation settings.** Temperature and "thinking" dramatically change a reasoning model's output. Try the [Unsloth-recommended Qwen3.5 presets](https://unsloth.ai/docs/models/qwen3.5) without leaving the chat:
```text
/preset thinking     # creative/general: temp 1.0, top_p 0.95, top_k 20, thinking on
/preset coding       # precise code:     temp 0.6, top_p 0.95, top_k 20, thinking on
/preset instruct     # fast answers:     temp 0.7, top_p 0.8,  top_k 20, thinking off
/temp 0.3            # or tune one knob; /set top_p 0.9 ; /config to view
```
Lower temperature → more deterministic/precise; higher → more diverse/creative. Turn `/think on` for step-by-step reasoning, `/think off` for direct answers.

---

## ③ NVIDIA Build API backend (cloud)

For models too large for the Jetson, use NVIDIA's free, OpenAI-compatible [Build API](https://build.nvidia.com).

**1. Get & store an API key** (one time):
```bash
sjsujetsontool setup-nvapi     # paste your key from build.nvidia.com; saved to .env.local + tested
```

**2. Chat with a cloud model:**
```bash
sjsujetsontool chat --nvidia   # then choose a Nemotron model (Nano 8B … Ultra 253B)
```

**3. Or call it from a script:**
```bash
python3 jetson/jetson-llm/nvidia_api_client.py -p "Explain speculative decoding."
python3 jetson/jetson-llm/nvidia_api_client.py -m nvidia/llama-3.3-nemotron-super-49b-v1 -p "Hi"
```
[`nvidia_api_client.py`](../../jetson/jetson-llm/nvidia_api_client.py) reads the key from `$NVIDIA_API_KEY` or `.env.local`. See also the legacy **[`nv-chat` tutorial](00_sjsujetsontool_guide.md#-sjsujetsontool-nv-chat)**.

---

## ④ Shared GPU server over Headscale (no IPs)

A powerful node (e.g. an RTX 5080 board) can serve a big model that **every Jetson shares by name over HTTPS**:

```bash
sjsujetsontool chat --server   # base URL e.g. https://llm.forgengi.org/node05/v1
# or directly:
python3 jetson/jetson-llm/llama_client.py \
  --url https://llm.forgengi.org/node05/v1 --api-key <token> --no-think -p "Hello!"
```
How the gateway works (Headscale + nginx + a friendly `llm.forgengi.org/<node>` name): **[gputool guide → Share the server by name](00c_gputool_guide.md)**.

---

## 🐍 llama-cpp-python (Python bindings)

[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) gives a Python/OpenAI-style API and is prebuilt with CUDA inside our container. Minimal example:
```python
from llama_cpp import Llama
llm = Llama(model_path="/models/gemma-4-E2B-it-Q4_K_S.gguf", n_ctx=2048, n_gpu_layers=99)
print(llm("Explain what is NVIDIA Jetson?", max_tokens=128)["choices"][0]["text"])
```
A complete, timed example is in [`llama_cpp_pythontest.py`](../../jetson/jetson-llm/llama_cpp_pythontest.py):
```bash
python3 jetson/jetson-llm/llama_cpp_pythontest.py
```

---

## 📊 Benchmarking

Measure tokens/sec and time-to-first-token against any served endpoint:
```bash
# local Gemma 4 E2B
python3 jetson/jetson-llm/benchmark_llm.py
# the shared server
python3 jetson/jetson-llm/benchmark_llm.py --url https://llm.forgengi.org/node05/v1 --api-key <token>
```
[`benchmark_llm.py`](../../jetson/jetson-llm/benchmark_llm.py) reports **prefill vs generation** rates (from llama.cpp `timings`) averaged over several prompts and runs.

---

## 🧰 Memory & performance utilities

The board-specific tuning, memory cleanup, and a resource-monitoring context manager are collected in [`jetson_llm_utils.py`](../../jetson/jetson-llm/jetson_llm_utils.py):
```bash
python3 jetson/jetson-llm/jetson_llm_utils.py     # prints env + recommended config for this board
```
```python
from jetson_llm_utils import optimize_memory, system_monitor, get_jetson_config
cfg = get_jetson_config()                 # {'n_gpu_layers': 99, 'n_ctx': 4096, ...}
with system_monitor("generate"):          # times the block, reports CPU/RAM/GPU
    ...                                    # your inference call
optimize_memory()                          # free GC + CUDA cache between runs
```

---

## 🔄 GGUF model format

**GGUF** is llama.cpp's single-file format: memory-mapped (no full load into RAM), embedded metadata, and multiple quantization levels.

| Format | ~Size (8B) | Quality | Speed | Best For |
|--------|-----------|---------|-------|----------|
| FP16   | 16 GB | 100% | 1× | Maximum quality (too big for Orin Nano) |
| Q6_K   | 6.6 GB | 99% | 1.5× | Best quality that fits |
| Q4_K_M | 4.9 GB | 96% | 2× | **Recommended balance** |
| Q4_K_S | 4.5 GB | 95% | 2.1× | Default for Gemma 4 E2B |
| Q3_K_M | 4.0 GB | 90% | 2.5× | Memory constrained |

On Orin Nano, prefer **Q4_K_M / Q4_K_S**; for sub-4B models you can afford Q6_K.

---

## ⚠️ Common issues & tips

| Symptom | Fix |
|---|---|
| **CUDA out of memory** | use a smaller/quantized model, lower `n_ctx`, ensure `-ngl 99` fits; call `optimize_memory()` between runs |
| **Empty reply from a reasoning model** | Qwen3/Nemotron "think" first — raise `--max-tokens` or pass `--no-think` (sends `enable_thinking:false`) |
| **`llama-server` not reachable from another host** | it binds `0.0.0.0` already; open the port: `sudo ufw allow 8080/tcp` |
| **NVIDIA `403 Forbidden`** | model unavailable on the free plan or quota exhausted — pick another model or re-run `sjsujetsontool setup-nvapi` |
| **Slow first token** | first call loads the model + warms the KV cache; subsequent calls are faster |

---

## 🧭 Summary

- **Local**: `sjsujetsontool llama` (Gemma 4 E2B) → query with `llama_client.py` or `chat --local`.
- **Cloud**: `sjsujetsontool setup-nvapi` → `chat --nvidia` / `nvidia_api_client.py`.
- **Shared**: `chat --server` → `https://llm.forgengi.org/<node>` over Headscale.
- **Measure & tune**: `benchmark_llm.py`, `jetson_llm_utils.py`.

All hands-on code is in [`jetson/jetson-llm/`](../../jetson/jetson-llm/); the tooling is documented in [`00_sjsujetsontool_guide.md`](00_sjsujetsontool_guide.md) and [`00c_gputool_guide.md`](00c_gputool_guide.md).
