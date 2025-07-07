import torch
import transformers
import subprocess
import platform
import os
import sys

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"Error: {e}"

def check_ollama():
    path = run_cmd("which ollama")
    if "ollama" not in path:
        return "❌ Ollama not installed (binary not found in PATH)"
    version = run_cmd("ollama --version")
    return f"✅ Ollama installed: {version}"
    # try:
    #     import requests
    #     r = requests.post("http://localhost:11434/api/generate", json={
    #         "model": "phi3",
    #         "prompt": "Hello",
    #         "stream": False
    #     }, timeout=5)
    #     if r.ok:
    #         return f"Ollama OK: {r.json().get('response', '')}"
    #     return f"Ollama error: {r.text}"
    # except Exception as e:
    #     return f"Ollama not available: {e}"

def check_llama_cpp():
    llama_bin = "/Developer/llama.cpp/build_cuda/bin/llama-server"
    return "Found" if os.path.exists(llama_bin) else "Not found"

def test_torch_cnn():
    from torchvision import models
    import torch.nn.functional as F

    model = models.resnet18(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    return f"ResNet18 output shape: {y.shape}, top-5 logits: {y[0].topk(5).values.tolist()}"

def test_llm():
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=10)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("📦 Python:", sys.version)
    print("🧠 Torch:", torch.__version__)
    print("⚙️  CUDA available:", torch.cuda.is_available())
    print("🖥️  CUDA version:", run_cmd("nvcc --version | grep release") or "Unavailable")
    print("📚 Transformers:", transformers.__version__)
    print("🧬 HuggingFace hub:", run_cmd("pip show huggingface-hub | grep Version") or "Unknown")
    print("💡 Platform:", platform.platform())
    print("🔍 Ollama:", check_ollama())
    print("🔍 llama.cpp:", check_llama_cpp())
    print("🧪 Torch CNN test:", test_torch_cnn())
    print("🧪 LLM test:", test_llm())