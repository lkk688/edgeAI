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
    try:
        import requests
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3",
            "prompt": "Hello",
            "stream": False
        }, timeout=5)
        if r.ok:
            return f"Ollama OK: {r.json().get('response', '')}"
        return f"Ollama error: {r.text}"
    except Exception as e:
        return f"Ollama not available: {e}"

def check_llama_cpp():
    llama_bin = "/Developer/llama.cpp/build_cuda/bin/llama-server"
    return "Found" if os.path.exists(llama_bin) else "Not found"

def test_torch_cnn():
    import torch.nn as nn
    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
            self.fc1 = nn.Linear(1440, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(-1, 1440)
            x = self.fc1(x)
            return F.log_softmax(x, dim=1)

    model = Net()
    x = torch.randn(1, 1, 12, 12)
    y = model(x)
    return f"CNN output: {y.shape}"

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