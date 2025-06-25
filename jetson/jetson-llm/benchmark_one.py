import argparse
import time
import json
import requests
from pathlib import Path
from llama_cpp import Llama
from langchain.llms import LlamaCpp

# Constants
PROMPT = "Explain the importance of edge AI in smart cities."
MAX_TOKENS = 100
RESULTS_FILE = "benchmark_results.json"

# Model info
MODEL_MAP = {
    "mistral": {
        "path": "./models/mistral.gguf",
        "ollama": "mistral"
    },
    "qwen": {
        "path": "./models/qwen.gguf",
        "ollama": "qwen"
    },
    "llama3": {
        "path": "./models/llama3.gguf",
        "ollama": "llama3"
    },
    "deepseek": {
        "path": "./models/deepseek.gguf",
        "ollama": "deepseek"
    }
}

def benchmark_llama_cpp_python(name, path):
    try:
        llm = Llama(
            model_path=path,
            n_ctx=1024,
            n_gpu_layers=20,
            use_mlock=True,
            use_mmap=True
        )
        start = time.time()
        out = llm(PROMPT, max_tokens=MAX_TOKENS)
        duration = time.time() - start
        tokens = out["usage"]["completion_tokens"]
        speed = tokens / duration
        return {"method": "llama-cpp-python", "model": name, "tokens": tokens, "time": duration, "speed": speed}
    except Exception as e:
        return {"method": "llama-cpp-python", "model": name, "error": str(e)}

def benchmark_langchain(name, path):
    try:
        llm = LlamaCpp(
            model_path=path,
            n_ctx=1024,
            n_gpu_layers=20,
            temperature=0.7,
            verbose=False
        )
        start = time.time()
        output = llm.invoke(PROMPT)
        duration = time.time() - start
        tokens = len(output.split())
        speed = tokens / duration
        return {"method": "langchain-llama-cpp", "model": name, "tokens": tokens, "time": duration, "speed": speed}
    except Exception as e:
        return {"method": "langchain-llama-cpp", "model": name, "error": str(e)}

def benchmark_llama_server(name):
    try:
        start = time.time()
        r = requests.post("http://localhost:8000/completion", json={
            "prompt": PROMPT,
            "n_predict": MAX_TOKENS
        })
        duration = time.time() - start
        text = r.json()["content"]
        tokens = len(text.split())
        speed = tokens / duration
        return {"method": "llama.cpp REST", "model": name, "tokens": tokens, "time": duration, "speed": speed}
    except Exception as e:
        return {"method": "llama.cpp REST", "model": name, "error": str(e)}

def benchmark_ollama(name, ollama_name):
    try:
        start = time.time()
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": ollama_name,
            "prompt": PROMPT,
            "stream": False
        })
        duration = time.time() - start
        response = r.json()["response"]
        tokens = len(response.split())
        speed = tokens / duration
        return {"method": "ollama", "model": name, "tokens": tokens, "time": duration, "speed": speed}
    except Exception as e:
        return {"method": "ollama", "model": name, "error": str(e)}

def append_result(result):
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("\n✅ Result appended to benchmark_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark one LLM backend")
    parser.add_argument("--method", type=str, required=True,
                        choices=["llama-cpp-python", "langchain", "llama-server", "ollama"],
                        help="Backend to benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_MAP.keys(),
                        help="Model to test")
    args = parser.parse_args()

    name = args.model
    method = args.method
    info = MODEL_MAP[name]

    print(f"\n🚀 Benchmarking: Method={method}, Model={name}")

    if method == "llama-cpp-python":
        result = benchmark_llama_cpp_python(name, info["path"])
    elif method == "langchain":
        result = benchmark_langchain(name, info["path"])
    elif method == "llama-server":
        result = benchmark_llama_server(name)
    elif method == "ollama":
        result = benchmark_ollama(name, info["ollama"])
    else:
        raise ValueError("Invalid method")

    # Print result
    print("\n📊 Benchmark Result:")
    print(json.dumps(result, indent=2))

    append_result(result)