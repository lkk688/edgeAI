import os
import time
from llama_cpp import Llama

# GGUF model: pass a local path, or let llama-cpp-python pull from Hugging Face.
# Defaults to Gemma 4 E2B (the model `sjsujetsontool llama` serves).
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")  # e.g. /models/gemma-4-E2B-it-Q4_K_S.gguf
HF_REPO = os.environ.get("LLAMA_HF_REPO", "unsloth/gemma-4-E2B-it-GGUF")
HF_FILE = os.environ.get("LLAMA_HF_FILE", "*Q4_K_S.gguf")

if MODEL_PATH:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=99,
                use_mlock=True, use_mmap=True)
else:
    # Downloads (and caches) the GGUF from Hugging Face on first run.
    llm = Llama.from_pretrained(repo_id=HF_REPO, filename=HF_FILE,
                                n_ctx=2048, n_gpu_layers=99, use_mmap=True)
#To adjust n_gpu_layers in llama-cpp-python, 
# you pass it as an argument when initializing the Llama object. This controls how many layers of the model run on the GPU, while the rest run on CPU.
#Applies to quantized models only.
# Too high? You’ll get an out-of-memory (OOM) or crash.

# Input prompt
prompt = "Explain what is Nvidia jetson?"

# Measure inference time
start_time = time.time()

# Run inference
output = llm(
    prompt,
    max_tokens=128,
    temperature=0.7,
    echo=False
)

end_time = time.time()

# Print the response
print("Generated Text:\n", output["choices"][0]["text"].strip())

# Measure and print token speed
tokens_generated = output["usage"]["completion_tokens"]
duration = end_time - start_time
tokens_per_sec = tokens_generated / duration

print(f"\n🕒 Inference time: {duration:.2f} seconds")
print(f"🔢 Tokens generated: {tokens_generated}")
print(f"⚡ Tokens/sec: {tokens_per_sec:.2f}")