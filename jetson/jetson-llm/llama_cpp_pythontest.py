import time
from llama_cpp import Llama

# Path to your quantized GGUF model (e.g., Mistral 7B Q4_K_M)
MODEL_PATH = "./models/mistral.gguf"

# Initialize the Llama model with GPU acceleration
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_gpu_layers=20,       # Adjust based on Jetson memory (8GB: try 20–28)
    use_mlock=True,        # Lock model in memory (optional)
    use_mmap=True          # Use memory-mapped file (faster loading)
)
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