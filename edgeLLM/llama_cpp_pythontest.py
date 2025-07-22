# llama_cpp_pythontest.py
import time
from llama_cpp import Llama

# Path to your GGUF model
MODEL_PATH = "/models/mistral.gguf"  # Change if your model is elsewhere

# Create Llama instance with GPU support if built with CUDA
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,      # -1 means fully offloaded to GPU (if available)
    #n_gpu_layers=20, # adjust based on Jetson memory, for 8GB, 20-28
    n_ctx=2048,           # Context length
    verbose=True,
    use_mlock=True, # lock model in memory (optional)
    use_mmap=True,  # memory map file (optional, faster loading)
)
#n_gpu_layers controls how many layers of the model run on the GPU, while the rest run on CPU
# Prompt to test
prompt = "Explain what Nvidia Jetson is."

start_time = time.time()

# Generate a response
#output = llm(prompt, max_tokens=128, stop=["</s>"])
output = llm(prompt, max_tokens=128, temperature=0.7, echo= False)

end_time = time.time()
elapsed_time = end_time - start_time

# Print the response
print("Generated Text:\n", output["choices"][0]["text"].strip())

# Measure and print token speed
tokens_generated = output["usage"]["completion_tokens"]
tokens_per_sec = tokens_generated / elapsed_time

print(f"\n🕒 Inference time: {elapsed_time:.2f} seconds")
print(f"🔢 Tokens generated: {tokens_generated}")
print(f"⚡ Tokens/sec: {tokens_per_sec:.2f}")