import time
import subprocess
import json
import os
import threading
import sys
import argparse
import platform
import requests
import gradio as gr
from gradio.themes.soft import Soft  # Explicit and clean


# Import our modularized system monitoring utilities
# Import system monitoring utilities from local directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import system_monitor # pylint: disable=import-error

# Settings
BACKENDS = ["ollama", "llama.cpp"]
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMACPP_API_URL = "http://localhost:8000/completion"
CHAT_LOG_PATH = "./chat_logs"
os.makedirs(CHAT_LOG_PATH, exist_ok=True)

# Initialize system monitoring
platform_info = system_monitor.init_monitoring()
IS_JETSON = platform_info["is_jetson"]
IS_APPLE_SILICON = platform_info["is_apple_silicon"]
HAS_NVIDIA_GPU = platform_info["has_nvidia_gpu"]

# Start the appropriate monitoring thread and get the system_info dictionary
system_info = system_monitor.start_monitoring()

# Fetch available models
def list_models(backend="ollama"):
    try:
        if backend == "ollama":
            r = requests.get("http://localhost:11434/api/tags")
            if r.ok:
                return [m["name"] for m in r.json().get("models", [])]
        elif backend == "llama.cpp":
            return ["llama.cpp-default"]
    except:
        return []
    return []

# Chat streaming
def chat_with_backend_stream(prompt, model, backend, history=None):
    if history is None:
        history = []
    
    start = time.time()
    response = ""
    tokens = 0
    error_msg = None
    
    # Add user message to history immediately for better UX
    history.append({"role": "user", "content": prompt})
    # Create a placeholder for assistant response
    history.append({"role": "assistant", "content": "Thinking..."})
    
    # Check if model is available
    if backend == "ollama":
        try:
            # First check if Ollama server is running
            try:
                server_check = requests.get("http://localhost:11434/", timeout=2)
                if not server_check.ok:
                    error_msg = "[ERROR] Ollama server is not responding. Make sure it's running."
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
            except requests.RequestException:
                error_msg = "[ERROR] Cannot connect to Ollama server. Make sure it's running with 'ollama serve'."
                history[-1]["content"] = error_msg
                return history, history, "N/A"
            
            # Check if the model exists
            try:
                models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if not models_response.ok:
                    error_msg = f"[ERROR] Failed to get models list: {models_response.status_code} - {models_response.text}"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                models_data = models_response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]
                
                if not available_models:
                    error_msg = "[ERROR] No models available in Ollama. Please pull a model first using 'ollama pull <model>'"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                if model not in available_models:
                    error_msg = f"[ERROR] Selected model '{model}' not found. Available models: {', '.join(available_models)}"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
            except json.JSONDecodeError:
                error_msg = "[ERROR] Invalid response from Ollama API. Check if Ollama is running correctly."
                history[-1]["content"] = error_msg
                return history, history, "N/A"
                
        except requests.RequestException as e:
            error_msg = f"[ERROR] Cannot connect to Ollama API: {str(e)}"
            history[-1]["content"] = error_msg
            return history, history, "N/A"
        except Exception as e:
            error_msg = f"[ERROR] Unexpected error checking models: {str(e)}"
            history[-1]["content"] = error_msg
            return history, history, "N/A"
    
    # Prepare API request based on backend - no streaming
    if backend == "ollama":
        payload = {"model": model, "prompt": prompt, "stream": False}
        url = OLLAMA_API_URL
    else:
        payload = {"prompt": prompt, "n_predict": 128, "stream": False}
        url = LLAMACPP_API_URL

    # Make a non-streaming request
    try:
        with requests.post(url, json=payload, timeout=60) as r:
            if not r.ok:
                error_msg = f"[ERROR] API returned status code {r.status_code}: {r.text}"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
                
            try:
                data = r.json()
                # Extract response based on backend
                if backend == "ollama":
                    response = data.get("response", "")
                else:  # llama.cpp
                    response = data.get("content", "")
                
                # Update the assistant's message in history
                history[-1]["content"] = response
                # Calculate tokens
                tokens = len(response.split())
            except json.JSONDecodeError:
                error_msg = "[ERROR] Invalid JSON response from API"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
            except Exception as e:
                error_msg = f"[ERROR] Failed to process response: {str(e)}"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
    except requests.RequestException as e:
        error_msg = f"[ERROR] Connection error: {str(e)}"
        history[-1]["content"] = error_msg
        return history, history, "N/A"
    except Exception as e:
        error_msg = f"[ERROR] Unexpected error: {str(e)}"
        history[-1]["content"] = error_msg
        return history, history, "N/A"
    
    # If there was no response
    if not response:
        # Provide a more detailed error message
        if backend == "ollama":
            history[-1]["content"] = "[ERROR] No response received from model. This could be due to:\n\n" \
                                    "1. The model is still loading (first run can take time)\n" \
                                    "2. Insufficient system resources (memory/GPU)\n" \
                                    "3. Ollama server issue - try restarting with 'ollama serve'\n\n" \
                                    f"Selected model: {model}"
        else:
            history[-1]["content"] = "[ERROR] No response received from model. Check if the model is properly loaded."
        return history, history, "N/A"
    
    elapsed = time.time() - start
    tps = f"{tokens / elapsed:.2f} tokens/sec" if tokens > 0 and elapsed > 0 else "N/A"
    
    return history, history, tps

# Export chat history
def export_chat(history):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.md")
    json_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.json")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    with open(md_path, "w") as md_file:
        # Process messages in pairs (user, assistant)
        i = 0
        while i < len(history):
            user_msg = history[i] if i < len(history) else None
            assistant_msg = history[i+1] if i+1 < len(history) else None
            
            if user_msg and user_msg.get("role") == "user":
                md_file.write(f"**User:** {user_msg.get('content', '')}\n\n")
            
            if assistant_msg and assistant_msg.get("role") == "assistant":
                md_file.write(f"**Assistant:** {assistant_msg.get('content', '')}\n\n---\n")
            
            i += 2
    
    with open(json_path, "w") as json_file:
        json.dump(history, json_file, indent=2)
    
    return f"✅ Exported:\n- {md_path}\n- {json_path}"

# UI
with gr.Blocks(title="Ollama Chat UI", theme=Soft(primary_hue="indigo")) as demo:
    # Get platform name from system_monitor
    platform_name = system_monitor.get_platform_name()
    
    gr.Markdown(f"## 🧠 Ollama / llama.cpp Chat UI ({platform_name})")
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                backend_select = gr.Radio(
                    BACKENDS, 
                    value="ollama", 
                    label="Backend",
                    interactive=True,
                    scale=1
                )
                model_dropdown = gr.Dropdown(
                    choices=list_models("ollama"), 
                    label="Model",
                    interactive=True,
                    scale=2
                )
            
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                type="messages",
                height=600,  # Increased height for better visibility
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png"),
                bubble_full_width=False,
                show_label=True
            )
            
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Ask something...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    scale=5
                )
                send = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear Chat", scale=1)
                refresh_btn = gr.Button("🔄 Refresh Models", scale=1)
                export_btn = gr.Button("💾 Export Chat", scale=1)
        
        # Right sidebar for system info
        with gr.Column(scale=1):
            dashboard = gr.Textbox(
                label="Live System Info", 
                lines=15, 
                max_lines=20,
                interactive=False, 
                value=system_info["text"],
                show_label=True
            )
            token_speed = gr.Textbox(label="Token Speed", interactive=False)
            
            # Platform-specific info
            if IS_JETSON:
                gr.Markdown("### Jetson Device")
                gr.Markdown("Optimized for NVIDIA Jetson hardware")
                gr.Markdown("*For best performance, ensure jetson-stats is installed*")
            elif IS_APPLE_SILICON:
                gr.Markdown("### Apple Silicon")
                gr.Markdown("Optimized for Apple M1/M2/M3 hardware")
                gr.Markdown("*For detailed GPU stats, run 'sudo powermetrics --samplers gpu_power' in a terminal*")
            elif HAS_NVIDIA_GPU:
                gr.Markdown("### NVIDIA GPU")
                gr.Markdown("Optimized for NVIDIA graphics cards")
                gr.Markdown("*Using GPUtil or nvidia-smi for monitoring*")
            else:
                gr.Markdown("### CPU Mode")
                gr.Markdown("Running in CPU-only mode")
                gr.Markdown("*For better performance, consider using a GPU*")
    
    # Hidden state
    state = gr.State([])
    
    # Functions
    def refresh_dashboard():
        return system_info["text"]
    
    def update_model_list(backend):
        # In newer versions of Gradio, we return a list directly instead of using .update()
        models = list_models(backend)
        return gr.Dropdown(choices=models)
    
    def export_trigger(history):
        result = export_chat(history)
        gr.Info("Chat history exported successfully")
        return result
    
    def clear_chat():
        return [], []
    
    # Event handlers
    send.click(
        fn=chat_with_backend_stream,
        inputs=[prompt, model_dropdown, backend_select, state],
        outputs=[chatbot, state, token_speed],
        api_name="chat"
    )
    
    # Also trigger on Enter key
    prompt.submit(
        fn=chat_with_backend_stream,
        inputs=[prompt, model_dropdown, backend_select, state],
        outputs=[chatbot, state, token_speed]
    )
    
    refresh_btn.click(
        fn=update_model_list, 
        inputs=[backend_select], 
        outputs=[model_dropdown]
    )
    
    export_btn.click(
        fn=export_trigger, 
        inputs=[state], 
        outputs=[dashboard]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, state]
    )
    
    # Real-time dashboard refresh every 2 seconds
    timer = gr.Timer(2)
    timer.tick(fn=refresh_dashboard, outputs=[dashboard])

if __name__ == "__main__":
    # Performance optimizations based on platform
    if IS_JETSON:
        print("Running on Jetson device - applying optimizations")
        # Reduce memory usage by limiting thread pool
        try:
            import torch
            if torch.cuda.is_available():
                # Set lower memory usage for CUDA if available
                #caps the memory usage to 70% of total GPU memory to avoid out-of-memory crashes and allow other GPU tasks to run.
                torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except ImportError:
            print("PyTorch not available, skipping CUDA optimizations")
        
        # Set environment variables for better performance
        #Gradio by default collects usage analytics.
	    #On a resource-constrained Jetson device, turning this off can save bandwidth and slightly improve startup performance.
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        #PCI_BUS_ID ensures that CUDA device enumeration follows the PCI bus order, which is reliable for multi-GPU setups.
	    #On Jetson devices, which usually only have one GPU, this line is not necessary
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Use PCI bus ID for CUDA devices
    
    elif IS_APPLE_SILICON:
        print("Running on Apple Silicon - applying optimizations")
        # Set environment variables for better performance on Apple Silicon
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        
        # Check if PyTorch is available and configured for MPS (Metal Performance Shaders)
        try:
            import torch
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) is available for GPU acceleration")
                # No need to set memory fraction as Apple Silicon manages memory differently
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for operations not supported by MPS
        except ImportError:
            print("PyTorch not available, skipping MPS optimizations")
        except AttributeError:
            print("PyTorch available but MPS not supported in this version")
    
    elif HAS_NVIDIA_GPU:
        print("Running on system with NVIDIA GPU - applying optimizations")
        # Optimize for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                # Set reasonable memory usage for CUDA
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except ImportError:
            print("PyTorch not available, skipping CUDA optimizations")
        
        # Set environment variables for better performance
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Use PCI bus ID for CUDA devices
    
    else:
        print("Running on CPU-only system")
        # Set environment variables for better performance on CPU
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
    
    # Check if we need to start ollama server
    parser = argparse.ArgumentParser(description="Ollama Gradio UI")
    parser.add_argument("--start-ollama", action="store_true", help="Start ollama server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio server on")
    args = parser.parse_args()
    
    if args.start_ollama:
        print("Starting ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Waiting for ollama server to start...")
        time.sleep(2)  # Give ollama time to start
    
    # Launch the Gradio app
    print(f"Starting Gradio UI on port {args.port}...")
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0", 
        server_port=args.port,
        share=False,
        show_error=True
        # Removed favicon_path 
    )
    
    # Command for Docker/container environments:
    # $EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /workspace/scripts/ollama_gradio_ui.py"