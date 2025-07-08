import gradio as gr
import requests
import time
import subprocess
import json
import os
import threading

# Settings
BACKENDS = ["ollama", "llama.cpp"]
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMACPP_API_URL = "http://localhost:8000/completion"
CHAT_LOG_PATH = "/workspace/chat_logs"
os.makedirs(CHAT_LOG_PATH, exist_ok=True)
system_info = {"text": "Initializing system monitor..."}

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

# Background system monitor
def poll_system_usage():
    while True:
        try:
            gpu = subprocess.check_output("tegrastats --interval 1000 --count 1", shell=True).decode()
        except:
            gpu = "tegrastats not available"
        try:
            cpu = subprocess.check_output("top -b -n1 | head -n 5", shell=True).decode()
        except:
            cpu = "top not available"
        system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n🎮 GPU Info:\n{gpu}"
        time.sleep(2)

threading.Thread(target=poll_system_usage, daemon=True).start()

# Chat streaming
def chat_with_backend_stream(prompt, model, backend, history=[]):
    start = time.time()
    response = ""
    tokens = 0
    if backend == "ollama":
        payload = {"model": model, "prompt": prompt, "stream": True}
        url = OLLAMA_API_URL
    else:
        payload = {"prompt": prompt, "n_predict": 128, "stream": True}
        url = LLAMACPP_API_URL

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("response") or data.get("content") or ""
                        response += chunk
                    except:
                        continue
        tokens = len(response.split())
    except Exception as e:
        response = f"[ERROR] {e}"

    elapsed = time.time() - start
    tps = f"{tokens / elapsed:.2f} tokens/sec" if tokens > 0 else "N/A"

    #history.append((prompt, response))
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": response})
    return history, history, tps

# Export chat history
def export_chat(history):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.md")
    json_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.json")
    with open(md_path, "w") as md_file:
        for user, bot in history:
            md_file.write(f"**User:** {user}\n\n**Assistant:** {bot}\n\n---\n")
    with open(json_path, "w") as json_file:
        json.dump(history, json_file, indent=2)
    return f"✅ Exported:\n- {md_path}\n- {json_path}"

# def get_dashboard_text():
#     return system_info["text"]

# UI
with gr.Blocks(title="Ollama Chat UI") as demo:
    gr.Markdown("## 🧠 Ollama / llama.cpp Chat UI (Jetson + Gradio)")

    with gr.Row():
        backend_select = gr.Radio(BACKENDS, value="ollama", label="Backend")
        model_dropdown = gr.Dropdown(choices=list_models("ollama"), label="Model")
        refresh_btn = gr.Button("🔄 Refresh Models")
        export_btn = gr.Button("💾 Export Chat")

    with gr.Row():
        dashboard = gr.Textbox(label="Live System Info", lines=10, interactive=False, value=system_info["text"])
        token_speed = gr.Textbox(label="Token Speed", interactive=False)

    chatbot = gr.Chatbot(label="Chat History",type="messages")
    prompt = gr.Textbox(label="Prompt", placeholder="Ask something...")
    send = gr.Button("Send")
    state = gr.State([])

    def refresh_dashboard():
        return system_info["text"]

    def update_model_list(backend):
        return gr.Dropdown.update(choices=list_models(backend))

    def export_trigger(history):
        return export_chat(history)

    send.click(fn=chat_with_backend_stream,
               inputs=[prompt, model_dropdown, backend_select, state],
               outputs=[chatbot, state, token_speed])

    refresh_btn.click(fn=update_model_list, inputs=[backend_select], outputs=[model_dropdown])
    export_btn.click(fn=export_trigger, inputs=[state], outputs=[dashboard])

    # Dashboard updater
    #gr.Textbox.update(value=system_info["text"])
    gr.Textbox(label="System Info", lines=10, value=system_info["text"], interactive=False, show_label=True)
    gr.Timer(2.0, refresh_dashboard, every=2).output(dashboard)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
    #$EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /workspace/scripts/ollama_gradio_ui.py"