import gradio as gr
import requests
import json

# Ollama uses a different API format than OpenAI
# For direct Ollama API (used in the main UI)
OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_API_BASE}/api/generate"

# Create a simple chat interface that uses the Ollama API directly
#message: a str representing the user's most recent message.
#history: a list of openai-style dictionaries with role and content keys, representing the previous conversation history.
#the history could look like this:
# [
#     {"role": "user", "content": "What is the capital of France?"},
#     {"role": "assistant", "content": "Paris"}
# ]
def chat_with_ollama(message, history):
    # Format the conversation history for Ollama
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    # Check if Ollama is running
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llama3.2:latest",
                "prompt": message,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from model")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Launch a simple Gradio interface
#To create a chat application with gr.ChatInterface(), the first thing you should do is define your chat function, e.g., chat_with_ollama
#In the simplest case, your chat function should accept two arguments: message and history (the arguments can be named anything, but must be in this order).
gr.ChatInterface(chat_with_ollama).launch()
#call the .launch() method to create the web interface