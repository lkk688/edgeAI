#!/usr/bin/env python3
"""
Ollama MCP Gradio Client
Optimized for Nvidia Jetson Orin Nano and Apple Silicon

Based on Gradio MCP tutorial but adapted for local Ollama models
instead of Claude API for edge AI applications.
"""

import asyncio
import os
import json
import time
import platform
import subprocess
import threading
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

import gradio as gr
from gradio.components.chatbot import ChatMessage
import requests

# Import our modularized system monitoring utilities
from utils import system_monitor # pylint: disable=import-error

# Try to import MCP components (optional)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

# Settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
CHAT_LOG_PATH = "./chat_logs"
os.makedirs(CHAT_LOG_PATH, exist_ok=True)

# Initialize system monitoring
platform_info = system_monitor.init_monitoring()
IS_JETSON = platform_info["is_jetson"]
IS_APPLE_SILICON = platform_info["is_apple_silicon"]
HAS_NVIDIA_GPU = platform_info["has_nvidia_gpu"]

# Start the appropriate monitoring thread and get the system_info dictionary
system_info = system_monitor.start_monitoring()

# Global event loop for async operations
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class OllamaMCPClient:
    """MCP Client wrapper for Ollama integration"""
    
    def __init__(self):
        self.session = None
        self.exit_stack = None
        self.tools = []
        self.connected_servers = []
        
    def connect_to_server(self, server_path: str) -> str:
        """Connect to an MCP server"""
        if not MCP_AVAILABLE:
            return "MCP not available. Please install with: pip install mcp"
            
        return loop.run_until_complete(self._connect_async(server_path))
    
    async def _connect_async(self, server_path: str) -> str:
        """Async connection to MCP server"""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            
            self.exit_stack = AsyncExitStack()
            
            is_python = server_path.endswith('.py')
            command = "python3" if is_python else "node"
            
            server_params = StdioServerParameters(
                command=command,
                args=[server_path],
                env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            
            response = await self.session.list_tools()
            self.tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
            
            self.connected_servers.append(server_path)
            tool_names = [tool["name"] for tool in self.tools]
            return f"✅ Connected to MCP server: {server_path}\nAvailable tools: {', '.join(tool_names)}"
            
        except Exception as e:
            return f"❌ Failed to connect to MCP server: {str(e)}"
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools"""
        return self.tools
    
    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool asynchronously"""
        if not self.session:
            return "No MCP server connected"
            
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return json.dumps(result.content, indent=2)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool synchronously"""
        return loop.run_until_complete(self.call_tool_async(tool_name, arguments))

# Global MCP client instance
mcp_client = OllamaMCPClient()

def list_ollama_models() -> List[str]:
    """Fetch available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
    except Exception as e:
        print(f"Error fetching models: {e}")
    return []

def check_ollama_server() -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        return response.ok
    except:
        return False

def process_with_ollama(messages: List[Dict], model: str, use_tools: bool = False) -> str:
    """Process messages with Ollama, optionally using MCP tools"""
    
    # Check if server is running
    if not check_ollama_server():
        return "❌ Ollama server is not running. Please start it with 'ollama serve'"
    
    # If tools are enabled and available, check if we need to use them
    if use_tools and mcp_client.tools:
        # Simple tool detection - look for keywords in the last message
        last_message = messages[-1]["content"] if messages else ""
        
        # Check for image generation requests
        if any(keyword in last_message.lower() for keyword in ["generate image", "create image", "draw", "picture"]):
            # Try to use image generation tool if available
            for tool in mcp_client.tools:
                if "image" in tool["name"].lower() or "generate" in tool["name"].lower():
                    # Extract prompt for image generation
                    prompt = last_message.replace("generate image", "").replace("create image", "").strip()
                    if not prompt:
                        prompt = "a beautiful landscape"
                    
                    try:
                        tool_result = mcp_client.call_tool(tool["name"], {"prompt": prompt})
                        return f"🎨 Generated image using {tool['name']}:\n\n{tool_result}"
                    except Exception as e:
                        return f"❌ Error using image tool: {str(e)}"
    
    # Standard Ollama chat completion
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        
        if not response.ok:
            return f"❌ Ollama API error: {response.status_code} - {response.text}"
        
        data = response.json()
        return data.get("message", {}).get("content", "No response received")
        
    except requests.RequestException as e:
        return f"❌ Connection error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def chat_with_ollama(message: str, history: List[List[str]], model: str, use_tools: bool) -> tuple:
    """Main chat function with Ollama integration"""
    if not message.strip():
        return history, ""
    
    # Convert Gradio history to Ollama message format
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from Ollama
    start_time = time.time()
    response = process_with_ollama(messages, model, use_tools)
    elapsed_time = time.time() - start_time
    
    # Calculate approximate tokens per second
    token_count = len(response.split())
    tps = f"{token_count / elapsed_time:.2f} tokens/sec" if elapsed_time > 0 else "N/A"
    
    # Update history
    history.append([message, response])
    
    return history, "", tps

def export_chat_history(history: List[List[str]]) -> str:
    """Export chat history to files"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = os.path.join(CHAT_LOG_PATH, f"mcp_chat_{timestamp}.md")
    json_path = os.path.join(CHAT_LOG_PATH, f"mcp_chat_{timestamp}.json")
    
    # Create markdown export
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Chat Export - {timestamp}\n\n")
        for i, (user_msg, assistant_msg) in enumerate(history):
            f.write(f"## Exchange {i+1}\n\n")
            f.write(f"**User:** {user_msg}\n\n")
            f.write(f"**Assistant:** {assistant_msg}\n\n---\n\n")
    
    # Create JSON export
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    return f"✅ Chat exported to:\n- {md_path}\n- {json_path}"

def refresh_system_info() -> str:
    """Refresh system monitoring information"""
    return system_info["text"]

def connect_mcp_server(server_path: str) -> str:
    """Connect to an MCP server"""
    if not server_path.strip():
        return "Please provide a server path"
    
    return mcp_client.connect_to_server(server_path.strip())

def get_mcp_tools_info() -> str:
    """Get information about available MCP tools"""
    if not mcp_client.tools:
        return "No MCP tools available. Connect to an MCP server first."
    
    info = "Available MCP Tools:\n\n"
    for tool in mcp_client.tools:
        info += f"**{tool['name']}**\n"
        info += f"Description: {tool['description']}\n"
        info += f"Schema: {json.dumps(tool['input_schema'], indent=2)}\n\n"
    
    return info

# Create Gradio interface
with gr.Blocks(title="Ollama MCP Client", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    platform_name = system_monitor.get_platform_name()
    
    gr.Markdown(f"# 🤖 Ollama MCP Client ({platform_name})")
    gr.Markdown("*Local AI with Model Context Protocol integration*")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list_ollama_models(),
                    label="Ollama Model",
                    value=list_ollama_models()[0] if list_ollama_models() else None,
                    interactive=True,
                    scale=2
                )
                refresh_models_btn = gr.Button("🔄 Refresh", scale=1)
                use_tools_checkbox = gr.Checkbox(
                    label="Enable MCP Tools",
                    value=False,
                    scale=1
                )
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_copy_button=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask something or request an image...",
                    lines=2,
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", scale=1)
                export_btn = gr.Button("💾 Export", scale=1)
                token_speed = gr.Textbox(
                    label="Speed",
                    interactive=False,
                    scale=2
                )
        
        with gr.Column(scale=2):
            # MCP Server Connection
            gr.Markdown("### MCP Server Connection")
            with gr.Row():
                server_path_input = gr.Textbox(
                    label="Server Path",
                    placeholder="path/to/mcp_server.py",
                    scale=3
                )
                connect_btn = gr.Button("Connect", scale=1)
            
            mcp_status = gr.Textbox(
                label="MCP Status",
                value="No MCP server connected",
                interactive=False,
                lines=3
            )
            
            mcp_tools_info = gr.Textbox(
                label="Available Tools",
                value="Connect to an MCP server to see available tools",
                interactive=False,
                lines=8
            )
            
            # System monitoring
            gr.Markdown("### System Monitor")
            system_monitor_display = gr.Textbox(
                label="System Info",
                value=system_info["text"],
                interactive=False,
                lines=12
            )
            
            # Platform-specific info
            if IS_JETSON:
                gr.Markdown("**🚀 Jetson Optimized**")
                gr.Markdown("NVIDIA Jetson Orin Nano detected")
            elif IS_APPLE_SILICON:
                gr.Markdown("**🍎 Apple Silicon Optimized**")
                gr.Markdown("Apple M-series processor detected")
            elif HAS_NVIDIA_GPU:
                gr.Markdown("**⚡ NVIDIA GPU Detected**")
            else:
                gr.Markdown("**💻 CPU Mode**")
    
    # Event handlers
    def refresh_models():
        models = list_ollama_models()
        return gr.Dropdown(choices=models, value=models[0] if models else None)
    
    def clear_chat():
        return [], ""
    
    # Connect event handlers
    send_btn.click(
        fn=chat_with_ollama,
        inputs=[msg_input, chatbot, model_dropdown, use_tools_checkbox],
        outputs=[chatbot, msg_input, token_speed]
    )
    
    msg_input.submit(
        fn=chat_with_ollama,
        inputs=[msg_input, chatbot, model_dropdown, use_tools_checkbox],
        outputs=[chatbot, msg_input, token_speed]
    )
    
    refresh_models_btn.click(
        fn=refresh_models,
        outputs=[model_dropdown]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input]
    )
    
    export_btn.click(
        fn=export_chat_history,
        inputs=[chatbot],
        outputs=[mcp_status]
    )
    
    connect_btn.click(
        fn=connect_mcp_server,
        inputs=[server_path_input],
        outputs=[mcp_status]
    )
    
    # Update MCP tools info when status changes
    mcp_status.change(
        fn=get_mcp_tools_info,
        outputs=[mcp_tools_info]
    )
    
    # Real-time system monitoring
    timer = gr.Timer(3)  # Update every 3 seconds
    timer.tick(
        fn=refresh_system_info,
        outputs=[system_monitor_display]
    )

if __name__ == "__main__":
    # Apply platform-specific optimizations
    if IS_JETSON:
        print("🚀 Applying Jetson Orin Nano optimizations...")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        # Set memory limits for CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.7)
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
            
    elif IS_APPLE_SILICON:
        print("🍎 Applying Apple Silicon optimizations...")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        try:
            import torch
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) available")
        except (ImportError, AttributeError):
            pass
    
    elif HAS_NVIDIA_GPU:
        print("⚡ Applying NVIDIA GPU optimizations...")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
    
    else:
        print("💻 Running in CPU mode")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    # Check if Ollama is running
    if not check_ollama_server():
        print("⚠️  Ollama server not detected. Starting...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)  # Wait for server to start
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
    
    # Launch the application
    print("🚀 Starting Ollama MCP Gradio Client...")
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from original
        share=False,
        show_error=True,
        show_api=False
    )