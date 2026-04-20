import os
import json
import requests
import httpx
from openai import OpenAI

# ==============================================================================
# 0. Environment & Proxy Setup
# ==============================================================================
# Force Python, requests, and httpx to ignore all system proxies.
# This ensures traffic to localhost/0.0.0.0 is not routed through a VPN/Proxy.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

BASE_URL = "http://100.81.219.32:8000/v1"
# Ensure this matches the exact model name running in your vLLM Docker container
MODEL_NAME = "nvidia/Gemma-4-31B-IT-NVFP4" 
# MODEL_NAME = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"

# Initialize the OpenAI client pointing to your local Jetson Thor.
# Passing a custom httpx client with trust_env=False acts as a bulletproof way 
# to prevent the library from using any OS-level proxy settings.
client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",  # API key is not required for local vLLM instances
    http_client=httpx.Client(trust_env=False) 
)

# ==============================================================================
# Feature 1: Query Model Context Length
# ==============================================================================
def check_model_context_length(base_url, model_name):
    """Queries the vLLM server REST API to retrieve the max_model_len."""
    print("\n" + "="*50)
    print("🔍 FEATURE 1: Querying Model Configuration")
    print("="*50)
    try:
        session = requests.Session()
        session.trust_env = False 
        
        response = session.get(f"{base_url}/models")
        response.raise_for_status()
        
        models_data = response.json().get("data", [])
        for model in models_data:
            if model.get("id") == model_name:
                max_len = model.get("max_model_len", "Not specified in API response")
                print(f"[✅ Success] Model: {model_name}")
                print(f"[✅ Success] Max Context Length: {max_len} tokens")
                return
        
        print(f"[⚠️ Warning] Model '{model_name}' not found on the server.")
    except Exception as e:
        print(f"[❌ Error] Failed to fetch model info: {e}")

# ==============================================================================
# Feature 2: Basic Chat (Standard blocking request)
# ==============================================================================
def test_basic_chat(client, model_name):
    """Sends a standard prompt and waits for the complete response."""
    print("\n" + "="*50)
    print("💬 FEATURE 2: Basic Chat (Standard Request)")
    print("="*50)
    
    prompt = "Explain the concept of 'Edge AI' in exactly two sentences."
    print(f"User: {prompt}\n")
    print("Waiting for response...\n")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Assistant: {response.choices[0].message.content}")

# ==============================================================================
# Feature 3: Streaming Chat (Real-time token generation)
# ==============================================================================
def test_streaming_chat(client, model_name):
    """Sends a prompt and prints the response token-by-token in real time."""
    print("\n" + "="*50)
    print("🌊 FEATURE 3: Streaming Chat (Real-time text generation)")
    print("="*50)
    
    prompt = "Write a short, creative haiku about a GPU."
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)
    
    # Adding stream=True enables Server-Sent Events (SSE)
    stream_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
        stream=True
    )
    
    # Iterate over the incoming stream chunks
    for chunk in stream_response:
        # Check if the chunk contains text delta
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # Print without newline and force flush to terminal immediately
            print(content, end="", flush=True)
    print("\n") # Add a final newline when streaming is done

# ==============================================================================
# Feature 4: Tool Calling (Function Calling)
# ==============================================================================
def get_current_weather(location, unit="celsius"):
    """Mock local Python function to simulate an external API call."""
    print(f"\n[💻 Executing Local Tool] Fetching weather for '{location}'...")
    return json.dumps({"location": location, "weather": "Cloudy", "temperature": 18, "unit": unit})

def test_tool_call(client, model_name):
    """Tests the model's ability to trigger a local tool, parse args, and reply."""
    print("\n" + "="*50)
    print("🛠️ FEATURE 4: Tool Calling (OpenAI Standard Schema)")
    print("="*50)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather conditions for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., San Jose, California"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools if necessary."},
        {"role": "user", "content": "What is the weather like in Gilroy, California today in celsius?"}
    ]
    
    print(f"User: {messages[1]['content']}")
    
    # Round 1: Let the model decide to use the tool
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_msg = response.choices[0].message
    messages.append(response_msg)
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            print(f"\n[🤖 LLM requested tool] {func_name} with args: {func_args}")
            
            # Execute local logic
            if func_name == "get_current_weather":
                tool_result = get_current_weather(
                    location=func_args.get("location"),
                    unit=func_args.get("unit", "celsius")
                )
                print(f"  <- Local Function Returned: {tool_result}")
                
                # Append tool result to history
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": tool_result,
                })
                
        # Round 2: Get final answer combining tool result
        print("\n🧠 Sending tool results back to LLM for final answer...")
        final_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        print(f"\n[🎯 Final LLM Answer]:\n{final_response.choices[0].message.content}\n")
    else:
        print(f"\n[💬 LLM answered directly]:\n{response_msg.content}\n")

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == "__main__":
    check_model_context_length(BASE_URL, MODEL_NAME)
    test_basic_chat(client, MODEL_NAME)
    test_streaming_chat(client, MODEL_NAME)
    test_tool_call(client, MODEL_NAME)