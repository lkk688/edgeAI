import os
import json
import time
import argparse
import requests
import httpx
from openai import OpenAI

# ==============================================================================
# 0. Environment & Proxy Setup
# ==============================================================================
# Force Python, requests, and httpx to ignore all system proxies.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# ==============================================================================
# 1. Helper: Print Performance & Token Metrics
# ==============================================================================
def print_metrics(usage, ttft_seconds=None, total_seconds=None):
    """
    Calculates and prints token consumption and processing speed.
    - TTFT (Time To First Token) represents the Prefill phase.
    - The remaining time represents the Generation (Decode) phase.
    """
    print("\n" + "-"*40)
    print("📊 PERFORMANCE METRICS")
    print("-"*40)
    
    if not usage:
        print("[⚠️ Warning] No usage data returned by the server.")
        return

    p_tokens = usage.prompt_tokens
    c_tokens = usage.completion_tokens
    t_tokens = usage.total_tokens
    
    print(f"Tokens Consumed: [Prompt: {p_tokens}] + [Completion: {c_tokens}] = [Total: {t_tokens}]")

    # If timing data is available (usually gathered accurately via streaming)
    if ttft_seconds is not None and total_seconds is not None:
        generate_seconds = total_seconds - ttft_seconds
        
        prefill_speed = (p_tokens / ttft_seconds) if ttft_seconds > 0 else 0
        generate_speed = (c_tokens / generate_seconds) if generate_seconds > 0 else 0
        
        print(f"Time Taken     : Prefill (TTFT): {ttft_seconds:.3f}s | Generate: {generate_seconds:.3f}s")
        print(f"Prefill Speed  : {prefill_speed:.2f} tokens / second")
        print(f"Generate Speed : {generate_speed:.2f} tokens / second")
    
    # Fallback for blocking (non-streaming) calls where TTFT isn't natively exposed client-side
    elif total_seconds is not None:
        overall_speed = (t_tokens / total_seconds) if total_seconds > 0 else 0
        print(f"Total Time     : {total_seconds:.3f}s")
        print(f"Overall Speed  : {overall_speed:.2f} tokens / second (Prefill + Generate combined)")
    print("-"*40)

# ==============================================================================
# 2. Feature: Query Model Context Length
# ==============================================================================
def check_model_context_length(base_url, model_name):
    print("\n" + "="*60)
    print("🔍 FEATURE 1: Querying Model Configuration")
    print("="*60)
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
# 3. Feature: Basic Chat (Blocking)
# ==============================================================================
def test_basic_chat(client, model_name):
    print("\n" + "="*60)
    print("💬 FEATURE 2: Basic Chat (Standard Request)")
    print("="*60)
    
    prompt = "Explain the concept of 'Edge AI' in exactly two sentences."
    print(f"User: {prompt}\n")
    print("Waiting for response...\n")
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    total_time = time.perf_counter() - start_time
    
    print(f"Assistant: {response.choices[0].message.content}")
    # Print overall metrics (cannot split prefill/decode accurately without streaming)
    print_metrics(response.usage, total_seconds=total_time)

# ==============================================================================
# 4. Feature: Streaming Chat (Real-time with accurate TTFT metrics)
# ==============================================================================
def test_streaming_chat(client, model_name):
    print("\n" + "="*60)
    print("🌊 FEATURE 3: Streaming Chat (Real-time & Accurate Speeds)")
    print("="*60)
    
    prompt = "Write a short, creative haiku about a GPU."
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)
    
    start_time = time.perf_counter()
    first_token_time = None
    final_usage = None
    
    stream_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
        stream=True,
        # Critical: Requests the server to send a final chunk containing token usage
        stream_options={"include_usage": True} 
    )
    
    for chunk in stream_response:
        # Capture Time To First Token (TTFT)
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.perf_counter()
            
        # Print content if available
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            
        # The last chunk will have an empty choices list but will contain the usage object
        if chunk.usage:
            final_usage = chunk.usage
            
    end_time = time.perf_counter()
    print("\n") # Newline after generation completes
    
    # Calculate times
    ttft_seconds = (first_token_time - start_time) if first_token_time else 0
    total_seconds = end_time - start_time
    
    print_metrics(final_usage, ttft_seconds=ttft_seconds, total_seconds=total_seconds)

# ==============================================================================
# 5. Feature: Tool Calling
# ==============================================================================
def get_current_weather(location, unit="celsius"):
    print(f"\n[💻 Executing Local Tool] Fetching weather for '{location}'...")
    return json.dumps({"location": location, "weather": "Cloudy", "temperature": 18, "unit": unit})

def test_tool_call(client, model_name):
    print("\n" + "="*60)
    print("🛠️ FEATURE 4: Tool Calling (OpenAI Standard Schema)")
    print("="*60)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather conditions for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., Gilroy, California"},
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
    
    # Round 1: Model Tool Request
    start_time_r1 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    time_r1 = time.perf_counter() - start_time_r1
    response_msg = response.choices[0].message
    messages.append(response_msg)
    
    print("\n[Metrics for Round 1 (Tool Decision)]")
    print_metrics(response.usage, total_seconds=time_r1)
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            print(f"\n[🤖 LLM requested tool] {func_name} with args: {func_args}")
            
            if func_name == "get_current_weather":
                tool_result = get_current_weather(
                    location=func_args.get("location"),
                    unit=func_args.get("unit", "celsius")
                )
                print(f"  <- Local Function Returned: {tool_result}")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": tool_result,
                })
                
        # Round 2: Final Answer
        print("\n🧠 Sending tool results back to LLM for final answer...")
        start_time_r2 = time.perf_counter()
        final_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        time_r2 = time.perf_counter() - start_time_r2
        
        print(f"\n[🎯 Final LLM Answer]:\n{final_response.choices[0].message.content}")
        print("\n[Metrics for Round 2 (Final Synthesis)]")
        print_metrics(final_response.usage, total_seconds=time_r2)

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="vLLM Feature & Performance Tester")
    parser.add_argument(
        "--base_url", 
        type=str, 
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="The base URL of the vLLM OpenAI API endpoint."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=os.getenv("VLLM_MODEL_NAME", "nvidia/Gemma-4-31B-IT-NVFP4"),
        help="The exact model name registered in the vLLM server."
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"⚙️  INITIALIZING TEST SUITE")
    print(f"Target URL   : {args.base_url}")
    print(f"Target Model : {args.model_name}")
    print("="*60)

    # Initialize client (trust_env=False bypasses proxy)
    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",  
        http_client=httpx.Client(trust_env=False) 
    )

    # Run tests
    check_model_context_length(args.base_url, args.model_name)
    test_basic_chat(client, args.model_name)
    test_streaming_chat(client, args.model_name)
    test_tool_call(client, args.model_name)

"""
python test_llm_calls_tools.py --base_url "http://100.81.219.32:8000/v1" --model_name "nvidia/Gemma-4-31B-IT-NVFP4"
"""