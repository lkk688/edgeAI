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

    # If timing data is available (gathered accurately via streaming)
    if ttft_seconds is not None and total_seconds is not None and ttft_seconds > 0:
        generate_seconds = total_seconds - ttft_seconds
        
        prefill_speed = (p_tokens / ttft_seconds) if ttft_seconds > 0 else 0
        generate_speed = (c_tokens / generate_seconds) if generate_seconds > 0 else 0
        
        print(f"Time Taken     : Prefill (TTFT): {ttft_seconds:.3f}s | Generate: {generate_seconds:.3f}s")
        print(f"Prefill Speed  : {prefill_speed:.2f} tokens / second")
        print(f"Generate Speed : {generate_speed:.2f} tokens / second")
    
    # Fallback for blocking calls or if TTFT failed to capture
    elif total_seconds is not None:
        overall_speed = (t_tokens / total_seconds) if total_seconds > 0 else 0
        print(f"Total Time     : {total_seconds:.3f}s")
        print(f"Overall Speed  : {overall_speed:.2f} tokens / second (Prefill + Generate combined)")
    print("-"*40)

# ==============================================================================
# 2. Feature 1: Query Model Context Length
# ==============================================================================
def check_model_context_length(base_url, model_name, api_key):
    print("\n" + "="*60)
    print("🔍 FEATURE 1: Querying Model Configuration")
    print("="*60)
    try:
        session = requests.Session()
        session.trust_env = False 
        session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        response = session.get(f"{base_url}/models", timeout=15)
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
# 3. Feature 2: Basic Chat
# ==============================================================================
def test_basic_chat(client, model_name):
    print("\n" + "="*60)
    print("💬 FEATURE 2: Basic Chat (Standard Request)")
    print("="*60)
    
    prompt = "Explain the concept of 'Edge AI' in exactly two sentences."
    print(f"User: {prompt}\n")
    print("Waiting for response (This may take a while if thinking is enabled)...\n")
    
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and concise AI assistant."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2048, # Increased to prevent cut-off during hidden thinking
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        total_time = time.perf_counter() - start_time
        
        print(f"Assistant: {response.choices[0].message.content}")
        print_metrics(response.usage, total_seconds=total_time)
    except Exception as e:
        print(f"[❌ API Error Intercepted]: {e}")

# ==============================================================================
# 4. Feature 3: Streaming Chat (Now with Visible Thinking Process!)
# ==============================================================================
def test_streaming_chat(client, model_name):
    print("\n" + "="*60)
    print("🌊 FEATURE 3: Streaming Chat (Real-time & Accurate Speeds)")
    print("="*60)
    
    prompt = "Write a short, creative haiku about a GPU."
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)
    
    first_token_time = None
    final_usage = None
    has_printed_thinking_header = False
    has_printed_answer_header = False
    
    try:
        start_time = time.perf_counter()
        stream_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        
        for chunk in stream_response:
            # Usage chunk usually has empty choices
            if not chunk.choices:
                if chunk.usage:
                    final_usage = chunk.usage
                continue
                
            delta = chunk.choices[0].delta
            
            # Extract reasoning and normal content safely
            reasoning = getattr(delta, 'reasoning_content', None)
            content = getattr(delta, 'content', None)
            
            # Capture TTFT as soon as ANY token (thinking or content) arrives
            if first_token_time is None and (reasoning or content):
                first_token_time = time.perf_counter()
                
            # --- 1. Handle Thinking Process ---
            if reasoning:
                if not has_printed_thinking_header:
                    print("\n\033[90m[💡 Thinking Process Started...]\033[0m\n\033[90m", end="")
                    has_printed_thinking_header = True
                
                # Print thinking in Grey (\033[90m is ANSI grey, \033[0m resets)
                print(f"{reasoning}", end="", flush=True)
                
            # --- 2. Handle Final Content ---
            if content:
                if not has_printed_answer_header:
                    # Reset color and print a separator if we previously showed thinking
                    if has_printed_thinking_header:
                        print("\033[0m\n\n[✅ Final Answer]:\n", end="")
                    has_printed_answer_header = True
                
                # Print normal text
                print(content, end="", flush=True)
                
            # Capture final usage if attached to the last choice chunk
            if chunk.usage:
                final_usage = chunk.usage
                
        end_time = time.perf_counter()
        print("\n\033[0m") # Ensure color formatting is reset at the very end
        
        ttft_seconds = (first_token_time - start_time) if first_token_time else 0
        total_seconds = end_time - start_time
        
        print_metrics(final_usage, ttft_seconds=ttft_seconds, total_seconds=total_seconds)
    except Exception as e:
         print(f"\n[❌ API Error Intercepted]: {e}")


# ==============================================================================
# 5. Feature 4: Tool Calling (Basic)
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
    
    try:
        # Round 1: Model Tool Request
        start_time_r1 = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
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
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            time_r2 = time.perf_counter() - start_time_r2
            
            print(f"\n[🎯 Final LLM Answer]:\n{final_response.choices[0].message.content}")
            print("\n[Metrics for Round 2 (Final Synthesis)]")
            print_metrics(final_response.usage, total_seconds=time_r2)
    except Exception as e:
        print(f"\n[❌ API Error Intercepted]: {e}")

# ==============================================================================
# 6. Feature 5: Math Reasoning + Streaming Tool Call Separation
# ==============================================================================
def test_math_reasoning_tool(client, model_name):
    print("\n" + "="*60)
    print("🧮 FEATURE 5: Math Reasoning & Visible Tool Thinking")
    print("="*60)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_math_answer",
            "description": "Submit the final calculated answer for a math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_distance": {"type": "number", "description": "The distance in miles from City A where they meet"}
                },
                "required": ["final_distance"]
            }
        }
    }]
    
    prompt = "A train leaves City A at 60 mph. Another train leaves City B (300 miles away) 1 hour later at 40 mph heading towards City A. How many miles from City A will they meet? Think carefully step-by-step, then use the tool to submit the final distance."
    print(f"User: {prompt}\n")
    print("Assistant is processing...\n")
    
    first_token_time = None
    final_usage = None
    func_name = ""
    tool_args_str = ""
    has_printed_thinking_header = False
    
    try:
        start_time = time.perf_counter()
        stream_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_math_answer"}}, 
            temperature=0.1, 
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
                
        for chunk in stream_response:
            if not chunk.choices:
                if chunk.usage:
                    final_usage = chunk.usage
                continue
                
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, 'reasoning_content', None)
            
            # Record TTFT 
            if first_token_time is None and (reasoning or delta.tool_calls):
                first_token_time = time.perf_counter()
                
            # --- Print Thinking Process for Tool ---
            if reasoning:
                if not has_printed_thinking_header:
                    print("\033[90m[💡 Internal Math Reasoning...]\n", end="")
                    has_printed_thinking_header = True
                print(f"{reasoning}", end="", flush=True)
                
            # --- Parse streaming tool calls ---
            if getattr(delta, 'tool_calls', None):
                # Reset formatting when tool call starts
                if has_printed_thinking_header:
                    print("\033[0m\n\n[🛠️ Preparing Tool Call Parameters...]: ", end="", flush=True)
                    has_printed_thinking_header = False # Prevent multiple headers
                    
                tc_delta = delta.tool_calls[0]
                
                if tc_delta.function.name:
                    func_name = tc_delta.function.name
                    
                if tc_delta.function.arguments:
                    arg_chunk = tc_delta.function.arguments
                    tool_args_str += arg_chunk
                    print(arg_chunk, end="", flush=True)
                    
            if chunk.usage:
                final_usage = chunk.usage
                
        end_time = time.perf_counter()
        print("\033[0m\n") # Reset color buffer
        
        if func_name:
            print(f"[🔧 Tool Executed]: {func_name}")
            print(f"[✅ Final Answer Parsed]: {tool_args_str}")
        else:
            print("\n[⚠️ Error] Model failed to call the specified tool.")
        
        ttft_seconds = (first_token_time - start_time) if first_token_time else 0
        total_seconds = end_time - start_time
        
        print_metrics(final_usage, ttft_seconds=ttft_seconds, total_seconds=total_seconds)
        
    except Exception as e:
         print(f"\n[❌ API Error Intercepted]: {e}")

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM API Testing Suite")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for standard API")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-31b-it", help="Target model name")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="Standard API Key")
    parser.add_argument("--nvidia_api", action="store_true", help="Enable NVIDIA API mode (auto-sets URL and reads NVIDIA_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check if NVIDIA API mode is enabled
    if args.nvidia_api:
        args.base_url = "https://integrate.api.nvidia.com/v1"
        args.api_key = os.getenv("NVIDIA_API_KEY", "EMPTY")
        print("API Key:", args.api_key)
        if args.api_key == "EMPTY":
            print("\n[⚠️ WARNING] --nvidia_api flag used, but NVIDIA_API_KEY environment variable is not set. API calls will likely fail.\n")
    
    print("="*60)
    print(f"⚙️  INITIALIZING TEST SUITE")
    print(f"Target URL   : {args.base_url}")
    print(f"Target Model : {args.model_name}")
    print(f"Mode         : {'NVIDIA NIM API' if args.nvidia_api else 'Standard URL'}")
    print(f"API Key Used : {'Yes (Hidden)' if args.api_key != 'EMPTY' else 'No (EMPTY)'}")
    print("="*60)

    # API Protection: Retries and timeouts added
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,  
        timeout=180.0,            
        max_retries=4,            
        http_client=httpx.Client(trust_env=False) 
    )

    check_model_context_length(args.base_url, args.model_name, args.api_key)
    test_basic_chat(client, args.model_name)
    test_streaming_chat(client, args.model_name)
    test_tool_call(client, args.model_name)
    test_math_reasoning_tool(client, args.model_name)

"""
python test_llmcalls.py \
  --base_url "https://integrate.api.nvidia.com/v1" \
  --model_name "google/gemma-4-31b-it" \
  --nvidia_api

python test_llmcalls.py \
  --base_url "https://integrate.api.nvidia.com/v1" \
  --model_name "minimaxai/minimax-m2.7" \
  --nvidia_api
  

python test_llmcalls.py \
    --base_url "http://100.81.219.32:8000/v1" \
    --model_name "nvidia/Gemma-4-31B-IT-NVFP4"
"""