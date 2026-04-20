import os
import json
import time
import argparse
import requests
import httpx
import re
from openai import OpenAI

# ==============================================================================
# 0. Environment & Proxy Setup
# ==============================================================================
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# ANSI Color Codes for terminal output
COLOR_GREY = "\033[90m"
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"

# ==============================================================================
# 1. Helper: Performance Metrics
# ==============================================================================
def print_metrics(usage, ttft_seconds=None, total_seconds=None):
    print(f"\n{'-'*40}\n📊 PERFORMANCE METRICS\n{'-'*40}")
    if not usage:
        print("[⚠️ Warning] No usage data available.")
        return

    p_tokens = getattr(usage, 'prompt_tokens', 0)
    c_tokens = getattr(usage, 'completion_tokens', 0)
    t_tokens = getattr(usage, 'total_tokens', 0)
    
    print(f"Tokens: [Prompt: {p_tokens}] + [Completion: {c_tokens}] = [Total: {t_tokens}]")

    if ttft_seconds and total_seconds and ttft_seconds > 0:
        gen_seconds = max(total_seconds - ttft_seconds, 0.001)
        print(f"Time  : Prefill (TTFT): {ttft_seconds:.3f}s | Generate: {gen_seconds:.3f}s")
        print(f"Speed : Prefill: {p_tokens/ttft_seconds:.2f} t/s | Generate: {c_tokens/gen_seconds:.2f} t/s")
    elif total_seconds:
        print(f"Time  : Total: {total_seconds:.3f}s")
        print(f"Speed : Overall: {t_tokens/total_seconds:.2f} t/s")
    print("-" * 40)

# ==============================================================================
# 2. Core Feature: Unified Streaming with Thinking Detection
# ==============================================================================
def run_streaming_test(client, model_name, prompt, feature_name):
    print(f"\n{'='*60}\n{feature_name}\n{'='*60}")
    print(f"User: {prompt}\n\nAssistant: ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None
    final_usage = None
    
    # State tracking for different thinking styles
    in_thinking_block = False
    
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=4096, # High budget for reasoning
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
        )

        for chunk in stream:
            # Safely check for choices using NVIDIA's recommended approach
            if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                if getattr(chunk, "usage", None): 
                    final_usage = chunk.usage
                continue
            
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            
            reasoning = getattr(delta, 'reasoning_content', None)
            content = getattr(delta, 'content', None) or ""

            # 1. Capture TTFT safely
            if first_token_time is None and (reasoning or content):
                first_token_time = time.perf_counter()

            # 2. Handle Dedicated Reasoning Field (Gemma/GLM style)
            if reasoning:
                if not in_thinking_block:
                    print(f"\n{COLOR_GREY}[💡 Thinking Phase...]{COLOR_RESET}\n{COLOR_GREY}", end="")
                    in_thinking_block = True
                print(reasoning, end="", flush=True)
                continue

            # 3. Handle Inline Thinking Tags (MiniMax style)
            if "<think>" in content:
                print(f"\n{COLOR_GREY}[💡 Thinking Phase...]{COLOR_RESET}\n{COLOR_GREY}", end="")
                content = content.replace("<think>", "")
                in_thinking_block = True
            
            if "</think>" in content:
                content = content.replace("</think>", f"{COLOR_RESET}\n\n[✅ Final Answer]:\n")
                in_thinking_block = False
            
            # Final output print
            if content:
                print(content, end="", flush=True)

        end_time = time.perf_counter()
        print(COLOR_RESET)
        print_metrics(final_usage, first_token_time - start_time if first_token_time else 0, end_time - start_time)

    except Exception as e:
        print(f"\n[❌ API Error]: {e}")

# ==============================================================================
# 3. Feature 5: Math + Tool Call (Hardened Logic)
# ==============================================================================
def test_math_tool(client, model_name):
    print(f"\n{'='*60}\n🧮 FEATURE 5: Math Reasoning & Tool Execution\n{'='*60}")
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_math_answer",
            "description": "Submit final math distance",
            "parameters": {
                "type": "object",
                "properties": {"final_distance": {"type": "number"}},
                "required": ["final_distance"]
            }
        }
    }]
    prompt = "Train A (60mph) leaves City A. Train B (40mph) leaves City B (300m away) 1hr later. Where do they meet?"
    
    print(f"User: {prompt}\n")
    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_math_answer"}},
            temperature=0.1,
            # Note: Some models struggle with tools while 'enable_thinking' is active.
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        total_time = time.perf_counter() - start_time
        
        # Safely validate the response choices
        if not getattr(response, "choices", None) or len(response.choices) == 0:
            print("\n[❌ Tool Error]: Model returned an empty choices list.")
            return

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # Graceful fallback if the model ignored the tool choice and returned text instead
        if not tool_calls or len(tool_calls) == 0:
            print(f"[⚠️ Tool Warning]: Model bypassed the tool call and returned standard text.")
            print(f"[Model Fallback Response]: {getattr(message, 'content', 'No content provided.')}")
        else:
            tool_call = tool_calls[0]
            print(f"[🔧 Tool Call Received]: {tool_call.function.name}")
            print(f"[✅ Arguments]: {tool_call.function.arguments}")
        
        print_metrics(response.usage, total_seconds=total_time)
        
    except Exception as e:
        print(f"\n[❌ Execution Error]: {e}")

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--nvidia_api", action="store_true")
    args = parser.parse_args()

    if args.nvidia_api:
        args.base_url = "https://integrate.api.nvidia.com/v1"
        # Force clean environment variable loading
        env_key = os.getenv("NVIDIA_API_KEY", "EMPTY")
        args.api_key = env_key.strip().strip("'").strip('"')

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=180.0,
        max_retries=3,
        http_client=httpx.Client(trust_env=False)
    )

    print(f"{'='*60}\n🚀 RUNNING: {args.model_name} on {args.base_url}\n{'='*60}")
    
    run_streaming_test(client, args.model_name, "Write a haiku about a GPU.", "🌊 FEATURE 3: Streaming Haiku")
    test_math_tool(client, args.model_name)

"""
python test_llmcalls_v2.py --base_url "http://100.81.219.32:8000/v1" --model_name "nvidia/Gemma-4-31B-IT-NVFP4"

#export NVIDIA_API_KEY="nvapi-..."
python test_llmcalls_v2.py --model_name "minimaxai/minimax-m2.7" --nvidia_api

python test_llmcalls_v2.py --model_name "z-ai/glm4.7" --nvidia_api

"""