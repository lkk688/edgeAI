#!/usr/bin/env python3
"""
Demo script showing how to use the Ollama MCP Gradio Client
Optimized for Jetson Orin Nano and Apple Silicon
"""

import subprocess
import time
import sys
import os

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            models = result.stdout.strip()
            if models and "NAME" in models:
                print("✅ Ollama models available:")
                print(models)
                return True
            else:
                print("⚠️  No Ollama models found")
                return False
        else:
            print("❌ Ollama not responding")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not installed or not in PATH")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        "gradio",
        "requests", 
        "numpy"
    ]
    
    optional_packages = [
        "mcp",
        "gradio_client"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} missing")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} missing (optional)")
    
    return missing_required, missing_optional

def setup_ollama():
    """Setup Ollama with a lightweight model"""
    print("\n🚀 Setting up Ollama...")
    
    # Start Ollama server
    print("Starting Ollama server...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Wait for server to start
        print("✅ Ollama server started")
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first.")
        return False
    
    # Pull a lightweight model
    print("Pulling lightweight model (llama3.2:1b)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", "llama3.2:1b"], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minutes timeout
        )
        if result.returncode == 0:
            print("✅ Model downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Model download timed out (this is normal for first download)")
        return True  # Continue anyway
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def run_demo():
    """Run the MCP client demo"""
    print("\n🎯 Starting Ollama MCP Gradio Client...")
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the MCP client
        subprocess.run(["python3", "ollama_mcp_gradio_client.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")
    except FileNotFoundError:
        print("❌ ollama_mcp_gradio_client.py not found")

def main():
    print("🤖 Ollama MCP Gradio Client Demo")
    print("=" * 40)
    
    # Check system
    print("\n📋 Checking system requirements...")
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements_mcp.txt")
        return
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some features may not work. Install with: pip install mcp gradio-client")
    
    # Check Ollama
    if not check_ollama():
        print("\n🔧 Setting up Ollama...")
        if not setup_ollama():
            print("❌ Failed to setup Ollama")
            return
    
    print("\n✅ All checks passed!")
    print("\n📖 Usage Instructions:")
    print("1. The web interface will open at http://localhost:7861")
    print("2. Select a model from the dropdown")
    print("3. Type your message and click Send")
    print("4. To use MCP tools:")
    print("   - Enter './simple_mcp_server.py' in Server Path")
    print("   - Click Connect")
    print("   - Enable 'MCP Tools' checkbox")
    print("   - Try: 'Generate ASCII art of a cat'")
    print("5. Press Ctrl+C to stop")
    
    input("\nPress Enter to start the demo...")
    run_demo()

if __name__ == "__main__":
    main()