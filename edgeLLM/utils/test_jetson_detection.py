#!/usr/bin/env python3
"""
Test script to verify Jetson detection improvements.
This script tests the enhanced is_jetson() function with multiple fallback methods.
"""

import sys
import os
import subprocess

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from edgeLLM.utils.jetson_monitor import is_jetson, init_monitoring
except ImportError:
    print("Warning: Could not import jetson_monitor module")
# try:
#     from jetson_monitor import is_jetson, init_monitoring
# except ImportError:
#     # Try relative import if absolute import fails
#     try:
#         from .jetson_monitor import is_jetson, init_monitoring
#     except ImportError:
#         # Manual import as last resort
#          import importlib.util
#          spec = importlib.util.spec_from_file_location("jetson_monitor", 
#                                                       os.path.join(os.path.dirname(__file__), "jetson_monitor.py"))
#          if spec is not None and spec.loader is not None:
#              jetson_monitor = importlib.util.module_from_spec(spec)
#              spec.loader.exec_module(jetson_monitor)
#              is_jetson = jetson_monitor.is_jetson
#              init_monitoring = jetson_monitor.init_monitoring
#          else:
#              raise ImportError("Could not load jetson_monitor module")

def test_jetson_detection():
    """Test the improved Jetson detection logic"""
    print("=== Jetson Detection Test ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    print()
    
    # Test the main detection function
    print("Testing is_jetson() function...")
    jetson_detected = is_jetson()
    print(f"Jetson detected: {jetson_detected}")
    print()
    
    # Test individual detection methods
    print("Testing individual detection methods:")
    
    # 1. Device tree model check
    print("1. Checking /proc/device-tree/model...")
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            print(f"   Model: {model}")
            print(f"   Contains 'NVIDIA Jetson': {'NVIDIA Jetson' in model}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # 2. Tegrastats availability
    print("2. Checking tegrastats availability...")
    import subprocess
    try:
        result = subprocess.check_output("which tegrastats", shell=True, stderr=subprocess.DEVNULL)
        tegrastats_path = result.decode().strip()
        print(f"   tegrastats found at: {tegrastats_path}")
        print(f"   tegrastats available: True")
    except subprocess.CalledProcessError:
        print(f"   tegrastats available: False")
    print()
    
    # 3. Jetson-specific files/directories
    print("3. Checking Jetson-specific indicators...")
    jetson_indicators = [
        "/usr/bin/tegrastats",
        "/sys/devices/gpu.0",
        "/proc/device-tree/compatible"
    ]
    
    for indicator in jetson_indicators:
        try:
            if indicator.endswith("compatible"):
                with open(indicator, "r") as f:
                    content = f.read().lower()
                    has_tegra = "nvidia,tegra" in content
                    print(f"   {indicator}: exists, contains 'nvidia,tegra': {has_tegra}")
            else:
                exists = os.path.exists(indicator)
                print(f"   {indicator}: exists: {exists}")
        except Exception as e:
            print(f"   {indicator}: error: {e}")
    print()
    
    # Test monitoring initialization
    print("Testing monitoring initialization...")
    has_tegrastats = init_monitoring()
    print(f"Monitoring initialization successful: {has_tegrastats}")
    print()
    
    # Summary
    print("=== Summary ===")
    print(f"Final Jetson detection result: {jetson_detected}")
    print(f"Tegrastats available for monitoring: {has_tegrastats}")
    
    if jetson_detected:
        print("✅ Jetson environment detected successfully!")
    else:
        print("❌ Jetson environment not detected.")
        print("   This could be normal if running on a non-Jetson system,")
        print("   or indicate detection issues in a Jetson container.")

if __name__ == "__main__":
    test_jetson_detection()