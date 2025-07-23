#!/usr/bin/env python3
"""
Apple Silicon Monitor - Utility for monitoring Apple Silicon Macs

This script provides real-time monitoring of system resources on Apple Silicon Macs,
including CPU usage, memory usage, swap usage, and GPU information. It can display
information in a full or compact format, with or without colors, and can log output
to a file.

Usage:
    python apple_silicon_monitor.py [options]

Options:
    -i, --interval INTERVAL  Update interval in seconds (default: 2.0)
    --no-clear              Don't clear the screen between updates
    --log LOG               Log output to specified file
    --compact               Display information in compact format
    --no-color              Disable colored output
    --version               Show program's version number and exit
    -h, --help              Show this help message and exit

Examples:
    # Basic usage with default settings
    python apple_silicon_monitor.py
    
    # Update every 5 seconds and log to a file
    python apple_silicon_monitor.py --interval 5 --log system_stats.log
    
    # Compact display with no colors
    python apple_silicon_monitor.py --compact --no-color
"""

import threading
import platform
import subprocess
import shlex

# Check if running on Apple Silicon
def is_apple_silicon():
    """Check if the current system is running on Apple Silicon"""
    if platform.system() == "Darwin":
        # Check for arm64 architecture which indicates Apple Silicon (M1/M2/etc)
        return platform.machine() == "arm64"
    return False

# Initialize Apple Silicon monitoring
def init_monitoring():
    """Initialize Apple Silicon monitoring and return True if psutil is available"""
    has_psutil = False
    try:
        import psutil
        has_psutil = True
        print("Running on Apple Silicon")
        
        # Check for monitoring tools availability
        try:
            import importlib.util
            if importlib.util.find_spec("asitop") is not None:
                print("asitop found, can be used for detailed monitoring")
                print("Run 'asitop' in a separate terminal for detailed stats")
            
            # Check if brew-installed tools might be available
            try:
                result = subprocess.run(["which", "mactop"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("mactop found, can be used for detailed monitoring")
                    print("Run 'mactop' in a separate terminal for detailed stats")
                
                result = subprocess.run(["which", "macmon"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("macmon found, can be used for detailed monitoring")
                    print("Run 'macmon' in a separate terminal for detailed stats")
            except:
                pass
        except:
            pass
    except ImportError:
        print("psutil not found, using basic monitoring")
        print("To install: pip install psutil")
    
    return has_psutil

# Start Apple Silicon monitoring with psutil
def start_monitoring(system_info):
    """Start monitoring thread for Apple Silicon"""
    import psutil
    
    def apple_silicon_monitor():
        import time
        while True:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent)
                cpu_info = f"CPU: {cpu_avg:.1f}%\n"
                for i, cpu in enumerate(cpu_percent):
                    cpu_info += f"  CPU{i}: {cpu:.1f}%\n"
                
                # Get memory usage
                memory = psutil.virtual_memory()
                ram_info = f"RAM: {memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f}GB ({memory.percent}%)\n"
                
                # Get swap usage
                swap = psutil.swap_memory()
                swap_info = f"Swap: {swap.used / (1024**3):.1f}/{swap.total / (1024**3):.1f}GB ({swap.percent}%)\n"
                
                # Get GPU stats using system_profiler for basic info
                try:
                    # Get basic GPU info from system_profiler
                    gpu_info = "GPU: Apple Silicon GPU (Metal)\n"
                    gpu_usage = None
                    gpu_power = None
                    
                    # Try to get GPU model info
                    try:
                        result = subprocess.run(["system_profiler", "SPDisplaysDataType"], 
                                               capture_output=True, text=True, timeout=2)
                        if result.returncode == 0:
                            # Extract GPU model if available
                            output = result.stdout
                            if "Chipset Model:" in output:
                                for line in output.split('\n'):
                                    if "Chipset Model:" in line:
                                        gpu_model = line.split("Chipset Model:")[1].strip()
                                        gpu_info = f"GPU: {gpu_model}\n"
                                        break
                    except Exception:
                        pass  # Fallback to default GPU info
                    
                    # Try to get GPU info from macmon
                    try:
                        # Use macmon to get GPU usage and power info with a shorter timeout
                        # This prevents long hangs when macmon is unresponsive
                        macmon_result = subprocess.run(["macmon", "pipe", "-s", "1"], 
                                                     capture_output=True, text=True, timeout=5)
                        if macmon_result.returncode == 0:
                            import json
                            macmon_data = json.loads(macmon_result.stdout)
                            
                            # Extract GPU usage and power
                            if "gpu_usage" in macmon_data and len(macmon_data["gpu_usage"]) > 1:
                                gpu_usage = macmon_data["gpu_usage"][1] * 100  # Convert to percentage
                                gpu_info += f"  GPU Usage: {gpu_usage:.1f}%\n"
                            
                            if "gpu_power" in macmon_data:
                                gpu_power = macmon_data["gpu_power"]
                                gpu_info += f"  GPU Power: {gpu_power:.2f}W\n"
                                
                            if "temp" in macmon_data and "gpu_temp_avg" in macmon_data["temp"]:
                                gpu_temp = macmon_data["temp"]["gpu_temp_avg"]
                                gpu_info += f"  GPU Temp: {gpu_temp:.1f}°C\n"
                    except subprocess.TimeoutExpired:
                        # If macmon times out, just continue without GPU metrics
                        # This is a common issue with macmon and shouldn't crash the program
                        pass
                    except Exception as e:
                        # For other exceptions, just continue without GPU metrics
                        pass
                    
                    # Check if powermetrics is running
                    try:
                        # Use ps command to check for running powermetrics
                        ps_result = subprocess.run(shlex.split("ps aux"), 
                                                 capture_output=True, text=True, timeout=2)
                        
                        # Check if any monitoring tools are running
                        if ps_result.returncode == 0 and any(tool in ps_result.stdout for tool in ["powermetrics", "mactop", "macmon", "asitop"]):
                            if gpu_usage is None and gpu_power is None:
                                # Only add this if we didn't get GPU info from macmon
                                gpu_info += "  ✅ GPU monitoring tool is running in another terminal\n"
                                gpu_info += "  Check that terminal for detailed GPU metrics\n"
                        else:
                            # Only add instructions if we didn't get GPU info from macmon
                            if gpu_usage is None and gpu_power is None:
                                gpu_info += "  For detailed GPU metrics, try these tools:\n"
                                gpu_info += "  mactop (brew install mactop)\n"
                                gpu_info += "  macmon (brew install macmon)\n"
                                gpu_info += "  asitop (pip install asitop)\n"
                    except Exception:
                        # Add instructions for detailed metrics if checking fails
                        if gpu_usage is None and gpu_power is None:
                            gpu_info += "  For detailed GPU metrics, try these tools:\n"
                            gpu_info += "  mactop (brew install mactop)\n"
                            gpu_info += "  macmon (brew install macmon)\n"
                            gpu_info += "  asitop (pip install asitop)\n"
                except Exception as e:
                    gpu_info = "GPU: Apple Silicon GPU (Metal)\n"
                    gpu_info += f"Error getting GPU info: {str(e)}\n"
                    gpu_info += "For detailed GPU stats, try tools like 'mactop', 'macmon', or 'asitop'\n"
                
                # Store raw data for compact mode and trend analysis
                system_info["raw_data"] = {
                    "cpu": {
                        "total": cpu_avg,
                        "per_cpu": cpu_percent
                    },
                    "memory": {
                        "used_gb": memory.used / (1024**3),
                        "total_gb": memory.total / (1024**3),
                        "percent": memory.percent
                    },
                    "swap": {
                        "used_gb": swap.used / (1024**3),
                        "total_gb": swap.total / (1024**3),
                        "percent": swap.percent
                    },
                    "gpu": {
                        "model": gpu_info.split('\n')[0].replace("GPU: ", ""),
                        "monitoring_active": "monitoring tool is running" in gpu_info,
                        "usage": gpu_usage,
                        "power": gpu_power
                    }
                }
                
                system_info["text"] = f"🖥️ Apple Silicon Stats:\n{cpu_info}\n🎮 {gpu_info}\n💾 {ram_info}\n💿 {swap_info}"
            except Exception as e:
                system_info["text"] = f"Error getting system stats: {str(e)}"
            
            time.sleep(2)
    
    thread = threading.Thread(target=apple_silicon_monitor, daemon=True)
    thread.start()
    return thread

# Start basic monitoring (fallback)
def start_basic_monitoring(system_info):
    """Start basic monitoring thread (fallback method)"""
    def basic_monitor():
        import time
        while True:
            try:
                # Try to get CPU info using top command
                cpu = subprocess.check_output("top -l 1 | head -n 10", shell=True).decode()
            except Exception as e:
                cpu = f"top not available: {str(e)}"
            
            # Try to get memory info
            try:
                mem = subprocess.check_output("vm_stat", shell=True).decode()
            except Exception as e:
                mem = f"Memory info not available: {str(e)}"
            
            # Basic GPU info
            gpu_info = "GPU: Apple Silicon GPU (Metal)\n"
            gpu_info += "For detailed GPU stats, try tools like 'mactop' (brew install mactop), 'macmon' (brew install macmon), or 'asitop' (pip install asitop)\n"
            
            system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n💾 Memory Info:\n{mem}\n\n🎮 GPU Info:\n{gpu_info}"
            time.sleep(2)
    
    thread = threading.Thread(target=basic_monitor, daemon=True)
    thread.start()
    return thread


# Synchronous monitoring function for performance_monitor integration
def get_system_metrics():
    """Get current system metrics synchronously for performance monitoring"""
    import psutil
    import json
    
    metrics = {
        'cpu': {'total': 0, 'per_cpu': []},
        'memory': {'used_gb': 0, 'total_gb': 0, 'percent': 0, 'used_mb': 0, 'total_mb': 0},
        'gpu': {'utilization': None, 'power': None, 'temperature': None, 'model': 'Apple Silicon GPU'}
    }
    
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)
        metrics['cpu']['total'] = cpu_avg
        metrics['cpu']['per_cpu'] = cpu_percent
        
        # Get memory usage
        memory = psutil.virtual_memory()
        metrics['memory']['used_gb'] = memory.used / (1024**3)
        metrics['memory']['total_gb'] = memory.total / (1024**3)
        metrics['memory']['percent'] = memory.percent
        metrics['memory']['used_mb'] = memory.used / (1024**2)
        metrics['memory']['total_mb'] = memory.total / (1024**2)
        
        # Try to get GPU metrics from macmon
        try:
            macmon_result = subprocess.run(["macmon", "pipe", "-s", "1"], 
                                         capture_output=True, text=True, timeout=3)
            if macmon_result.returncode == 0:
                macmon_data = json.loads(macmon_result.stdout)
                
                # Extract GPU usage and power
                if "gpu_usage" in macmon_data and len(macmon_data["gpu_usage"]) > 1:
                    metrics['gpu']['utilization'] = macmon_data["gpu_usage"][1] * 100  # Convert to percentage
                
                if "gpu_power" in macmon_data:
                    metrics['gpu']['power'] = macmon_data["gpu_power"]
                    
                if "temp" in macmon_data and "gpu_temp_avg" in macmon_data["temp"]:
                    metrics['gpu']['temperature'] = macmon_data["temp"]["gpu_temp_avg"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # macmon not available or failed, continue without GPU metrics
            pass
        except Exception:
            # Other exceptions, continue without GPU metrics
            pass
            
    except Exception as e:
        # If psutil fails, return basic fallback metrics
        metrics = {
            'cpu': {'total': 0, 'per_cpu': []},
            'memory': {'used_gb': 0, 'total_gb': 0, 'percent': 0, 'used_mb': 0, 'total_mb': 0},
            'gpu': {'utilization': None, 'power': None, 'temperature': None, 'model': 'Apple Silicon GPU'}
        }
    
    return metrics


# Main function for testing
def main():
    """Main function to test Apple Silicon monitoring functionality"""
    import argparse
    import time
    import sys
    import os
    import signal
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apple Silicon System Monitor")
    parser.add_argument("-i", "--interval", type=float, default=2.0,
                        help="Update interval in seconds (default: 2.0)")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear the screen between updates")
    parser.add_argument("--log", type=str,
                        help="Log output to specified file")
    parser.add_argument("--compact", action="store_true",
                        help="Display information in compact format")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--version", action="version", version="Apple Silicon Monitor v1.0.0")
    args = parser.parse_args()
    
    # Set up colored output
    if args.no_color:
        # Disable colors
        BLUE = GREEN = YELLOW = RED = RESET = ""
    else:
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
    
    print(f"{GREEN}Apple Silicon System Monitor{RESET}")
    print(f"{GREEN}============================{RESET}")
    
    # Check if running on Apple Silicon
    if not is_apple_silicon():
        print(f"{RED}⚠️  This script is designed for Apple Silicon Macs.{RESET}")
        print(f"Current system: {platform.system()} {platform.machine()}")
        sys.exit(1)
    
    # Initialize monitoring
    print(f"{BLUE}🔄 Initializing monitoring...{RESET}")
    has_psutil = init_monitoring()
    
    # Create a shared dictionary for system info
    system_info = {"text": "Initializing...", "compact": args.compact}
    
    # Set up signal handler for graceful exit
    def signal_handler(sig, frame):
        print(f"\n{GREEN}👋 Monitoring stopped by user.{RESET}")
        if log_file:
            log_file.close()
            print(f"{BLUE}📝 Log file closed: {args.log}{RESET}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start appropriate monitoring based on available tools
    if has_psutil:
        print(f"{GREEN}✅ Starting monitoring with psutil...{RESET}")
        monitor_thread = start_monitoring(system_info)
    else:
        print(f"{YELLOW}⚠️ Starting basic monitoring (limited functionality)...{RESET}")
        monitor_thread = start_basic_monitoring(system_info)
    
    # Open log file if specified
    log_file = None
    if args.log:
        try:
            log_file = open(args.log, 'w')
            print(f"{BLUE}📝 Logging to {args.log}{RESET}")
        except Exception as e:
            print(f"{RED}⚠️ Error opening log file: {e}{RESET}")
            log_file = None
    
    # Display system info updates
    try:
        print(f"\n{GREEN}🚀 Monitoring started. Update interval: {args.interval}s{RESET}")
        print(f"Press {YELLOW}Ctrl+C{RESET} to exit.\n")
        
        # Track CPU/RAM trends
        cpu_history = []
        ram_history = []
        
        while True:
            # Clear screen if enabled
            if not args.no_clear:
                print("\033[H\033[J", end="")
            
            # Get current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Update system info with display preferences
            if "text" in system_info and system_info["text"] != "Initializing...":
                # Extract CPU and RAM values for trend tracking
                try:
                    lines = system_info["text"].split('\n')
                    for line in lines:
                        if line.startswith("CPU:"):
                            cpu_val = float(line.split('%')[0].split(': ')[1])
                            cpu_history.append(cpu_val)
                            # Keep only the last 10 values
                            if len(cpu_history) > 10:
                                cpu_history.pop(0)
                        elif line.startswith("💾 RAM:"):
                            ram_val = float(line.split('GB')[0].split('/')[-2])
                            ram_history.append(ram_val)
                            # Keep only the last 10 values
                            if len(ram_history) > 10:
                                ram_history.pop(0)
                except Exception:
                    pass  # Ignore parsing errors
            
            # Add trend indicators if we have history
            trend_info = ""
            if len(cpu_history) > 1 and len(ram_history) > 1:
                # CPU trend
                cpu_diff = cpu_history[-1] - cpu_history[0]
                cpu_trend = "↑" if cpu_diff > 2 else "↓" if cpu_diff < -2 else "→"
                cpu_color = RED if cpu_diff > 2 else GREEN if cpu_diff < -2 else BLUE
                
                # RAM trend
                ram_diff = ram_history[-1] - ram_history[0]
                ram_trend = "↑" if ram_diff > 0.1 else "↓" if ram_diff < -0.1 else "→"
                ram_color = RED if ram_diff > 0.1 else GREEN if ram_diff < -0.1 else BLUE
                
                trend_info = f"\n{BLUE}Trends (last {len(cpu_history)} samples):{RESET}\n"
                trend_info += f"CPU: {cpu_color}{cpu_trend}{RESET} | RAM: {ram_color}{ram_trend}{RESET}"
            
            # Print current system info
            header = f"{YELLOW}[{timestamp}]{RESET}"
            if args.compact:
                # Compact display - one line summary
                if "text" in system_info and system_info["text"] != "Initializing...":
                    try:
                        lines = system_info["text"].split('\n')
                        cpu = "CPU: N/A"
                        ram = "RAM: N/A"
                        gpu = "GPU: N/A"
                        
                        for line in lines:
                            if line.startswith("CPU:"):
                                cpu = line
                            elif line.startswith("💾 RAM:"):
                                ram = line.replace("💾 ", "")
                            elif line.startswith("🎮 GPU:"):
                                gpu = line.replace("🎮 ", "")
                                gpu = gpu.split('\n')[0]  # Just the first line
                        
                        # Add GPU usage if available in raw_data
                        if "raw_data" in system_info and system_info["raw_data"]["gpu"]["usage"] is not None:
                            gpu_usage = system_info["raw_data"]["gpu"]["usage"]
                            gpu += f" ({gpu_usage:.1f}%)"
                        
                        compact_info = f"{GREEN}{cpu}{RESET} | {BLUE}{ram}{RESET} | {YELLOW}{gpu}{RESET}"
                        output = f"{header}\n{compact_info}{trend_info}"
                    except Exception:
                        output = f"{header}\n{system_info['text']}"
                else:
                    output = f"{header}\n{system_info['text']}"
            else:
                # Full display
                colored_text = system_info["text"]
                # Add colors to the output
                colored_text = colored_text.replace("🖥️ Apple Silicon Stats:", f"{GREEN}🖥️ Apple Silicon Stats:{RESET}")
                colored_text = colored_text.replace("🎮 GPU:", f"{YELLOW}🎮 GPU:{RESET}")
                colored_text = colored_text.replace("  GPU Usage:", f"  {YELLOW}GPU Usage:{RESET}")
                colored_text = colored_text.replace("  GPU Power:", f"  {YELLOW}GPU Power:{RESET}")
                colored_text = colored_text.replace("  GPU Temp:", f"  {YELLOW}GPU Temp:{RESET}")
                colored_text = colored_text.replace("💾 RAM:", f"{BLUE}💾 RAM:{RESET}")
                colored_text = colored_text.replace("💿 Swap:", f"{BLUE}💿 Swap:{RESET}")
                
                output = f"{header}\n{colored_text}{trend_info}"
            
            print(output)
            
            # Log to file if enabled (without color codes)
            if log_file:
                # Remove ANSI color codes for log file
                clean_output = output
                for color in [BLUE, GREEN, YELLOW, RED, RESET]:
                    clean_output = clean_output.replace(color, "")
                
                log_file.write(f"{clean_output}\n{'=' * 40}\n")
                log_file.flush()
            
            # Wait before next update
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print(f"\n{GREEN}👋 Monitoring stopped by user.{RESET}")
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{RESET}")
    finally:
        # Close log file if open
        if log_file:
            log_file.close()
            print(f"{BLUE}📝 Log file closed: {args.log}{RESET}")


# Run main function if script is executed directly
if __name__ == "__main__":
    main()