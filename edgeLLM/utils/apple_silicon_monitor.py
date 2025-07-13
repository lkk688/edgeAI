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
        
        # Check if asitop is available (optional)
        try:
            import importlib.util
            if importlib.util.find_spec("asitop") is not None:
                print("asitop found, can be used for detailed monitoring")
                print("Run 'sudo asitop' in a separate terminal for detailed stats")
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
                    
                    # Check if powermetrics is running
                    try:
                        # Use ps command to check for running powermetrics
                        ps_result = subprocess.run(shlex.split("ps aux"), 
                                                 capture_output=True, text=True, timeout=2)
                        
                        # Check if powermetrics is in the output
                        if ps_result.returncode == 0 and "powermetrics" in ps_result.stdout:
                            gpu_info += "  ✅ powermetrics is running in another terminal\n"
                            gpu_info += "  Check that terminal for detailed GPU metrics\n"
                        else:
                            # Add instructions for detailed metrics
                            gpu_info += "  For detailed GPU metrics, run:\n"
                            gpu_info += "  sudo powermetrics --samplers gpu_power\n"
                    except Exception:
                        # Add instructions for detailed metrics if checking fails
                        gpu_info += "  For detailed GPU metrics, run:\n"
                        gpu_info += "  sudo powermetrics --samplers gpu_power\n"
                except Exception as e:
                    gpu_info = "GPU: Apple Silicon GPU (Metal)\n"
                    gpu_info += "For detailed GPU stats, run 'sudo powermetrics --samplers gpu_power'\n"
                
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
            gpu_info += "For detailed GPU stats, run 'sudo powermetrics --samplers gpu_power'\n"
            
            system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n💾 Memory Info:\n{mem}\n\n🎮 GPU Info:\n{gpu_info}"
            time.sleep(2)
    
    thread = threading.Thread(target=basic_monitor, daemon=True)
    thread.start()
    return thread