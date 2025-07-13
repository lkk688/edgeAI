import threading
import subprocess
import platform

# Check if system has NVIDIA GPU
def has_nvidia_gpu():
    """Check if the system has an NVIDIA GPU"""
    try:
        # Try to detect NVIDIA GPU using nvidia-smi
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except:
        return False

# Initialize NVIDIA GPU monitoring
def init_monitoring():
    """Initialize NVIDIA GPU monitoring and return True if GPUtil is available"""
    has_gputil = False
    print("NVIDIA GPU detected")
    
    # Try to import GPUtil if available
    try:
        import GPUtil
        has_gputil = True
        print("Using GPUtil for GPU monitoring")
    except ImportError:
        print("GPUtil not found, using nvidia-smi for basic monitoring")
        print("To install: pip install gputil")
    
    return has_gputil

# Start NVIDIA GPU monitoring with GPUtil
def start_gputil_monitoring(system_info):
    """Start monitoring thread using GPUtil"""
    import GPUtil
    import psutil
    
    def nvidia_monitor():
        import time
        while True:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent)
                cpu_info = f"CPU: {cpu_avg:.1f}%\n"
                for i, cpu in enumerate(cpu_percent[:4]):  # Limit to first 4 CPUs to save space
                    cpu_info += f"  CPU{i}: {cpu:.1f}%\n"
                if len(cpu_percent) > 4:
                    cpu_info += f"  ... and {len(cpu_percent) - 4} more CPUs\n"
                
                # Get memory usage
                memory = psutil.virtual_memory()
                ram_info = f"RAM: {memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f}GB ({memory.percent}%)\n"
                
                # Get GPU info using GPUtil
                gpus = GPUtil.getGPUs()
                gpu_info = ""
                for i, gpu in enumerate(gpus):
                    gpu_info += f"GPU {i}: {gpu.name}\n"
                    gpu_info += f"  Load: {gpu.load*100:.1f}%\n"
                    gpu_info += f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)\n"
                    gpu_info += f"  Temperature: {gpu.temperature}°C\n"
                
                system_info["text"] = f"🖥️ System Stats:\n{cpu_info}\n🎮 NVIDIA GPU Stats:\n{gpu_info}\n💾 {ram_info}"
            except Exception as e:
                system_info["text"] = f"Error getting system stats: {str(e)}"
            
            time.sleep(2)
    
    thread = threading.Thread(target=nvidia_monitor, daemon=True)
    thread.start()
    return thread

# Start NVIDIA GPU monitoring with nvidia-smi (fallback)
def start_nvidia_smi_monitoring(system_info):
    """Start monitoring thread using nvidia-smi directly (fallback method)"""
    def nvidia_smi_monitor():
        import time
        while True:
            try:
                # Get CPU info
                if platform.system() == "Darwin":  # macOS
                    cpu = subprocess.check_output("top -l 1 | head -n 10", shell=True).decode()
                else:  # Linux and others
                    cpu = subprocess.check_output("top -b -n1 | head -n 5", shell=True).decode()
                
                # Get GPU info using nvidia-smi
                gpu = subprocess.check_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv", 
                                            shell=True).decode()
                
                system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n🎮 NVIDIA GPU Info:\n{gpu}"
            except Exception as e:
                system_info["text"] = f"Error getting system stats: {str(e)}"
            
            time.sleep(2)
    
    thread = threading.Thread(target=nvidia_smi_monitor, daemon=True)
    thread.start()
    return thread