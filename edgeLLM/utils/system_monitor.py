import platform
import threading
import subprocess

# Import platform-specific monitoring modules
from . import jetson_monitor
from . import apple_silicon_monitor
from . import nvidia_gpu_monitor

# System info dictionary to be updated by monitoring threads
system_info = {"text": "Initializing system monitor..."}

# Platform detection flags
IS_JETSON = False
IS_APPLE_SILICON = False
HAS_NVIDIA_GPU = False

# Initialize monitoring based on platform
def init_monitoring():
    """Initialize system monitoring based on detected platform"""
    global IS_JETSON, IS_APPLE_SILICON, HAS_NVIDIA_GPU
    
    # Check for Jetson platform
    IS_JETSON = jetson_monitor.is_jetson()
    
    # Check for Apple Silicon
    IS_APPLE_SILICON = apple_silicon_monitor.is_apple_silicon()
    
    # Check for NVIDIA GPU on other systems
    if not IS_JETSON and not IS_APPLE_SILICON:
        HAS_NVIDIA_GPU = nvidia_gpu_monitor.has_nvidia_gpu()
    
    return {
        "is_jetson": IS_JETSON,
        "is_apple_silicon": IS_APPLE_SILICON,
        "has_nvidia_gpu": HAS_NVIDIA_GPU
    }

# Start appropriate monitoring thread based on platform
def start_monitoring():
    """Start the appropriate monitoring thread based on detected platform"""
    # Initialize platform detection
    platform_info = init_monitoring()
    
    if platform_info["is_jetson"]:
        # Jetson monitoring
        has_jtop = jetson_monitor.init_monitoring()
        if has_jtop:
            jetson_monitor.start_jtop_monitoring(system_info)
        else:
            jetson_monitor.start_basic_monitoring(system_info)
    
    elif platform_info["is_apple_silicon"]:
        # Apple Silicon monitoring
        has_psutil = apple_silicon_monitor.init_monitoring()
        if has_psutil:
            apple_silicon_monitor.start_monitoring(system_info)
        else:
            apple_silicon_monitor.start_basic_monitoring(system_info)
    
    elif platform_info["has_nvidia_gpu"]:
        # NVIDIA GPU monitoring
        has_gputil = nvidia_gpu_monitor.init_monitoring()
        try:
            import psutil
            if has_gputil:
                nvidia_gpu_monitor.start_gputil_monitoring(system_info)
            else:
                nvidia_gpu_monitor.start_nvidia_smi_monitoring(system_info)
        except ImportError:
            print("psutil not found, using basic monitoring with nvidia-smi")
            print("To install: pip install psutil")
            nvidia_gpu_monitor.start_nvidia_smi_monitoring(system_info)
    
    else:
        # Basic monitoring for other systems
        start_basic_monitoring()
    
    return system_info

# Basic monitoring for unsupported platforms
def start_basic_monitoring():
    """Start basic monitoring for unsupported platforms"""
    def basic_monitor():
        import time
        while True:
            try:
                # Try to get CPU info using top command
                if platform.system() == "Darwin":  # macOS
                    cpu = subprocess.check_output("top -l 1 | head -n 10", shell=True).decode()
                else:  # Linux and others
                    cpu = subprocess.check_output("top -b -n1 | head -n 5", shell=True).decode()
            except Exception as e:
                cpu = f"top not available: {str(e)}"
            
            # Try to get memory info
            try:
                if platform.system() == "Darwin":  # macOS
                    mem = subprocess.check_output("vm_stat", shell=True).decode()
                else:  # Linux and others
                    mem = subprocess.check_output("free -h", shell=True).decode()
            except Exception as e:
                mem = f"Memory info not available: {str(e)}"
            
            system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n💾 Memory Info:\n{mem}\n\n🎮 GPU Info:\nGPU info not available"
            time.sleep(2)
    
    thread = threading.Thread(target=basic_monitor, daemon=True)
    thread.start()
    return thread

# Get current system info text
def get_system_info():
    """Get the current system info text"""
    return system_info["text"]

# Get platform information
def get_platform_info():
    """Get information about the detected platform"""
    return {
        "is_jetson": IS_JETSON,
        "is_apple_silicon": IS_APPLE_SILICON,
        "has_nvidia_gpu": HAS_NVIDIA_GPU,
        "platform_name": get_platform_name()
    }

# Get platform name for display
def get_platform_name():
    """Get a user-friendly platform name for display"""
    if IS_JETSON:
        return "Jetson"
    elif IS_APPLE_SILICON:
        return "Apple Silicon"
    elif HAS_NVIDIA_GPU:
        return "NVIDIA GPU"
    else:
        return "Cross-Platform"