import threading
import subprocess

# Check if running on Jetson
def is_jetson():
    """Check if the current system is a Jetson device"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "NVIDIA Jetson" in f.read()
    except:
        return False

# Initialize Jetson monitoring
def init_monitoring():
    """Initialize Jetson monitoring and return True if jtop is available"""
    has_jtop = False
    try:
        # Try to import jetson-stats if available
        from jtop import jtop
        has_jtop = True
        print("Using jetson-stats for monitoring")
    except ImportError:
        print("jetson-stats not found, falling back to tegrastats")
        print("To install: sudo -H pip install -U jetson-stats")
    
    return has_jtop

# Start Jetson monitoring with jtop
def start_jtop_monitoring(system_info):
    """Start monitoring thread using jtop"""
    from jtop import jtop
    
    def jetson_monitor():
        with jtop() as jetson:
            while jetson.ok():
                stats = jetson.stats
                cpu_info = f"CPU: {stats['CPU']['val']}% @ {stats['CPU']['frq']}MHz\n"
                for i, cpu in enumerate(stats['CPU']['CPU']):
                    cpu_info += f"  CPU{i}: {cpu['val']}% @ {cpu['frq']}MHz\n"
                
                gpu_info = f"GPU: {stats['GPU']['val']}% @ {stats['GPU']['frq']}MHz\n"
                ram_info = f"RAM: {stats['RAM']['use']}/{stats['RAM']['tot']}MB ({stats['RAM']['val']}%)\n"
                temp_info = f"Temperature: {stats['Temp']['CPU']}°C\n"
                power_info = ""
                if 'Power' in stats and stats['Power']:
                    power_info = f"Power: {stats['Power']['tot']}mW\n"
                
                system_info["text"] = f"🖥️ Jetson Stats:\n{cpu_info}\n🎮 {gpu_info}\n💾 {ram_info}\n🌡️ {temp_info}\n⚡ {power_info}"
                # jtop handles the sleep internally
    
    thread = threading.Thread(target=jetson_monitor, daemon=True)
    thread.start()
    return thread

# Start Jetson monitoring with tegrastats (fallback)
def start_basic_monitoring(system_info):
    """Start monitoring thread using tegrastats (fallback method)"""
    def basic_monitor():
        while True:
            try:
                # Try to get CPU info using top command
                cpu = subprocess.check_output("top -b -n1 | head -n 5", shell=True).decode()
            except Exception as e:
                cpu = f"top not available: {str(e)}"
            
            # Try to get memory info
            try:
                mem = subprocess.check_output("free -h", shell=True).decode()
            except Exception as e:
                mem = f"Memory info not available: {str(e)}"
            
            # Try tegrastats with proper error handling
            try:
                gpu = subprocess.check_output("tegrastats --interval 1000 --count 1", 
                                            shell=True, stderr=subprocess.STDOUT).decode()
                gpu_info = gpu
            except Exception as e:
                gpu_info = f"tegrastats not available: {str(e)}"
            
            system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n💾 Memory Info:\n{mem}\n\n🎮 GPU Info:\n{gpu_info}"
            import time
            time.sleep(2)
    
    thread = threading.Thread(target=basic_monitor, daemon=True)
    thread.start()
    return thread