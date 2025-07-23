import threading
import subprocess

# Check if running on Jetson
def is_jetson():
    """Check if the current system is a Jetson device"""
    # Primary method: check device tree model
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "NVIDIA Jetson" in f.read():
                return True
    except:
        pass
    
    # Fallback method for containers: check if tegrastats is available
    # tegrastats is Jetson-specific and typically mounted into containers
    try:
        subprocess.check_output("which tegrastats", shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        pass
    
    # Additional fallback: check for Jetson-specific files/directories
    jetson_indicators = [
        "/usr/bin/tegrastats",
        "/sys/devices/gpu.0",
        "/proc/device-tree/compatible"
    ]
    
    for indicator in jetson_indicators:
        try:
            if indicator.endswith("compatible"):
                with open(indicator, "r") as f:
                    if "nvidia,tegra" in f.read().lower():
                        return True
            else:
                import os
                if os.path.exists(indicator):
                    return True
        except:
            continue
    
    return False

# Initialize Jetson monitoring
def init_monitoring():
    """Initialize Jetson monitoring and check if tegrastats is available"""
    has_tegrastats = False
    try:
        # Check if tegrastats is available
        subprocess.check_output("which tegrastats", shell=True)
        has_tegrastats = True
        print("Using tegrastats for monitoring")
    except subprocess.CalledProcessError:
        print("tegrastats not found, monitoring may not work properly")
        print("tegrastats should be available by default on Jetson devices")
        print("In containers, ensure tegrastats is mounted: -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro")
    
    return has_tegrastats

# Start Jetson monitoring with tegrastats
def start_tegrastats_monitoring(system_info):
    """Start monitoring thread using tegrastats"""
    import time
    import re
    
    def jetson_monitor():
        while True:
            try:
                # Get CPU usage with top
                cpu_output = subprocess.check_output("top -bn1 | grep '%Cpu'", shell=True).decode().strip()
                cpu_usage = re.search(r'(\d+\.\d+)\s+us', cpu_output)
                cpu_usage = float(cpu_usage.group(1)) if cpu_usage else 0
                cpu_info = f"CPU: {cpu_usage}%\n"
                
                # Get CPU frequency
                try:
                    cpu_freq_output = subprocess.check_output("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", shell=True).decode().strip()
                    cpu_freq = int(cpu_freq_output) / 1000  # Convert to MHz
                    cpu_info += f"CPU Frequency: {cpu_freq} MHz\n"
                except:
                    pass
                
                # Get per-CPU usage
                try:
                    cpu_detailed = subprocess.check_output("top -bn1 | grep '%Cpu' | tail -n +2", shell=True).decode().strip()
                    for i, line in enumerate(cpu_detailed.split('\n')):
                        usage = re.search(r'(\d+\.\d+)\s+us', line)
                        if usage:
                            cpu_info += f"  CPU{i}: {usage.group(1)}%\n"
                except:
                    pass
                
                # Get GPU and memory info from tegrastats
                tegrastats_output = subprocess.check_output("tegrastats --interval 1000 --count 1", shell=True).decode().strip()
                
                # Extract GPU usage
                gpu_match = re.search(r'GR3D_FREQ (\d+)%', tegrastats_output)
                gpu_usage = gpu_match.group(1) if gpu_match else "N/A"
                
                # Extract GPU frequency
                gpu_freq_match = re.search(r'GR3D_FREQ\s+\d+%@(\d+)', tegrastats_output)
                gpu_freq = gpu_freq_match.group(1) if gpu_freq_match else "N/A"
                
                gpu_info = f"GPU: {gpu_usage}% @ {gpu_freq}MHz\n"
                
                # Extract RAM usage
                ram_match = re.search(r'RAM (\d+)/(\d+)MB', tegrastats_output)
                if ram_match:
                    ram_used = ram_match.group(1)
                    ram_total = ram_match.group(2)
                    ram_percent = round(int(ram_used) / int(ram_total) * 100, 1)
                    ram_info = f"RAM: {ram_used}/{ram_total}MB ({ram_percent}%)\n"
                else:
                    ram_info = "RAM: N/A\n"
                
                # Extract temperature
                temp_match = re.search(r'CPU@(\d+\.\d+)C', tegrastats_output)
                temp = temp_match.group(1) if temp_match else "N/A"
                temp_info = f"Temperature: {temp}°C\n"
                
                # Extract power consumption
                power_match = re.search(r'VDD_IN (\d+)/(\d+)mW', tegrastats_output)
                power = power_match.group(1) if power_match else "N/A"
                power_info = f"Power: {power}mW\n" if power != "N/A" else ""
                
                system_info["text"] = f"🖥️ Jetson Stats:\n{cpu_info}\n🎮 {gpu_info}\n💾 {ram_info}\n🌡️ {temp_info}\n⚡ {power_info}"
            except Exception as e:
                system_info["text"] = f"Error getting tegrastats: {str(e)}"
            
            time.sleep(2)  # Update every 2 seconds
    
    thread = threading.Thread(target=jetson_monitor, daemon=True)
    thread.start()
    return thread

# Start Jetson monitoring with basic commands (fallback)
def start_basic_monitoring(system_info):
    """Start monitoring thread using basic commands (fallback method)"""
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
            
            # Try to get GPU info with nvidia-smi if available
            try:
                gpu = subprocess.check_output("nvidia-smi", 
                                            shell=True, stderr=subprocess.STDOUT).decode()
                gpu_info = gpu
            except Exception as e:
                gpu_info = f"nvidia-smi not available: {str(e)}"
            
            system_info["text"] = f"🖥️ CPU Info:\n{cpu}\n\n💾 Memory Info:\n{mem}\n\n🎮 GPU Info:\n{gpu_info}"
            import time
            time.sleep(2)
    
    thread = threading.Thread(target=basic_monitor, daemon=True)
    thread.start()
    return thread

# Start monitoring based on available tools
def start_monitoring(system_info=None):
    """Start Jetson monitoring with the best available method"""
    if system_info is None:
        system_info = {"text": "Initializing monitoring..."}
    
    if not is_jetson():
        print("Not running on a Jetson device (or Jetson detection failed in container), using basic monitoring")
        return start_basic_monitoring(system_info)
    
    has_tegrastats = init_monitoring()
    
    if has_tegrastats:
        return start_tegrastats_monitoring(system_info)
    else:
        return start_basic_monitoring(system_info)

# Main function to test the monitoring
if __name__ == "__main__":
    import time
    
    print("Starting Jetson monitoring test...")
    system_info = {"text": "Initializing..."}
    
    # Start monitoring
    monitor_thread = start_monitoring(system_info)
    
    # Display monitoring information for 30 seconds
    try:
        for _ in range(15):  # 15 iterations * 2 seconds = 30 seconds
            print("\n" + "-"*50)
            print(system_info["text"])
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring test stopped by user")
    
    print("\nJetson monitoring test completed")