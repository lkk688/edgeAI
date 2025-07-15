#!/usr/bin/env python3
import psutil
import time
import subprocess

def get_gpu_temp():
    """Get GPU temperature from thermal zone"""
    try:
        with open('/sys/class/thermal/thermal_zone1/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
        return temp
    except:
        return None

def get_cpu_freq():
    """Get current CPU frequencies"""
    freqs = []
    for i in range(psutil.cpu_count()):
        try:
            with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq', 'r') as f:
                freq = int(f.read().strip()) / 1000  # Convert to MHz
                freqs.append(freq)
        except:
            freqs.append(0)
    return freqs

def monitor_system():
    """Monitor system performance"""
    print("System Performance Monitor")
    print("=" * 50)
    
    while True:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freqs = get_cpu_freq()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # GPU temperature
            gpu_temp = get_gpu_temp()
            
            # Display information
            print(f"\rCPU Usage: {[f'{p:5.1f}%' for p in cpu_percent]}")
            print(f"CPU Freqs: {[f'{f:6.0f}' for f in cpu_freqs]} MHz")
            print(f"Memory: {memory.percent:5.1f}% ({memory.used//1024//1024:,} MB / {memory.total//1024//1024:,} MB)")
            if gpu_temp:
                print(f"GPU Temp: {gpu_temp:5.1f}°C")
            print("\033[4A", end='')  # Move cursor up 4 lines
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n" * 4)
            print("Monitoring stopped.")
            break

if __name__ == "__main__":
    monitor_system()