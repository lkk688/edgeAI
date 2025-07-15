#!/usr/bin/env python3
# performance_monitor.py
import psutil
import time
import json
import argparse
from datetime import datetime

class JetsonMonitor:
    def __init__(self):
        self.data = []
        
    def get_cpu_info(self):
        """Get CPU information"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        cpu_info = {
            'usage_percent': cpu_percent,
            'frequencies': [f.current for f in cpu_freq] if cpu_freq else [],
            'load_average': psutil.getloadavg()
        }
        
        return cpu_info
        
    def get_memory_info(self):
        """Get memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
        
    def get_disk_info(self):
        """Get disk information"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        return {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': (disk_usage.used / disk_usage.total) * 100,
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }
        
    def get_network_info(self):
        """Get network information"""
        net_io = psutil.net_io_counters()
        
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
    def get_thermal_info(self):
        """Get thermal information"""
        thermal_info = {}
        
        try:
            # Read thermal zones
            import glob
            for zone_path in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
                zone_name = zone_path.split('/')[-2]
                with open(zone_path, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    thermal_info[zone_name] = temp
        except:
            pass
            
        return thermal_info
        
    def get_gpu_info(self):
        """Get GPU information (if available)"""
        gpu_info = {}
        
        try:
            # Try to read GPU memory usage
            with open('/proc/driver/nvidia/gpus/0000:00:00.0/used_memory', 'r') as f:
                gpu_info['memory_used'] = int(f.read().strip())
        except:
            pass
            
        return gpu_info
        
    def collect_data(self):
        """Collect all system data"""
        timestamp = datetime.now().isoformat()
        
        data_point = {
            'timestamp': timestamp,
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'network': self.get_network_info(),
            'thermal': self.get_thermal_info(),
            'gpu': self.get_gpu_info()
        }
        
        self.data.append(data_point)
        return data_point
        
    def display_data(self, data_point):
        """Display data in human-readable format"""
        print(f"\n=== System Monitor - {data_point['timestamp']} ===")
        
        # CPU
        cpu = data_point['cpu']
        print(f"CPU Usage: {[f'{p:5.1f}%' for p in cpu['usage_percent']]}")
        if cpu['frequencies']:
            print(f"CPU Freq:  {[f'{f:6.0f}' for f in cpu['frequencies']]} MHz")
        print(f"Load Avg:  {cpu['load_average']}")
        
        # Memory
        mem = data_point['memory']
        print(f"Memory:    {mem['percent']:5.1f}% ({mem['used']//1024//1024:,} MB / {mem['total']//1024//1024:,} MB)")
        if mem['swap_total'] > 0:
            print(f"Swap:      {mem['swap_percent']:5.1f}% ({mem['swap_used']//1024//1024:,} MB / {mem['swap_total']//1024//1024:,} MB)")
        
        # Disk
        disk = data_point['disk']
        print(f"Disk:      {disk['percent']:5.1f}% ({disk['used']//1024//1024//1024:,} GB / {disk['total']//1024//1024//1024:,} GB)")
        
        # Thermal
        thermal = data_point['thermal']
        if thermal:
            temps = [f"{zone}: {temp:.1f}°C" for zone, temp in thermal.items()]
            print(f"Thermal:   {', '.join(temps)}")
        
        # GPU
        gpu = data_point['gpu']
        if gpu:
            if 'memory_used' in gpu:
                print(f"GPU Mem:   {gpu['memory_used']} bytes")
                
    def save_data(self, filename):
        """Save collected data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved to {filename}")
        
    def monitor(self, duration=60, interval=1, save_file=None):
        """Monitor system for specified duration"""
        print(f"Starting system monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                data_point = self.collect_data()
                self.display_data(data_point)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            
        if save_file:
            self.save_data(save_file)

def main():
    parser = argparse.ArgumentParser(description='Jetson System Performance Monitor')
    parser.add_argument('-d', '--duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('-i', '--interval', type=int, default=1, help='Sampling interval in seconds')
    parser.add_argument('-s', '--save', type=str, help='Save data to JSON file')
    
    args = parser.parse_args()
    
    monitor = JetsonMonitor()
    monitor.monitor(duration=args.duration, interval=args.interval, save_file=args.save)

if __name__ == "__main__":
    main()