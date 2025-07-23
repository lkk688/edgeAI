#!/usr/bin/env python3
"""Performance Monitor - Cross-platform performance monitoring utility

This module provides a unified interface for monitoring system performance
across different platforms including:
- NVIDIA GPUs (via GPUtil or nvidia-smi)
- Jetson devices (via tegrastats)
- Apple Silicon (using platform-specific tools like macmon, asitop, or basic psutil monitoring)
- Generic fallback for other platforms

It tracks metrics such as execution time, memory usage, GPU utilization,
temperature, and power consumption when available.

Usage:
    # Create a monitor instance
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run your code here
    # ...
    
    # Stop monitoring and get results
    results = monitor.stop_monitoring()
    print(f"Execution time: {results['execution_time']:.4f} seconds")
    print(f"Memory delta: {results['memory_delta']:.2f} MB")
"""

import os
import time
import logging
import subprocess
import numpy as np
import psutil
import re
from typing import Dict, Optional
from threading import Thread, Event

# Import system monitoring utilities
from edgeLLM.utils import system_monitor
from edgeLLM.utils import jetson_monitor
from edgeLLM.utils import apple_silicon_monitor
from edgeLLM.utils import nvidia_gpu_monitor
from edgeLLM.utils.system_monitor import get_platform_info, get_platform_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPUtil only works with standard NVIDIA GPUs using nvidia-smi
# Import GPUtil with error handling
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.debug("Warning: GPUtil not available. Using alternative monitoring methods.")

class PerformanceMonitor:
    """Monitor system performance during inference across different platforms"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.gpu_stats = []
        self.power_readings = []
        self.stop_event = Event()
        
        # Detect platform using system_monitor
        platform_info = get_platform_info()
        self.is_jetson = platform_info["is_jetson"]
        self.is_apple_silicon = platform_info["is_apple_silicon"]
        self.has_nvidia_gpu = platform_info["has_nvidia_gpu"]
        self.platform_name = get_platform_name()
        
        logger.info(f"Platform detected: {self.platform_name}")
        
    def start_monitoring(self):
        """Start performance monitoring based on detected platform"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Reset monitoring state
        self.gpu_stats = []
        self.power_readings = []
        self.stop_event.clear()
        
        # Start the monitoring thread
        self.monitoring_thread = Thread(target=self._monitoring_thread)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def _monitoring_thread(self):
        """Background thread for continuous monitoring."""
        while not self.stop_event.is_set():
            # Platform-specific monitoring
            if self.is_jetson:
                # Jetson-specific monitoring using tegrastats
                try:
                    # Use tegrastats to get system information
                    tegrastats_output = subprocess.check_output("tegrastats --interval 1000 --count 1", shell=True).decode().strip()
                    
                    # Extract GPU usage
                    gpu_match = re.search(r'GR3D_FREQ (\d+)%', tegrastats_output)
                    gpu_usage = int(gpu_match.group(1)) if gpu_match else 0
                    
                    # Extract RAM usage
                    ram_match = re.search(r'RAM (\d+)/(\d+)MB', tegrastats_output)
                    ram_used = int(ram_match.group(1)) if ram_match else 0
                    ram_total = int(ram_match.group(2)) if ram_match else 0
                    
                    # Extract temperature
                    temp_match = re.search(r'CPU@(\d+\.\d+)C', tegrastats_output)
                    temp = float(temp_match.group(1)) if temp_match else 0
                    
                    # Extract power consumption
                    power_match = re.search(r'VDD_IN (\d+)/(\d+)mW', tegrastats_output)
                    power = int(power_match.group(1)) if power_match else None
                    
                    self.gpu_stats.append({
                        'memory_used': ram_used,
                        'memory_total': ram_total,
                        'utilization': gpu_usage,
                        'temperature': temp,
                        'power': power
                    })
                    
                    if power is not None:
                        self.power_readings.append(power / 1000)  # Convert mW to W
                except Exception as e:
                    logger.debug(f"Jetson tegrastats monitoring error: {e}")
                    # Fallback to basic monitoring using jetson_monitor
                    try:
                        system_info = {}
                        jetson_monitor.start_basic_monitoring(system_info)
                        
                        self.gpu_stats.append({
                            'memory_used': system_info.get('memory', {}).get('used', 0),
                            'memory_total': system_info.get('memory', {}).get('total', 0),
                            'utilization': system_info.get('gpu', {}).get('utilization', 0),
                            'temperature': system_info.get('temperature', 0),
                            'power': None
                        })
                    except Exception as e2:
                        logger.debug(f"Jetson fallback monitoring error: {e2}")
                        pass
                
            elif self.is_apple_silicon:
                # Apple Silicon monitoring
                try:
                    # Use apple_silicon_monitor's synchronous monitoring
                    from edgeLLM.utils.apple_silicon_monitor import get_system_metrics
                    
                    # Get current system metrics
                    metrics = get_system_metrics()
                    
                    # Extract information from metrics
                    gpu_utilization = metrics['gpu']['utilization']
                    gpu_temp = metrics['gpu']['temperature']
                    gpu_power = metrics['gpu']['power']
                    memory_used = metrics['memory']['used_mb']
                    memory_total = metrics['memory']['total_mb']
                    
                    self.gpu_stats.append({
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'utilization': gpu_utilization,
                        'temperature': gpu_temp,
                        'power': gpu_power
                    })
                    
                    # Add power reading if available
                    if gpu_power is not None:
                        self.power_readings.append(gpu_power)
                except Exception as e:
                    logger.debug(f"Apple Silicon monitoring error: {e}")
                    # Fallback to basic monitoring
                    try:
                        memory = psutil.virtual_memory()
                        self.gpu_stats.append({
                            'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                            'memory_total': memory.total / 1024 / 1024,
                            'utilization': 0,  # No direct utilization access
                            'temperature': 0,  # No direct temperature access
                            'power': None  # No direct power access
                        })
                    except Exception as e2:
                        logger.debug(f"Apple Silicon fallback monitoring error: {e2}")
                        pass
                
            elif self.has_nvidia_gpu:
                # Standard NVIDIA GPU monitoring
                gpu_stats_added = False
                
                # Try GPUtil first if available
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            self.gpu_stats.append({
                                'memory_used': gpus[0].memoryUsed,
                                'memory_total': gpus[0].memoryTotal,
                                'utilization': gpus[0].load * 100,
                                'temperature': gpus[0].temperature,
                                'power': None  # GPUtil doesn't provide power info
                            })
                            gpu_stats_added = True
                    except Exception as e:
                        logger.debug(f"NVIDIA GPU monitoring error with GPUtil: {e}")
                        # Will fall through to nvidia-smi fallback
                
                # Try nvidia-smi as fallback if GPUtil failed or isn't available
                if not gpu_stats_added:
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            values = result.stdout.strip().split(',')
                            if len(values) >= 5:
                                self.gpu_stats.append({
                                    'utilization': float(values[0]) if values[0].strip() else 0,
                                    'memory_used': float(values[1]) if values[1].strip() else 0,
                                    'memory_total': float(values[2]) if values[2].strip() else 0,
                                    'temperature': float(values[3]) if values[3].strip() else 0,
                                    'power': float(values[4]) if values[4].strip() else None
                                })
                                if values[4].strip():
                                    self.power_readings.append(float(values[4]))
                                gpu_stats_added = True
                    except Exception as e:
                        logger.debug(f"nvidia-smi monitoring error: {e}")
                        # Will fall through to generic monitoring
                
                # If both methods failed, use generic monitoring
                if not gpu_stats_added:
                    memory = psutil.virtual_memory()
                    self.gpu_stats.append({
                        'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                        'memory_total': memory.total / 1024 / 1024,
                        'utilization': 0,
                        'temperature': 0,
                        'power': None
                    })
            else:
                # Generic monitoring for unsupported platforms
                memory = psutil.virtual_memory()
                self.gpu_stats.append({
                    'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                    'memory_total': memory.total / 1024 / 1024,
                    'utilization': 0,
                    'temperature': 0,
                    'power': None
                })
            
            # Sleep for a short interval before next monitoring cycle
            time.sleep(1.0)
            
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        # Stop the monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=2.0)  # Wait for thread to finish with timeout
            
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Take one final measurement if we have no stats yet
        if not self.gpu_stats:
            # Platform-specific final monitoring
            if self.is_jetson:
                # Jetson-specific monitoring using tegrastats
                try:
                    # Use tegrastats to get system information
                    tegrastats_output = subprocess.check_output("tegrastats --interval 1000 --count 1", shell=True).decode().strip()
                    
                    # Extract GPU usage
                    gpu_match = re.search(r'GR3D_FREQ (\d+)%', tegrastats_output)
                    gpu_usage = int(gpu_match.group(1)) if gpu_match else 0
                    
                    # Extract RAM usage
                    ram_match = re.search(r'RAM (\d+)/(\d+)MB', tegrastats_output)
                    ram_used = int(ram_match.group(1)) if ram_match else 0
                    ram_total = int(ram_match.group(2)) if ram_match else 0
                    
                    # Extract temperature
                    temp_match = re.search(r'CPU@(\d+\.\d+)C', tegrastats_output)
                    temp = float(temp_match.group(1)) if temp_match else 0
                    
                    # Extract power consumption
                    power_match = re.search(r'VDD_IN (\d+)/(\d+)mW', tegrastats_output)
                    power = int(power_match.group(1)) if power_match else None
                    
                    self.gpu_stats.append({
                        'memory_used': ram_used,
                        'memory_total': ram_total,
                        'utilization': gpu_usage,
                        'temperature': temp,
                        'power': power
                    })
                    
                    if power is not None:
                        self.power_readings.append(power / 1000)  # Convert mW to W
                except Exception as e:
                    logger.debug(f"Jetson tegrastats monitoring error: {e}")
                    pass
                
        elif self.is_apple_silicon and not self.gpu_stats:
            # Apple Silicon monitoring - only if we don't have stats from the monitoring thread
            try:
                # Use apple_silicon_monitor's synchronous monitoring
                from edgeLLM.utils.apple_silicon_monitor import get_system_metrics
                
                # Get current system metrics
                metrics = get_system_metrics()
                
                # Extract information from metrics
                gpu_utilization = metrics['gpu']['utilization']
                gpu_temp = metrics['gpu']['temperature']
                gpu_power = metrics['gpu']['power']
                memory_used = metrics['memory']['used_mb']
                memory_total = metrics['memory']['total_mb']
                
                self.gpu_stats.append({
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'utilization': gpu_utilization,
                    'temperature': gpu_temp,
                    'power': gpu_power
                })
                
                # Add power reading if available
                if gpu_power is not None:
                    self.power_readings.append(gpu_power)
            except Exception as e:
                logger.debug(f"Apple Silicon monitoring error: {e}")
                # Fallback to basic monitoring
                try:
                    memory = psutil.virtual_memory()
                    self.gpu_stats.append({
                        'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                        'memory_total': memory.total / 1024 / 1024,
                        'utilization': 0,  # No direct utilization access
                        'temperature': 0,  # No direct temperature access
                        'power': None  # No direct power access
                    })
                except Exception as e2:
                    logger.debug(f"Apple Silicon fallback monitoring error: {e2}")
                    pass
                
        elif self.has_nvidia_gpu and not self.gpu_stats:
            # Standard NVIDIA GPU monitoring - only if we don't have stats from the monitoring thread
            gpu_stats_added = False
            
            # Try GPUtil first if available
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.gpu_stats.append({
                            'memory_used': gpus[0].memoryUsed,
                            'memory_total': gpus[0].memoryTotal,
                            'utilization': gpus[0].load * 100,
                            'temperature': gpus[0].temperature,
                            'power': None  # GPUtil doesn't provide power info
                        })
                        gpu_stats_added = True
                except Exception as e:
                    logger.debug(f"NVIDIA GPU monitoring error with GPUtil: {e}")
                    # Will fall through to nvidia-smi fallback
            
            # Try nvidia-smi as fallback if GPUtil failed or isn't available
            if not gpu_stats_added:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        values = result.stdout.strip().split(',')
                        if len(values) >= 5:
                            self.gpu_stats.append({
                                'utilization': float(values[0]) if values[0].strip() else 0,
                                'memory_used': float(values[1]) if values[1].strip() else 0,
                                'memory_total': float(values[2]) if values[2].strip() else 0,
                                'temperature': float(values[3]) if values[3].strip() else 0,
                                'power': float(values[4]) if values[4].strip() else None
                            })
                            if values[4].strip():
                                self.power_readings.append(float(values[4]))
                            gpu_stats_added = True
                except Exception as e:
                    logger.debug(f"nvidia-smi monitoring error: {e}")
                    # Will fall through to generic monitoring
            
            # If both methods failed, use generic monitoring
            if not gpu_stats_added:
                memory = psutil.virtual_memory()
                self.gpu_stats.append({
                    'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                    'memory_total': memory.total / 1024 / 1024,
                    'utilization': 0,
                    'temperature': 0,
                    'power': None
                })
        elif not self.gpu_stats:
            # Generic monitoring for unsupported platforms - only if we don't have stats from the monitoring thread
            memory = psutil.virtual_memory()
            self.gpu_stats.append({
                'memory_used': memory.used / 1024 / 1024,  # Convert to MB
                'memory_total': memory.total / 1024 / 1024,
                'utilization': 0,
                'temperature': 0,
                'power': None
            })
            
        # Calculate results
        # Ensure all values are properly initialized to avoid None subtraction
        execution_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Handle None values in memory calculations
        if self.end_memory is not None and self.start_memory is not None:
            memory_delta = float(self.end_memory - self.start_memory)
        else:
            memory_delta = 0.0
        
        # Calculate GPU utilization and memory used
        # Filter out None values for GPU utilization calculation
        utilization_values = [stat['utilization'] for stat in self.gpu_stats if stat.get('utilization') is not None]
        gpu_utilization = np.mean(utilization_values) if utilization_values else 0
        gpu_memory_used = self.gpu_stats[-1]['memory_used'] if self.gpu_stats and 'memory_used' in self.gpu_stats[-1] else 0
        
        # Calculate power consumption if available
        power_consumption = np.mean(self.power_readings) if self.power_readings else None
        
        results = {
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'gpu_utilization': gpu_utilization,
            'gpu_memory_used': gpu_memory_used
        }
        
        # Add power consumption if available
        if power_consumption is not None:
            results['power_consumption'] = power_consumption
            
        return results


def main():
    """Test the performance monitor functionality."""
    import time
    import argparse
    import logging
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test the performance monitor')
    parser.add_argument('--duration', type=int, default=10, help='Duration of the test in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Start monitoring
    print(f"Starting performance monitoring for {args.duration} seconds...")
    monitor.start_monitoring()
    
    # Simulate workload
    start_time = time.time()
    while time.time() - start_time < args.duration:
        # Create some memory load
        data = [i for i in range(1000000)]
        # Create some CPU load
        for i in range(1000000):
            _ = i * i
        # Sleep for the interval
        time.sleep(args.interval)
    
    # Stop monitoring and get results
    results = monitor.stop_monitoring()
    
    # Print results
    print("\nPerformance Monitoring Results:")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Memory Delta: {results['memory_delta']:.2f} MB")
    print(f"GPU Utilization: {results['gpu_utilization'] if results['gpu_utilization'] is not None else 'N/A'}")
    print(f"GPU Memory Used: {results['gpu_memory_used'] if results['gpu_memory_used'] is not None else 'N/A'} MB")
    print(f"Power Consumption: {results['power_consumption'] if 'power_consumption' in results else 'N/A'} W")
    
    # Print platform information
    from edgeLLM.utils.system_monitor import get_platform_info, get_platform_name
    platform_info = get_platform_info()
    print(f"\nPlatform: {get_platform_name()}")
    print(f"Jetson: {platform_info['is_jetson']}")
    print(f"Apple Silicon: {platform_info['is_apple_silicon']}")
    print(f"NVIDIA GPU: {platform_info['has_nvidia_gpu']}")


if __name__ == "__main__":
    main()