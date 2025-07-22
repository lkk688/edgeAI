#!/usr/bin/env python3
"""
Performance Monitor - Cross-platform performance monitoring utility

This module provides a unified interface for monitoring system performance
across different platforms including:
- NVIDIA GPUs (via GPUtil or nvidia-smi)
- Jetson devices (via jetson-stats/jtop)
- Apple Silicon (via system monitoring)
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

import time
import logging
import subprocess
import numpy as np
import psutil
from typing import Dict, Optional

# Import system monitoring utilities
from edgeLLM.utils import system_monitor
from edgeLLM.utils import jetson_monitor
from edgeLLM.utils import apple_silicon_monitor
from edgeLLM.utils import nvidia_gpu_monitor

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
        self.platform_info = None
        self.power_readings = []
        
        # Detect platform
        self.platform_info = system_monitor.init_monitoring()
        self.is_jetson = self.platform_info["is_jetson"]
        self.is_apple_silicon = self.platform_info["is_apple_silicon"]
        self.has_nvidia_gpu = self.platform_info["has_nvidia_gpu"]
        
        logger.info(f"Platform detected: {system_monitor.get_platform_name()}")
        
    def start_monitoring(self):
        """Start performance monitoring based on detected platform"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Platform-specific monitoring
        if self.is_jetson:
            # Jetson-specific monitoring
            try:
                # Try to use jtop if available
                from jtop import jtop
                jetson = jtop()
                jetson.start()
                if jetson.ok():
                    stats = jetson.stats
                    self.gpu_stats.append({
                        'memory_used': stats['RAM']['use'],
                        'memory_total': stats['RAM']['tot'],
                        'utilization': stats['GPU']['val'],
                        'temperature': stats['Temp']['GPU'] if 'GPU' in stats['Temp'] else stats['Temp']['CPU'],
                        'power': stats['Power']['tot'] if 'Power' in stats and stats['Power'] else None
                    })
                    if 'Power' in stats and stats['Power']:
                        self.power_readings.append(stats['Power']['tot'] / 1000)  # Convert mW to W
                jetson.close()
            except (ImportError, Exception) as e:
                logger.debug(f"Jetson jtop monitoring error: {e}")
                # Fallback to basic monitoring
                pass
                
        elif self.is_apple_silicon:
            # Apple Silicon monitoring
            try:
                # Basic monitoring for Apple Silicon
                self.gpu_stats.append({
                    'memory_used': self.start_memory,  # No direct GPU memory access
                    'memory_total': psutil.virtual_memory().total / 1024 / 1024,
                    'utilization': 0,  # No direct utilization access
                    'temperature': 0,  # No direct temperature access
                    'power': None  # No direct power access
                })
            except Exception as e:
                logger.debug(f"Apple Silicon monitoring error: {e}")
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
                self.gpu_stats.append({
                    'memory_used': self.start_memory,
                    'memory_total': psutil.virtual_memory().total / 1024 / 1024,
                    'utilization': 0,
                    'temperature': 0,
                    'power': None
                })
        else:
            # Generic monitoring for unsupported platforms
            self.gpu_stats.append({
                'memory_used': self.start_memory,
                'memory_total': psutil.virtual_memory().total / 1024 / 1024,
                'utilization': 0,
                'temperature': 0,
                'power': None
            })
            
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Platform-specific final monitoring
        if self.is_jetson:
            # Jetson-specific monitoring
            try:
                # Try to use jtop if available
                from jtop import jtop
                jetson = jtop()
                jetson.start()
                if jetson.ok():
                    stats = jetson.stats
                    self.gpu_stats.append({
                        'memory_used': stats['RAM']['use'],
                        'memory_total': stats['RAM']['tot'],
                        'utilization': stats['GPU']['val'],
                        'temperature': stats['Temp']['GPU'] if 'GPU' in stats['Temp'] else stats['Temp']['CPU'],
                        'power': stats['Power']['tot'] if 'Power' in stats and stats['Power'] else None
                    })
                    if 'Power' in stats and stats['Power']:
                        self.power_readings.append(stats['Power']['tot'] / 1000)  # Convert mW to W
                jetson.close()
            except (ImportError, Exception) as e:
                logger.debug(f"Jetson jtop monitoring error: {e}")
                pass
                
        elif self.is_apple_silicon:
            # Apple Silicon monitoring
            try:
                # Basic monitoring for Apple Silicon
                self.gpu_stats.append({
                    'memory_used': self.end_memory,  # No direct GPU memory access
                    'memory_total': psutil.virtual_memory().total / 1024 / 1024,
                    'utilization': 0,  # No direct utilization access
                    'temperature': 0,  # No direct temperature access
                    'power': None  # No direct power access
                })
            except Exception as e:
                logger.debug(f"Apple Silicon monitoring error: {e}")
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
                self.gpu_stats.append({
                    'memory_used': self.end_memory,
                    'memory_total': psutil.virtual_memory().total / 1024 / 1024,
                    'utilization': 0,
                    'temperature': 0,
                    'power': None
                })
        else:
            # Generic monitoring for unsupported platforms
            self.gpu_stats.append({
                'memory_used': self.end_memory,
                'memory_total': psutil.virtual_memory().total / 1024 / 1024,
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
        
        results = {
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'gpu_utilization': np.mean([stat['utilization'] for stat in self.gpu_stats]) if self.gpu_stats else 0,
            'gpu_memory_used': self.gpu_stats[-1]['memory_used'] if self.gpu_stats and 'memory_used' in self.gpu_stats[-1] else 0
        }
        
        # Add power consumption if available
        if self.power_readings:
            results['power_consumption'] = np.mean(self.power_readings)
            
        return results