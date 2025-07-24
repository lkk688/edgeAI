#!/usr/bin/env python3
"""
Test script to debug tegrastats parsing and GPU information extraction
"""

import subprocess
import re

def test_tegrastats_parsing():
    """Test tegrastats parsing with improved patterns"""
    print("Testing tegrastats parsing...")
    
    try:
        # Get tegrastats output
        print("Running tegrastats command...")
        tegrastats_output = subprocess.check_output(
            "timeout 2 tegrastats --interval 1000 | head -n 1", 
            shell=True
        ).decode().strip()
        
        print(f"Raw tegrastats output: {tegrastats_output}")
        print("\n" + "="*50)
        
        # Test GPU usage patterns
        print("Testing GPU usage patterns:")
        gpu_usage = "N/A"
        gpu_patterns = [
            r'GR3D_FREQ (\d+)%',  # Standard pattern
            r'GPU (\d+)%',         # Alternative pattern
            r'gr3d (\d+)%',        # Lowercase pattern
            r'GR3D\s+(\d+)%'       # With whitespace
        ]
        
        for i, pattern in enumerate(gpu_patterns):
            gpu_match = re.search(pattern, tegrastats_output, re.IGNORECASE)
            print(f"  Pattern {i+1}: {pattern} -> {'MATCH: ' + gpu_match.group(1) + '%' if gpu_match else 'NO MATCH'}")
            if gpu_match and gpu_usage == "N/A":
                gpu_usage = gpu_match.group(1)
        
        print(f"Final GPU usage: {gpu_usage}%")
        
        # Test GPU frequency patterns
        print("\nTesting GPU frequency patterns:")
        gpu_freq = "N/A"
        gpu_freq_patterns = [
            r'GR3D_FREQ\s+\d+%@(\d+)',  # Standard pattern
            r'GPU\s+\d+%@(\d+)',        # Alternative pattern
            r'gr3d\s+\d+%@(\d+)',      # Lowercase pattern
            r'@(\d+)MHz',               # Simple MHz pattern
            r'(\d+)MHz'                 # Just MHz
        ]
        
        for i, pattern in enumerate(gpu_freq_patterns):
            gpu_freq_match = re.search(pattern, tegrastats_output, re.IGNORECASE)
            print(f"  Pattern {i+1}: {pattern} -> {'MATCH: ' + gpu_freq_match.group(1) + 'MHz' if gpu_freq_match else 'NO MATCH'}")
            if gpu_freq_match and gpu_freq == "N/A":
                gpu_freq = gpu_freq_match.group(1)
        
        print(f"Final GPU frequency: {gpu_freq}MHz")
        
        # Test RAM patterns
        print("\nTesting RAM patterns:")
        ram_info = "N/A"
        ram_patterns = [
            r'RAM (\d+)/(\d+)MB',     # Standard pattern
            r'MEM (\d+)/(\d+)MB',     # Alternative pattern
            r'ram (\d+)/(\d+)mb',     # Lowercase pattern
            r'RAM\s+(\d+)/(\d+)\s*MB' # With whitespace
        ]
        
        for i, pattern in enumerate(ram_patterns):
            ram_match = re.search(pattern, tegrastats_output, re.IGNORECASE)
            if ram_match:
                ram_used = ram_match.group(1)
                ram_total = ram_match.group(2)
                ram_percent = round(int(ram_used) / int(ram_total) * 100, 1)
                ram_info = f"{ram_used}/{ram_total}MB ({ram_percent}%)"
                print(f"  Pattern {i+1}: {pattern} -> MATCH: {ram_info}")
                break
            else:
                print(f"  Pattern {i+1}: {pattern} -> NO MATCH")
        
        print(f"Final RAM info: {ram_info}")
        
        # Test temperature patterns
        print("\nTesting temperature patterns:")
        temp_match = re.search(r'CPU@(\d+\.\d+)C', tegrastats_output)
        temp = temp_match.group(1) if temp_match else "N/A"
        print(f"Temperature: {temp}°C")
        
        # Test power patterns
        print("\nTesting power patterns:")
        power_match = re.search(r'VDD_IN (\d+)/(\d+)mW', tegrastats_output)
        power = power_match.group(1) if power_match else "N/A"
        print(f"Power: {power}mW")
        
        print("\n" + "="*50)
        print(f"Final GPU info: GPU: {gpu_usage}% @ {gpu_freq}MHz")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tegrastats: {e}")
        print("Make sure you're running on a Jetson device with tegrastats available")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_tegrastats_parsing()