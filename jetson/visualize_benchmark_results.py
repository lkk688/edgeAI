#!/usr/bin/env python3
"""
Benchmark Results Visualization Tool for Unified LLM Demo

This script loads benchmark results from JSON files generated by the unified_llm_demo.py
and creates visualizations to compare performance across different backends and models.

Usage:
    python visualize_benchmark_results.py --file benchmark_results/20230615_123456_llm_comparison_results.json
    python visualize_benchmark_results.py --dir benchmark_results --latest
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization will be disabled.")


def load_benchmark_results(file_path):
    """
    Load benchmark results from a JSON file.
    
    Args:
        file_path (str): Path to the benchmark results JSON file
        
    Returns:
        dict: Loaded benchmark results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading benchmark results: {str(e)}")
        return None


def find_latest_benchmark_file(directory):
    """
    Find the most recent benchmark results file in the specified directory.
    
    Args:
        directory (str): Directory containing benchmark results files
        
    Returns:
        str: Path to the most recent benchmark results file
    """
    try:
        files = list(Path(directory).glob("*_llm_comparison_results.json"))
        if not files:
            print(f"No benchmark results found in {directory}")
            return None
        
        # Sort by modification time (most recent first)
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        print(f"Found latest benchmark file: {latest_file}")
        return str(latest_file)
    except Exception as e:
        print(f"Error finding latest benchmark file: {str(e)}")
        return None


def create_performance_comparison(results, output_file="benchmark_comparison.png"):
    """
    Create a visualization comparing performance across backends and models.
    
    Args:
        results (dict): Benchmark results data
        output_file (str): Output file path for the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create visualization.")
        return
    
    try:
        # Extract data for plotting
        backends = []
        avg_times = []
        std_devs = []
        memory_usages = []
        
        for result in results["backend_results"]:
            backend_name = f"{result['backend']}\n{result['model'].split('/')[-1]}"
            backends.append(backend_name)
            
            stats = result["overall_stats"]
            avg_times.append(stats.get("avg_time", 0))
            std_devs.append(stats.get("std_dev", 0))
            memory_usages.append(stats.get("avg_memory_gb", 0))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot inference times
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(backends)))
        bars = ax1.bar(backends, avg_times, yerr=std_devs, capsize=5, color=colors)
        ax1.set_ylabel('Average Inference Time (seconds)', fontsize=12)
        ax1.set_title('Inference Time Comparison', fontsize=14)
        ax1.set_ylim(0, max(avg_times) * 1.2 if avg_times else 1)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # Plot memory usage if available
        if any(memory_usages):
            ax2.bar(backends, memory_usages, color=colors)
            ax2.set_ylabel('Memory Usage (GB)', fontsize=12)
            ax2.set_title('Memory Efficiency', fontsize=14)
            ax2.set_ylim(0, max(memory_usages) * 1.2 if memory_usages else 1)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on memory bars
            for i, mem_val in enumerate(memory_usages):
                if mem_val > 0:
                    ax2.text(i, mem_val + max(memory_usages)*0.01, 
                            f'{mem_val:.2f} GB', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Memory usage data not available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # Add title and metadata
        device = results.get("device", "Unknown")
        num_runs = results.get("num_runs", 0)
        max_tokens = results.get("max_tokens", 0)
        timestamp = results.get("timestamp", "")
        
        # Format timestamp if available
        formatted_time = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
        
        plt.suptitle(f'LLM Backend Comparison on {device.upper()} Device', fontsize=16)
        plt.figtext(0.5, 0.01, 
                   f'Benchmark: {num_runs} runs, {max_tokens} tokens, '
                   f'{len(results["backend_results"])} backends | {formatted_time}', 
                   ha='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        
        print(f"Visualization saved as '{output_file}'")
        
        # Create a second visualization for per-prompt performance
        create_prompt_comparison(results, output_file.replace('.png', '_per_prompt.png'))
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")


def create_prompt_comparison(results, output_file="benchmark_per_prompt.png"):
    """
    Create a visualization comparing performance across different prompts.
    
    Args:
        results (dict): Benchmark results data
        output_file (str): Output file path for the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        # Check if we have per-prompt data
        if not results.get("backend_results") or len(results["backend_results"]) == 0:
            return
        
        # Get the first backend result that has prompt_results
        backend_data = None
        for result in results["backend_results"]:
            if "prompt_results" in result and result["prompt_results"]:
                backend_data = result
                break
        
        if not backend_data or not backend_data.get("prompt_results"):
            return
        
        # Extract prompt data
        prompts = []
        avg_times = []
        
        for i, prompt_result in enumerate(backend_data["prompt_results"]):
            # Truncate prompt for display
            prompt_text = prompt_result.get("prompt", f"Prompt {i+1}")
            if len(prompt_text) > 30:
                prompt_text = prompt_text[:27] + "..."
            prompts.append(f"P{i+1}: {prompt_text}")
            
            # Get average time if available
            if "stats" in prompt_result and "avg_time" in prompt_result["stats"]:
                avg_times.append(prompt_result["stats"]["avg_time"])
            else:
                # Calculate from runs if stats not available
                times = [run["inference_time"] for run in prompt_result.get("runs", []) 
                         if "inference_time" in run]
                avg_time = sum(times) / len(times) if times else 0
                avg_times.append(avg_time)
        
        if not prompts or not avg_times:
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bars = plt.bar(prompts, avg_times, color=plt.cm.viridis(np.linspace(0, 0.8, len(prompts))))
        plt.ylabel('Average Inference Time (seconds)', fontsize=12)
        plt.title(f'Performance by Prompt - {backend_data["backend"]} with {backend_data["model"].split("/")[-1]}', 
                 fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Per-prompt visualization saved as '{output_file}'")
        
    except Exception as e:
        print(f"Error creating per-prompt visualization: {str(e)}")


def create_backend_comparison_table(results):
    """
    Print a table comparing the performance of different backends.
    
    Args:
        results (dict): Benchmark results data
    """
    if not results or "backend_results" not in results or not results["backend_results"]:
        print("No backend results available for comparison.")
        return
    
    # Print header
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS SUMMARY - {results.get('device', 'Unknown').upper()} DEVICE")
    print("=" * 80)
    
    # Print metadata
    print(f"Runs per prompt: {results.get('num_runs', 'N/A')}")
    print(f"Max tokens: {results.get('max_tokens', 'N/A')}")
    if "timestamp" in results:
        try:
            dt = datetime.fromisoformat(results["timestamp"])
            print(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"Timestamp: {results['timestamp']}")
    print("-" * 80)
    
    # Print table header
    print(f"{'Backend':<15} {'Model':<25} {'Avg Time':<10} {'Min Time':<10} {'Max Time':<10} {'Memory (GB)':<12}")
    print("-" * 80)
    
    # Print each backend's results
    for result in results["backend_results"]:
        backend = result.get("backend", "Unknown")
        model = result.get("model", "Unknown").split("/")[-1]  # Get last part of model name
        
        stats = result.get("overall_stats", {})
        avg_time = f"{stats.get('avg_time', 0):.3f}s"
        min_time = f"{stats.get('min_time', 0):.3f}s"
        max_time = f"{stats.get('max_time', 0):.3f}s"
        memory = f"{stats.get('avg_memory_gb', 0):.2f} GB" if stats.get('avg_memory_gb', 0) > 0 else "N/A"
        
        print(f"{backend:<15} {model:<25} {avg_time:<10} {min_time:<10} {max_time:<10} {memory:<12}")
    
    print("=" * 80)
    
    # Calculate speedup compared to slowest backend
    if len(results["backend_results"]) > 1:
        print("\nPERFORMANCE COMPARISON:")
        
        # Find slowest backend
        backends_by_speed = sorted(results["backend_results"], 
                                  key=lambda x: x.get("overall_stats", {}).get("avg_time", float('inf')))
        
        slowest = backends_by_speed[-1]
        slowest_time = slowest.get("overall_stats", {}).get("avg_time", 0)
        
        if slowest_time > 0:
            print(f"Speedup relative to {slowest['backend']} with {slowest['model'].split('/')[-1]}:")
            
            for backend in backends_by_speed[:-1]:  # All except the slowest
                backend_time = backend.get("overall_stats", {}).get("avg_time", 0)
                if backend_time > 0:
                    speedup = slowest_time / backend_time
                    print(f"  - {backend['backend']} with {backend['model'].split('/')[-1]}: {speedup:.2f}x faster")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results from unified LLM demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to benchmark results JSON file")
    group.add_argument("--dir", type=str, help="Directory containing benchmark results files")
    parser.add_argument("--latest", action="store_true", help="Use the most recent benchmark file in the directory")
    parser.add_argument("--output", type=str, default="benchmark_comparison.png", 
                      help="Output file path for visualization (default: benchmark_comparison.png)")
    
    args = parser.parse_args()
    
    # Determine which file to use
    file_path = None
    if args.file:
        file_path = args.file
    elif args.dir and args.latest:
        file_path = find_latest_benchmark_file(args.dir)
    else:
        print("Please specify either --file or --dir with --latest")
        return
    
    if not file_path:
        return
    
    # Load benchmark results
    results = load_benchmark_results(file_path)
    if not results:
        return
    
    # Print comparison table
    create_backend_comparison_table(results)
    
    # Create visualization
    if MATPLOTLIB_AVAILABLE:
        create_performance_comparison(results, args.output)
    else:
        print("Matplotlib not available. Install matplotlib to enable visualizations.")


if __name__ == "__main__":
    main()