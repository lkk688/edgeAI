#!/usr/bin/env python3
"""
jetson_llm_utils.py — memory, performance, and device helpers for LLMs on Jetson.

Importable helpers (used by other scripts / notebooks):
    optimize_memory()                 -> clears GC + CUDA cache, prints RAM/GPU
    system_monitor()                  -> context manager that times a block + reports resources
    get_jetson_config()               -> dict of recommended settings for this board

Run directly to print a quick environment + device report:
    python3 jetson_llm_utils.py

`torch` and `psutil` are optional — the script degrades gracefully without them.
"""
import gc, os, time, platform
from contextlib import contextmanager

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def optimize_memory():
    """Free Python/CUDA memory and print a short RAM/GPU summary."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print("Available RAM: %.1f GB / %.1f GB" % (mem.available / 1024**3, mem.total / 1024**3))
    if _HAS_TORCH and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        print("GPU: %.1f GB total · %.2f GB allocated" % (total, alloc))


@contextmanager
def system_monitor(label="inference"):
    """Time a block and report CPU/RAM/GPU deltas (best-effort)."""
    t0 = time.time()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        dt = time.time() - t0
        print("\n📊 %s: %.3fs" % (label, dt))
        if _HAS_PSUTIL:
            print("   CPU %.0f%% · RAM %.0f%%" % (psutil.cpu_percent(), psutil.virtual_memory().percent))
        if _HAS_TORCH and torch.cuda.is_available():
            print("   GPU peak %.0f MB" % (torch.cuda.max_memory_allocated() / 1024**2))


def get_jetson_config():
    """Recommended LLM settings based on the detected board."""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip("\x00").strip()
    except OSError:
        model = platform.node()

    if "Orin Nano" in model:
        cfg = {"max_memory_gb": 6, "n_gpu_layers": 99, "n_ctx": 4096, "use_fp16": True}
    elif "Orin NX" in model:
        cfg = {"max_memory_gb": 14, "n_gpu_layers": 99, "n_ctx": 8192, "use_fp16": True}
    elif "AGX" in model:
        cfg = {"max_memory_gb": 28, "n_gpu_layers": 99, "n_ctx": 8192, "use_fp16": True}
    else:
        cfg = {"max_memory_gb": 8, "n_gpu_layers": 0, "n_ctx": 2048, "use_fp16": False}
    cfg["model"] = model
    return cfg


def main():
    print("=" * 56)
    print(" Jetson LLM environment report")
    print("=" * 56)
    print("Platform : %s (%s)" % (platform.system(), platform.machine()))
    print("torch    : %s" % (torch.__version__ if _HAS_TORCH else "not installed"))
    if _HAS_TORCH:
        print("CUDA     : %s" % ("available" if torch.cuda.is_available() else "no"))
    print("\nRecommended config for this device:")
    for k, v in get_jetson_config().items():
        print("   %-14s : %s" % (k, v))
    print("\nMemory:")
    optimize_memory()
    with system_monitor("demo idle block"):
        time.sleep(0.2)


if __name__ == "__main__":
    main()
