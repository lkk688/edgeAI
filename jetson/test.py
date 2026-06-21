#!/usr/bin/env python3
"""Minimal check: is PyTorch CUDA available in this container / on this Jetson?

Run it with:  sjsujetsontool run /Developer/edgeAI/jetson/test.py
"""
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    # tiny GPU sanity op
    x = torch.randn(1000, device="cuda")
    print("GPU tensor sum (sanity):", float(x.sum()))
else:
    print("⚠️  CUDA is NOT available — PyTorch is running on CPU.")
