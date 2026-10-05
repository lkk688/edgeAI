"""bf16 matmul vs torch._int_mm (int8 x int8 -> int32 tensor cores) at pi05's shapes.

W8A8 with per-token activation scales is the quantisation that needs no Triton
(Orin's PyTorch has none) and can be captured in a CUDA graph. This only times
the GEMMs, including the activation quantise + rescale overhead.
"""
import time, torch

dev = "cuda"
# Gemma-2B layer GEMMs (prefix, M = tokens) and Gemma-300M expert GEMMs (M = 50 actions)
SHAPES = {
    "2B q/kv/o  (M=tok, 2048->2560)": (2048, 2560),
    "2B gate+up (M=tok, 2048->32768)": (2048, 32768),
    "2B down    (M=tok, 16384->2048)": (16384, 2048),
}
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000

print(torch.cuda.get_device_name(0))
for M in (712, 1224):
    for name, (K, N) in SHAPES.items():
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        wq = (w / (w.abs().amax(0, keepdim=True) / 127)).round().to(torch.int8)
        ws = (w.abs().amax(0) / 127).to(torch.bfloat16)
        def int8():
            xs = x.abs().amax(1, keepdim=True).float() / 127
            xq = (x.float() / xs).round().to(torch.int8)
            return (torch._int_mm(xq, wq).float() * xs * ws.float()).to(torch.bfloat16)
        tb, ti = bench(lambda: x @ w), bench(int8)
        err = ((int8() - x @ w).float().norm() / (x @ w).float().norm()).item()
        print(f"M={M:5d} {name:<34} bf16 {tb:7.2f} ms   int8 W8A8 {ti:7.2f} ms   x{tb/ti:4.2f}   rel err {err:.4f}")
