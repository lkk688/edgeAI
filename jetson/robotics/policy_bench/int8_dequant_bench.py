"""Cost of weight-only int8 storage for pi05's Gemma-2B trunk: dequantise each layer
just before use, then run the normal bf16 GEMM. The trunk runs once per chunk (the
prefix), so this cost is paid once per chunk, not once per denoising step.
"""
import time, torch
dev = "cuda"
K, N, I = 2048, 2560, 16384          # one Gemma-2B layer: qkv/o + gate/up/down
shapes = [(K, 2048), (K, 512), (2048, K), (K, I), (K, I), (I, K)]   # q, kv, o, gate, up, down
params = sum(a * b for a, b in shapes)
wq = [torch.randint(-127, 127, s, device=dev, dtype=torch.int8) for s in shapes]
sc = [torch.rand(s[1], device=dev, dtype=torch.bfloat16) for s in shapes]
buf = [torch.empty(s, device=dev, dtype=torch.bfloat16) for s in shapes]
def deq():
    for q, s, b in zip(wq, sc, buf):
        torch.mul(q, s, out=b)
for _ in range(3): deq()
torch.cuda.synchronize(); t = time.time()
for _ in range(20): deq()
torch.cuda.synchronize(); ms = (time.time() - t) / 20 * 1000
print(f"{torch.cuda.get_device_name(0)}: one layer {params/1e6:.0f}M params dequant {ms:.2f} ms "
      f"-> x18 layers {ms*18:.0f} ms per chunk; saves {params*18/2**30:.2f} GB vs bf16")
