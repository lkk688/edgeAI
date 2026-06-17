# 🏛️ Advanced CNNs, Optimization & Image Processing on Jetson

> **Prerequisite:** [04 — Deep Learning & CNNs](04_deeplearning_cnn.md) covers the fundamentals (what a CNN is, layer types, the `jetson_cnn_toolkit.py` quick-start). This chapter goes deeper: advanced architectures, Jetson-specific optimization (quantization, TensorRT), and real-time image-processing pipelines.

## 🎯 Learning Objectives
- Compare advanced CNN architectures (ResNet, MobileNet, EfficientNet) on Jetson
- Apply Jetson optimizations: FP16/INT8 and the PyTorch → ONNX → TensorRT pipeline
- Build an image-classification + camera pipeline with OpenCV

---

## 🚀 Environment: the Jetson container

Everything here runs inside the `jetson-dev` container (PyTorch CUDA, torchvision, OpenCV, TensorRT all preinstalled). Start it and move to the toolkit:

```bash
sjsujetsontool shell                       # host folder /Developer is mounted into the container
cd /Developer/edgeAI/edgeLLM               # this repo lives at /Developer/edgeAI
python3 jetson_cnn_toolkit.py --help
```

The toolkit exposes four models — `basiccnn`, `resnet`, `mobilenet`, `efficientnet` — and the modes `demo | benchmark | train | inference | optimize`. We avoid `train` in labs (it downloads CIFAR-10 and runs for a long time); `demo`/`benchmark` need neither training nor data.

---

## 🏗️ Advanced CNN Architectures

### ResNet — Residual Networks
ResNet introduced **skip connections** so gradients flow directly through deep networks, solving the vanishing-gradient/degradation problem.
- **Residual block:** `output = F(x) + x` (identity shortcut)
- **Bottleneck design:** 1×1 convs cut compute while preserving capacity
- In the toolkit: `CustomResNet` (built from `ResidualBlock`), ~11M params.

### MobileNet — efficient by design
MobileNet replaces standard convolutions with **depthwise-separable convolutions** (a depthwise conv + a 1×1 pointwise conv), cutting compute ~8–9×.
- **Width multiplier** scales channels (0.25–1.0) to trade accuracy for speed
- Best speed-to-size ratio on the Orin Nano → the go-to edge architecture
- In the toolkit: `MobileNet`, ~2.2M params.

### EfficientNet — compound scaling
EfficientNet **compound-scales** depth, width, and resolution together, using MBConv blocks with squeeze-and-excitation attention.
- Strong accuracy per FLOP; the base B0 is edge-friendly
- In the toolkit: `EfficientNet` (B0-style), ~3.5M params.

### See the trade-offs yourself
```bash
python3 jetson_cnn_toolkit.py --mode demo --model all --device cuda
```
```text
  basiccnn     | params=  1,147,914 | out=(32, 10) |   2.9 ms/batch | 10965 img/s
  resnet       | params= 11,181,642 | out=(32, 10) |   8.6 ms/batch |  3707 img/s
  mobilenet    | params=  2,155,338 | out=(32, 10) |   6.0 ms/batch |  5333 img/s
  efficientnet | params=  3,472,714 | out=(32, 10) |   9.6 ms/batch |  3330 img/s
```
Note how `mobilenet` delivers far more img/s per parameter than `resnet` — the essence of edge-efficient design.

---

## ⚙️ Jetson-Specific Optimization

### 1. Precision: FP16 and INT8
Lower precision means less memory and faster math on Jetson's Tensor Cores:
- **FP16** (half precision): ~2× speedup, negligible accuracy loss — the easy default.
- **INT8** (quantization): ~4× smaller, fastest, needs calibration data to preserve accuracy.

### 2. Benchmark a model's inference
`benchmark` mode times inference without training (uses synthetic data unless `--dataset cifar10`):
```bash
python3 jetson_cnn_toolkit.py --mode benchmark --model mobilenet --dataset custom
python3 jetson_cnn_toolkit.py --mode benchmark --model all --dataset custom   # compare all
```
Results (FPS, latency, memory) are written to `outputs/benchmark_results.json` with a chart in `outputs/benchmark_comparison.png`.

### 3. TensorRT optimization
`optimize` mode exports your **trained** model to a TensorRT engine. It needs a weights file (from `--mode train`) and a Jetson with TensorRT:
```bash
python3 jetson_cnn_toolkit.py --mode optimize --model resnet \
    --weights outputs/resnet_cifar10_best.pth --precision fp16
```
> Requires a trained `.pth`. If you haven't trained, use the manual PyTorch → ONNX → TensorRT workflow below, which works with pretrained torchvision weights.

---

## ⚡ TensorRT: PyTorch → ONNX → TensorRT

TensorRT is NVIDIA's inference optimizer/runtime. The standard path converts a model to ONNX, then builds a TensorRT engine.

**1. Export a pretrained model to ONNX** (run in the container):
```python
import torch, torchvision.models as models
model = models.resnet18(weights="IMAGENET1K_V1").eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "resnet18.onnx", opset_version=13,
                  input_names=["input"], output_names=["output"])
print("exported resnet18.onnx")
```

**2. Build a TensorRT engine with `trtexec`:**
```bash
# FP16 engine
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.trt --fp16
# this also prints throughput/latency for the optimized engine
```

**3. Compare** the PyTorch vs TensorRT latency — TensorRT typically gives 2–5× lower latency on Jetson.

---

## 🖼️ Image Processing Tools on Jetson

| Tool / Library | Purpose |
|---|---|
| **OpenCV (`cv2`)** | Real-time image/video processing, camera capture |
| **Pillow (`PIL`)** | Image loading and conversion |
| **PyTorch / TensorRT** | CNN inference |
| **v4l2-ctl** | Inspect/configure cameras |
| **GStreamer** | Hardware-accelerated media/camera pipelines |

### Classify an image with a pretrained CNN
A complete runnable example (uses torchvision ImageNet weights — no training):
```python
import torch, torchvision.models as models
import torchvision.transforms as T
from PIL import Image

model = models.mobilenet_v2(weights="IMAGENET1K_V1").eval().cuda()
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
x = tf(Image.open("cat.jpg").convert("RGB")).unsqueeze(0).cuda()
with torch.no_grad():
    probs = model(x).softmax(1)
top = probs.topk(3)
print("top-3 class ids:", top.indices[0].tolist(), "scores:", top.values[0].tolist())
```

### Capture from a camera with OpenCV
```python
import cv2
cap = cv2.VideoCapture(0)              # USB camera; CSI cameras use a GStreamer pipeline
ok, frame = cap.read()
if ok:
    cv2.imwrite("frame.jpg", frame)
    print("captured", frame.shape)
cap.release()
```

---

## 🧪 Lab: Architecture comparison & optimization

**Goal:** quantify the accuracy/speed/size trade-offs and accelerate a model — without long training.

1. **Compare architectures** (no training):
   ```bash
   python3 jetson_cnn_toolkit.py --mode demo --model all --device cuda
   ```
   Record params and img/s for each. Which gives the best img/s per million params?

2. **Benchmark a chosen model** and inspect the JSON/chart:
   ```bash
   python3 jetson_cnn_toolkit.py --mode benchmark --model mobilenet --dataset custom
   ```

3. **Accelerate with TensorRT** using the PyTorch → ONNX → `trtexec --fp16` workflow above on `resnet18`, and compare the reported latency to plain PyTorch.

4. *(Optional, slow)* Train `basiccnn` on CIFAR-10 for a few epochs and run `--mode inference` on the saved weights.

### Deliverables
- A table of model · params · img/s (from step 1)
- `benchmark_results.json` + the comparison chart (step 2)
- PyTorch-vs-TensorRT latency numbers (step 3)
- One paragraph: which model would you deploy on the Orin Nano, and why?

---

## 📌 Summary
- **Architecture matters:** depthwise-separable (MobileNet) and compound scaling (EfficientNet) buy speed/size on the edge; skip connections (ResNet) buy depth.
- **Optimize for Jetson:** FP16 is the easy 2× win; INT8/TensorRT go further with calibration.
- **`jetson_cnn_toolkit.py`** gives you `demo`/`benchmark` for instant, training-free experiments, plus `train`/`inference`/`optimize` for the full pipeline.
- **OpenCV + TensorRT** turn a trained model into a real-time camera classifier.

→ **Next:** [Transformers & NLP on Jetson](05_transformers_nlp_applications.md)
