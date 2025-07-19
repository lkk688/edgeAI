import torch
import numpy as np
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import onnx
import onnxruntime as ort
from PIL import Image

# ------------------------
# STEP 1: Load HuggingFace ViT
# ------------------------
model_name = "google/vit-base-patch16-224-in21k"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

extractor = AutoFeatureExtractor.from_pretrained(model_name)

# ------------------------
# STEP 2: Prepare dummy input
# ------------------------
img = Image.new('RGB', (224, 224), color='blue')
inputs = extractor(images=img, return_tensors="pt")
input_names = list(inputs.keys())

# ------------------------
# STEP 3: Export to ONNX
# ------------------------
onnx_path = "vit.onnx"
torch.onnx.export(
    model,
    args=tuple(inputs.values()),
    f=onnx_path,
    input_names=input_names,
    output_names=["logits"],
    dynamic_axes={k: {0: "batch"} for k in input_names},
    opset_version=13,
)

print(f"✅ Exported to: {onnx_path}")

# ------------------------
# STEP 4: Inference benchmark
# ------------------------
providers_to_test = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

print("\n🔍 Running Inference on Each Provider:\n")

for provider in providers_to_test:
    if provider not in ort.get_available_providers():
        print(f"❌ {provider} not available, skipping.")
        continue

    print(f"🚀 Using {provider}...")
    session = ort.InferenceSession(onnx_path, providers=[provider])

    input_feed = {k: v.cpu().numpy() for k, v in inputs.items()}

    # Warm-up
    for _ in range(3):
        _ = session.run(None, input_feed)

    # Timed inference
    start = time.time()
    for _ in range(100):
        _ = session.run(None, input_feed)
    end = time.time()

    total_time = end - start
    fps = 100 / total_time
    print(f"✅ {provider}: {fps:.2f} FPS, Total time: {total_time:.2f}s\n")