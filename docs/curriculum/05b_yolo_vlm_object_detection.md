# 🎯 Real-Time Object Detection on Jetson: YOLO + Vision-Language Models

This guide introduces how to use NVIDIA Jetson Orin Nano for fast, real-time object detection using:

* **YOLOv8 with TensorRT acceleration**
* **Zero-shot object detection using vision-language models (VLMs)**

---

## 🧠 What is Object Detection?

Object detection identifies and classifies objects in an image while locating them with bounding boxes.

> Object Detection = Classification + Localization

---

## 🚀 YOLO with TensorRT Acceleration

### 🔧 Why TensorRT?

TensorRT is NVIDIA’s high-performance inference optimizer for deep learning. It improves model performance on Jetson devices significantly.

### 🛠️ Install YOLOv8 + TensorRT on Jetson

```bash
pip install ultralytics
sudo apt install python3-libnvinfer-dev libnvinfer-bin
```

### 🔍 Convert YOLO to TensorRT (Optional Optimization)

Use `onnx` export + `trtexec`:

```bash
# Export to ONNX
yolo export model=yolov8n.pt format=onnx

# Convert ONNX to TRT
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.trt --fp16
```

### 📦 Run Inference on Jetson

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # or use .trt engine
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8", annotated)
    if cv2.waitKey(1) == ord('q'):
        break
```

---

## 🧠 Zero-Shot Object Detection with Vision-Language Models

Instead of training on fixed classes, VLMs detect objects based on text prompts like:

> "a red backpack next to a bicycle"

### 📦 Popular Models

* **OWL-ViT** (Google Research)
* **GroundingDINO**
* **GLIP** (Grounded Language Image Pretraining)

### 🛠️ Install OWL-ViT or GroundingDINO

```bash
pip install transformers torchvision timm opencv-python
```

### 🔍 OWL-ViT Example

```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch, cv2
from PIL import Image

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
image = Image.open("input.jpg")
prompts = [["laptop", "person", "backpack"]]

inputs = processor(text=prompts, images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    results = processor.post_process(outputs, inputs, threshold=0.3)

# TODO: Draw bounding boxes from results[0]
```

---

## ⚖️ YOLO vs VLM Comparison

| Feature       | YOLOv8                   | OWL-ViT / VLMs             |
| ------------- | ------------------------ | -------------------------- |
| Speed (FPS)   | ✅ Real-time (30+ FPS)    | ❌ Slower (\~1–2 FPS)       |
| Accuracy      | ✅ High for known classes | 🔄 Variable, context-aware |
| Customization | ❌ Retrain needed         | ✅ Just change prompt       |
| Model Size    | Small (10–50MB)          | Large (200MB–1GB+)         |

---

## 🧪 Lab: Comparing YOLO and VLMs

1. Use a webcam or image stream
2. Run YOLOv8 on live camera
3. Run OWL-ViT on the same input
4. Time both methods and compare
5. Try custom prompts like:

   * "an orange traffic cone"
   * "a green bottle on table"

---

## 🧠 BONUS: Combine Detection + LLM for Scene Understanding

Once bounding boxes are identified, use a local LLM to describe the scene:

### 🔗 Integrate with llama-cpp-python

```python
from llama_cpp import Llama
llm = Llama(model_path="/models/qwen.gguf")

summary = llm("""
Objects Detected: person, chair, table, laptop
Generate a natural language summary of the scene:
""")
print(summary)
```

### 🧠 Sample Output

> "A person is sitting at a table with a laptop open, next to a chair."

This setup mimics real-world perception: detect → interpret → describe → act.

---

## 🧠 Takeaway

* Use YOLO for real-time detection where speed matters.
* Use OWL-ViT or GroundingDINO when you need **zero-shot detection** flexibility.
* Combine both with LLMs to enable **full-scene language understanding**.

Next: Build interactive visual assistants on Jetson!
