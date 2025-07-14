# 🎯 Deep Learning Object Detection on Jetson: From Classical to Zero-Shot Approaches

This comprehensive guide covers modern object detection techniques on NVIDIA Jetson, from classical two-stage detectors to cutting-edge zero-shot models:

* **Two-Stage Detectors**: Faster R-CNN, Mask R-CNN
* **One-Stage Detectors**: YOLO family, SSD, RetinaNet
* **Zero-Shot Detection**: GroundingDINO, OWL-ViT
* **TensorRT Optimization**: Performance acceleration on Jetson
* **Comparative Analysis**: Speed vs. accuracy trade-offs

---

## 🧠 Object Detection Fundamentals

Object detection is a computer vision task that combines:

> **Object Detection = Classification + Localization + Multiple Objects**

### 📊 Detection Pipeline Components

| Component | Purpose | Output |
|-----------|---------|--------|
| **Backbone** | Feature extraction | Feature maps |
| **Neck** | Feature fusion/enhancement | Multi-scale features |
| **Head** | Classification + Regression | Bounding boxes + Classes |
| **Post-processing** | NMS, confidence filtering | Final detections |

### 🎯 Evaluation Metrics

- **mAP (mean Average Precision)**: Primary metric for detection accuracy
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth boxes
- **FPS (Frames Per Second)**: Inference speed metric
- **Model Size**: Memory footprint and storage requirements

---

## 🏗️ Two-Stage Object Detectors

Two-stage detectors separate object detection into two phases: region proposal generation and classification/refinement.

### 🎯 Faster R-CNN Architecture

```
Input Image → Backbone (ResNet/VGG) → RPN → ROI Pooling → Classification Head
                                    ↓
                              Region Proposals
```

#### 🔧 Key Components

1. **Region Proposal Network (RPN)**: Generates object proposals
2. **ROI Pooling**: Extracts fixed-size features from proposals
3. **Classification Head**: Final object classification and bbox regression

#### 🛠️ Faster R-CNN Implementation on Jetson

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FasterRCNNDetector:
    def __init__(self, device='cuda'):
        self.device = device
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        
        # COCO class names
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def detect(self, image, confidence_threshold=0.5):
        """Perform object detection on input image"""
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Post-process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return boxes, scores, labels
    
    def draw_predictions(self, image, boxes, scores, labels):
        """Draw bounding boxes and labels on image"""
        image_copy = image.copy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            class_name = self.classes[label]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(image_copy, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_copy

# Usage example
detector = FasterRCNNDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    boxes, scores, labels = detector.detect(frame)
    
    # Draw results
    result_frame = detector.draw_predictions(frame, boxes, scores, labels)
    
    cv2.imshow('Faster R-CNN Detection', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 🎭 Mask R-CNN for Instance Segmentation

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt

class MaskRCNNDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
    
    def detect_with_masks(self, image, confidence_threshold=0.5):
        """Detect objects and generate segmentation masks"""
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        masks = predictions[0]['masks'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > confidence_threshold
        return boxes[mask], scores[mask], labels[mask], masks[mask]
```

---

## ⚡ One-Stage Object Detectors

One-stage detectors perform detection in a single forward pass, trading some accuracy for speed.

### 🎯 YOLO Family Evolution

| Model | Year | Key Innovation | Speed (FPS) | mAP |
|-------|------|----------------|-------------|-----|
| **YOLOv1** | 2016 | Grid-based detection | 45 | 63.4 |
| **YOLOv3** | 2018 | Multi-scale prediction | 20 | 55.3 |
| **YOLOv5** | 2020 | Efficient architecture | 140 | 56.8 |
| **YOLOv8** | 2023 | Anchor-free design | 80 | 53.9 |
| **YOLOv10** | 2024 | NMS-free training | 120 | 54.4 |

---

## 🚀 YOLO with TensorRT Acceleration

### 🔧 Why TensorRT?

TensorRT provides significant performance improvements on Jetson:
- **2-5x speedup** compared to standard PyTorch inference
- **Reduced memory usage** through layer fusion and optimization
- **Mixed precision support** (FP16/INT8) for faster inference
- **Dynamic shape optimization** for variable input sizes

### 🛠️ Complete YOLOv8 Setup on Jetson

```bash
# Install dependencies
pip install ultralytics torch torchvision
sudo apt update
sudo apt install python3-libnvinfer-dev libnvinfer-bin

# Verify TensorRT installation
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### 🎯 Enhanced YOLOv8 Implementation

```python
import torch
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class OptimizedYOLODetector:
    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self, iterations=10):
        """Warm up the model for consistent performance"""
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            _ = self.model.predict(dummy_input, verbose=False)
        print("Model warmed up successfully")
    
    def detect_optimized(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Optimized detection with performance tracking"""
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False,
            device=self.device
        )
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Update performance tracking
        self.fps_history.append(fps)
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        
        # Extract results
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
        else:
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'inference_time': inference_time * 1000,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0
        }

# Usage example with performance monitoring
detector = OptimizedYOLODetector()
cap = cv2.VideoCapture(0)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = detector.detect_optimized(frame)
    
    # Draw results
    for box, score, class_id in zip(results['boxes'], results['scores'], results['classes']):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Class {int(class_id)}: {score:.2f}', 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display performance metrics
    cv2.putText(frame, f'FPS: {results["avg_fps"]:.1f}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Inference: {results["inference_time"]:.1f}ms', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Optimized YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 🔍 Advanced TensorRT Optimization

#### Step 1: Export to ONNX with Optimization

```python
from ultralytics import YOLO

# Load and export YOLOv8 model
model = YOLO('yolov8n.pt')

# Export with optimizations
model.export(
    format='onnx',
    dynamic=True,  # Enable dynamic batch size
    simplify=True,  # Simplify the model
    opset=11,      # ONNX opset version
    imgsz=640      # Input image size
)

print("Model exported to ONNX format")
```

#### Step 2: Convert to TensorRT Engine

```bash
# Basic FP16 conversion
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_fp16.trt \
        --fp16 \
        --workspace=1024

# Advanced optimization for Jetson Orin
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_optimized.trt \
        --fp16 \
        --workspace=2048 \
        --minShapes=input:1x3x320x320 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x1280x1280 \
        --verbose \
        --buildOnly

# INT8 calibration (requires calibration dataset)
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_int8.trt \
        --int8 \
        --workspace=2048 \
        --verbose
```

---

## 🧠 Zero-Shot Object Detection with Vision-Language Models

Instead of training on fixed classes, VLMs detect objects based on text prompts like:

> "a red backpack next to a bicycle"

### 📦 Popular Models

* **OWL-ViT** (Google Research) - Vision Transformer based
* **GroundingDINO** - DETR-based with superior performance
* **GLIP** (Grounded Language Image Pretraining)
* **OWL-v2** - Improved version with better accuracy

### 🛠️ Complete Installation for Jetson

```bash
# Install base dependencies
pip install transformers torchvision timm opencv-python pillow

# For GroundingDINO (more complex setup)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# Download pre-trained weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 🔍 Enhanced OWL-ViT Implementation

```python
import torch
import cv2
import numpy as np
import time
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from collections import deque

class OptimizedOWLViTDetector:
    def __init__(self, model_name="google/owlvit-base-patch32", device="cuda"):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = deque(maxlen=30)
        
        # Enable optimizations
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        print(f"OWL-ViT model loaded on {device}")
    
    def detect_with_prompts(self, image, text_prompts, confidence_threshold=0.1):
        """Detect objects using text prompts"""
        start_time = time.time()
        
        # Convert image format if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = self.processor(
            text=[text_prompts], 
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time * 1000)
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"].cpu().numpy()
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'text_prompts': text_prompts,
            'inference_time': inference_time * 1000,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0
        }
    
    def visualize_results(self, image, results, prompt_colors=None):
        """Visualize detection results with colored boxes per prompt"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        image_copy = image.copy()
        
        # Default colors for different prompts
        if prompt_colors is None:
            prompt_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                           (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for box, score, label_idx in zip(results['boxes'], results['scores'], results['labels']):
            x1, y1, x2, y2 = box.astype(int)
            color = prompt_colors[label_idx % len(prompt_colors)]
            prompt_text = results['text_prompts'][label_idx]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{prompt_text}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_copy

# Usage example
detector = OptimizedOWLViTDetector()
cap = cv2.VideoCapture(0)

# Define detection prompts
prompts = ["person", "laptop", "cell phone", "bottle", "backpack"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection every few frames to maintain reasonable FPS
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Press space to detect
        results = detector.detect_with_prompts(frame, prompts, confidence_threshold=0.15)
        annotated_frame = detector.visualize_results(frame, results)
        
        # Display performance info
        cv2.putText(annotated_frame, f'Inference: {results["inference_time"]:.0f}ms', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Avg: {results["avg_inference_time"]:.0f}ms', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('OWL-ViT Zero-Shot Detection', annotated_frame)
    else:
        cv2.imshow('OWL-ViT Zero-Shot Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 🚀 GroundingDINO: Superior Zero-Shot Detection

```python
import torch
import cv2
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict

class GroundingDINODetector:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device
        
        # Load model
        args = type('Args', (), {
            'config_file': config_path,
            'checkpoint': checkpoint_path,
            'device': device
        })()
        
        self.model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        self.model.eval().to(device)
        
        # Image preprocessing
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("GroundingDINO model loaded successfully")
    
    def detect_with_text(self, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """Detect objects using natural language descriptions"""
        start_time = time.time()
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        image_tensor, _ = self.transform(image_pil, None)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor[None].to(self.device), 
                               captions=[text_prompt])
        
        # Post-process
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter predictions
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)
        
        # Convert to absolute coordinates
        H, W = image_pil.size[1], image_pil.size[0]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        
        inference_time = time.time() - start_time
        
        return {
            'boxes': boxes.numpy(),
            'scores': logits.max(dim=1)[0].numpy(),
            'inference_time': inference_time * 1000,
            'text_prompt': text_prompt
        }

# Example usage with complex prompts
# detector = GroundingDINODetector(
#     config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     checkpoint_path="groundingdino_swint_ogc.pth"
# )
# 
# # Complex natural language prompts
# prompts = [
#     "a person wearing a red shirt",
#     "a laptop computer on a desk",
#     "a smartphone in someone's hand",
#     "a coffee cup or water bottle"
# ]
```

### ⚡ Optimization Techniques for Jetson

#### 1. Model Quantization for VLMs

```python
import torch.quantization as quantization

def optimize_owl_vit_for_jetson(model):
    """Apply optimizations for better Jetson performance"""
    # Enable mixed precision
    model.half()  # Convert to FP16
    
    # Enable torch.jit compilation
    model.eval()
    
    # Optimize for inference
    with torch.no_grad():
        model = torch.jit.optimize_for_inference(model)
    
    return model

# Apply optimizations
# optimized_model = optimize_owl_vit_for_jetson(detector.model)
```

#### 2. Batch Processing for Better Throughput

```python
class BatchedVLMDetector:
    def __init__(self, model_detector, batch_size=4):
        self.detector = model_detector
        self.batch_size = batch_size
        self.frame_buffer = []
        self.result_buffer = []
    
    def add_frame(self, frame, prompts):
        """Add frame to batch buffer"""
        self.frame_buffer.append((frame, prompts))
        
        if len(self.frame_buffer) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        """Process accumulated frames in batch"""
        if not self.frame_buffer:
            return []
        
        # Process all frames in batch
        results = []
        for frame, prompts in self.frame_buffer:
            result = self.detector.detect_with_prompts(frame, prompts)
            results.append(result)
        
        self.frame_buffer.clear()
        return results

# Usage for video processing
# batched_detector = BatchedVLMDetector(detector, batch_size=2)
```

#### 3. Caching and Frame Skipping

```python
class CachedVLMDetector:
    def __init__(self, base_detector, cache_frames=5):
        self.detector = base_detector
        self.cache_frames = cache_frames
        self.frame_count = 0
        self.last_results = None
    
    def detect_with_caching(self, frame, prompts):
        """Only run detection every N frames"""
        self.frame_count += 1
        
        if self.frame_count % self.cache_frames == 0 or self.last_results is None:
            self.last_results = self.detector.detect_with_prompts(frame, prompts)
        
        return self.last_results

# Use cached detection for real-time video
# cached_detector = CachedVLMDetector(detector, cache_frames=3)
```

---

## 🔄 Zero-Shot vs Two-Step Detection Approaches

### 🎯 Approach Comparison

| Approach | Method | Advantages | Disadvantages |
|----------|--------|------------|---------------|
| **Zero-Shot VLM** | Single model (OWL-ViT, GroundingDINO) | • Natural language prompts<br>• No retraining needed<br>• Complex scene understanding | • Slower inference<br>• Higher memory usage<br>• Less accurate for common objects |
| **Two-Step (YOLO + CLIP)** | Detection → Classification | • Faster inference<br>• Better accuracy for known objects<br>• Modular design | • Two-stage complexity<br>• Limited to detected objects<br>• Requires object detection first |
| **Two-Step (YOLO + BLIP)** | Detection → Captioning | • Rich descriptions<br>• Context understanding<br>• Good for scene analysis | • Slowest approach<br>• Most memory intensive<br>• Overkill for simple detection |

### 🚀 Two-Step Approach Implementation

#### YOLO + CLIP for Semantic Classification

```python
import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

class YOLOCLIPDetector:
    def __init__(self, yolo_model="yolov8n.pt", device="cuda"):
        self.device = device
        
        # Load YOLO for object detection
        self.yolo = YOLO(yolo_model)
        self.yolo.to(device)
        
        # Load CLIP for classification
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"YOLO + CLIP loaded on {device}")
    
    def detect_and_classify(self, image, text_queries, conf_threshold=0.25):
        """Two-step detection: YOLO detection + CLIP classification"""
        start_time = time.time()
        
        # Step 1: YOLO Detection
        yolo_start = time.time()
        yolo_results = self.yolo.predict(image, conf=conf_threshold, verbose=False)
        yolo_time = time.time() - yolo_start
        
        if yolo_results[0].boxes is None:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'yolo_time': yolo_time * 1000,
                'clip_time': 0,
                'total_time': (time.time() - start_time) * 1000
            }
        
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        yolo_scores = yolo_results[0].boxes.conf.cpu().numpy()
        
        # Step 2: CLIP Classification
        clip_start = time.time()
        
        # Extract crops for each detection
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        clip_scores = []
        clip_labels = []
        
        # Prepare text queries
        text_tokens = clip.tokenize(text_queries).to(self.device)
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Crop the detected region
            crop = pil_image.crop((x1, y1, x2, y2))
            
            # Preprocess for CLIP
            crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            
            # Get CLIP features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(crop_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get best match
                best_match_idx = similarities.argmax().item()
                best_score = similarities[0, best_match_idx].item()
                
                clip_scores.append(best_score)
                clip_labels.append(best_match_idx)
        
        clip_time = time.time() - clip_start
        total_time = time.time() - start_time
        
        return {
            'boxes': boxes,
            'yolo_scores': yolo_scores,
            'clip_scores': np.array(clip_scores),
            'labels': np.array(clip_labels),
            'text_queries': text_queries,
            'yolo_time': yolo_time * 1000,
            'clip_time': clip_time * 1000,
            'total_time': total_time * 1000
        }
    
    def visualize_results(self, image, results):
        """Visualize YOLO + CLIP results"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        image_copy = image.copy()
        
        if len(results['boxes']) == 0:
            return image_copy
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (box, yolo_score, clip_score, label_idx) in enumerate(
            zip(results['boxes'], results['yolo_scores'], results['clip_scores'], results['labels'])):
            
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            query_text = results['text_queries'][label_idx]
            label = f"{query_text}: Y{yolo_score:.2f} C{clip_score:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_copy

# Usage example
detector = YOLOCLIPDetector()
cap = cv2.VideoCapture(0)

# Define semantic queries
queries = ["a person", "a laptop computer", "a mobile phone", "a bottle", "a backpack"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run two-step detection
    results = detector.detect_and_classify(frame, queries)
    annotated_frame = detector.visualize_results(frame, results)
    
    # Display timing information
    cv2.putText(annotated_frame, f'YOLO: {results["yolo_time"]:.0f}ms', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'CLIP: {results["clip_time"]:.0f}ms', 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Total: {results["total_time"]:.0f}ms', 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('YOLO + CLIP Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### YOLO + BLIP for Rich Scene Understanding

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

class YOLOBLIPDetector:
    def __init__(self, yolo_model="yolov8n.pt", device="cuda"):
        self.device = device
        
        # Load YOLO
        self.yolo = YOLO(yolo_model)
        self.yolo.to(device)
        
        # Load BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(device)
        
        print(f"YOLO + BLIP loaded on {device}")
    
    def detect_and_caption(self, image, conf_threshold=0.25):
        """Detect objects and generate captions for each"""
        start_time = time.time()
        
        # Step 1: YOLO Detection
        yolo_results = self.yolo.predict(image, conf=conf_threshold, verbose=False)
        
        if yolo_results[0].boxes is None:
            return {'boxes': [], 'captions': [], 'total_time': 0}
        
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        # Step 2: BLIP Captioning
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        captions = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            crop = pil_image.crop((x1, y1, x2, y2))
            
            # Generate caption
            inputs = self.blip_processor(crop, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
        
        total_time = time.time() - start_time
        
        return {
            'boxes': boxes,
            'captions': captions,
            'total_time': total_time * 1000
        }

# Example usage for rich scene understanding
# blip_detector = YOLOBLIPDetector()
# results = blip_detector.detect_and_caption(frame)
# for box, caption in zip(results['boxes'], results['captions']):
#     print(f"Object at {box}: {caption}")
```

### 📊 Performance Comparison on Jetson Orin Nano

```python
def comprehensive_benchmark():
    """Benchmark all detection approaches"""
    import time
    import psutil
    
    # Initialize all detectors
    yolo_detector = OptimizedYOLODetector()
    owl_vit_detector = OptimizedOWLViTDetector()
    yolo_clip_detector = YOLOCLIPDetector()
    
    # Test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_prompts = ["person", "laptop", "phone", "bottle"]
    
    results = {}
    
    # Benchmark YOLO (traditional)
    times = []
    for _ in range(50):
        start = time.time()
        _ = yolo_detector.detect_optimized(test_image)
        times.append(time.time() - start)
    
    results['YOLO Only'] = {
        'avg_time': np.mean(times[5:]) * 1000,
        'fps': 1000 / (np.mean(times[5:]) * 1000),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
    }
    
    # Benchmark OWL-ViT (zero-shot)
    times = []
    for _ in range(20):  # Fewer iterations due to slower inference
        start = time.time()
        _ = owl_vit_detector.detect_with_prompts(test_image, test_prompts)
        times.append(time.time() - start)
    
    results['OWL-ViT Zero-Shot'] = {
        'avg_time': np.mean(times[2:]) * 1000,
        'fps': 1000 / (np.mean(times[2:]) * 1000),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
    }
    
    # Benchmark YOLO + CLIP (two-step)
    times = []
    for _ in range(30):
        start = time.time()
        _ = yolo_clip_detector.detect_and_classify(test_image, test_prompts)
        times.append(time.time() - start)
    
    results['YOLO + CLIP'] = {
        'avg_time': np.mean(times[3:]) * 1000,
        'fps': 1000 / (np.mean(times[3:]) * 1000),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
    }
    
    # Print comparison table
    print("\n🚀 Comprehensive Detection Benchmark on Jetson Orin Nano")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Time (ms)':<12} | {'FPS':<6} | {'Memory (MB)':<12}")
    print("=" * 70)
    
    for method, stats in results.items():
        print(f"{method:<20} | {stats['avg_time']:>10.1f} | {stats['fps']:>4.1f} | {stats['memory_mb']:>10.0f}")
    
    return results

# Run comprehensive benchmark
# benchmark_results = comprehensive_benchmark()
```

### 🎯 When to Use Each Approach

#### Use **Zero-Shot VLMs** when:
- ✅ Need flexible, natural language queries
- ✅ Working with novel object categories
- ✅ Prototype development and experimentation
- ✅ Complex scene understanding required
- ❌ Real-time performance not critical

#### Use **YOLO + CLIP** when:
- ✅ Need balance between flexibility and speed
- ✅ Working with known object categories
- ✅ Want semantic classification beyond COCO classes
- ✅ Moderate real-time requirements
- ❌ Can accept two-stage complexity

#### Use **Traditional YOLO** when:
- ✅ Maximum speed required
- ✅ Working with standard object categories
- ✅ Resource-constrained environments
- ✅ Production deployment
- ❌ Limited to pre-trained classes

### 🔧 Optimization Strategies for Each Approach

#### Zero-Shot VLM Optimization
```python
# Model quantization
model = model.half()  # FP16

# Reduce input resolution
image = cv2.resize(image, (512, 512))  # Instead of 640x640

# Batch processing
batch_size = 2  # Process multiple frames together

# Frame skipping
process_every_n_frames = 3  # Only process every 3rd frame
```

#### Two-Step Approach Optimization
```python
# Pipeline optimization
# 1. Use smaller YOLO model (yolov8n vs yolov8s)
# 2. Higher confidence threshold to reduce CLIP workload
# 3. Crop size optimization for CLIP
# 4. Async processing between YOLO and CLIP

class OptimizedYOLOCLIP:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')  # Smallest model
        self.clip_model, _ = clip.load("ViT-B/16")  # Smaller CLIP
    
    def detect_optimized(self, image):
        # Higher threshold = fewer detections = faster CLIP
        yolo_results = self.yolo.predict(image, conf=0.5)  # vs 0.25
        # ... rest of implementation
```

---

## 🧪 Comprehensive Lab Exercise: Detection Approaches Comparison

### 🎯 Lab Objectives
1. **Performance Analysis**: Compare inference speed, memory usage, and accuracy
2. **Flexibility Testing**: Evaluate adaptability to novel objects and scenarios
3. **Optimization Impact**: Measure the effect of various optimization techniques
4. **Real-world Application**: Test on diverse scenarios (indoor, outdoor, crowded scenes)

### 📋 Lab Setup

```python
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import psutil
import torch

class DetectionComparator:
    def __init__(self):
        # Initialize all detection approaches
        self.yolo_detector = OptimizedYOLODetector()
        self.owl_vit_detector = OptimizedOWLViTDetector()
        self.yolo_clip_detector = YOLOCLIPDetector()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        
        print("🚀 All detectors initialized for comparison")
    
    def run_comprehensive_test(self, test_scenarios):
        """Run comprehensive comparison across multiple scenarios"""
        results = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            print(f"\n🧪 Testing Scenario: {scenario_name}")
            
            image = scenario_data['image']
            prompts = scenario_data['prompts']
            expected_objects = scenario_data.get('expected_objects', [])
            
            scenario_results = {}
            
            # Test each detection approach
            approaches = {
                'YOLO Only': lambda img: self.yolo_detector.detect_optimized(img),
                'OWL-ViT Zero-Shot': lambda img: self.owl_vit_detector.detect_with_prompts(img, prompts),
                'YOLO + CLIP': lambda img: self.yolo_clip_detector.detect_and_classify(img, prompts)
            }
            
            for approach_name, detect_func in approaches.items():
                print(f"  Testing {approach_name}...")
                
                # Warm-up runs
                for _ in range(3):
                    _ = detect_func(image)
                
                # Actual measurement runs
                times = []
                memory_usage = []
                
                for _ in range(10):
                    # Memory before
                    mem_before = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Time the detection
                    start_time = time.time()
                    results_data = detect_func(image)
                    end_time = time.time()
                    
                    # Memory after
                    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                    memory_usage.append(mem_after - mem_before)
                
                # Calculate metrics
                avg_time = np.mean(times[2:])  # Skip first 2 for stability
                fps = 1000 / avg_time
                avg_memory = np.mean(memory_usage)
                
                # Count detections
                if hasattr(results_data, 'boxes') and hasattr(results_data.boxes, 'xyxy'):
                    num_detections = len(results_data.boxes.xyxy)
                elif isinstance(results_data, dict) and 'boxes' in results_data:
                    num_detections = len(results_data['boxes'])
                else:
                    num_detections = 0
                
                scenario_results[approach_name] = {
                    'avg_time_ms': avg_time,
                    'fps': fps,
                    'memory_delta_mb': avg_memory,
                    'num_detections': num_detections,
                    'std_time': np.std(times[2:])
                }
            
            results[scenario_name] = scenario_results
        
        return results
    
    def visualize_comparison(self, results):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔍 Object Detection Approaches Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        approaches = list(next(iter(results.values())).keys())
        scenarios = list(results.keys())
        
        # 1. Average Inference Time
        ax1 = axes[0, 0]
        time_data = []
        for approach in approaches:
            times = [results[scenario][approach]['avg_time_ms'] for scenario in scenarios]
            time_data.append(times)
        
        x = np.arange(len(scenarios))
        width = 0.2
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (approach, times) in enumerate(zip(approaches, time_data)):
            ax1.bar(x + i * width, times, width, label=approach, color=colors[i % len(colors)])
        
        ax1.set_xlabel('Test Scenarios')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_title('⏱️ Inference Time Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. FPS Comparison
        ax2 = axes[0, 1]
        fps_data = []
        for approach in approaches:
            fps = [results[scenario][approach]['fps'] for scenario in scenarios]
            fps_data.append(fps)
        
        for i, (approach, fps) in enumerate(zip(approaches, fps_data)):
            ax2.bar(x + i * width, fps, width, label=approach, color=colors[i % len(colors)])
        
        ax2.set_xlabel('Test Scenarios')
        ax2.set_ylabel('FPS')
        ax2.set_title('🚀 Frames Per Second Comparison')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Usage
        ax3 = axes[1, 0]
        memory_data = []
        for approach in approaches:
            memory = [results[scenario][approach]['memory_delta_mb'] for scenario in scenarios]
            memory_data.append(memory)
        
        for i, (approach, memory) in enumerate(zip(approaches, memory_data)):
            ax3.bar(x + i * width, memory, width, label=approach, color=colors[i % len(colors)])
        
        ax3.set_xlabel('Test Scenarios')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.set_title('💾 Memory Usage Comparison')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(scenarios, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Detection Count
        ax4 = axes[1, 1]
        detection_data = []
        for approach in approaches:
            detections = [results[scenario][approach]['num_detections'] for scenario in scenarios]
            detection_data.append(detections)
        
        for i, (approach, detections) in enumerate(zip(approaches, detection_data)):
            ax4.bar(x + i * width, detections, width, label=approach, color=colors[i % len(colors)])
        
        ax4.set_xlabel('Test Scenarios')
        ax4.set_ylabel('Number of Detections')
        ax4.set_title('🎯 Detection Count Comparison')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(scenarios, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results):
        """Generate detailed comparison report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DETECTION APPROACHES ANALYSIS REPORT")
        print("="*80)
        
        for scenario_name, scenario_results in results.items():
            print(f"\n🧪 Scenario: {scenario_name.upper()}")
            print("-" * 60)
            
            # Sort by FPS for ranking
            sorted_approaches = sorted(scenario_results.items(), 
                                     key=lambda x: x[1]['fps'], reverse=True)
            
            print(f"{'Rank':<4} | {'Approach':<20} | {'Time(ms)':<10} | {'FPS':<6} | {'Memory(MB)':<12} | {'Detections':<11}")
            print("-" * 75)
            
            for rank, (approach, metrics) in enumerate(sorted_approaches, 1):
                print(f"{rank:<4} | {approach:<20} | {metrics['avg_time_ms']:>8.1f} | "
                      f"{metrics['fps']:>4.1f} | {metrics['memory_delta_mb']:>9.1f} | "
                      f"{metrics['num_detections']:>9}")
        
        # Overall recommendations
        print("\n" + "="*80)
        print("🎯 RECOMMENDATIONS")
        print("="*80)
        
        print("\n🚀 For Real-time Applications (>15 FPS):")
        print("   1. YOLO Only - Best for standard object categories")
        print("   2. YOLO + CLIP - Good balance of speed and flexibility")
        
        print("\n🎨 For Flexible/Novel Object Detection:")
        print("   1. GroundingDINO - Best zero-shot performance")
        print("   2. OWL-ViT - Good for research and prototyping")
        
        print("\n⚖️ For Balanced Performance:")
        print("   1. YOLO + CLIP - Modular, optimizable")
        print("   2. Consider ensemble approaches for critical applications")

# Example test scenarios
test_scenarios = {
    'Indoor_Office': {
        'image': cv2.imread('office_scene.jpg'),  # Replace with actual image
        'prompts': ['person', 'laptop', 'chair', 'monitor', 'phone'],
        'expected_objects': ['person', 'laptop', 'chair']
    },
    'Outdoor_Street': {
        'image': cv2.imread('street_scene.jpg'),  # Replace with actual image
        'prompts': ['car', 'person', 'bicycle', 'traffic light', 'building'],
        'expected_objects': ['car', 'person']
    },
    'Kitchen_Scene': {
        'image': cv2.imread('kitchen_scene.jpg'),  # Replace with actual image
        'prompts': ['microwave', 'refrigerator', 'bottle', 'apple', 'knife'],
        'expected_objects': ['microwave', 'bottle']
    }
}

# Run comprehensive comparison
comparator = DetectionComparator()
results = comparator.run_comprehensive_test(test_scenarios)
comparator.visualize_comparison(results)
comparator.generate_report(results)
```

### 🔬 Advanced Analysis Tasks

#### Task 1: Optimization Impact Study
```python
def optimization_impact_study():
    """Study the impact of various optimizations"""
    
    # Test different optimization levels
    optimization_configs = {
        'Baseline': {'model': 'yolov8n.pt', 'precision': 'fp32', 'batch_size': 1},
        'FP16': {'model': 'yolov8n.pt', 'precision': 'fp16', 'batch_size': 1},
        'TensorRT': {'model': 'yolov8n.engine', 'precision': 'fp16', 'batch_size': 1},
        'TensorRT_Batch': {'model': 'yolov8n.engine', 'precision': 'fp16', 'batch_size': 4}
    }
    
    for config_name, config in optimization_configs.items():
        print(f"Testing {config_name} configuration...")
        # Implement optimization testing logic
        # Measure: inference time, memory usage, accuracy retention

# Run optimization study
optimization_impact_study()
```

#### Task 2: Novel Object Detection Challenge
```python
def novel_object_challenge():
    """Test detection of unusual/novel objects"""
    
    novel_prompts = [
        "a vintage typewriter",
        "a 3D printed object",
        "a handmade craft item",
        "an unusual kitchen gadget",
        "a piece of electronic art"
    ]
    
    # Compare how well each approach handles novel objects
    # Measure: detection confidence, accuracy, false positives

# Run novel object challenge
novel_object_challenge()
```

### 📝 Lab Report Template

```markdown
# Object Detection Approaches Comparison Report

## Executive Summary
- **Best Overall Performance**: [Approach Name]
- **Best for Real-time**: [Approach Name]
- **Best for Flexibility**: [Approach Name]
- **Most Resource Efficient**: [Approach Name]

## Detailed Results

### Performance Metrics
| Approach | Avg Time (ms) | FPS | Memory (MB) | Accuracy Score |
|----------|---------------|-----|-------------|----------------|
| YOLO Only | X.X | X.X | X.X | X.X |
| OWL-ViT | X.X | X.X | X.X | X.X |
| YOLO+CLIP | X.X | X.X | X.X | X.X |
| GroundingDINO | X.X | X.X | X.X | X.X |

### Key Findings
1. **Speed**: [Analysis]
2. **Accuracy**: [Analysis]
3. **Flexibility**: [Analysis]
4. **Resource Usage**: [Analysis]

### Recommendations
- **For Production Deployment**: [Recommendation]
- **For Research/Prototyping**: [Recommendation]
- **For Edge Devices**: [Recommendation]
```

---

## 🎯 Advanced Integration: Multi-Modal Scene Understanding

For the most comprehensive scene analysis, combine multiple approaches:

```python
class MultiModalSceneAnalyzer:
    def __init__(self):
        self.yolo_detector = OptimizedYOLODetector()
        self.clip_classifier = YOLOCLIPDetector()
        self.scene_captioner = YOLOBLIPDetector()
        self.zero_shot_detector = GroundingDINODetector()
    
    def comprehensive_analysis(self, image, context_prompts):
        """Multi-stage scene understanding"""
        
        # Stage 1: Fast object detection
        yolo_results = self.yolo_detector.detect_optimized(image)
        
        # Stage 2: Semantic classification of detected objects
        if len(yolo_results.boxes) > 0:
            clip_results = self.clip_classifier.detect_and_classify(image, context_prompts)
        
        # Stage 3: Scene-level captioning
        scene_caption = self.scene_captioner.detect_and_caption(image)
        
        # Stage 4: Zero-shot detection for missed objects
        zero_shot_results = self.zero_shot_detector.detect_with_caption(
            image, f"objects not detected: {' '.join(context_prompts)}"
        )
        
        return {
            'fast_detection': yolo_results,
            'semantic_classification': clip_results,
            'scene_description': scene_caption,
            'additional_objects': zero_shot_results
        }

# Usage for comprehensive scene understanding
analyzer = MultiModalSceneAnalyzer()
results = analyzer.comprehensive_analysis(frame, ['person', 'vehicle', 'furniture'])

print(f"Fast Detection: {len(results['fast_detection'].boxes)} objects")
print(f"Scene Description: {results['scene_description']}")
print(f"Additional Objects: {len(results['additional_objects'])} found")
```

This multi-modal approach provides the most comprehensive scene understanding by leveraging the strengths of each detection method while mitigating their individual weaknesses.

### 🔗 Integration with Local LLMs

```python
from llama_cpp import Llama

class SceneNarrator:
    def __init__(self, model_path="/models/qwen.gguf"):
        self.llm = Llama(model_path=model_path)
    
    def generate_scene_narrative(self, detection_results):
        """Generate natural language scene description"""
        
        # Extract detected objects
        objects = []
        for result in detection_results:
            if 'objects' in result:
                objects.extend(result['objects'])
        
        prompt = f"""
Objects detected in the scene: {', '.join(objects)}
Generate a natural, descriptive summary of what's happening in this scene:
"""
        
        narrative = self.llm(prompt, max_tokens=100)
        return narrative

# Usage
narrator = SceneNarrator()
scene_story = narrator.generate_scene_narrative(detection_results)
print(f"Scene Narrative: {scene_story}")
```

### 🧠 Sample Output

> "A person is working at a desk with a laptop computer open. There's a coffee cup nearby and a smartphone on the table. The scene suggests a typical office or home workspace environment."

This complete pipeline mimics human-like perception: **detect → classify → understand → narrate → act**.

---

## 🧠 Takeaway

* Use YOLO for real-time detection where speed matters.
* Use OWL-ViT or GroundingDINO when you need **zero-shot detection** flexibility.
* Combine both with LLMs to enable **full-scene language understanding**.

Next: Build interactive visual assistants on Jetson!
