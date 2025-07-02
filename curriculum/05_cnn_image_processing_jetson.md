🧠 CNNs + Image Processing on Jetson

📸 What is Image Processing?

Image processing involves transforming, analyzing, and extracting information from digital images. It’s foundational to computer vision, robotics, and AI perception systems.

Common image processing tasks include:
	•	Edge detection
	•	Object recognition
	•	Color space conversion
	•	Feature extraction

⸻

🧠 Convolutional Neural Networks (CNNs)

CNNs are specialized deep neural networks designed for image-based tasks. They use filters (kernels) that convolve across the image to detect features like edges, textures, and shapes.

🔄 Typical CNN Workflow
	1.	Input: Image (e.g., 224x224 RGB)
	2.	Convolution → ReLU → Pooling (repeat)
	3.	Fully connected layers
	4.	Softmax output (classification)

🧱 Layers in CNNs

Layer Type	Function
Convolution	Extracts features with sliding kernels
Activation (ReLU)	Adds non-linearity
Pooling	Downsamples spatial dimensions
Fully Connected	Learns classification decision


⸻

⚙️ CNNs on Jetson

Jetson Orin Nano provides accelerated inference using:
	•	✅ TensorRT (for optimized execution)
	•	✅ CUDA cores (for parallelism)
	•	✅ cuDNN (optimized DNN kernels)

Models like ResNet, YOLOv5, and MobileNet run efficiently on Jetson using PyTorch or ONNX.

Example: Load and Run CNN on Jetson (PyTorch)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

img = Image.open("cat.jpg")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted = output.argmax().item()
    print("Predicted class index:", predicted)


⸻

⚡️ TensorRT Acceleration on Jetson

🧩 What is TensorRT?

TensorRT is NVIDIA’s high-performance deep learning inference optimizer and runtime engine. It converts trained models (ONNX, PyTorch, TensorFlow) into fast, deployable engines.

🔁 Workflow: PyTorch → ONNX → TensorRT
	1.	Export model to ONNX

import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx", opset_version=11)

	2.	Convert ONNX to TensorRT engine

/opt/nvidia/onnxruntime/bin/trtexec --onnx=resnet18.onnx --saveEngine=resnet18.trt --explicitBatch

	3.	Run inference with TensorRT Python API
Use libraries like tensorrt, pycuda, or NVIDIA’s onnxruntime-gpu backend.

⸻

🧪 Jetson Image Processing Tools

Tool/Library	Purpose
OpenCV (cv2)	Real-time image processing
Pillow (PIL)	Image loading and conversion
PyTorch/TensorRT	CNN inference
v4l2-ctl	Access and configure camera
GStreamer	Media pipeline and camera stream


⸻

🧪 Lab: Classify Images with ResNet on Jetson

🎯 Objective

Run a pretrained CNN to classify local image data using PyTorch on Jetson. Then accelerate with TensorRT.

✅ Setup

pip install torch torchvision pillow opencv-python onnx
sudo apt install python3-pycuda

🛠️ Tasks
	1.	Run PyTorch-based ResNet inference on a test image
	2.	Export to ONNX and convert to TensorRT engine
	3.	Run and compare performance (fps / latency)

📋 Deliverables
	•	Output of predicted class
	•	Timing comparison between PyTorch and TensorRT
	•	Screenshot of TensorRT engine build or trtexec results

⸻

🧪 Lab: TensorRT-Based Image Classification in PyTorch Container

🎯 Objective

Use a Docker container with PyTorch and TensorRT pre-installed to classify an image using an optimized engine.

✅ Container Setup

Use NVIDIA’s PyTorch container image with TensorRT:

docker run --rm -it --runtime nvidia \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash

Inside container:

pip install pillow onnx

🛠️ Tasks
	1.	Inside the container, download and convert a ResNet model to ONNX.
	2.	Run trtexec to convert ONNX → TensorRT.
	3.	Write a Python script to load and run inference on the image using the onnxruntime-gpu or tensorrt backend.

Sample trtexec command

trtexec --onnx=resnet18.onnx --saveEngine=resnet18.trt --explicitBatch

Sample Python snippet

import onnxruntime as ort
from PIL import Image
import numpy as np

# Preprocess input
img = Image.open("cat.jpg").resize((224, 224))
img_np = np.asarray(img).astype(np.float32) / 255.0
img_np = img_np.transpose(2, 0, 1).reshape(1, 3, 224, 224)

# Run inference
session = ort.InferenceSession("resnet18.onnx")
outputs = session.run(None, {session.get_inputs()[0].name: img_np})
print("Predicted index:", np.argmax(outputs[0]))

📋 Deliverables
	•	Screenshot of classification output
	•	Screenshot of trtexec performance results
	•	Brief reflection: how does TensorRT help?

⸻

📌 Summary
	•	CNNs are essential for vision-based AI
	•	Jetson accelerates inference with CUDA + TensorRT
	•	TensorRT reduces latency and improves FPS
	•	PyTorch models can be deployed as optimized engines

→ Next: Transformers & LLMs