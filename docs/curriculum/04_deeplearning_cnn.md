🧠 Deep Learning & CNNs for Image Classification on Jetson

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand deep learning fundamentals and CNN architecture
- Implement basic and advanced CNN models using the Jetson CNN Toolkit
- Optimize CNN inference on Jetson devices using various techniques
- Deploy production-ready image classification systems

---

## 🚀 Getting Started: the Jetson container

All hands-on examples run **inside the `jetson-dev` container**, which already has PyTorch (CUDA), torchvision, matplotlib, OpenCV, and the rest preinstalled. Start it with one command from the host:

```bash
sjsujetsontool shell
```

This launches/attaches the persistent container and drops you into a shell. The host folder **`/Developer` is mounted into the container at the same path**, and this course repo lives at **`/Developer/edgeAI`**, so your edits on the host are instantly visible inside the container (and vice-versa). The CNN toolkit used below is at `/Developer/edgeAI/edgeLLM/jetson_cnn_toolkit.py`:

```bash
# inside the container (after `sjsujetsontool shell`)
cd /Developer/edgeAI/edgeLLM
python3 jetson_cnn_toolkit.py --help
```

> [!TIP]
> If a Python package is missing, install it once inside the container (`pip install <pkg>`); an instructor can then `commit` the container image and publish it so every node picks it up via `sjsujetsontool update`.

---

## 🧠 Deep Learning Theoretical Foundations

### **What is Deep Learning?**

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. For image classification, deep learning has revolutionized computer vision by automatically learning hierarchical feature representations.

### **Key Concepts**

#### **1. Neural Network Basics**
- **Neuron**: Basic computational unit that applies weights, bias, and activation function
- **Layer**: Collection of neurons that process input simultaneously
- **Forward Propagation**: Data flows from input to output through layers
- **Backpropagation**: Error flows backward to update weights during training

#### **2. Deep Learning vs Traditional ML**

| Aspect | Traditional ML | Deep Learning |
|--------|----------------|---------------|
| **Feature Engineering** | Manual feature extraction | Automatic feature learning |
| **Data Requirements** | Works with small datasets | Requires large datasets |
| **Computational Cost** | Lower | Higher |
| **Performance** | Good for simple patterns | Excellent for complex patterns |
| **Interpretability** | Higher | Lower (black box) |

#### **3. Why Deep Learning for Images?**

- **Hierarchical Learning**: Lower layers detect edges, higher layers detect objects
- **Translation Invariance**: Can recognize objects regardless of position
- **Scale Invariance**: Can handle objects of different sizes
- **Robustness**: Handles variations in lighting, rotation, and occlusion

### **Mathematical Foundations**

#### **Convolution Operation**
The convolution operation is fundamental to CNNs. It involves sliding a filter (kernel) across an input image to detect features. The mathematical representation is:

**Output[i,j] = Σ Σ Input[i+m, j+n] * Kernel[m,n]**

The Jetson CNN Toolkit includes a demonstration of 2D convolution operations for educational purposes, showing how edge detection kernels work on sample images.

#### **Activation Functions**
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns:

- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x) - Most commonly used
- **Sigmoid**: f(x) = 1/(1+e^(-x)) - Outputs between 0 and 1
- **Tanh**: f(x) = tanh(x) - Outputs between -1 and 1
- **Leaky ReLU**: f(x) = max(0.01x, x) - Prevents dying ReLU problem

The toolkit includes visualization capabilities for comparing different activation functions and their characteristics.

---

## 🏗️ CNN Architecture Deep Dive

### **Convolutional Neural Networks (CNNs)**

CNNs are specialized deep neural networks designed for processing grid-like data such as images. They use convolution operations to detect local features and build hierarchical representations.

### **CNN Layer Types**

#### **1. Convolutional Layer**
- **Purpose**: Feature extraction using learnable filters
- **Parameters**: Filter size, stride, padding, number of filters
- **Output**: Feature maps highlighting detected patterns

#### **2. Activation Layer**
- **Purpose**: Introduce non-linearity
- **Common**: ReLU, Leaky ReLU, ELU
- **Effect**: Enables learning of complex patterns

#### **3. Pooling Layer**
- **Purpose**: Spatial downsampling and translation invariance
- **Types**: Max pooling, Average pooling, Global pooling
- **Benefits**: Reduces computational cost and overfitting

#### **4. Normalization Layer**
- **Purpose**: Stabilize training and improve convergence
- **Types**: Batch Normalization, Layer Normalization, Group Normalization
- **Benefits**: Faster training, better generalization

#### **5. Fully Connected Layer**
- **Purpose**: Final classification or regression
- **Position**: Usually at the end of the network
- **Function**: Maps features to output classes

### **CNN Architecture Implementation**

The Jetson CNN Toolkit provides a comprehensive `BasicCNN` class that demonstrates proper CNN architecture design:

- **Feature Extraction Layers**: Three convolutional blocks with increasing channel depth (32→64→128)
- **Batch Normalization**: Applied after each convolution for training stability
- **Pooling Strategy**: Max pooling for spatial downsampling, adaptive pooling for variable input sizes
- **Classification Head**: Fully connected layers with dropout for regularization

The toolkit's `BasicCNN` implementation supports configurable input channels and output classes, making it suitable for various image classification tasks from CIFAR-10 to ImageNet-scale problems.

---

## 💻 CNN Implementation with Jetson Toolkit

### **CIFAR-10 Classification Example**

The Jetson CNN Toolkit provides a comprehensive implementation for image classification tasks. The toolkit includes several CNN architectures optimized for NVIDIA Jetson devices.

#### **Basic CNN Architecture**

The toolkit's `BasicCNN` class demonstrates a well-structured CNN design:

- **Convolutional Blocks**: Three sequential blocks with increasing feature depth (32→64→128 channels)
- **Batch Normalization**: Applied after each convolution for training stability
- **Pooling Strategy**: Max pooling for spatial downsampling
- **Classification Head**: Fully connected layers with dropout regularization
- **Adaptive Design**: Supports variable input sizes through adaptive pooling

#### **Data Preparation and Augmentation**

The Jetson CNN Toolkit includes comprehensive data handling capabilities:

- **Dataset Support**: CIFAR-10, ImageNet, and custom datasets
- **Data Augmentation**: Random horizontal flip, rotation, color jittering, and normalization
- **Efficient Loading**: Optimized data loaders with configurable batch sizes and worker processes
- **Preprocessing Pipeline**: Automatic image preprocessing with dataset-specific normalization values

#### **Training Pipeline**

The Jetson CNN Toolkit provides a comprehensive training system:

- **Optimizer Support**: Adam, SGD, and other optimizers with configurable learning rates
- **Loss Functions**: Cross-entropy, focal loss, and custom loss implementations
- **Learning Rate Scheduling**: Step decay, cosine annealing, and adaptive scheduling
- **Training Monitoring**: Real-time loss and accuracy tracking with progress visualization
- **Validation**: Automatic validation during training with early stopping capabilities
- **Device Management**: Automatic GPU/CPU detection and memory optimization for Jetson devices

#### **Visualization and Monitoring**

The toolkit includes comprehensive visualization capabilities:

- **Training Curves**: Real-time plotting of loss and accuracy metrics
- **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score tracking
- **Model Visualization**: Architecture diagrams and feature map visualization
- **Export Options**: Save training history and model checkpoints automatically

### **Hands-on: run the toolkit (no training required)**

The toolkit has four models — `basiccnn`, `resnet`, `mobilenet`, `efficientnet` — and several modes: `demo`, `benchmark`, `train`, `inference`, `optimize`. Training downloads CIFAR-10 and takes a long time, so for this lab we use the **`demo`** mode, which builds each model, runs a forward pass on synthetic data, and times GPU inference — **instantly, with no training and no downloads**.

```bash
# inside the container, in /Developer/edgeAI/edgeLLM
python3 jetson_cnn_toolkit.py --mode demo --model all --device cuda
```
Example output on an Orin Nano (your numbers will vary):
```text
Input size: (3, 32, 32), batch size: 32, device: cuda
  basiccnn     | params=  1,147,914 | out=(32, 10) |   2.9 ms/batch | 10965 img/s
  resnet       | params= 11,181,642 | out=(32, 10) |   8.6 ms/batch |  3707 img/s
  mobilenet    | params=  2,155,338 | out=(32, 10) |   6.0 ms/batch |  5333 img/s
  efficientnet | params=  3,472,714 | out=(32, 10) |   9.6 ms/batch |  3330 img/s
```
This single command teaches the core trade-off: **`basiccnn`** is smallest/fastest, **`resnet`** has the most parameters, and **`mobilenet`** gives a strong speed-to-size balance — exactly why it's favored on edge devices.

**Benchmark inference** (synthetic 224×224 data, still no training):
```bash
python3 jetson_cnn_toolkit.py --mode benchmark --model mobilenet --dataset custom
```

> [!NOTE]
> **Training is optional and slow.** It downloads CIFAR-10 (~170 MB) and runs many epochs — run it on your own time, not during the lab:
> ```bash
> python3 jetson_cnn_toolkit.py --mode train --model basiccnn --dataset cifar10 --epochs 10
> ```
> Then test the saved weights with `--mode inference --model basiccnn --weights outputs/basiccnn_cifar10_best.pth`.

---