🧠 Deep Learning & CNNs for Image Classification on Jetson

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand deep learning fundamentals and CNN architecture
- Implement basic and advanced CNN models from scratch
- Optimize CNN inference on Jetson devices using various techniques
- Deploy production-ready image classification systems

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
```python
# Mathematical representation of 2D convolution
# Output[i,j] = Σ Σ Input[i+m, j+n] * Kernel[m,n]
#              m n

import numpy as np

def convolution_2d(image, kernel):
    """
    Perform 2D convolution operation
    
    Args:
        image: Input image (H x W)
        kernel: Convolution kernel (K x K)
    
    Returns:
        Convolved output
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            # Element-wise multiplication and sum
            output[i, j] = np.sum(
                image[i:i+kernel_h, j:j+kernel_w] * kernel
            )
    
    return output

# Example: Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

# Apply to sample image
sample_image = np.random.rand(10, 10)
convolved = convolution_2d(sample_image, edge_kernel)
print(f"Original shape: {sample_image.shape}")
print(f"Convolved shape: {convolved.shape}")
```

#### **Activation Functions**
```python
import numpy as np
import matplotlib.pyplot as plt

def activation_functions_demo():
    x = np.linspace(-5, 5, 100)
    
    # ReLU (Rectified Linear Unit)
    relu = np.maximum(0, x)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, relu, 'b-', linewidth=2)
    plt.title('ReLU: f(x) = max(0, x)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, sigmoid, 'r-', linewidth=2)
    plt.title('Sigmoid: f(x) = 1/(1+e^(-x))')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, tanh, 'g-', linewidth=2)
    plt.title('Tanh: f(x) = tanh(x)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu, 'm-', linewidth=2)
    plt.title('Leaky ReLU: f(x) = max(0.01x, x)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run demonstration
activation_functions_demo()
```

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

### **CNN Architecture Patterns**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """
    Basic CNN architecture for image classification
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(BasicCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classification layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model instance
model = BasicCNN(num_classes=1000, input_channels=3)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Model summary
from torchsummary import summary
summary(model, (3, 224, 224))
```

---

## 💻 Basic CNN Implementation from Scratch

### **Simple CNN for CIFAR-10 Classification**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)  # CIFAR-10 has 10 classes
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Data preparation
def prepare_cifar10_data(batch_size=32):
    """
    Prepare CIFAR-10 dataset with augmentation
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, test_loader, num_epochs=10, device='cuda'):
    """
    Train the CNN model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training accuracy
        train_acc = 100. * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        test_acc = 100. * correct_test / total_test
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    return train_losses, train_accuracies, test_accuracies

# Visualization function
def plot_training_history(train_losses, train_accuracies, test_accuracies):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, test_loader = prepare_cifar10_data(batch_size=64)
    
    # Create model
    model = SimpleCNN()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    train_losses, train_accs, test_accs = train_model(
        model, train_loader, test_loader, num_epochs=20, device=device
    )
    
    # Plot results
    plot_training_history(train_losses, train_accs, test_accs)
    
    # Save model
    torch.save(model.state_dict(), 'simple_cnn_cifar10.pth')
    print("Model saved successfully!")
```

---

## 🏛️ Advanced CNN Architectures

### **ResNet (Residual Networks)**

ResNet introduced skip connections to solve the vanishing gradient problem in deep networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        
        return out

class CustomResNet(nn.Module):
    """
    Custom ResNet implementation
    """
    def __init__(self, num_classes=1000):
        super(CustomResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Create ResNet model
resnet_model = CustomResNet(num_classes=10)
print(f"ResNet parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
```

### **MobileNet - Efficient CNN for Mobile Devices**

MobileNet uses depthwise separable convolutions to reduce computational cost.

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution used in MobileNet
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNet(nn.Module):
    """
    MobileNet architecture for efficient inference
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        
        # Calculate channel numbers based on width multiplier
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Initial convolution
        input_channel = _make_divisible(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # Depthwise separable convolutions
        self.features = nn.Sequential(
            DepthwiseSeparableConv(input_channel, _make_divisible(64 * width_multiplier)),
            DepthwiseSeparableConv(_make_divisible(64 * width_multiplier), 
                                 _make_divisible(128 * width_multiplier), stride=2),
            DepthwiseSeparableConv(_make_divisible(128 * width_multiplier), 
                                 _make_divisible(128 * width_multiplier)),
            DepthwiseSeparableConv(_make_divisible(128 * width_multiplier), 
                                 _make_divisible(256 * width_multiplier), stride=2),
            DepthwiseSeparableConv(_make_divisible(256 * width_multiplier), 
                                 _make_divisible(256 * width_multiplier)),
            DepthwiseSeparableConv(_make_divisible(256 * width_multiplier), 
                                 _make_divisible(512 * width_multiplier), stride=2),
        )
        
        # Additional depthwise separable layers
        for _ in range(5):
            self.features.add_module(
                f'dw_conv_{len(self.features)}',
                DepthwiseSeparableConv(_make_divisible(512 * width_multiplier), 
                                     _make_divisible(512 * width_multiplier))
            )
        
        self.features.add_module(
            'dw_conv_final',
            DepthwiseSeparableConv(_make_divisible(512 * width_multiplier), 
                                 _make_divisible(1024 * width_multiplier), stride=2)
        )
        
        self.features.add_module(
            'dw_conv_last',
            DepthwiseSeparableConv(_make_divisible(1024 * width_multiplier), 
                                 _make_divisible(1024 * width_multiplier))
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(_make_divisible(1024 * width_multiplier), num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Create MobileNet model
mobilenet_model = MobileNet(num_classes=10, width_multiplier=0.5)  # Smaller for Jetson
print(f"MobileNet parameters: {sum(p.numel() for p in mobilenet_model.parameters()):,}")
```

### **EfficientNet - Compound Scaling**

EfficientNet uses compound scaling to balance network depth, width, and resolution.

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                                       stride=stride, padding=kernel_size//2, 
                                       groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
            self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)
        
        # Output projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_ratio != 1:
            x = F.relu6(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze and Excitation
        if self.se_ratio > 0:
            se = F.adaptive_avg_pool2d(x, 1)
            se = F.relu(self.se_reduce(se))
            se = torch.sigmoid(self.se_expand(se))
            x = x * se
        
        # Output projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x

class EfficientNet(nn.Module):
    """
    EfficientNet implementation
    """
    def __init__(self, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0):
        super(EfficientNet, self).__init__()
        
        # Stem
        stem_channels = int(32 * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True)
        )
        
        # MBConv blocks configuration
        # [expand_ratio, channels, repeats, stride, kernel_size]
        blocks_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in blocks_config:
            out_channels = int(channels * width_coefficient)
            num_repeats = int(repeats * depth_coefficient)
            
            for i in range(num_repeats):
                self.blocks.append(
                    MBConvBlock(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size,
                        stride if i == 0 else 1,
                        expand_ratio
                    )
                )
                in_channels = out_channels
        
        # Head
        head_channels = int(1280 * width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(head_channels, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.head(x)
        return x

# Create EfficientNet-B0 variant for Jetson
efficientnet_model = EfficientNet(num_classes=10, width_coefficient=0.75, depth_coefficient=0.75)
print(f"EfficientNet parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}")
```

---

## ⚙️ Jetson-Specific Optimizations

### **1. Memory Management for Jetson**

```python
import torch
import gc
import psutil
import subprocess

class JetsonMemoryManager:
    """
    Memory management utilities for Jetson devices
    """
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information"""
        # System memory
        memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory['total'] = torch.cuda.get_device_properties(0).total_memory
            gpu_memory['allocated'] = torch.cuda.memory_allocated(0)
            gpu_memory['cached'] = torch.cuda.memory_reserved(0)
            gpu_memory['free'] = gpu_memory['total'] - gpu_memory['allocated']
        
        return {
            'system': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent
            },
            'gpu': gpu_memory
        }
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Memory optimization completed")
    
    @staticmethod
    def set_jetson_power_mode(mode='MAXN'):
        """Set Jetson power mode for optimal performance"""
        try:
            if mode == 'MAXN':
                subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
            elif mode == '15W':
                subprocess.run(['sudo', 'nvpmodel', '-m', '1'], check=True)
            elif mode == '10W':
                subprocess.run(['sudo', 'nvpmodel', '-m', '2'], check=True)
            
            print(f"Power mode set to {mode}")
        except subprocess.CalledProcessError:
            print("Failed to set power mode. Make sure you have sudo privileges.")
    
    @staticmethod
    def enable_jetson_clocks():
        """Enable maximum clock speeds"""
        try:
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            print("Jetson clocks enabled for maximum performance")
        except subprocess.CalledProcessError:
            print("Failed to enable jetson_clocks")

# Usage example
memory_manager = JetsonMemoryManager()
memory_info = memory_manager.get_memory_info()
print(f"System Memory: {memory_info['system']['percentage']:.1f}% used")
if memory_info['gpu']:
    gpu_usage = (memory_info['gpu']['allocated'] / memory_info['gpu']['total']) * 100
    print(f"GPU Memory: {gpu_usage:.1f}% used")
```

### **2. Model Quantization for Jetson**

```python
import torch
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub

class QuantizedCNN(nn.Module):
    """
    Quantization-ready CNN model
    """
    def __init__(self, num_classes=10):
        super(QuantizedCNN, self).__init__()
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Model layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.quant(x)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = self.dequant(x)
        return x

def quantize_model(model, train_loader, device='cuda'):
    """
    Apply post-training quantization
    """
    # Set quantization configuration
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = quant.prepare(model, inplace=False)
    
    # Calibrate with representative data
    model_prepared.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= 100:  # Use subset for calibration
                break
            data = data.to(device)
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = quant.convert(model_prepared, inplace=False)
    
    return model_quantized

# Example usage
model = QuantizedCNN(num_classes=10)
# quantized_model = quantize_model(model, train_loader)
```

### **3. TensorRT Optimization Pipeline**

```python
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from collections import OrderedDict

class TensorRTOptimizer:
    """
    TensorRT optimization for Jetson deployment
    """
    
    def __init__(self, logger=None):
        self.logger = logger or trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.config = None
        self.engine = None
        
    def build_engine(self, onnx_path, engine_path, max_batch_size=1, 
                    max_workspace_size=1 << 30, fp16_mode=True):
        """
        Build TensorRT engine from ONNX model
        """
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        
        # Parse ONNX
        parser = trt.OnnxParser(self.network, self.logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Create builder config
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = max_workspace_size
        
        # Enable FP16 precision for Jetson
        if fp16_mode and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        
        # Build engine
        print("Building TensorRT engine... This may take a while.")
        self.engine = self.builder.build_engine(self.network, self.config)
        
        if self.engine is None:
            print("Failed to build engine")
            return None
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(self.engine.serialize())
        
        print(f"Engine saved to {engine_path}")
        return self.engine
    
    def load_engine(self, engine_path):
        """
        Load TensorRT engine from file
        """
        runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        return self.engine
    
    def infer(self, input_data):
        """
        Run inference with TensorRT engine
        """
        if self.engine is None:
            raise RuntimeError("Engine not loaded")
        
        # Create execution context
        context = self.engine.create_execution_context()
        
        # Allocate GPU memory
        inputs, outputs, bindings, stream = self._allocate_buffers()
        
        # Copy input data to GPU
        np.copyto(inputs[0].host, input_data.ravel())
        
        # Run inference
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        
        # Synchronize
        stream.synchronize()
        
        return [out.host for out in outputs]
    
    def _allocate_buffers(self):
        """
        Allocate GPU memory buffers
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream

class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

# Example usage
def export_to_onnx(model, dummy_input, onnx_path):
    """
    Export PyTorch model to ONNX
    """
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")

# Complete optimization pipeline
def optimize_for_jetson(model, sample_input, model_name):
    """
    Complete optimization pipeline for Jetson deployment
    """
    # 1. Export to ONNX
    onnx_path = f"{model_name}.onnx"
    export_to_onnx(model, sample_input, onnx_path)
    
    # 2. Build TensorRT engine
    optimizer = TensorRTOptimizer()
    engine_path = f"{model_name}.trt"
    engine = optimizer.build_engine(onnx_path, engine_path, fp16_mode=True)
    
    if engine:
        print(f"Optimization complete! Engine saved to {engine_path}")
        return engine_path
    else:
        print("Optimization failed")
        return None

# Example usage
# model = SimpleCNN()
# sample_input = torch.randn(1, 3, 224, 224)
# engine_path = optimize_for_jetson(model, sample_input, "simple_cnn")
```

---

## 🚀 Production Deployment on Jetson

### **Real-time Image Classification Pipeline**

```python
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from collections import deque
import threading

class JetsonImageClassifier:
    """
    Real-time image classification system for Jetson
    """
    
    def __init__(self, model_path, class_names, device='cuda', input_size=(224, 224)):
        self.device = torch.device(device)
        self.input_size = input_size
        self.class_names = class_names
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.inference_times = deque(maxlen=100)
        
        # Threading for camera
        self.frame_queue = deque(maxlen=2)
        self.result_queue = deque(maxlen=5)
        self.running = False
        
    def _load_model(self, model_path):
        """Load and prepare model for inference"""
        if model_path.endswith('.pth'):
            # PyTorch model
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        elif model_path.endswith('.trt'):
            # TensorRT engine
            from .tensorrt_inference import TensorRTInference
            return TensorRTInference(model_path)
        else:
            raise ValueError("Unsupported model format")
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for inference"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(rgb_frame)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        return input_batch
    
    def predict(self, frame):
        """Run inference on single frame"""
        start_time = time.time()
        
        # Preprocess
        input_batch = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top prediction
        top_prob, top_class = torch.topk(probabilities, 1)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'class_id': top_class.item(),
            'class_name': self.class_names[top_class.item()],
            'confidence': top_prob.item(),
            'inference_time': inference_time
        }
    
    def camera_thread(self, camera_id=0):
        """Camera capture thread"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                if len(self.frame_queue) >= 2:
                    self.frame_queue.popleft()
                self.frame_queue.append(frame)
            
            time.sleep(0.01)  # Small delay to prevent overwhelming
        
        cap.release()
    
    def inference_thread(self):
        """Inference processing thread"""
        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                result = self.predict(frame)
                
                if len(self.result_queue) >= 5:
                    self.result_queue.popleft()
                self.result_queue.append((frame, result))
            
            time.sleep(0.001)
    
    def run_realtime(self, camera_id=0, display=True):
        """Run real-time classification"""
        self.running = True
        
        # Start threads
        camera_thread = threading.Thread(target=self.camera_thread, args=(camera_id,))
        inference_thread = threading.Thread(target=self.inference_thread)
        
        camera_thread.start()
        inference_thread.start()
        
        # Main display loop
        last_fps_time = time.time()
        frame_count = 0
        
        try:
            while self.running:
                if self.result_queue:
                    frame, result = self.result_queue.popleft()
                    
                    if display:
                        # Draw results on frame
                        self._draw_results(frame, result)
                        
                        # Calculate FPS
                        frame_count += 1
                        current_time = time.time()
                        if current_time - last_fps_time >= 1.0:
                            fps = frame_count / (current_time - last_fps_time)
                            self.fps_counter.append(fps)
                            frame_count = 0
                            last_fps_time = current_time
                        
                        # Display frame
                        cv2.imshow('Jetson Image Classification', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("Stopping...")
        
        finally:
            self.running = False
            camera_thread.join()
            inference_thread.join()
            cv2.destroyAllWindows()
    
    def _draw_results(self, frame, result):
        """Draw classification results on frame"""
        # Draw prediction text
        text = f"{result['class_name']}: {result['confidence']:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Draw performance info
        if self.fps_counter:
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 0, 0), 2)
        
        if self.inference_times:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            inference_text = f"Inference: {avg_inference*1000:.1f}ms"
            cv2.putText(frame, inference_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 0, 0), 2)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}
        
        if self.fps_counter:
            stats['avg_fps'] = sum(self.fps_counter) / len(self.fps_counter)
            stats['max_fps'] = max(self.fps_counter)
            stats['min_fps'] = min(self.fps_counter)
        
        if self.inference_times:
            stats['avg_inference_time'] = sum(self.inference_times) / len(self.inference_times)
            stats['max_inference_time'] = max(self.inference_times)
            stats['min_inference_time'] = min(self.inference_times)
        
        return stats

# Example usage
if __name__ == "__main__":
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Initialize classifier
    classifier = JetsonImageClassifier(
        model_path='simple_cnn_cifar10.pth',
        class_names=class_names,
        device='cuda'
    )
    
    # Run real-time classification
    print("Starting real-time classification. Press 'q' to quit.")
    classifier.run_realtime(camera_id=0, display=True)
    
    # Print performance stats
    stats = classifier.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}")

---

## 🧪 Comprehensive Lab: Advanced CNN Implementation and Optimization

### **Lab Overview**
This comprehensive lab will guide you through implementing, training, and optimizing CNN models for real-time image classification on Jetson devices.

### **Learning Objectives**
- Implement multiple CNN architectures from scratch
- Apply Jetson-specific optimizations
- Deploy real-time inference pipelines
- Compare performance across different optimization techniques

### **Prerequisites**
- Jetson Orin Nano with JetPack 5.0+
- PyTorch 1.12+ with CUDA support
- Camera module (USB or CSI)
- 16GB+ storage space

---

## **Part 1: Environment Setup and Data Preparation**

### **1.1 Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python matplotlib seaborn psutil
pip3 install tensorrt pycuda  # For TensorRT optimization
pip3 install onnx onnxruntime-gpu  # For ONNX optimization

# Verify CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **1.2 Dataset Preparation**
```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split

# Enhanced data transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
print("Downloading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=test_transform)

# Create validation split
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

# Data loaders
batch_size = 64  # Optimized for Jetson memory
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {len(trainset)}")
print(f"Validation samples: {len(valset)}")
print(f"Test samples: {len(testset)}")

# Visualize sample data
def show_sample_images(loader, num_samples=8):
    """Display sample images from the dataset"""
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(num_samples):
        img = images[i]
        # Denormalize for display
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        ax = axes[i//4, i%4]
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(f'{class_names[labels[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

show_sample_images(train_loader)
```

---

## **Part 2: Model Implementation and Training**

### **2.1 Training Framework**
```python
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict

class ModelTrainer:
    """
    Comprehensive training framework for CNN models
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = defaultdict(list)
        
    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              weight_decay=1e-4, scheduler_step=20, scheduler_gamma=0.1):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, 
                                            gamma=scheduler_gamma)
        
        best_val_acc = 0
        best_model_state = None
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, epoch_time = self.train_epoch(
                train_loader, criterion, optimizer, epoch + 1
            )
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Time: {epoch_time:.2f}s')
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Training time per epoch
        ax3.plot(self.history['epoch_time'])
        ax3.set_title('Training Time per Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history:
            ax4.plot(self.history['lr'])
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.show()
```

### **2.2 Model Comparison Framework**
```python
class ModelComparison:
    """
    Framework for comparing different CNN architectures
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}
        self.results = {}
        
    def add_model(self, name, model):
        """Add a model to comparison"""
        self.models[name] = model
        
    def train_all_models(self, train_loader, val_loader, epochs=20):
        """Train all models and collect results"""
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
            
            trainer = ModelTrainer(model, self.device)
            history = trainer.train(train_loader, val_loader, epochs=epochs)
            
            # Store results
            self.results[name] = {
                'model': trainer.model,
                'history': history,
                'params': sum(p.numel() for p in model.parameters()),
                'best_val_acc': max(history['val_acc'])
            }
            
            # Save model
            torch.save(trainer.model.state_dict(), f'{name.lower()}_cifar10.pth')
            print(f"Model saved as {name.lower()}_cifar10.pth")
    
    def compare_results(self):
        """Generate comparison report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison table
        print(f"{'Model':<15} {'Parameters':<12} {'Best Val Acc':<12} {'Avg Epoch Time':<15}")
        print("-" * 60)
        
        for name, result in self.results.items():
            params = f"{result['params']:,}"
            best_acc = f"{result['best_val_acc']:.2f}%"
            avg_time = f"{np.mean(result['history']['epoch_time']):.2f}s"
            print(f"{name:<15} {params:<12} {best_acc:<12} {avg_time:<15}")
        
        # Plot comparison
        self.plot_comparison()
    
    def plot_comparison(self):
        """Plot model comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Validation accuracy comparison
        for name, result in self.results.items():
            ax1.plot(result['history']['val_acc'], label=name, linewidth=2)
        ax1.set_title('Validation Accuracy Comparison', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training loss comparison
        for name, result in self.results.items():
            ax2.plot(result['history']['train_loss'], label=name, linewidth=2)
        ax2.set_title('Training Loss Comparison', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Parameter count vs accuracy
        names = list(self.results.keys())
        params = [self.results[name]['params'] for name in names]
        best_accs = [self.results[name]['best_val_acc'] for name in names]
        
        ax3.scatter(params, best_accs, s=100, alpha=0.7)
        for i, name in enumerate(names):
            ax3.annotate(name, (params[i], best_accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax3.set_title('Parameters vs Best Accuracy', fontsize=14)
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Best Validation Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        
        # Training time comparison
        avg_times = [np.mean(self.results[name]['history']['epoch_time']) for name in names]
        ax4.bar(names, avg_times, alpha=0.7)
        ax4.set_title('Average Training Time per Epoch', fontsize=14)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

---

## **Part 3: Model Training and Comparison**

### **3.1 Train Multiple Architectures**
```python
# Initialize model comparison
comparison = ModelComparison(device='cuda')

# Add models to comparison
comparison.add_model('SimpleCNN', SimpleCNN(num_classes=10))
comparison.add_model('ResNet', CustomResNet(num_classes=10))
comparison.add_model('MobileNet', MobileNet(num_classes=10, width_multiplier=0.5))
comparison.add_model('EfficientNet', EfficientNet(num_classes=10, width_coefficient=0.5, depth_coefficient=0.5))

# Train all models (reduce epochs for faster execution)
comparison.train_all_models(train_loader, val_loader, epochs=15)

# Compare results
comparison.compare_results()
```

---

## **Part 4: Jetson Optimization and Deployment**

### **4.1 Performance Benchmarking**
```python
import time
import psutil
import torch.profiler

class JetsonBenchmark:
    """
    Comprehensive benchmarking for Jetson deployment
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def benchmark_inference(self, input_size=(1, 3, 32, 32), num_runs=100, warmup=10):
        """Benchmark inference performance"""
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {num_runs} inference iterations...")
        times = []
        
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_runs} iterations")
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to milliseconds
        
        results = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'fps': 1000 / np.mean(times),
            'throughput': input_size[0] * 1000 / np.mean(times)  # samples per second
        }
        
        return results
    
    def memory_profile(self, input_size=(1, 3, 32, 32)):
        """Profile memory usage"""
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure memory before inference
        if self.device.type == 'cuda':
            memory_before = torch.cuda.memory_allocated()
        
        # Run inference
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Measure memory after inference
        if self.device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        
        memory_info = {
            'model_memory': memory_after - memory_before,
            'peak_memory': peak_memory,
            'total_gpu_memory': torch.cuda.get_device_properties(0).total_memory if self.device.type == 'cuda' else 0
        }
        
        return memory_info
    
    def profile_with_pytorch_profiler(self, input_size=(1, 3, 32, 32), num_steps=10):
        """Detailed profiling with PyTorch profiler"""
        dummy_input = torch.randn(input_size).to(self.device)
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(num_steps):
                    _ = self.model(dummy_input)
        
        # Print profiling results
        print("\nCPU Time Breakdown:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if self.device.type == 'cuda':
            print("\nGPU Time Breakdown:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return prof

# Benchmark all trained models
print("\n" + "="*60)
print("JETSON PERFORMANCE BENCHMARKING")
print("="*60)

for name, result in comparison.results.items():
    print(f"\nBenchmarking {name}...")
    
    benchmark = JetsonBenchmark(result['model'])
    
    # Inference benchmark
    perf_results = benchmark.benchmark_inference()
    print(f"Average inference time: {perf_results['mean_time']:.2f} ± {perf_results['std_time']:.2f} ms")
    print(f"FPS: {perf_results['fps']:.1f}")
    print(f"Throughput: {perf_results['throughput']:.1f} samples/sec")
    
    # Memory profile
    memory_results = benchmark.memory_profile()
    print(f"Model memory usage: {memory_results['model_memory'] / 1024**2:.1f} MB")
    print(f"Peak memory usage: {memory_results['peak_memory'] / 1024**2:.1f} MB")
    
    # Store benchmark results
    result['benchmark'] = perf_results
    result['memory'] = memory_results
```

---

## **Part 5: Real-time Deployment**

### **5.1 Deploy Best Model for Real-time Inference**
```python
# Select best model based on accuracy and performance trade-off
best_model_name = max(comparison.results.keys(), 
                     key=lambda x: comparison.results[x]['best_val_acc'])
best_model = comparison.results[best_model_name]['model']

print(f"Deploying {best_model_name} for real-time inference...")

# Initialize real-time classifier
classifier = JetsonImageClassifier(
    model_path=None,  # We'll pass the model directly
    class_names=class_names,
    device='cuda'
)

# Override model loading to use our trained model
classifier.model = best_model
classifier.model.eval()

# Run real-time classification
print("Starting real-time classification...")
print("Press 'q' to quit, 's' to save screenshot")

try:
    classifier.run_realtime(camera_id=0, display=True)
except KeyboardInterrupt:
    print("\nStopping real-time inference...")

# Print final performance statistics
stats = classifier.get_performance_stats()
print("\nFinal Performance Statistics:")
for key, value in stats.items():
    print(f"{key}: {value:.3f}")
```

---

## **Lab Deliverables**

### **Required Deliverables**

1. **Model Implementation Report**
   - Implementation of at least 3 different CNN architectures
   - Training curves and accuracy comparisons
   - Analysis of parameter count vs performance trade-offs

2. **Optimization Analysis**
   - Benchmark results for all models
   - Memory usage analysis
   - Performance profiling reports
   - Jetson-specific optimization recommendations

3. **Real-time Deployment Demo**
   - Working real-time inference pipeline
   - Performance metrics (FPS, latency, accuracy)
   - Video demonstration or screenshots

4. **Technical Documentation**
   - Code documentation and comments
   - Setup instructions for reproduction
   - Troubleshooting guide

### **Bonus Challenges**

1. **🏆 TensorRT Master**
   - Convert best model to TensorRT engine
   - Achieve >50% inference speedup
   - Maintain <2% accuracy loss

2. **🧠 Architecture Innovator**
   - Design and implement custom CNN architecture
   - Achieve competitive accuracy with fewer parameters
   - Document design decisions

3. **⚡ Speed Demon**
   - Achieve >30 FPS real-time inference
   - Implement multi-threading optimization
   - Add performance monitoring dashboard

4. **🎯 Accuracy Champion**
   - Achieve >90% validation accuracy on CIFAR-10
   - Implement advanced training techniques
   - Use ensemble methods or knowledge distillation

5. **🔧 Production Ready**
   - Create complete deployment package
   - Add error handling and logging
   - Implement model versioning and updates

---

## **Summary and Next Steps**

### **What You've Accomplished**
- ✅ Implemented multiple CNN architectures from scratch
- ✅ Applied comprehensive training and validation frameworks
- ✅ Performed Jetson-specific optimizations
- ✅ Deployed real-time inference pipelines
- ✅ Conducted performance benchmarking and analysis

### **Key Takeaways**
1. **Architecture Matters**: Different CNN architectures have distinct trade-offs between accuracy, speed, and memory usage
2. **Optimization is Critical**: Jetson-specific optimizations can significantly improve performance
3. **Real-time Constraints**: Production deployment requires careful balance of accuracy and speed
4. **Profiling is Essential**: Understanding bottlenecks is key to effective optimization

### **Next Steps**
1. Explore advanced optimization techniques (TensorRT, quantization)
2. Implement object detection and segmentation models
3. Study transformer-based vision models
4. Investigate edge AI deployment strategies
5. Learn about model compression and pruning techniques

---

📌 **Summary**
- ✅ CNNs are essential for computer vision tasks
- ✅ Jetson devices provide excellent edge AI capabilities
- ✅ Multiple optimization strategies can improve performance
- ✅ Real-time deployment requires careful engineering
- ✅ Benchmarking and profiling guide optimization decisions

→ **Next**: Transformers & LLMs on Jetson
```

---


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