#!/usr/bin/env python3
"""
Jetson CNN Toolkit - Comprehensive Image Classification System

A complete toolkit for training, optimizing, and deploying CNN models on multiple platforms.
Now uses the modular CNN toolkit architecture for better maintainability and platform support.

Features:
- Multi-platform support (Apple Silicon, Jetson, NVIDIA GPU, CPU)
- Multiple CNN architectures (BasicCNN, ResNet, MobileNet, EfficientNet)
- Comprehensive training pipeline with validation and visualization
- TensorRT optimization for NVIDIA deployment
- Performance monitoring and benchmarking
- Support for multiple datasets (CIFAR-10, ImageNet, custom)
- Inference engine with batch processing
- Model comparison and analysis tools

Usage:
    # Train a model
    python jetson_cnn_toolkit.py --mode train --model resnet --dataset cifar10 --epochs 50
    
    # Run inference
    python jetson_cnn_toolkit.py --mode inference --model mobilenet --weights model.pth --input image.jpg
    
    # Optimize with TensorRT
    python jetson_cnn_toolkit.py --mode optimize --model efficientnet --weights model.pth --precision fp16
    
    # Benchmark performance
    python jetson_cnn_toolkit.py --mode benchmark --model all --dataset cifar10

Version: 2.0.0
"""

import argparse
import os
import time
import json
import logging
import subprocess
from typing import Dict, List, Tuple, Optional #, Any, Union, cast, Number
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import psutil

# Simplified imports - use built-in components only
MODULAR_TOOLKIT_AVAILABLE = False

# Optional TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available. Optimization features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system monitoring utilities
import sys
from pathlib import Path

# Add parent directory to path to import edgeLLM utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from edgeLLM.utils.performance_monitor import PerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    print("Warning: PerformanceMonitor not available. Using basic monitoring.")

@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results"""
    model_name: str
    dataset: str
    batch_size: int
    inference_time_ms: float
    throughput_fps: float
    accuracy: float
    memory_usage_mb: float
    gpu_utilization: float
    power_consumption_w: Optional[float] = None

# PerformanceMonitor class has been moved to edgeLLM.utils.performance_monitor

# ============================================================================
# CNN ARCHITECTURES
# ============================================================================

class BasicCNN(nn.Module):
    """Basic CNN architecture for image classification"""
    
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

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    
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
    """Custom ResNet implementation"""
    
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

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used in MobileNet"""
    
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
    """MobileNet architecture for efficient inference"""
    
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        
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
        
        # Additional layers
        for _ in range(5):
            self.features.add_module(
                f'dw_conv_{len(self.features)}',
                DepthwiseSeparableConv(_make_divisible(512 * width_multiplier), 
                                     _make_divisible(512 * width_multiplier))
            )
            
        self.features.add_module(
            'final_dw_conv',
            DepthwiseSeparableConv(_make_divisible(512 * width_multiplier), 
                                 _make_divisible(1024 * width_multiplier), stride=2)
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

class EfficientNet(nn.Module):
    """Simplified EfficientNet implementation"""
    
    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks (simplified)
        self.blocks = nn.Sequential(
            self._make_mbconv_block(32, 64, 1, 1),
            self._make_mbconv_block(64, 128, 2, 2),
            self._make_mbconv_block(128, 256, 2, 2),
            self._make_mbconv_block(256, 512, 3, 2),
            self._make_mbconv_block(512, 1024, 3, 1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
    def _make_mbconv_block(self, in_channels, out_channels, num_layers, stride):
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            layers.append(DepthwiseSeparableConv(in_channels if i == 0 else out_channels, out_channels, s))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# ============================================================================
# DATA HANDLING
# ============================================================================

class DatasetManager:
    """Manage different datasets for training and inference"""
    
    @staticmethod
    def get_cifar10_loaders(batch_size=32, data_dir='./data'):
        """Get CIFAR-10 data loaders with augmentation"""
        
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
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
    
    @staticmethod
    def get_imagenet_loaders(batch_size=32, data_dir='./imagenet'):
        """Get ImageNet data loaders"""
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader
    
    @staticmethod
    def get_custom_loader(data_dir, batch_size=32, image_size=224):
        """Get custom dataset loader"""
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ImageFolder(data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return loader

# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory class for creating different CNN models"""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """Create a model by name"""
        
        model_name = model_name.lower()
        
        if model_name == 'basiccnn':
            return BasicCNN(num_classes=num_classes, **kwargs)
        elif model_name == 'resnet':
            return CustomResNet(num_classes=num_classes)
        elif model_name == 'mobilenet':
            width_multiplier = kwargs.get('width_multiplier', 1.0)
            return MobileNet(num_classes=num_classes, width_multiplier=width_multiplier)
        elif model_name == 'efficientnet':
            return EfficientNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models"""
        return ['basiccnn', 'resnet', 'mobilenet', 'efficientnet']

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training class for CNN models"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, save_path=None):
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc and save_path:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")
            
            self.scheduler.step()
        
        return self.train_losses, self.train_accuracies, self.val_accuracies
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.info(f"Training history saved to {save_path}")

# ============================================================================
# INFERENCE
# ============================================================================

class InferenceEngine:
    """Inference engine for CNN models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict_single(self, image_path: str, transform=None) -> Tuple[int, float]:
        """Predict single image"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        # Transform the PIL image to tensor first, then add batch dimension
        # PIL Image doesn't have unsqueeze method, only tensors do
        if transform is not None:
            image_tensor = transform(image)  # This transforms PIL Image to tensor
        else:
            # Fallback: manual conversion if no transform provided
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = default_transform(image)
        
        # Add batch dimension (unsqueeze) and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            # Get predicted class as an integer
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            # Use integer indexing for tensor
            confidence = float(probabilities[0, predicted_class].item())
        
        return predicted_class, confidence
    
    def predict_batch(self, data_loader) -> Tuple[List[int], List[float]]:
        """Predict batch of images"""
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                
                batch_predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
                batch_confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
                
                predictions.extend(batch_predictions)
                confidences.extend(batch_confidences)
        
        return predictions, confidences
    
    def benchmark_inference(self, data_loader, num_warmup=10) -> BenchmarkResult:
        """Benchmark inference performance"""
        logger.info("Starting inference benchmark...")
        
        # Warmup
        for i, (data, _) in enumerate(data_loader):
            if i >= num_warmup:
                break
            data = data.to(self.device)
            with torch.no_grad():
                _ = self.model(data)
        
        # Actual benchmark
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                _, predicted = torch.max(output, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted == target).sum().item()
        
        end_time = time.time()
        
        perf_stats = monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        throughput = total_samples / total_time
        accuracy = 100. * correct_predictions / total_samples
        avg_inference_time = (total_time / total_samples) * 1000  # ms
        
        return BenchmarkResult(
            model_name=self.model.__class__.__name__,
            dataset="benchmark",
            batch_size=data_loader.batch_size,
            inference_time_ms=avg_inference_time,
            throughput_fps=throughput,
            accuracy=accuracy,
            memory_usage_mb=perf_stats['memory_delta'],
            gpu_utilization=perf_stats['gpu_utilization']
        )

# ============================================================================
# TENSORRT OPTIMIZATION
# ============================================================================

class TensorRTOptimizer:
    """TensorRT optimization for Jetson devices"""
    
    def __init__(self):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def optimize_model(self, model, input_shape, precision='fp16', max_batch_size=1, workspace_size=1<<30):
        """Optimize PyTorch model with TensorRT"""
        logger.info(f"Optimizing model with TensorRT (precision: {precision})")
        
        # Export to ONNX first
        # Create a tuple from input_shape to ensure proper unpacking
        input_shape_tuple = tuple(input_shape) if isinstance(input_shape, (list, tuple)) else (input_shape,)
        # Create dummy input with proper dimensions
        if isinstance(input_shape_tuple[0], int):
            # If input_shape_tuple contains integers, use it directly
            dummy_input = torch.randn(1, *input_shape_tuple).cuda()
        else:
            # If input_shape_tuple contains a nested tuple/list, extract the values
            flattened_shape = [item for sublist in input_shape_tuple for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
            dummy_input = torch.randn(1, *flattened_shape).cuda()
        onnx_path = "temp_model.onnx"
        
        torch.onnx.export(
            model,
            (dummy_input,), #PyTorch expects it to be a tuple
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Build TensorRT engine
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        engine_path = "optimized_model.trt"
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        
        # Cleanup
        os.remove(onnx_path)
        
        logger.info(f"TensorRT engine saved to {engine_path}")
        return engine_path
    
    def load_engine(self, engine_path):
        """Load TensorRT engine"""
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        return engine
    
    def benchmark_tensorrt(self, engine_path, input_shape, num_iterations=1000):
        """Benchmark TensorRT engine"""
        engine = self.load_engine(engine_path)
        context = engine.create_execution_context()
        
        # Allocate buffers
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = 1000 * np.dtype(np.float32).itemsize  # Assuming 1000 classes
        
        h_input = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(1000, dtype=np.float32)
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        # Create random input
        np.random.seed(42)
        h_input[:] = np.random.random(np.prod(input_shape))
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod(d_input, h_input)
            context.execute_v2([int(d_input), int(d_output)])
            cuda.memcpy_dtoh(h_output, d_output)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            cuda.memcpy_htod(d_input, h_input)
            context.execute_v2([int(d_input), int(d_output)])
            cuda.memcpy_dtoh(h_output, d_output)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = 1000 / avg_time  # FPS
        
        logger.info(f"TensorRT Benchmark Results:")
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} FPS")
        
        return avg_time, throughput

# ============================================================================
# BENCHMARKING
# ============================================================================

class BenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
        
    def benchmark_model(self, model_name: str, dataset_name: str, num_classes: int = 10, 
                       batch_size: int = 32, num_samples: int = 1000) -> BenchmarkResult:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name} on {dataset_name}")
        
        # Create model
        model = ModelFactory.create_model(model_name, num_classes=num_classes)
        
        # Create dummy data loader for benchmarking
        if dataset_name == 'cifar10':
            _, test_loader = DatasetManager.get_cifar10_loaders(batch_size=batch_size)
        else:
            # Create dummy dataset for benchmarking
            dummy_data = torch.randn(num_samples, 3, 224, 224)
            dummy_targets = torch.randint(0, num_classes, (num_samples,))
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
            test_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
        
        # Run inference benchmark
        inference_engine = InferenceEngine(model, device=self.device)
        result = inference_engine.benchmark_inference(test_loader)
        
        self.results.append(result)
        return result
    
    def benchmark_all_models(self, dataset_name: str = 'cifar10', num_classes: int = 10):
        """Benchmark all available models"""
        models = ModelFactory.get_available_models()
        
        for model_name in models:
            try:
                result = self.benchmark_model(model_name, dataset_name, num_classes)
                logger.info(f"{model_name}: {result.throughput_fps:.2f} FPS, {result.accuracy:.2f}% accuracy")
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save benchmark results to JSON"""
        results_dict = []
        for result in self.results:
            results_dict.append({
                'model_name': result.model_name,
                'dataset': result.dataset,
                'batch_size': result.batch_size,
                'inference_time_ms': result.inference_time_ms,
                'throughput_fps': result.throughput_fps,
                'accuracy': result.accuracy,
                'memory_usage_mb': result.memory_usage_mb,
                'gpu_utilization': result.gpu_utilization
            })
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")
    
    def plot_results(self, save_path: str = 'benchmark_comparison.png'):
        """Plot benchmark results"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        models = [r.model_name for r in self.results]
        throughputs = [r.throughput_fps for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput
        ax1.bar(models, throughputs, color='skyblue')
        ax1.set_title('Inference Throughput (FPS)')
        ax1.set_ylabel('FPS')
        ax1.tick_params(axis='x', rotation=45)
        
        # Accuracy
        ax2.bar(models, accuracies, color='lightgreen')
        ax2.set_title('Model Accuracy (%)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage
        ax3.bar(models, memory_usage, color='salmon')
        ax3.set_title('Memory Usage (MB)')
        ax3.set_ylabel('Memory (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Efficiency (FPS per MB)
        efficiency = [t/m if m > 0 else 0 for t, m in zip(throughputs, memory_usage)]
        ax4.bar(models, efficiency, color='gold')
        ax4.set_title('Efficiency (FPS/MB)')
        ax4.set_ylabel('FPS/MB')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.info(f"Benchmark plots saved to {save_path}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def get_best_device(device_choice):
    """Simple function to choose the best available device for beginners"""
    if device_choice == 'auto':
        if torch.cuda.is_available():
            logger.info("Found NVIDIA GPU - using CUDA for fast AI processing!")
            return 'cuda'
        elif torch.backends.mps.is_available():
            logger.info("Found Apple Silicon - using MPS for fast AI processing!")
            return 'mps'
        else:
            logger.info("Using CPU - AI will work but might be slower")
            return 'cpu'
    else:
        # Check if requested device is available
        if device_choice == 'cuda' and not torch.cuda.is_available():
            logger.warning("NVIDIA GPU not available, using CPU instead")
            return 'cpu'
        elif device_choice == 'mps' and not torch.backends.mps.is_available():
            logger.warning("Apple Silicon not available, using CPU instead")
            return 'cpu'
        else:
            return device_choice

def parse_arguments():
    """Parse command line arguments for the CNN toolkit"""
    parser = argparse.ArgumentParser(description='Simple CNN Toolkit for Beginners')
    
    # Basic operation modes
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference', 'benchmark', 'optimize'],
                        help='What do you want to do?')
    
    # Model selection
    parser.add_argument('--model', type=str, default='basiccnn',
                        choices=['basiccnn', 'resnet18', 'resnet50', 'mobilenetv2', 'efficientnetb0', 'all'],
                        help='Which AI model to use?')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'custom'],
                        help='Which dataset to use?')
    
    # File paths
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Where is your data stored?')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Where to save results?')
    parser.add_argument('--weights', type=str,
                        help='Path to trained model file')
    parser.add_argument('--input', type=str,
                        help='Path to image for testing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='How many training rounds?')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='How many images to process at once?')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='How fast should the AI learn?')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Which device to use for AI processing?')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of different categories')
    
    # Optimization parameters
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision for optimization')
    
    # Verbose logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cnn_toolkit.log')
        ]
    )

# Create logger
logger = logging.getLogger(__name__)

def main():
    """Main function to run the simple CNN toolkit for beginners"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Choose the best device for AI processing
    device = get_best_device(args.device)
    args.device = device
    logger.info(f"Ready to run AI on: {device}")
    
    if args.mode == 'train':
        # Simple training mode
        logger.info("Starting training - this will teach the AI to recognize images!")
        
        # Create the AI model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        logger.info(f"Created {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Get the training data
        if args.dataset == 'cifar10':
            train_loader, val_loader = DatasetManager.get_cifar10_loaders(
                batch_size=args.batch_size, data_dir=args.data_dir
            )
        elif args.dataset == 'custom':
            # For custom datasets, we'll use a simple approach
            train_loader = DatasetManager.get_custom_loader(
                args.data_dir, batch_size=args.batch_size
            )
            val_loader = train_loader  # Simplified - use same for validation
        else:
            raise ValueError(f"Dataset {args.dataset} not supported for training")
        
        # Create trainer
        trainer = Trainer(model, device=args.device, learning_rate=args.learning_rate)
        
        # Train the model
        save_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_best.pth")
        logger.info(f"Training for {args.epochs} epochs...")
        
        train_losses, train_accs, val_accs = trainer.train(
            train_loader, val_loader, num_epochs=args.epochs, save_path=save_path
        )
        
        # Save training results
        plot_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_training_history.png")
        trainer.plot_training_history(save_path=plot_path)
        
        logger.info(f"Training completed! Best model saved to {save_path}")
        logger.info(f"Training history plot saved to {plot_path}")
    
    elif args.mode == 'inference':
        # Simple inference mode - test the AI on new images
        logger.info("Starting inference - testing the AI on images!")
        
        if not args.weights:
            raise ValueError("You need to provide a trained model file with --weights")
        
        # Create the AI model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.weights, map_location=args.device))
        logger.info(f"Loaded trained AI model from {args.weights}")
        
        # Create inference engine
        inference_engine = InferenceEngine(model, device=args.device)
        
        if args.input:
            # Test on a single image
            logger.info(f"Testing AI on single image: {args.input}")
            predicted_class, confidence = inference_engine.predict_single(args.input)
            logger.info(f"AI thinks this is class {predicted_class} with {confidence:.1%} confidence")
        else:
            # Test on multiple images
            logger.info(f"Testing AI on {args.dataset} dataset...")
            
            if args.dataset == 'cifar10':
                _, test_loader = DatasetManager.get_cifar10_loaders(
                    batch_size=args.batch_size, data_dir=args.data_dir
                )
            elif args.dataset == 'custom':
                test_loader = DatasetManager.get_custom_loader(
                    args.data_dir, batch_size=args.batch_size
                )
            else:
                raise ValueError(f"Dataset {args.dataset} not supported for testing")
            
            predictions, confidences = inference_engine.predict_batch(test_loader)
            logger.info(f"Tested {len(predictions)} images")
            logger.info(f"Average confidence: {np.mean(confidences):.1%}")
    
    elif args.mode == 'optimize':
        # Simple optimization mode - make the AI run faster
        logger.info("Starting optimization - making the AI faster!")
        
        if not args.weights:
            raise ValueError("You need to provide a trained model file with --weights")
        
        # Only support TensorRT optimization for now (simplified)
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Optimization requires NVIDIA GPU with TensorRT.")
            logger.info("Your AI will still work, but won't be optimized for speed.")
            return
        
        # Load the trained AI model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        model = model.cuda().eval()
        logger.info(f"Loaded AI model from {args.weights}")
        
        # Make the AI faster with TensorRT
        optimizer = TensorRTOptimizer()
        
        # Figure out what size images the AI expects
        if args.dataset == 'cifar10':
            input_shape = (3, 32, 32)
            logger.info("Optimizing for small 32x32 images (CIFAR-10)")
        else:
            input_shape = (3, 224, 224)
            logger.info("Optimizing for standard 224x224 images")
        
        # Create the faster version
        logger.info("Creating optimized AI... this may take a few minutes")
        engine_path = optimizer.optimize_model(
            model, input_shape, precision=getattr(args, 'precision', 'fp16')
        )
        
        # Test how much faster it is
        logger.info("Testing the optimized AI speed...")
        avg_time, throughput = optimizer.benchmark_tensorrt(engine_path, input_shape)
        
        logger.info(f"Success! Your AI is now optimized and saved to {engine_path}")
        logger.info(f"Speed improvement: {avg_time:.2f} ms per image, {throughput:.2f} images per second")
        logger.info("Your AI should now run much faster on this device!")
    
    elif args.mode == 'benchmark':
        # Simple benchmark mode - test how fast the AI runs
        logger.info("Starting benchmark - testing AI speed!")
        
        # Create benchmark suite
        benchmark_suite = BenchmarkSuite(device=args.device)
        
        if args.model == 'all':
            # Test all AI models
            logger.info("Testing speed of all AI models...")
            models_to_test = ['basiccnn', 'resnet', 'mobilenet', 'efficientnet']
            
            for model_name in models_to_test:
                try:
                    logger.info(f"Testing {model_name} speed...")
                    
                    # Test the speed
                    result = benchmark_suite.benchmark_model(model_name, args.dataset, args.num_classes)
                    logger.info(f"{model_name}: {result.throughput_fps:.2f} images/second, {result.accuracy:.2f}% accuracy")
                    
                except Exception as e:
                    logger.error(f"Failed to test {model_name}: {e}")
                    continue
        else:
            # Test single AI model
            logger.info(f"Testing {args.model} speed...")
            
            try:
                # Test the speed
                result = benchmark_suite.benchmark_model(args.model, args.dataset, args.num_classes)
                logger.info(f"Speed test results: {result.throughput_fps:.2f} images/second")
                logger.info(f"Accuracy: {result.accuracy:.2f}%")
            except Exception as e:
                logger.error(f"Failed to test {args.model}: {e}")
        
        # Save the results
        results_path = os.path.join(args.output_dir, 'benchmark_results.json')
        benchmark_suite.save_results(results_path)
        
        # Create a chart showing the results
        plot_path = os.path.join(args.output_dir, 'benchmark_comparison.png')
        benchmark_suite.plot_results(plot_path)
        
        logger.info(f"Speed test results saved to {results_path}")
        logger.info(f"Speed comparison chart saved to {plot_path}")
        logger.info("Benchmark complete! Check the files to see how fast your AI runs.")
    
    logger.info("🎉 All done! Your AI operation completed successfully!")

if __name__ == '__main__':
    main()