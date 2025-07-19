#!/usr/bin/env python3
"""
Jetson CNN Toolkit - All-in-One Image Classification System

A comprehensive toolkit for CNN-based image classification on NVIDIA Jetson devices.
Includes multiple CNN architectures, training, inference, and TensorRT optimization.

Features:
- Multiple CNN architectures (BasicCNN, ResNet, MobileNet, EfficientNet)
- Support for popular datasets (CIFAR-10, ImageNet, custom datasets)
- Training with data augmentation and optimization
- Inference with performance benchmarking
- TensorRT optimization for Jetson devices
- Comprehensive performance monitoring

Usage:
    # Train a model
    python jetson_cnn_toolkit.py --mode train --model resnet --dataset cifar10 --epochs 50
    
    # Run inference
    python jetson_cnn_toolkit.py --mode inference --model mobilenet --weights model.pth --input image.jpg
    
    # Optimize with TensorRT
    python jetson_cnn_toolkit.py --mode optimize --model efficientnet --weights model.pth --precision fp16
    
    # Benchmark performance
    python jetson_cnn_toolkit.py --mode benchmark --model all --dataset cifar10

Author: Jetson AI Team
Version: 1.0.0
"""

import argparse
import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
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
import GPUtil

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

class PerformanceMonitor:
    """Monitor system performance during inference"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.gpu_stats = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # GPU monitoring if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_stats.append({
                    'memory_used': gpus[0].memoryUsed,
                    'memory_total': gpus[0].memoryTotal,
                    'utilization': gpus[0].load * 100
                })
        except:
            pass
            
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Final GPU stats
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_stats.append({
                    'memory_used': gpus[0].memoryUsed,
                    'memory_total': gpus[0].memoryTotal,
                    'utilization': gpus[0].load * 100
                })
        except:
            pass
            
        return {
            'execution_time': self.end_time - self.start_time,
            'memory_delta': self.end_memory - self.start_memory,
            'gpu_utilization': np.mean([stat['utilization'] for stat in self.gpu_stats]) if self.gpu_stats else 0,
            'gpu_memory_used': self.gpu_stats[-1]['memory_used'] if self.gpu_stats else 0
        }

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
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
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
        dummy_input = torch.randn(1, *input_shape).cuda()
        onnx_path = "temp_model.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
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

def main():
    parser = argparse.ArgumentParser(description='Jetson CNN Toolkit')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'inference', 'optimize', 'benchmark'],
                       help='Operation mode')
    
    # Model selection
    parser.add_argument('--model', type=str, default='basiccnn',
                       choices=ModelFactory.get_available_models() + ['all'],
                       help='Model architecture to use')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'imagenet', 'custom'],
                       help='Dataset to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training/inference')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for training')
    
    # File paths
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing dataset')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--input', type=str, default=None,
                       help='Input image for inference')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    # Optimization parameters
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision for TensorRT optimization')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    
    # Number of classes
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes in dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    if args.mode == 'train':
        # Training mode
        logger.info("Starting training mode")
        
        # Create model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        logger.info(f"Created {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Get data loaders
        if args.dataset == 'cifar10':
            train_loader, val_loader = DatasetManager.get_cifar10_loaders(
                batch_size=args.batch_size, data_dir=args.data_dir
            )
        elif args.dataset == 'imagenet':
            train_loader, val_loader = DatasetManager.get_imagenet_loaders(
                batch_size=args.batch_size, data_dir=args.data_dir
            )
        else:
            raise ValueError(f"Training not supported for dataset: {args.dataset}")
        
        # Create trainer
        trainer = Trainer(model, device=args.device, learning_rate=args.learning_rate)
        
        # Train model
        save_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_best.pth")
        train_losses, train_accs, val_accs = trainer.train(
            train_loader, val_loader, num_epochs=args.epochs, save_path=save_path
        )
        
        # Plot training history
        plot_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_training_history.png")
        trainer.plot_training_history(save_path=plot_path)
        
        logger.info(f"Training completed. Best model saved to {save_path}")
    
    elif args.mode == 'inference':
        # Inference mode
        logger.info("Starting inference mode")
        
        if not args.weights:
            raise ValueError("Weights path required for inference mode")
        
        # Create and load model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.weights, map_location=args.device))
        
        # Create inference engine
        inference_engine = InferenceEngine(model, device=args.device)
        
        if args.input:
            # Single image inference
            predicted_class, confidence = inference_engine.predict_single(args.input)
            logger.info(f"Prediction: Class {predicted_class}, Confidence: {confidence:.4f}")
        else:
            # Batch inference
            if args.dataset == 'cifar10':
                _, test_loader = DatasetManager.get_cifar10_loaders(
                    batch_size=args.batch_size, data_dir=args.data_dir
                )
            elif args.dataset == 'custom':
                test_loader = DatasetManager.get_custom_loader(
                    args.data_dir, batch_size=args.batch_size
                )
            else:
                raise ValueError(f"Batch inference not supported for dataset: {args.dataset}")
            
            predictions, confidences = inference_engine.predict_batch(test_loader)
            logger.info(f"Processed {len(predictions)} images")
            logger.info(f"Average confidence: {np.mean(confidences):.4f}")
    
    elif args.mode == 'optimize':
        # TensorRT optimization mode
        logger.info("Starting TensorRT optimization mode")
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available for optimization")
        
        if not args.weights:
            raise ValueError("Weights path required for optimization mode")
        
        # Create and load model
        model = ModelFactory.create_model(args.model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        model = model.cuda().eval()
        
        # Optimize with TensorRT
        optimizer = TensorRTOptimizer()
        
        # Determine input shape based on dataset
        if args.dataset == 'cifar10':
            input_shape = (3, 32, 32)
        else:
            input_shape = (3, 224, 224)
        
        engine_path = optimizer.optimize_model(
            model, input_shape, precision=args.precision
        )
        
        # Benchmark optimized model
        avg_time, throughput = optimizer.benchmark_tensorrt(engine_path, input_shape)
        
        logger.info(f"Optimization completed. Engine saved to {engine_path}")
        logger.info(f"Optimized performance: {avg_time:.2f} ms, {throughput:.2f} FPS")
    
    elif args.mode == 'benchmark':
        # Benchmarking mode
        logger.info("Starting benchmark mode")
        
        benchmark_suite = BenchmarkSuite(device=args.device)
        
        if args.model == 'all':
            benchmark_suite.benchmark_all_models(args.dataset, args.num_classes)
        else:
            benchmark_suite.benchmark_model(args.model, args.dataset, args.num_classes)
        
        # Save and plot results
        results_path = os.path.join(args.output_dir, 'benchmark_results.json')
        plot_path = os.path.join(args.output_dir, 'benchmark_comparison.png')
        
        benchmark_suite.save_results(results_path)
        benchmark_suite.plot_results(plot_path)
        
        logger.info("Benchmarking completed")
    
    logger.info("Operation completed successfully")

if __name__ == '__main__':
    main()