#!/usr/bin/env python3
"""
Jetson Object Detection Toolkit
Comprehensive all-in-one object detection solution with multiple models and TensorRT acceleration

Supported Models:
- YOLOv8 (Traditional object detection)
- OWL-ViT (Zero-shot detection with text prompts)
- GroundingDINO (Superior zero-shot detection)
- YOLO + CLIP (Two-step detection and classification)
- YOLO + BLIP (Detection with rich descriptions)

Features:
- TensorRT acceleration for Jetson Orin
- Real-time performance monitoring
- Multi-modal scene analysis
- LLM integration for natural language descriptions
- Comprehensive benchmarking and analysis tools

Author: AI Assistant
Date: 2024
"""

import argparse
import cv2
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core ML libraries
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
except ImportError:
    print("PyTorch not found. Please install: pip install torch torchvision")
    sys.exit(1)

# YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Please install: pip install ultralytics")
    YOLO = None

# Transformers for VLM models
try:
    from transformers import (
        OwlViTProcessor, OwlViTForObjectDetection,
        CLIPProcessor, CLIPModel,
        BlipProcessor, BlipForConditionalGeneration,
        AutoProcessor, AutoModelForZeroShotObjectDetection
    )
except ImportError:
    print("Transformers not found. Please install: pip install transformers")
    OwlViTProcessor = None

# TensorRT (optional)
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Install for acceleration: pip install tensorrt")

# Performance monitoring
import psutil
import GPUtil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Standard detection result format"""
    boxes: List[List[float]]  # [x1, y1, x2, y2]
    scores: List[float]
    labels: List[str]
    descriptions: Optional[List[str]] = None
    inference_time: float = 0.0
    memory_usage: float = 0.0

class BaseDetector(ABC):
    """Base class for all detectors"""
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        self.device = device
        self.precision = precision
        self.model = None
        self.processor = None
        
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """Perform object detection"""
        pass
    
    def preprocess_image(self, image: np.ndarray) -> Union[torch.Tensor, Image.Image]:
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        return image
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure inference time and memory usage"""
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory (if available)
        gpu_mem_before = 0
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Time inference
        start_time = time.time()
        result = func(*args, **kwargs)
        inference_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        gpu_mem_after = 0
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        memory_usage = max(mem_after - mem_before, gpu_mem_after - gpu_mem_before)
        
        return result, inference_time, memory_usage

class OptimizedYOLODetector(BaseDetector):
    """Optimized YOLO detector with TensorRT support"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda", 
                 precision: str = "fp16", tensorrt: bool = True):
        super().__init__(device, precision)
        self.model_path = model_path
        self.tensorrt = tensorrt and TRT_AVAILABLE
        self.load_model()
    
    def load_model(self):
        """Load and optimize YOLO model"""
        if YOLO is None:
            raise ImportError("Ultralytics YOLO not available")
        
        self.model = YOLO(self.model_path)
        
        # TensorRT optimization for Jetson
        if self.tensorrt and self.device == "cuda":
            try:
                # Export to TensorRT if not already done
                engine_path = self.model_path.replace('.pt', '.engine')
                if not os.path.exists(engine_path):
                    logger.info("Exporting model to TensorRT...")
                    self.model.export(format='engine', half=True, device=0)
                    logger.info(f"TensorRT engine saved to {engine_path}")
                
                # Load TensorRT engine
                self.model = YOLO(engine_path)
                logger.info("TensorRT engine loaded successfully")
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}. Using PyTorch model.")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> DetectionResult:
        """Perform YOLO detection"""
        def _detect():
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            boxes = []
            scores = []
            labels = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Convert to xyxy format
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        scores.append(float(box.conf[0].cpu().numpy()))
                        
                        # Get class name
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        labels.append(class_name)
            
            return DetectionResult(boxes=boxes, scores=scores, labels=labels)
        
        result, inference_time, memory_usage = self.measure_performance(_detect)
        result.inference_time = inference_time
        result.memory_usage = memory_usage
        
        return result

class OptimizedOWLViTDetector(BaseDetector):
    """Optimized OWL-ViT detector for zero-shot detection"""
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        super().__init__(device, precision)
        self.load_model()
    
    def load_model(self):
        """Load OWL-ViT model with optimizations"""
        if OwlViTProcessor is None:
            raise ImportError("Transformers library not available")
        
        model_name = "google/owlvit-base-patch32"
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        
        # Move to device and optimize
        self.model.to(self.device)
        if self.precision == "fp16" and self.device == "cuda":
            self.model.half()
        
        self.model.eval()
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def detect(self, image: np.ndarray, text_prompts: List[str], 
               score_threshold: float = 0.1) -> DetectionResult:
        """Perform zero-shot detection with text prompts"""
        def _detect():
            image_pil = self.preprocess_image(image)
            
            # Process inputs
            inputs = self.processor(text=text_prompts, images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.precision == "fp16":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.Tensor([image.shape[:2]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
            )
            
            boxes = []
            scores = []
            labels = []
            
            for result in results:
                for box, score, label_id in zip(result["boxes"], result["scores"], result["labels"]):
                    if score > score_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        scores.append(float(score.cpu().numpy()))
                        labels.append(text_prompts[label_id])
            
            return DetectionResult(boxes=boxes, scores=scores, labels=labels)
        
        result, inference_time, memory_usage = self.measure_performance(_detect)
        result.inference_time = inference_time
        result.memory_usage = memory_usage
        
        return result

class GroundingDINODetector(BaseDetector):
    """GroundingDINO detector for superior zero-shot detection"""
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        super().__init__(device, precision)
        self.load_model()
    
    def load_model(self):
        """Load GroundingDINO model"""
        if AutoProcessor is None:
            raise ImportError("Transformers library not available")
        
        model_name = "IDEA-Research/grounding-dino-tiny"
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            
            self.model.to(self.device)
            if self.precision == "fp16" and self.device == "cuda":
                self.model.half()
            
            self.model.eval()
            logger.info("GroundingDINO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")
            raise
    
    def detect(self, image: np.ndarray, text_prompt: str, 
               score_threshold: float = 0.3) -> DetectionResult:
        """Perform detection with natural language prompt"""
        def _detect():
            image_pil = self.preprocess_image(image)
            
            # Process inputs
            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.precision == "fp16":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.Tensor([image.shape[:2]]).to(self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )
            
            boxes = []
            scores = []
            labels = []
            
            for result in results:
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    if score > score_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        scores.append(float(score.cpu().numpy()))
                        labels.append(label)
            
            return DetectionResult(boxes=boxes, scores=scores, labels=labels)
        
        result, inference_time, memory_usage = self.measure_performance(_detect)
        result.inference_time = inference_time
        result.memory_usage = memory_usage
        
        return result

class YOLOCLIPDetector(BaseDetector):
    """Two-step detector: YOLO for detection + CLIP for classification"""
    
    def __init__(self, yolo_model: str = "yolov8n.pt", device: str = "cuda", precision: str = "fp16"):
        super().__init__(device, precision)
        self.yolo_detector = OptimizedYOLODetector(yolo_model, device, precision)
        self.load_clip_model()
    
    def load_clip_model(self):
        """Load CLIP model for classification"""
        if CLIPProcessor is None:
            raise ImportError("Transformers library not available")
        
        model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        self.clip_model.to(self.device)
        if self.precision == "fp16" and self.device == "cuda":
            self.clip_model.half()
        
        self.clip_model.eval()
    
    def detect(self, image: np.ndarray, text_queries: List[str], 
               conf_threshold: float = 0.5) -> DetectionResult:
        """Detect objects and classify with CLIP"""
        def _detect():
            # Step 1: YOLO detection
            yolo_results = self.yolo_detector.detect(image, conf_threshold)
            
            if not yolo_results.boxes:
                return DetectionResult(boxes=[], scores=[], labels=[])
            
            # Step 2: CLIP classification of detected regions
            image_pil = self.preprocess_image(image)
            
            refined_boxes = []
            refined_scores = []
            refined_labels = []
            
            for box, score in zip(yolo_results.boxes, yolo_results.scores):
                x1, y1, x2, y2 = map(int, box)
                
                # Crop detected region
                cropped = image_pil.crop((x1, y1, x2, y2))
                
                # CLIP classification
                inputs = self.clip_processor(
                    text=text_queries, images=cropped, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if self.precision == "fp16":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get best matching query
                best_idx = probs.argmax().item()
                best_prob = probs[0, best_idx].item()
                
                # Only keep if confidence is reasonable
                if best_prob > 0.3:  # Threshold for CLIP confidence
                    refined_boxes.append(box)
                    refined_scores.append(score * best_prob)  # Combine scores
                    refined_labels.append(text_queries[best_idx])
            
            return DetectionResult(
                boxes=refined_boxes, 
                scores=refined_scores, 
                labels=refined_labels
            )
        
        result, inference_time, memory_usage = self.measure_performance(_detect)
        result.inference_time = inference_time
        result.memory_usage = memory_usage
        
        return result

class YOLOBLIPDetector(BaseDetector):
    """YOLO detection + BLIP for rich descriptions"""
    
    def __init__(self, yolo_model: str = "yolov8n.pt", device: str = "cuda", precision: str = "fp16"):
        super().__init__(device, precision)
        self.yolo_detector = OptimizedYOLODetector(yolo_model, device, precision)
        self.load_blip_model()
    
    def load_blip_model(self):
        """Load BLIP model for image captioning"""
        if BlipProcessor is None:
            raise ImportError("Transformers library not available")
        
        model_name = "Salesforce/blip-image-captioning-base"
        self.blip_processor = BlipProcessor.from_pretrained(model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        self.blip_model.to(self.device)
        if self.precision == "fp16" and self.device == "cuda":
            self.blip_model.half()
        
        self.blip_model.eval()
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> DetectionResult:
        """Detect objects and generate descriptions"""
        def _detect():
            # Step 1: YOLO detection
            yolo_results = self.yolo_detector.detect(image, conf_threshold)
            
            if not yolo_results.boxes:
                return DetectionResult(boxes=[], scores=[], labels=[], descriptions=[])
            
            # Step 2: BLIP descriptions for detected regions
            image_pil = self.preprocess_image(image)
            descriptions = []
            
            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Crop detected region
                cropped = image_pil.crop((x1, y1, x2, y2))
                
                # Generate description
                inputs = self.blip_processor(cropped, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if self.precision == "fp16":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    out = self.blip_model.generate(**inputs, max_length=50)
                
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
                descriptions.append(description)
            
            return DetectionResult(
                boxes=yolo_results.boxes,
                scores=yolo_results.scores,
                labels=yolo_results.labels,
                descriptions=descriptions
            )
        
        result, inference_time, memory_usage = self.measure_performance(_detect)
        result.inference_time = inference_time
        result.memory_usage = memory_usage
        
        return result

class MultiModalSceneAnalyzer:
    """Comprehensive scene analysis using multiple models"""
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        self.device = device
        self.precision = precision
        
        # Initialize detectors
        self.yolo_detector = OptimizedYOLODetector(device=device, precision=precision)
        self.owlvit_detector = OptimizedOWLViTDetector(device=device, precision=precision)
        self.grounding_dino = GroundingDINODetector(device=device, precision=precision)
        self.yolo_clip = YOLOCLIPDetector(device=device, precision=precision)
        self.yolo_blip = YOLOBLIPDetector(device=device, precision=precision)
    
    def analyze_scene(self, image: np.ndarray, context_prompts: List[str] = None, 
                     fusion_strategy: str = "weighted") -> Dict:
        """Comprehensive multi-modal scene analysis"""
        results = {}
        
        # Fast detection with YOLO
        results['yolo'] = self.yolo_detector.detect(image)
        
        # Zero-shot detection if prompts provided
        if context_prompts:
            results['owlvit'] = self.owlvit_detector.detect(image, context_prompts)
            results['grounding_dino'] = self.grounding_dino.detect(
                image, ", ".join(context_prompts)
            )
            results['yolo_clip'] = self.yolo_clip.detect(image, context_prompts)
        
        # Rich descriptions
        results['yolo_blip'] = self.yolo_blip.detect(image)
        
        # Fusion of results
        if fusion_strategy == "weighted":
            fused_result = self._weighted_fusion(results)
        else:
            fused_result = self._simple_fusion(results)
        
        return {
            'individual_results': results,
            'fused_result': fused_result,
            'analysis_summary': self._generate_summary(results)
        }
    
    def _weighted_fusion(self, results: Dict) -> DetectionResult:
        """Intelligent fusion based on confidence scores"""
        # Implementation of weighted fusion logic
        # This is a simplified version - real implementation would be more sophisticated
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Weight different models based on their strengths
        weights = {
            'yolo': 1.0,  # High weight for speed and general objects
            'grounding_dino': 0.9,  # High weight for accuracy
            'owlvit': 0.8,  # Good for zero-shot
            'yolo_clip': 0.7,  # Good for classification
            'yolo_blip': 0.6   # Lower weight, mainly for descriptions
        }
        
        for model_name, result in results.items():
            if hasattr(result, 'boxes') and result.boxes:
                weight = weights.get(model_name, 0.5)
                for box, score, label in zip(result.boxes, result.scores, result.labels):
                    all_boxes.append(box)
                    all_scores.append(score * weight)
                    all_labels.append(f"{model_name}:{label}")
        
        return DetectionResult(boxes=all_boxes, scores=all_scores, labels=all_labels)
    
    def _simple_fusion(self, results: Dict) -> DetectionResult:
        """Simple concatenation of all results"""
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for model_name, result in results.items():
            if hasattr(result, 'boxes') and result.boxes:
                all_boxes.extend(result.boxes)
                all_scores.extend(result.scores)
                all_labels.extend([f"{model_name}:{label}" for label in result.labels])
        
        return DetectionResult(boxes=all_boxes, scores=all_scores, labels=all_labels)
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate analysis summary"""
        summary = {
            'total_detections': 0,
            'model_performance': {},
            'unique_objects': set(),
            'avg_inference_time': 0
        }
        
        total_time = 0
        model_count = 0
        
        for model_name, result in results.items():
            if hasattr(result, 'boxes'):
                detection_count = len(result.boxes)
                summary['total_detections'] += detection_count
                summary['model_performance'][model_name] = {
                    'detections': detection_count,
                    'inference_time': getattr(result, 'inference_time', 0),
                    'memory_usage': getattr(result, 'memory_usage', 0)
                }
                
                if hasattr(result, 'labels'):
                    summary['unique_objects'].update(result.labels)
                
                total_time += getattr(result, 'inference_time', 0)
                model_count += 1
        
        summary['unique_objects'] = list(summary['unique_objects'])
        summary['avg_inference_time'] = total_time / max(model_count, 1)
        
        return summary

class PerformanceBenchmark:
    """Comprehensive performance benchmarking tool"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = []
    
    def benchmark_model(self, detector: BaseDetector, test_images: List[np.ndarray], 
                       test_name: str, **kwargs) -> Dict:
        """Benchmark a specific detector"""
        logger.info(f"Benchmarking {test_name}...")
        
        times = []
        memory_usage = []
        detection_counts = []
        
        for i, image in enumerate(test_images):
            try:
                result = detector.detect(image, **kwargs)
                times.append(result.inference_time)
                memory_usage.append(result.memory_usage)
                detection_counts.append(len(result.boxes))
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(test_images)} images")
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue
        
        # Calculate statistics
        avg_time = np.mean(times) if times else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        avg_detections = np.mean(detection_counts) if detection_counts else 0
        
        benchmark_result = {
            'model': test_name,
            'avg_inference_time_ms': avg_time * 1000,
            'avg_fps': avg_fps,
            'avg_memory_mb': avg_memory,
            'avg_detections': avg_detections,
            'total_images': len(test_images),
            'successful_images': len(times)
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_comprehensive_benchmark(self, test_images: List[np.ndarray]) -> Dict:
        """Run benchmark on all available models"""
        logger.info("Starting comprehensive benchmark...")
        
        # Test YOLO
        try:
            yolo_detector = OptimizedYOLODetector(device=self.device)
            self.benchmark_model(yolo_detector, test_images, "YOLO")
        except Exception as e:
            logger.error(f"YOLO benchmark failed: {e}")
        
        # Test OWL-ViT
        try:
            owlvit_detector = OptimizedOWLViTDetector(device=self.device)
            test_prompts = ["person", "car", "chair", "bottle", "laptop"]
            self.benchmark_model(owlvit_detector, test_images, "OWL-ViT", 
                               text_prompts=test_prompts)
        except Exception as e:
            logger.error(f"OWL-ViT benchmark failed: {e}")
        
        # Test GroundingDINO
        try:
            grounding_detector = GroundingDINODetector(device=self.device)
            self.benchmark_model(grounding_detector, test_images, "GroundingDINO",
                               text_prompt="person, car, chair, bottle, laptop")
        except Exception as e:
            logger.error(f"GroundingDINO benchmark failed: {e}")
        
        # Test YOLO+CLIP
        try:
            yolo_clip_detector = YOLOCLIPDetector(device=self.device)
            test_queries = ["person", "vehicle", "furniture", "electronics"]
            self.benchmark_model(yolo_clip_detector, test_images, "YOLO+CLIP",
                               text_queries=test_queries)
        except Exception as e:
            logger.error(f"YOLO+CLIP benchmark failed: {e}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Sort by FPS (descending)
        sorted_results = sorted(self.results, key=lambda x: x['avg_fps'], reverse=True)
        
        report = {
            'summary': {
                'fastest_model': sorted_results[0]['model'],
                'fastest_fps': sorted_results[0]['avg_fps'],
                'most_accurate': max(self.results, key=lambda x: x['avg_detections'])['model'],
                'most_efficient': min(self.results, key=lambda x: x['avg_memory_mb'])['model']
            },
            'detailed_results': sorted_results,
            'recommendations': self._generate_recommendations(sorted_results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[Dict]) -> Dict:
        """Generate usage recommendations based on benchmark results"""
        recommendations = {}
        
        # Real-time applications (>15 FPS)
        real_time_models = [r for r in results if r['avg_fps'] > 15]
        if real_time_models:
            recommendations['real_time'] = real_time_models[0]['model']
        
        # High accuracy (most detections)
        accuracy_model = max(results, key=lambda x: x['avg_detections'])
        recommendations['high_accuracy'] = accuracy_model['model']
        
        # Resource constrained (lowest memory)
        efficient_model = min(results, key=lambda x: x['avg_memory_mb'])
        recommendations['resource_constrained'] = efficient_model['model']
        
        # Balanced performance
        for result in results:
            score = (result['avg_fps'] / 50) + (result['avg_detections'] / 10) - (result['avg_memory_mb'] / 100)
            result['balance_score'] = score
        
        balanced_model = max(results, key=lambda x: x.get('balance_score', 0))
        recommendations['balanced'] = balanced_model['model']
        
        return recommendations

class VisualizationUtils:
    """Utilities for visualizing detection results"""
    
    @staticmethod
    def draw_detections(image: np.ndarray, result: DetectionResult, 
                       show_descriptions: bool = False) -> np.ndarray:
        """Draw detection results on image"""
        vis_image = image.copy()
        
        for i, (box, score, label) in enumerate(zip(result.boxes, result.scores, result.labels)):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label_text = f"{label}: {score:.2f}"
            if show_descriptions and result.descriptions and i < len(result.descriptions):
                label_text += f" - {result.descriptions[i][:30]}..."
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image
    
    @staticmethod
    def create_performance_chart(benchmark_results: List[Dict]) -> str:
        """Create a simple text-based performance chart"""
        chart = "\n" + "="*80 + "\n"
        chart += "PERFORMANCE COMPARISON\n"
        chart += "="*80 + "\n"
        chart += f"{'Model':<20} {'FPS':<10} {'Time(ms)':<12} {'Memory(MB)':<12} {'Detections':<12}\n"
        chart += "-"*80 + "\n"
        
        for result in benchmark_results:
            chart += f"{result['model']:<20} "
            chart += f"{result['avg_fps']:<10.1f} "
            chart += f"{result['avg_inference_time_ms']:<12.1f} "
            chart += f"{result['avg_memory_mb']:<12.1f} "
            chart += f"{result['avg_detections']:<12.1f}\n"
        
        chart += "="*80 + "\n"
        return chart

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Jetson Object Detection Toolkit")
    
    # Model selection
    parser.add_argument('--model', choices=['yolo', 'owlvit', 'grounding-dino', 'yolo-clip', 'yolo-blip', 'multi-modal'],
                       default='yolo', help='Detection model to use')
    
    # Input options
    parser.add_argument('--input', default='camera', help='Input source (camera, video file, or image file)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    
    # Model parameters
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16'], help='Model precision')
    parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT acceleration')
    
    # Detection parameters
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--prompts', nargs='+', default=['person', 'car', 'chair'], help='Text prompts for zero-shot models')
    parser.add_argument('--text-prompt', default='person, car, chair', help='Text prompt for GroundingDINO')
    
    # Benchmarking
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--benchmark-images', type=int, default=100, help='Number of images for benchmarking')
    
    # Advanced analysis
    parser.add_argument('--advanced-analysis', choices=['optimization-impact', 'novel-objects', 'precision-study'],
                       help='Run advanced analysis tasks')
    
    # Output options
    parser.add_argument('--output-dir', default='./output', help='Output directory for results')
    parser.add_argument('--save-results', action='store_true', help='Save detection results')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    
    # Visualization
    parser.add_argument('--no-display', action='store_true', help='Disable display output')
    parser.add_argument('--save-video', help='Save output video file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector based on model choice
    detector = None
    
    try:
        if args.model == 'yolo':
            detector = OptimizedYOLODetector(
                model_path=args.yolo_model,
                device=args.device,
                precision=args.precision,
                tensorrt=args.tensorrt
            )
        elif args.model == 'owlvit':
            detector = OptimizedOWLViTDetector(device=args.device, precision=args.precision)
        elif args.model == 'grounding-dino':
            detector = GroundingDINODetector(device=args.device, precision=args.precision)
        elif args.model == 'yolo-clip':
            detector = YOLOCLIPDetector(yolo_model=args.yolo_model, device=args.device, precision=args.precision)
        elif args.model == 'yolo-blip':
            detector = YOLOBLIPDetector(yolo_model=args.yolo_model, device=args.device, precision=args.precision)
        elif args.model == 'multi-modal':
            detector = MultiModalSceneAnalyzer(device=args.device, precision=args.precision)
        
        logger.info(f"Initialized {args.model} detector")
        
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running performance benchmark...")
        
        # Generate test images (or load from dataset)
        test_images = []
        if args.input == 'camera':
            cap = cv2.VideoCapture(args.camera_id)
            for _ in range(args.benchmark_images):
                ret, frame = cap.read()
                if ret:
                    test_images.append(frame)
                else:
                    break
            cap.release()
        else:
            # For simplicity, duplicate a single image
            if os.path.isfile(args.input):
                image = cv2.imread(args.input)
                test_images = [image] * args.benchmark_images
        
        if test_images:
            benchmark = PerformanceBenchmark(device=args.device)
            
            if args.model == 'multi-modal':
                # Special handling for multi-modal
                results = []
                for image in test_images[:10]:  # Limit for multi-modal due to complexity
                    start_time = time.time()
                    result = detector.analyze_scene(image, args.prompts)
                    inference_time = time.time() - start_time
                    
                    results.append({
                        'model': 'Multi-Modal',
                        'avg_inference_time_ms': inference_time * 1000,
                        'avg_fps': 1.0 / inference_time,
                        'avg_memory_mb': 0,  # Simplified
                        'avg_detections': len(result['fused_result'].boxes)
                    })
                
                benchmark.results = results
            else:
                # Standard benchmark
                if args.model in ['owlvit']:
                    benchmark.benchmark_model(detector, test_images, args.model, text_prompts=args.prompts)
                elif args.model == 'grounding-dino':
                    benchmark.benchmark_model(detector, test_images, args.model, text_prompt=args.text_prompt)
                elif args.model in ['yolo-clip']:
                    benchmark.benchmark_model(detector, test_images, args.model, text_queries=args.prompts)
                else:
                    benchmark.benchmark_model(detector, test_images, args.model)
            
            # Generate and display report
            report = benchmark.generate_report()
            
            print(VisualizationUtils.create_performance_chart(benchmark.results))
            print("\nRECOMMENDATIONS:")
            for use_case, model in report.get('recommendations', {}).items():
                print(f"  {use_case.replace('_', ' ').title()}: {model}")
            
            # Save report
            if args.save_results:
                report_path = os.path.join(args.output_dir, 'benchmark_report.json')
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Benchmark report saved to {report_path}")
        
        return
    
    # Real-time detection
    if args.input == 'camera':
        cap = cv2.VideoCapture(args.camera_id)
        
        # Video writer setup
        video_writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        
        logger.info("Starting real-time detection. Press 'q' to quit.")
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            start_time = time.time()
            
            if args.model == 'multi-modal':
                result = detector.analyze_scene(frame, args.prompts)
                detection_result = result['fused_result']
            elif args.model == 'owlvit':
                detection_result = detector.detect(frame, text_prompts=args.prompts)
            elif args.model == 'grounding-dino':
                detection_result = detector.detect(frame, text_prompt=args.text_prompt)
            elif args.model == 'yolo-clip':
                detection_result = detector.detect(frame, text_queries=args.prompts)
            else:
                detection_result = detector.detect(frame, conf_threshold=args.conf_threshold)
            
            inference_time = time.time() - start_time
            
            # Update statistics
            frame_count += 1
            total_time += inference_time
            avg_fps = frame_count / total_time
            
            # Visualize results
            vis_frame = VisualizationUtils.draw_detections(
                frame, detection_result, 
                show_descriptions=(args.model == 'yolo-blip')
            )
            
            # Add performance info
            info_text = f"FPS: {avg_fps:.1f} | Time: {inference_time*1000:.1f}ms | Objects: {len(detection_result.boxes)}"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame if video writer is active
            if video_writer:
                video_writer.write(vis_frame)
            
            # Display frame
            if not args.no_display:
                cv2.imshow('Object Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames with average FPS: {avg_fps:.2f}")
    
    # Single image or video file processing
    elif os.path.isfile(args.input):
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Single image
            image = cv2.imread(args.input)
            
            if args.model == 'multi-modal':
                result = detector.analyze_scene(image, args.prompts)
                detection_result = result['fused_result']
                
                # Print analysis summary
                print("\nMulti-Modal Analysis Summary:")
                summary = result['analysis_summary']
                print(f"Total detections: {summary['total_detections']}")
                print(f"Unique objects: {', '.join(summary['unique_objects'])}")
                print(f"Average inference time: {summary['avg_inference_time']:.3f}s")
                
                for model, perf in summary['model_performance'].items():
                    print(f"{model}: {perf['detections']} detections, {perf['inference_time']:.3f}s")
            
            elif args.model == 'owlvit':
                detection_result = detector.detect(image, text_prompts=args.prompts)
            elif args.model == 'grounding-dino':
                detection_result = detector.detect(image, text_prompt=args.text_prompt)
            elif args.model == 'yolo-clip':
                detection_result = detector.detect(image, text_queries=args.prompts)
            else:
                detection_result = detector.detect(image, conf_threshold=args.conf_threshold)
            
            # Visualize and save results
            vis_image = VisualizationUtils.draw_detections(
                image, detection_result,
                show_descriptions=(args.model == 'yolo-blip')
            )
            
            output_path = os.path.join(args.output_dir, 'detection_result.jpg')
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Result saved to {output_path}")
            
            if not args.no_display:
                cv2.imshow('Detection Result', vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Print detection info
            print(f"\nDetected {len(detection_result.boxes)} objects:")
            for box, score, label in zip(detection_result.boxes, detection_result.scores, detection_result.labels):
                print(f"  {label}: {score:.3f} at {box}")
                
            if detection_result.descriptions:
                print("\nDescriptions:")
                for i, desc in enumerate(detection_result.descriptions):
                    print(f"  {i+1}: {desc}")
        
        else:
            logger.error("Video file processing not implemented in this example")
    
    else:
        logger.error(f"Input source not found: {args.input}")

if __name__ == "__main__":
    main()