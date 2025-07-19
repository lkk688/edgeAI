# Jetson Object Detection Toolkit

A comprehensive all-in-one object detection solution optimized for NVIDIA Jetson devices, featuring multiple state-of-the-art models with TensorRT acceleration support.

## 🚀 Features

### Supported Models
- **YOLOv8**: Fast traditional object detection with TensorRT optimization
- **OWL-ViT**: Zero-shot detection with natural language prompts
- **GroundingDINO**: Superior zero-shot detection with complex queries
- **YOLO + CLIP**: Two-step detection and semantic classification
- **YOLO + BLIP**: Detection with rich natural language descriptions
- **Multi-Modal**: Comprehensive scene analysis using multiple models

### Key Capabilities
- 🔥 **TensorRT Acceleration**: Optimized inference for Jetson Orin
- 📊 **Real-time Performance Monitoring**: FPS, memory usage, inference time
- 🎯 **Zero-shot Detection**: Detect any object using text descriptions
- 🧠 **Multi-modal Analysis**: Combine multiple models for comprehensive understanding
- 📈 **Comprehensive Benchmarking**: Compare model performance across scenarios
- 🎨 **Rich Visualizations**: Advanced detection result display
- 📝 **Automated Reporting**: Generate detailed analysis reports

## 📦 Installation

### Prerequisites
- NVIDIA Jetson Orin (recommended) or compatible CUDA device
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd edgeAI/docs/curriculum

# Install dependencies
pip install -r requirements.txt

# For TensorRT support on Jetson (optional but recommended)
# Follow NVIDIA's TensorRT installation guide for your Jetson device
```

### Jetson-Specific Setup

```bash
# Install JetPack SDK (includes TensorRT)
sudo apt update
sudo apt install nvidia-jetpack

# Install Python packages optimized for Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics transformers

# Verify TensorRT installation
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

## 🎯 Quick Start

### Basic Usage

```bash
# Real-time detection with webcam using YOLO
python3 jetson_object_detection_toolkit.py --model yolo --input camera

# Zero-shot detection with custom prompts
python3 jetson_object_detection_toolkit.py --model owlvit --input camera --prompts "person" "laptop" "coffee cup"

# Process single image with GroundingDINO
python3 jetson_object_detection_toolkit.py --model grounding-dino --input image.jpg --text-prompt "person holding a laptop"

# Multi-modal scene analysis
python3 jetson_object_detection_toolkit.py --model multi-modal --input camera --prompts "person" "vehicle" "furniture"
```

### Advanced Usage

```bash
# Enable TensorRT acceleration
python3 jetson_object_detection_toolkit.py --model yolo --tensorrt --precision fp16

# Two-step detection with CLIP classification
python3 jetson_object_detection_toolkit.py --model yolo-clip --input camera --prompts "office equipment" "furniture" "electronics"

# Rich descriptions with BLIP
python3 jetson_object_detection_toolkit.py --model yolo-blip --input image.jpg

# Save results and generate report
python3 jetson_object_detection_toolkit.py --model yolo --input camera --save-results --generate-report --output-dir ./results
```

## 📊 Performance Benchmarking

### Run Comprehensive Benchmark

```bash
# Benchmark all models with 100 test images
python3 jetson_object_detection_toolkit.py --benchmark --benchmark-images 100

# Benchmark specific model
python3 jetson_object_detection_toolkit.py --model yolo --benchmark --tensorrt

# Advanced analysis
python3 jetson_object_detection_toolkit.py --advanced-analysis optimization-impact
```

### Sample Benchmark Results

```
================================================================================
PERFORMANCE COMPARISON
================================================================================
Model                FPS        Time(ms)     Memory(MB)   Detections  
--------------------------------------------------------------------------------
YOLO                 45.2       22.1         156.3        4.2         
YOLO+TensorRT        67.8       14.7         142.1        4.2         
YOLO+CLIP            19.1       52.3         234.7        3.8         
OWL-ViT              3.5        287.4        312.1        3.1         
GroundingDINO        2.4        412.8        398.9        4.7         
================================================================================

RECOMMENDATIONS:
  Real Time: YOLO+TensorRT
  High Accuracy: GroundingDINO
  Resource Constrained: YOLO
  Balanced: YOLO+CLIP
```

## 🔧 Configuration Options

### Model Parameters

```bash
# YOLO model selection
--yolo-model yolov8n.pt    # Nano (fastest)
--yolo-model yolov8s.pt    # Small (balanced)
--yolo-model yolov8m.pt    # Medium (accurate)
--yolo-model yolov8l.pt    # Large (most accurate)

# Precision settings
--precision fp32           # Full precision
--precision fp16           # Half precision (recommended for Jetson)

# Device selection
--device cuda              # GPU acceleration
--device cpu               # CPU inference
```

### Detection Parameters

```bash
# Confidence thresholds
--conf-threshold 0.5       # YOLO confidence threshold
--score-threshold 0.3      # Zero-shot model threshold

# Text prompts for zero-shot models
--prompts "person" "car" "laptop"                    # OWL-ViT prompts
--text-prompt "person holding a laptop in office"   # GroundingDINO prompt
```

### Output Options

```bash
# Save options
--save-results             # Save detection results as JSON
--save-video output.mp4    # Save annotated video
--output-dir ./results     # Output directory

# Visualization
--no-display              # Disable live display
--generate-report         # Create comprehensive report
```

## 🧪 Advanced Features

### Multi-Modal Scene Analysis

Combines multiple models for comprehensive scene understanding:

```python
# Example: Comprehensive scene analysis
analyzer = MultiModalSceneAnalyzer(device="cuda", precision="fp16")
result = analyzer.analyze_scene(image, context_prompts=["person", "laptop", "office"])

print(f"Total detections: {result['analysis_summary']['total_detections']}")
print(f"Unique objects: {result['analysis_summary']['unique_objects']}")
```

### Custom Model Integration

Extend the toolkit with your own models:

```python
class CustomDetector(BaseDetector):
    def __init__(self, model_path, device="cuda"):
        super().__init__(device)
        self.load_model(model_path)
    
    def detect(self, image, **kwargs):
        # Implement your detection logic
        return DetectionResult(boxes=[], scores=[], labels=[])
```

### Performance Optimization Tips

1. **TensorRT Acceleration**: Always use `--tensorrt` flag for Jetson devices
2. **Precision**: Use `--precision fp16` for 2x speedup with minimal accuracy loss
3. **Model Selection**: Choose appropriate YOLO model size based on requirements
4. **Batch Processing**: For multiple images, process in batches when possible
5. **Memory Management**: Monitor memory usage with built-in performance monitoring

## 📈 Use Case Examples

### Real-time Security Monitoring

```bash
# Detect people and vehicles in real-time
python3 jetson_object_detection_toolkit.py \
    --model yolo \
    --tensorrt \
    --input camera \
    --conf-threshold 0.6 \
    --save-video security_feed.mp4
```

### Flexible Object Search

```bash
# Search for specific objects using natural language
python3 jetson_object_detection_toolkit.py \
    --model grounding-dino \
    --input camera \
    --text-prompt "red fire extinguisher on the wall"
```

### Inventory Management

```bash
# Classify detected objects into categories
python3 jetson_object_detection_toolkit.py \
    --model yolo-clip \
    --input camera \
    --prompts "electronics" "furniture" "office supplies" "safety equipment"
```

### Scene Description

```bash
# Generate rich descriptions of detected objects
python3 jetson_object_detection_toolkit.py \
    --model yolo-blip \
    --input image.jpg \
    --save-results
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller model or reduce precision
   --yolo-model yolov8n.pt --precision fp16
   ```

2. **TensorRT Not Found**
   ```bash
   # Install TensorRT or disable acceleration
   pip install tensorrt  # or use --no-tensorrt
   ```

3. **Low FPS Performance**
   ```bash
   # Enable all optimizations
   --tensorrt --precision fp16 --yolo-model yolov8n.pt
   ```

4. **Model Download Issues**
   ```bash
   # Pre-download models
   python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

### Performance Tuning

- **Jetson Orin Nano**: Use YOLOv8n with TensorRT for 30+ FPS
- **Jetson Orin NX**: Use YOLOv8s with TensorRT for optimal balance
- **Jetson AGX Orin**: Use YOLOv8m or larger models for maximum accuracy

## 📚 API Reference

### Core Classes

- `OptimizedYOLODetector`: Fast traditional object detection
- `OptimizedOWLViTDetector`: Zero-shot detection with text prompts
- `GroundingDINODetector`: Advanced zero-shot detection
- `YOLOCLIPDetector`: Two-step detection and classification
- `YOLOBLIPDetector`: Detection with rich descriptions
- `MultiModalSceneAnalyzer`: Comprehensive scene analysis
- `PerformanceBenchmark`: Model performance evaluation

### Detection Result Format

```python
@dataclass
class DetectionResult:
    boxes: List[List[float]]      # [x1, y1, x2, y2] coordinates
    scores: List[float]           # Confidence scores
    labels: List[str]             # Object labels
    descriptions: List[str]       # Rich descriptions (optional)
    inference_time: float         # Inference time in seconds
    memory_usage: float           # Memory usage in MB
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black jetson_object_detection_toolkit.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Hugging Face](https://huggingface.co/) for Transformers library
- [NVIDIA](https://developer.nvidia.com/tensorrt) for TensorRT optimization
- [OpenAI](https://openai.com/) for CLIP model
- [Salesforce](https://github.com/salesforce/BLIP) for BLIP model
- [IDEA Research](https://github.com/IDEA-Research/GroundingDINO) for GroundingDINO

## 📞 Support

For questions, issues, or support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Happy Detecting! 🎯**