# Computer Vision Framework

A unified framework for computer vision models with a clean, modular design and consistent interface.

## Features

- **Unified Interface**: Run different models with the same interface and command line arguments
- **Multiple Model Support**: Includes YOLO, Faster R-CNN, SSD, DETR, and DINOv2 models (expandable)
- **Various Computer Vision Tasks**: Object detection, semantic segmentation, instance segmentation, pose estimation, depth estimation, and feature visualization
- **Visualization Utilities**: Common utilities for drawing bounding boxes, segmentation masks, etc.
- **Video Recording**: Automatically save processed video with timestamps
- **Runtime Configuration**: Adjust parameters like confidence threshold during execution
- **Modular Design**: Easily add new models to the framework
- **Benchmarking**: Compare performance of different models
- **SAM**: Segment Anything Model for interactive segmentation
- **Automatic Model Downloads**: Models are downloaded automatically when needed - no manual downloading required

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cv-framework

# Install dependencies
pip install -r requirements.txt
```

### Activating the Virtual Environment

#### Linux/macOS
```bash
source cv/bin/activate
```

#### Windows (PowerShell)
```powershell
.\cv\Scripts\Activate.ps1
```

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision 0.11+
- OpenCV 4.5+
- NumPy 1.20+
- scikit-learn 1.0+ (for DINOv2)
- Other model-specific dependencies (e.g., ultralytics for YOLO)
- Internet connection (for first-time model downloads)

## Automatic Model Management

This framework includes an automatic model download system. When you select a model that isn't already downloaded:

1. The system will check if the model exists locally
2. If not found, it will automatically download the model from official sources
3. Progress will be displayed during download
4. After downloading, the model is cached for future use

Supported models with auto-download:
- All YOLOv8 variants (nano, small, medium, large) for detection, segmentation and pose estimation
- DINOv2 models via PyTorch Hub

This ensures that you can use all features immediately without worrying about manually downloading model weights.

## Usage

### Running a Model

```bash
# Run the default model (YOLOv8s)
python -m cv_framework.run

# Run a specific model
python -m cv_framework.run --model yolov8s-seg

# Use a specific camera
python -m cv_framework.run --camera 1

# Set confidence threshold
python -m cv_framework.run --threshold 0.7

# Specify device
python -m cv_framework.run --device cuda:0
```

### List Available Models

```bash
python -m cv_framework.run --list
```

### Benchmark Models

The framework includes a benchmark utility to compare the performance of different models:

```bash
# Benchmark all available models
python -m cv_framework.benchmark

# Benchmark specific models
python -m cv_framework.benchmark --models yolov8s detr_resnet50 dinov2_vits14 sam_vit_b

# Specify the number of iterations
python -m cv_framework.benchmark --iterations 50

# Specify input size
python -m cv_framework.benchmark --input-size 640 480

# List models available for benchmarking
python -m cv_framework.benchmark --list
```

#### Benchmark Examples

Here are some practical examples for benchmarking models:

```bash
# Compare lightweight vs. heavier models on CPU
python benchmark.py --models yolov8n yolov8x --device cpu --iterations 20

# Compare models with larger input resolution
python benchmark.py --models yolov8s detr_resnet50 --input-size 1280 720 --iterations 30

# Benchmark all YOLO models to compare differences in size/performance
python benchmark.py --models yolov8n yolov8s yolov8m yolov8l yolov8x

# Compare segmentation models (YOLO-seg and SAM)
python benchmark.py --models yolov8n-seg yolov8s-seg sam_vit_b

# Benchmark on specific GPU if you have multiple
python benchmark.py --models dinov2_vits14 dinov2_vitl14 --device cuda:0

# Save benchmark results to custom directory
python benchmark.py --models yolov8s detr_resnet50 --output-dir my_benchmarks

# Quick benchmark with fewer iterations (useful for testing)
python benchmark.py --models yolov8s --iterations 10

# Maximum quality benchmark with many iterations for publication-quality results
python benchmark.py --models yolov8s detr_resnet50 dinov2_vits14 --iterations 200
```

The benchmark utility generates:
- A text report with performance metrics
- A chart comparing FPS and inference time across models

### Command Line Arguments

#### Run Script

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-m` | Model to use (e.g., yolov8s, faster_rcnn, ssd300, detr_resnet50, dinov2_vits14, sam_vit_b) |
| `--camera` | `-c` | Camera index to use |
| `--device` | `-d` | Device to run on (e.g., cpu, cuda:0) |
| `--threshold` | `-t` | Confidence threshold (0.0 to 1.0) |
| `--list` | `-l` | List available models and exit |

#### Benchmark Script

| Argument | Short | Description |
|----------|-------|-------------|
| `--models` | `-m` | Models to benchmark (space-separated) |
| `--device` | `-d` | Device to run on (e.g., cpu, cuda:0) |
| `--iterations` | `-i` | Number of iterations for benchmarking (default: 100) |
| `--input-size` | `-s` | Input size for benchmarking (width height, default: 640 640) |
| `--output-dir` | `-o` | Output directory for reports (default: benchmark_results) |
| `--list` | `-l` | List available models for benchmarking and exit |

## Model-specific Information

### YOLO Models (YOLOv5/YOLOv8)

Supports:
- Object detection
- Instance segmentation (with 'seg' suffix)
- Pose estimation (with 'pose' suffix)

Available models include:
- `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- `yolov8n-seg`, `yolov8s-seg`, etc.
- `yolov8n-pose`, `yolov8s-pose`, etc.
- `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`

### Faster R-CNN Models

Provides object detection with various backbones:
- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_mobilenet_v3_large_fpn`
- `fasterrcnn_mobilenet_v3_large_320_fpn`

### SSD Models

Simple and fast single-shot object detector:
- `ssd300`

### DETR Models

Detection Transformer models for object detection:
- `detr_resnet50`
- `detr_resnet101`

### DINOv2 Models

Self-supervised vision transformers with multiple visualization modes:
- `dinov2_vits14` (Small)
- `dinov2_vitb14` (Base)
- `dinov2_vitl14` (Large)
- `dinov2_vitg14` (Giant)

DINOv2 capabilities:
- Depth estimation
- Semantic segmentation
- Feature visualization

**Interactive controls** for DINOv2:
- Press 'v' to cycle through visualization modes
- Press 'c' to change depth colormaps

### SAM Models

- `sam_vit_b`, `sam_vit_l`, `sam_vit_h`: Original Meta AI Segment Anything Models with different ViT backbones
- `yolov8n-seg`, `yolov8s-seg`, etc.: Ultralytics YOLOv8 segmentation models compatible with SAM interface
- SAM provides:
  - Automatic segmentation of all objects in the frame
  - Interactive segmentation using mouse clicks
  - Controls:
    - 'i': Toggle between automatic and interactive modes
    - 'c': Clear all points in interactive mode
    - 'a': Return to automatic segmentation mode
    - Left-click: Add positive point (select object)
    - Right-click: Add negative point (remove background)

## Extending the Framework

### Adding a New Model

1. Create a new model class that inherits from `VisionModel` in `cv_framework/models/`
2. Implement the required methods:
   - `load_model()`
   - `preprocess_frame()`
   - `predict()`
   - `visualize_predictions()`
3. Update the `create_model()` function in `run.py` to include your model
4. (Optional) Add a method to list available models for your model type

Example:

```python
from ..models.base_model import VisionModel

class MyNewModel(VisionModel):
    def __init__(self, model_name='my_model', device=None):
        super().__init__(model_name, device)
        # Model-specific initialization
        
    def load_model(self):
        # Load your model
        
    def preprocess_frame(self, frame):
        # Preprocess frame for your model
        
    def predict(self, processed_input):
        # Run inference
        
    def visualize_predictions(self, frame, predictions):
        # Visualize results
```

## Output

Processed videos are saved to the `output_videos` directory with filenames that include the model name and timestamp.

## License

[MIT License](LICENSE) 