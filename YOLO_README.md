# Enhanced YOLO Model Runner

A flexible and interactive script for running any YOLO model from the Ultralytics family on a webcam feed. This tool supports all YOLOv5 and YOLOv8 models, including detection, segmentation, and pose estimation variants.

## Features

- **Multiple Model Support**: Supports all YOLOv5 and YOLOv8 models
- **Runtime Model Switching**: Change models during execution without restarting
- **Adjustable Threshold**: Increase or decrease detection confidence threshold in real-time
- **Interactive UI**: Clean display with real-time performance metrics
- **Video Recording**: Automatically saves processed video to file
- **Automatic Model Download**: Automatically downloads models if they're not found locally

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Alternatively, install directly
pip install torch ultralytics opencv-python numpy
```

Make sure you have the latest version of Ultralytics:

```bash
pip install -U ultralytics
```

## Usage

### Basic Usage

```bash
python run_yolo.py
```

This will run the default model (YOLOv8s) with standard settings. If the model isn't already downloaded, the script will automatically download it for you.

### Command Line Options

```bash
# List all available models
python run_yolo.py --list

# Use a specific model 
python run_yolo.py --model yolov8s-seg

# Set a custom confidence threshold
python run_yolo.py --threshold 0.6

# Specify a different camera
python run_yolo.py --camera 1

# Use specific device
python run_yolo.py --device cuda:0
```

### Interactive Controls

During execution, you can use the following keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `+` | Increase detection threshold by 0.05 |
| `-` | Decrease detection threshold by 0.05 |
| `m` | Open model selection menu to switch models |

### Model Management

The script handles models in the following way:

1. Checks if the model exists in the current directory
2. Checks if the model exists in the Ultralytics cache (`~/.cache/ultralytics/models/`)
3. If not found, automatically downloads the model

When switching models during runtime, the same process applies - if you select a model that isn't already on your system, it will be downloaded automatically.

## Supported Models

### YOLOv5 Models
- `yolov5n` - Nano (smallest and fastest)
- `yolov5s` - Small (balanced speed/accuracy)
- `yolov5m` - Medium 
- `yolov5l` - Large
- `yolov5x` - XLarge (most accurate)

### YOLOv8 Detection Models
- `yolov8n` - Nano
- `yolov8s` - Small
- `yolov8m` - Medium
- `yolov8l` - Large
- `yolov8x` - XLarge

### YOLOv8 Segmentation Models
- `yolov8n-seg` - Nano with segmentation
- `yolov8s-seg` - Small with segmentation
- `yolov8m-seg` - Medium with segmentation
- `yolov8l-seg` - Large with segmentation
- `yolov8x-seg` - XLarge with segmentation

### YOLOv8 Pose Models
- `yolov8n-pose` - Nano with pose estimation
- `yolov8s-pose` - Small with pose estimation
- `yolov8m-pose` - Medium with pose estimation
- `yolov8l-pose` - Large with pose estimation
- `yolov8x-pose` - XLarge with pose estimation

## Output

All processed videos are saved to the `output_videos` directory with filenames that include the model name and timestamp.

## System Requirements

- Python 3.8+
- Webcam
- GPU recommended for optimal performance
- For YOLOv8 models: 4GB+ RAM
- For larger models (L, XL): 8GB+ RAM and CUDA-capable GPU
- Latest ultralytics version recommended (`pip install -U ultralytics`) 