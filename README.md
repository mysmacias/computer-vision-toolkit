# DINOv2 Perception System

A real-time computer vision application leveraging Facebook Research's DINOv2 model to perform multiple perception tasks without task-specific training:

- **Depth Estimation**: Perceives relative depth in a scene
- **Semantic Segmentation**: Groups pixels into meaningful regions
- **Feature Visualization**: Visualizes the model's learned features in RGB space

## Features

- Real-time processing from webcam feed
- Multiple visualization modes
- Video recording with timestamp
- Intuitive keyboard controls
- Automatic initialization and calibration

## Quick Start

```bash
# Install dependencies
pip install torch torchvision opencv-python numpy scikit-learn pillow matplotlib

# Run the application
python run_dinov2.py
```

## Controls

- `v`: Cycle through visualization modes (Depth, Segmentation, Features, Side-by-Side)
- `c`: Change depth colormaps
- `q`: Quit the application

## System Requirements

- Python 3.x
- Webcam
- CUDA-compatible GPU recommended for faster processing

## Documentation

For detailed explanation of how the system works, see [DINOv2_Perception.md](DINOv2_Perception.md). 