# Computer Vision Toolkit GUI

A modern graphical user interface for the Computer Vision Toolkit, built with PySide6 (Qt).

## Features

- Real-time webcam feed with computer vision model processing
- Support for multiple models:
  - FasterRCNN
  - YOLOv8
  - YOLOv8 Segmentation
- Adjustable confidence threshold
- Camera resolution control
- Visualization options
- Model information display

## Requirements

- Python 3.8+
- PySide6
- PyTorch and torchvision
- OpenCV
- Ultralytics (for YOLOv8)
- NumPy

## Installation

1. Ensure all requirements are installed:

```bash
pip install PySide6 torch torchvision opencv-python ultralytics numpy
```

2. Navigate to the project directory:

```bash
cd cv_framework/gui
```

## Usage

Run the application:

```bash
python main.py
```

### Interface Overview

The GUI consists of:

1. **Central Display** - Shows the camera feed with model predictions
2. **Model Selection** (Left Dock) - Choose and load a model
3. **Camera Control** (Left Dock) - Start/stop the camera and select resolution
4. **Model Parameters** (Right Dock) - Adjust confidence threshold and visualization options

### Quick Start

1. Launch the application
2. Select a model from the dropdown in the Model Selection panel
3. Click "Load Model" to load the selected model
4. Click "Start" in the Camera Control panel to start the webcam feed
5. Adjust the confidence threshold slider to filter detections
6. Use the visualization checkboxes to customize the display

## Troubleshooting

- **Camera not starting**: Try selecting a different camera index from the dropdown
- **Model loading fails**: Ensure the model file exists in the expected location
- **Low performance**: Try reducing the camera resolution or using a GPU-enabled setup

## Development

To extend the application with new models:

1. Create a new class that inherits from `ModelInterface` in `model_manager.py`
2. Implement the `load()` and `process_frame()` methods
3. Add your model to the `ModelManager.load_model()` method

## License

This project is licensed under the MIT License - see the LICENSE file for details. 