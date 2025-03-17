"""
Computer Vision Toolkit GUI Package
-----------------------------------
This package contains the GUI components for the Computer Vision Toolkit.
"""

# Version information
__version__ = '0.1.0'

# Note: No imports are included here to avoid circular dependencies
# Import the specific modules directly in your code when needed

# Make important classes available at the package level
from .camera_thread import CameraThread
from .visualization_widget import VisualizationWidget
from .model_manager import (
    ModelInterface,
    FasterRCNNModel,
    YOLOv8Model,
    YOLOv8SegmentationModel
)
from .main import ComputerVisionApp