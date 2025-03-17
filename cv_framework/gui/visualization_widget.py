#!/usr/bin/env python
"""
Visualization Widget for Computer Vision Toolkit
Handles display of camera feed and model output
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen

class VisualizationWidget(QWidget):
    """Widget for visualizing camera feed and model output"""
    
    def __init__(self, parent=None):
        """Initialize the visualization widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: #222222;")
        
        # Add to layout
        self.layout.addWidget(self.video_label)
        
        # Visualization options
        self.options = {
            "show_boxes": True,
            "show_labels": True,
            "show_conf": True
        }
        
    def set_options(self, options):
        """Set visualization options
        
        Args:
            options (dict): Dictionary of visualization options
        """
        self.options.update(options)
        
    def update_frame(self, image):
        """Update the display with a new frame
        
        Args:
            image (QImage): New frame to display
        """
        if image is None:
            return
            
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Update the label with the new image
        self.video_label.setPixmap(scaled_pixmap)
        
    def clear(self):
        """Clear the display"""
        self.video_label.clear()
        
    def sizeHint(self):
        """Suggested size for the widget"""
        return QSize(800, 600) 