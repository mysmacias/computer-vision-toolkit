#!/usr/bin/env python
"""
Visualization Widget for Computer Vision Toolkit
Handles display of camera feed and model output
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
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
        
        # Create horizontal layout for side-by-side display
        self.horizontal_layout = QHBoxLayout()
        
        # Create video display labels
        self.video_label = QLabel("Original Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: #222222;")
        
        # Create second label for transformed feed
        self.transformed_video_label = QLabel("Transformed Feed")
        self.transformed_video_label.setAlignment(Qt.AlignCenter)
        self.transformed_video_label.setMinimumSize(320, 240)
        self.transformed_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transformed_video_label.setStyleSheet("background-color: #222222;")
        
        # Add labels to horizontal layout
        self.horizontal_layout.addWidget(self.video_label)
        self.horizontal_layout.addWidget(self.transformed_video_label)
        
        # Add horizontal layout to main layout
        self.layout.addLayout(self.horizontal_layout)
        
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
            image: QImage to display
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
        
    def update_transformed_frame(self, image):
        """Update the transformed display with a new frame
        
        Args:
            image: QImage to display
        """
        if image is None:
            return
            
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.transformed_video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Update the label with the new image
        self.transformed_video_label.setPixmap(scaled_pixmap)
        
    def clear(self):
        """Clear the display"""
        self.video_label.clear()
        self.transformed_video_label.clear()
        
    def sizeHint(self):
        """Return the preferred size of the widget"""
        return QSize(1280, 720) 