#!/usr/bin/env python
"""
Visualization Widget for Computer Vision Toolkit
Handles display of camera feed and model output
"""

import time
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QImage

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
        self.transformed_label = QLabel("Transformed Feed")
        self.transformed_label.setAlignment(Qt.AlignCenter)
        self.transformed_label.setMinimumSize(320, 240)
        self.transformed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transformed_label.setStyleSheet("background-color: #222222;")
        
        # Add labels to horizontal layout
        self.horizontal_layout.addWidget(self.video_label)
        self.horizontal_layout.addWidget(self.transformed_label)
        
        # Add horizontal layout to main layout
        self.layout.addLayout(self.horizontal_layout)
        
        # Visualization options
        self.options = {
            "show_boxes": True,
            "show_labels": True,
            "show_conf": True
        }
        
        # Set dual display mode by default
        self.dual_mode = True
        
        # Frame buffer for double-buffering
        self.current_frame = None
        self.current_transformed_frame = None
        self.buffer_pixmap_original = QPixmap(640, 480)
        self.buffer_pixmap_transformed = QPixmap(640, 480)
        self.buffer_pixmap_original.fill(Qt.black)
        self.buffer_pixmap_transformed.fill(Qt.black)
        
        # Frame rate control
        self.max_display_fps = 30  # Maximum display refresh rate
        self.last_display_time = 0
        self.display_interval = 1.0 / self.max_display_fps
        
        # Setup display timer for smoother rendering
        self.display_timer = QTimer(self)
        self.display_timer.setInterval(int(1000 / self.max_display_fps))
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start()
        
        # Optimization flags
        self.needs_update_original = False
        self.needs_update_transformed = False
        
    def set_options(self, options):
        """Set visualization options
        
        Args:
            options (dict): Dictionary of visualization options
        """
        self.options.update(options)
        
    def update_frame(self, image):
        """Receive a new frame for the original display
        
        This doesn't immediately update the display, but marks it for update
        in the next display refresh cycle.
        
        Args:
            image: QImage to display
        """
        if image is None:
            return
            
        # Store the image for later rendering
        self.current_frame = image
        self.needs_update_original = True
        
    def update_transformed_frame(self, image):
        """Receive a new frame for the transformed display
        
        This doesn't immediately update the display, but marks it for update
        in the next display refresh cycle.
        
        Args:
            image: QImage to display
        """
        if image is None:
            return
            
        # Store the image for later rendering
        self.current_transformed_frame = image
        self.needs_update_transformed = True
        
    def update_display(self):
        """Update the display with the latest frames at a controlled rate
        
        This is called by the timer to refresh the display at a steady rate.
        """
        current_time = time.time()
        
        # Throttle display updates to maintain a stable frame rate
        if current_time - self.last_display_time < self.display_interval:
            return
            
        self.last_display_time = current_time
        
        # Update original frame if needed and visible
        if self.needs_update_original and self.current_frame and (self.dual_mode or self.video_label.isVisible()):
            # Scale the image efficiently
            self.update_display_original()
            self.needs_update_original = False
            
        # Update transformed frame if needed and visible
        if self.needs_update_transformed and self.current_transformed_frame and (self.dual_mode or self.transformed_label.isVisible()):
            # Scale the image efficiently
            self.update_display_transformed()
            self.needs_update_transformed = False
    
    def update_display_original(self):
        """Update the original display pixmap"""
        if self.current_frame is None:
            return
            
        # Get the display size
        display_width = self.video_label.width()
        display_height = self.video_label.height()
        
        if display_width <= 1 or display_height <= 1:
            return  # Skip if display has invalid size
        
        # Convert to pixmap if it's a QImage
        if isinstance(self.current_frame, QImage):
            pixmap = QPixmap.fromImage(self.current_frame)
        else:
            # Assume it's already a pixmap
            pixmap = self.current_frame
        
        # Scale the image to fit the label while maintaining aspect ratio
        # Use FastTransformation for improved performance
        scaled_pixmap = pixmap.scaled(
            display_width, display_height,
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.FastTransformation
        )
        
        # Update the buffer pixmap
        self.buffer_pixmap_original = scaled_pixmap
        
        # Update the label with the scaled image
        self.video_label.setPixmap(self.buffer_pixmap_original)
        
    def update_display_transformed(self):
        """Update the transformed display pixmap"""
        if self.current_transformed_frame is None:
            return
            
        # Get the display size
        display_width = self.transformed_label.width()
        display_height = self.transformed_label.height()
        
        if display_width <= 1 or display_height <= 1:
            return  # Skip if display has invalid size
        
        # Convert to pixmap if it's a QImage
        if isinstance(self.current_transformed_frame, QImage):
            pixmap = QPixmap.fromImage(self.current_transformed_frame)
        else:
            # Assume it's already a pixmap
            pixmap = self.current_transformed_frame
        
        # Scale the image efficiently
        scaled_pixmap = pixmap.scaled(
            display_width, display_height,
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.FastTransformation
        )
        
        # Update the buffer pixmap
        self.buffer_pixmap_transformed = scaled_pixmap
        
        # Update the label with the scaled image
        self.transformed_label.setPixmap(self.buffer_pixmap_transformed)
        
    def toggle_display_mode(self):
        """Toggle between single and dual display modes"""
        self.dual_mode = not self.dual_mode
        
        if self.dual_mode:
            # Show both displays side by side
            self.video_label.setVisible(True)
            self.transformed_label.setVisible(True)
        else:
            # Only show the transformed display
            self.video_label.setVisible(False)
            self.transformed_label.setVisible(True)
        
    def clear(self):
        """Clear the display"""
        self.video_label.clear()
        self.transformed_label.clear()
        self.current_frame = None
        self.current_transformed_frame = None
        self.buffer_pixmap_original.fill(Qt.black)
        self.buffer_pixmap_transformed.fill(Qt.black)
        
    def sizeHint(self):
        """Return the preferred size of the widget"""
        return QSize(1280, 720)
        
    def set_max_display_fps(self, fps):
        """Set the maximum display refresh rate
        
        Args:
            fps (int): Maximum frames per second to display
        """
        self.max_display_fps = max(1, min(fps, 60))  # Limit between 1 and 60 FPS
        self.display_interval = 1.0 / self.max_display_fps
        self.display_timer.setInterval(int(1000 / self.max_display_fps)) 