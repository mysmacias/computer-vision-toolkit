#!/usr/bin/env python
"""
Transforms Panel Module
----------------------
Provides a UI panel for controlling image transformations to test model robustness.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QCheckBox, QPushButton, QGroupBox,
                               QScrollArea, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, Signal
import sys
from pathlib import Path

# Add parent directory to system path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from utils.transforms import TransformManager


class TransformSlider(QWidget):
    """Custom widget for transform slider with label and checkbox"""
    
    valueChanged = Signal(str, float, bool)  # name, strength, enabled
    
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header with name and checkbox
        header_layout = QHBoxLayout()
        self.label = QLabel(self.name)
        self.enabled_checkbox = QCheckBox("Enable")
        
        header_layout.addWidget(self.label)
        header_layout.addStretch()
        header_layout.addWidget(self.enabled_checkbox)
        
        # Create slider with value label
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.value_label = QLabel("0%")
        self.value_label.setMinimumWidth(40)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_label)
        
        # Add both layouts to main layout
        layout.addLayout(header_layout)
        layout.addLayout(slider_layout)
        
        # Add separating line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.enabled_checkbox.stateChanged.connect(self._on_enabled_changed)
        
    def _on_slider_changed(self, value):
        """Handle slider value changes"""
        strength = value / 100.0
        self.value_label.setText(f"{int(value)}%")
        self.valueChanged.emit(self.name, strength, self.enabled_checkbox.isChecked())
        
    def _on_enabled_changed(self, state):
        """Handle enabled checkbox changes"""
        enabled = state == Qt.Checked
        self.slider.setEnabled(enabled)
        self.valueChanged.emit(self.name, self.slider.value() / 100.0, enabled)
        
    def set_value(self, value):
        """Set the slider value (0-1)"""
        self.slider.setValue(int(value * 100))
        
    def set_enabled(self, enabled):
        """Set whether the transform is enabled"""
        self.enabled_checkbox.setChecked(enabled)
        self.slider.setEnabled(enabled)


class TransformsPanel(QWidget):
    """Panel for controlling image transformations"""
    
    transformsChanged = Signal()  # Signal when any transform changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.transform_manager = TransformManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        main_layout = QVBoxLayout(self)
        
        # Create a scroll area for the transforms
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Create a widget to hold all the transform controls
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        # Create header
        header_label = QLabel("Image Transforms")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.scroll_layout.addWidget(header_label)
        
        # Create description
        description = QLabel(
            "Use these controls to test model robustness against various image transformations. "
            "Enable the transforms you want to apply and adjust their strength."
        )
        description.setWordWrap(True)
        self.scroll_layout.addWidget(description)
        
        # Add transform sliders
        self.transform_sliders = {}
        for name in self.transform_manager.get_transform_names():
            slider = TransformSlider(name)
            slider.valueChanged.connect(self._on_transform_changed)
            self.transform_sliders[name] = slider
            self.scroll_layout.addWidget(slider)
        
        # Add buttons at the bottom
        buttons_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_all_transforms)
        
        self.enable_all_button = QPushButton("Enable All")
        self.enable_all_button.clicked.connect(self.enable_all_transforms)
        
        self.disable_all_button = QPushButton("Disable All")
        self.disable_all_button.clicked.connect(self.disable_all_transforms)
        
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.enable_all_button)
        buttons_layout.addWidget(self.disable_all_button)
        
        self.scroll_layout.addLayout(buttons_layout)
        self.scroll_layout.addStretch()
        
        # Add the scroll area to the main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Set up sizing
        self.setMinimumWidth(250)
        
    def _on_transform_changed(self, name, strength, enabled):
        """Handle transform slider changes"""
        self.transform_manager.set_transform_strength(name, strength)
        self.transform_manager.set_transform_enabled(name, enabled)
        self.transformsChanged.emit()
        
    def reset_all_transforms(self):
        """Reset all transforms to default values"""
        self.transform_manager.reset_all()
        
        # Update UI to match
        for name, slider in self.transform_sliders.items():
            slider.set_value(0.0)
            slider.set_enabled(False)
            
        self.transformsChanged.emit()
        
    def enable_all_transforms(self):
        """Enable all transforms"""
        for name, slider in self.transform_sliders.items():
            slider.set_enabled(True)
            self.transform_manager.set_transform_enabled(name, True)
            
        self.transformsChanged.emit()
        
    def disable_all_transforms(self):
        """Disable all transforms"""
        for name, slider in self.transform_sliders.items():
            slider.set_enabled(False)
            self.transform_manager.set_transform_enabled(name, False)
            
        self.transformsChanged.emit()
        
    def apply_transforms(self, image):
        """Apply all active transforms to the image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Transformed image
        """
        # Use the appropriate implementation based on what's available
        if hasattr(self, 'transform_manager'):
            # If using transform_manager
            return self.transform_manager.apply_transforms(image)
        elif hasattr(self, 'transform_sliders'):
            # If using transform_sliders directly
            result = image.copy()
            # Apply each active transform
            for transform_slider in self.transform_sliders:
                if transform_slider.enabled:
                    strength = transform_slider.get_value()
                    result = transform_slider.apply_transform(result, strength)
            return result
        
        # If neither is available, return the image unchanged
        return image
        
    def has_active_transforms(self):
        """Check if any transforms are currently active
        
        Returns:
            bool: True if any transforms are enabled with non-zero strength
        """
        # Get the transform manager and check if it has any active transforms
        if hasattr(self, 'transform_manager'):
            # If using transform_manager
            return self.transform_manager.has_active_transforms()
        elif hasattr(self, 'transform_sliders'):
            # If using transform_sliders directly
            for transform_slider in self.transform_sliders:
                if transform_slider.enabled and transform_slider.get_value() > 0:
                    return True
        return False
        
    def get_transform_params(self):
        """Get current transform parameters for caching
        
        Returns:
            tuple: Tuple of (transform_name, enabled, strength) for each transform
        """
        # Get the transform parameters based on implementation
        if hasattr(self, 'transform_manager'):
            # If using transform_manager
            return self.transform_manager.get_transform_params()
        elif hasattr(self, 'transform_sliders'):
            # If using transform_sliders directly
            return tuple(
                (slider.name, slider.enabled, slider.get_value())
                for slider in self.transform_sliders
            )
        return tuple()
        
    def get_transform_manager(self):
        """Get the transform manager"""
        return self.transform_manager 