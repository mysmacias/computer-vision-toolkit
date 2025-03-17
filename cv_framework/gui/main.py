#!/usr/bin/env python
"""
Computer Vision Toolkit - Modern GUI Application
Main application entry point
"""

import os
import sys
import time
import gc
from pathlib import Path

# Add parent directory to system path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QStatusBar, QDockWidget,
    QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QAction

# Import other modules we'll create
from gui.camera_thread import CameraThread
from gui.visualization_widget import VisualizationWidget
from gui.model_manager import FasterRCNNModel, YOLOv8Model, YOLOv8SegmentationModel, ModelInterface

class ComputerVisionApp(QMainWindow):
    """Main application window for the Computer Vision Toolkit"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("Computer Vision Toolkit")
        self.setMinimumSize(1000, 700)
        
        # Initialize variables
        self.camera_thread = None
        self.model = None
        self.model_type = None
        self.is_running = False
        self.last_gc_time = time.time()  # For periodic garbage collection
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Start with status update
        self.update_status("Ready. Select a model and start the camera.")
        
        # Handle application exit
        qApp.aboutToQuit.connect(self.cleanup_resources)
        
    def setup_ui(self):
        """Set up the user interface"""
        # Central widget - visualization area
        self.visualization = VisualizationWidget(self)
        self.setCentralWidget(self.visualization)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create dock widgets
        self.create_model_dock()
        self.create_camera_dock()
        self.create_parameters_dock()
        
    def create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menu_bar.addMenu("&Camera")
        
        # Start camera action
        start_camera_action = QAction("&Start Camera", self)
        start_camera_action.setStatusTip("Start camera feed")
        start_camera_action.triggered.connect(self.start_camera)
        camera_menu.addAction(start_camera_action)
        
        # Stop camera action
        stop_camera_action = QAction("S&top Camera", self)
        stop_camera_action.setStatusTip("Stop camera feed")
        stop_camera_action.triggered.connect(self.stop_camera)
        camera_menu.addAction(stop_camera_action)
        
        # Models menu
        models_menu = menu_bar.addMenu("&Models")
        
        # Load model action
        load_model_action = QAction("&Load Model", self)
        load_model_action.setStatusTip("Load a computer vision model")
        load_model_action.triggered.connect(self.load_model)
        models_menu.addAction(load_model_action)
        
    def create_model_dock(self):
        """Create the model selection dock widget"""
        dock = QDockWidget("Model Selection", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create widget for dock contents
        model_widget = QWidget()
        layout = QVBoxLayout(model_widget)
        
        # Model selection combo box
        model_group = QGroupBox("Available Models")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("FasterRCNN")
        self.model_combo.addItem("YOLOv8")
        self.model_combo.addItem("YOLOv8-Segmentation")
        
        model_layout.addWidget(self.model_combo)
        
        # Load button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_button)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Model status
        status_group = QGroupBox("Model Status")
        status_layout = QVBoxLayout()
        
        self.model_status_label = QLabel("No model loaded")
        status_layout.addWidget(self.model_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
        
        # Set the dock widget's content
        dock.setWidget(model_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        
    def create_camera_dock(self):
        """Create the camera control dock widget"""
        dock = QDockWidget("Camera Control", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create widget for dock contents
        camera_widget = QWidget()
        layout = QVBoxLayout(camera_widget)
        
        # Camera selection
        camera_group = QGroupBox("Camera Selection")
        camera_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera (0)")
        self.camera_combo.addItem("Camera 1")
        self.camera_combo.addItem("Camera 2")
        
        camera_layout.addWidget(self.camera_combo)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_camera)
        button_layout.addWidget(self.stop_button)
        
        camera_layout.addLayout(button_layout)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Resolution settings
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QVBoxLayout()
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640x480")
        self.resolution_combo.addItem("1280x720")
        self.resolution_combo.addItem("1920x1080")
        
        resolution_layout.addWidget(self.resolution_combo)
        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
        
        # Set the dock widget's content
        dock.setWidget(camera_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        
    def create_parameters_dock(self):
        """Create the parameters dock widget"""
        dock = QDockWidget("Model Parameters", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create widget for dock contents
        param_widget = QWidget()
        layout = QVBoxLayout(param_widget)
        
        # Confidence threshold
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QVBoxLayout()
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Default 0.5
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_label)
        
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        self.show_boxes_check = QCheckBox("Show Bounding Boxes")
        self.show_boxes_check.setChecked(True)
        viz_layout.addWidget(self.show_boxes_check)
        
        self.show_labels_check = QCheckBox("Show Labels")
        self.show_labels_check.setChecked(True)
        viz_layout.addWidget(self.show_labels_check)
        
        self.show_conf_check = QCheckBox("Show Confidence")
        self.show_conf_check.setChecked(True)
        viz_layout.addWidget(self.show_conf_check)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
        
        # Set the dock widget's content
        dock.setWidget(param_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def connect_signals(self):
        """Set up signal/slot connections"""
        # Connect confidence slider to label
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        # Connect visualization checkboxes
        self.show_boxes_check.toggled.connect(self.update_visualization_options)
        self.show_labels_check.toggled.connect(self.update_visualization_options)
        self.show_conf_check.toggled.connect(self.update_visualization_options)
        
        # Connect resolution combo box
        self.resolution_combo.currentIndexChanged.connect(self.update_resolution)
        
    def start_camera(self):
        """Start the camera and processing thread"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.update_status("Camera is already running")
            return
            
        # Update UI
        self.start_button.setText("Starting...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        QApplication.processEvents()
        
        try:
            # Create and start the camera thread
            resolution_text = self.resolution_combo.currentText()
            width, height = map(int, resolution_text.split('x'))
            
            self.camera_thread = CameraThread(self.model, width, height)
            
            # Connect signals
            self.camera_thread.new_frame.connect(self.update_frame)
            self.camera_thread.error.connect(self.handle_camera_error)
            
            # Start thread
            self.camera_thread.start()
            
            # Update state and UI
            self.is_running = True
            self.start_button.setText("Start Camera")
            self.start_button.setEnabled(False)
            self.update_status("Camera started successfully")
            
            # Force garbage collection to clean any unused memory
            gc.collect()
            
        except Exception as e:
            self.update_status(f"Failed to start camera: {str(e)}")
            self.start_button.setText("Start Camera")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            import traceback
            traceback.print_exc()

    def stop_camera(self):
        """Stop the camera and processing thread"""
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.update_status("Camera is not running")
            return
            
        # Update UI
        self.stop_button.setText("Stopping...")
        self.stop_button.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # Stop the thread
            self.camera_thread.stop()
            
            # Wait for a moment to ensure thread finishes
            for _ in range(10):
                if not self.camera_thread.isRunning():
                    break
                time.sleep(0.1)
                QApplication.processEvents()
            
            # Force thread termination if still running
            if self.camera_thread.isRunning():
                self.camera_thread.terminate()
                self.camera_thread.wait(1000)
            
            # Clean up
            self.camera_thread.disconnect()
            self.camera_thread = None
            
            # Update state and UI
            self.is_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setText("Stop Camera")
            self.stop_button.setEnabled(False)
            
            # Clear display
            self.reset_display("Camera stopped")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.update_status(f"Error stopping camera: {str(e)}")
            self.stop_button.setText("Stop Camera")
            self.stop_button.setEnabled(True)
            import traceback
            traceback.print_exc()

    def update_frame(self, image):
        """Update the display with a new frame"""
        if image is None:
            return
            
        # Check if it's time for garbage collection (every 10 seconds)
        current_time = time.time()
        if current_time - self.last_gc_time > 10:
            # Run garbage collection periodically to avoid memory buildup
            gc.collect()
            self.last_gc_time = current_time
        
        try:
            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            
            # Update the label with the new image
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.update_status(f"Error updating frame: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_model(self):
        """Load the selected model"""
        model_name = self.model_combo.currentText()
        
        # Check if already loaded
        if self.model is not None and self.model_type == model_name:
            self.update_status(f"Model {model_name} is already loaded")
            return
            
        # Update UI to indicate loading state
        self.model_status_label.setText(f"Loading {model_name}...")
        self.statusBar().showMessage(f"Loading {model_name}...")
        self.load_button.setText("Loading...")
        self.load_button.setEnabled(False)
        QApplication.processEvents()  # Force UI update
        
        try:
            # First clean up any existing model
            if self.model is not None:
                self.model = None
                # Force garbage collection
                gc.collect()
            
            # Create the model based on selection
            if model_name == "FasterRCNN":
                self.model = FasterRCNNModel()
            elif model_name == "YOLOv8":
                self.model = YOLOv8Model()
            elif model_name == "YOLOv8-Seg":
                self.model = YOLOv8SegmentationModel()
            else:
                self.update_status(f"Unknown model: {model_name}")
                return
                
            # Try to load the model
            success = self.model.load()
            
            # Update UI based on result
            if success:
                self.model_type = model_name
                self.model_status_label.setText(f"Model: {model_name} (loaded)")
                self.statusBar().showMessage(f"Model {model_name} loaded successfully")
                self.start_button.setEnabled(True)
                # If camera is running, update it with the new model
                if self.camera_thread is not None and self.camera_thread.isRunning():
                    self.camera_thread.set_model(self.model)
            else:
                self.model = None
                self.model_status_label.setText("Model: None")
                self.statusBar().showMessage(f"Failed to load {model_name}")
                # Show error dialog
                QMessageBox.critical(self, "Model Loading Error", 
                                   f"Failed to load {model_name}. Check console for details.")
                self.load_button.setText("Try Again")
        except Exception as e:
            self.model = None
            self.update_status(f"Error loading model: {str(e)}")
            self.model_status_label.setText("Model: Error")
            # Show error dialog
            QMessageBox.critical(self, "Model Loading Error", 
                               f"Error loading {model_name}: {str(e)}")
            self.load_button.setText("Try Again")
            import traceback
            traceback.print_exc()
        finally:
            # Always re-enable button
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Model")
            # Force garbage collection
            gc.collect()

    def handle_camera_error(self, error_msg):
        """Handle errors from the camera thread"""
        self.update_status(f"Camera error: {error_msg}")
        
        # Stop the camera thread in case of errors
        self.stop_camera()
        
        # Display error message
        QMessageBox.warning(self, "Camera Error", error_msg)

    def update_resolution(self, index):
        """Update camera resolution if camera is running"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            try:
                # Get new resolution
                resolution_text = self.resolution_combo.currentText()
                width, height = map(int, resolution_text.split('x'))
                
                # Restart camera with new resolution
                self.stop_camera()
                time.sleep(0.5)  # Give camera time to fully stop
                
                # Create new camera thread
                self.camera_thread = CameraThread(self.model, width, height)
                
                # Connect signals
                self.camera_thread.new_frame.connect(self.update_frame)
                self.camera_thread.error.connect(self.handle_camera_error)
                
                # Start thread
                self.camera_thread.start()
                
                # Update state and UI
                self.is_running = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.update_status(f"Resolution changed to {width}x{height}")
                
            except Exception as e:
                self.update_status(f"Error updating resolution: {str(e)}")
                import traceback
                traceback.print_exc()

    def cleanup_resources(self):
        """Clean up resources when application is closing"""
        try:
            # Stop camera thread if running
            if self.camera_thread is not None and self.camera_thread.isRunning():
                self.camera_thread.stop()
                # Wait for thread to finish
                self.camera_thread.wait(1000)
                # Force termination if still running
                if self.camera_thread.isRunning():
                    self.camera_thread.terminate()
                    self.camera_thread.wait(1000)
            
            # Release model resources
            self.model = None
            
            # Force final garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_conf_label(self, value):
        """Update the confidence threshold label"""
        conf_value = value / 100.0
        self.conf_label.setText(f"{conf_value:.2f}")
        
        # If we have a model, update its threshold
        if self.model is not None:
            self.model.set_threshold(conf_value)
            
    def update_visualization_options(self):
        """Update visualization options based on checkbox states"""
        options = {
            "show_boxes": self.show_boxes_check.isChecked(),
            "show_labels": self.show_labels_check.isChecked(),
            "show_conf": self.show_conf_check.isChecked()
        }
        
        # Update visualization widget with new options
        self.visualization.set_options(options)
        
    def update_status(self, message):
        """Update the status bar with a message"""
        self.statusBar().showMessage(message)
        
    def reset_display(self, message=None):
        """Reset the display to a blank state with optional message"""
        # Clear the video label
        self.video_label.clear()
        
        # Show message if provided
        if message:
            self.update_status(message)
            
    # Add missing video_label property which should have been part of the VisualizationWidget
    # but was referenced directly in update_frame
    @property
    def video_label(self):
        """Get the video label from the visualization widget"""
        return self.visualization.video_label


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern style
    
    window = ComputerVisionApp()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 