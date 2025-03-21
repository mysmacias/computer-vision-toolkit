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
import numpy as np

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
from gui.transforms_panel import TransformsPanel
from gui.model_manager import (
    FasterRCNNModel, YOLOv8Model, YOLOv8SegmentationModel, 
    YOLOv8NanoModel, YOLOv8MediumModel, YOLOv8LargeModel,
    YOLOv8NanoSegmentationModel, YOLOv8MediumSegmentationModel,
    YOLOv8PoseModel, DINOv2Model, ModelInterface
)

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
        
        # Performance optimization variables
        self.frame_count = 0
        self.transform_every_n_frames = 2  # Only apply transforms every N frames
        self.last_transform_params = None
        self.cached_transformed_image = None
        self.high_performance_mode = True  # Default to high performance mode
        
        # FPS tracking
        self.fps_display_timer = QTimer()
        self.fps_display_timer.setInterval(1000)  # Update every second
        self.fps_display_timer.timeout.connect(self.update_fps_display)
        self.current_fps = 0
        
        # Set up the UI
        self.setup_ui()
        
        # Set initial display refresh rate based on performance mode
        self.visualization.set_max_display_fps(60 if self.high_performance_mode else 30)
        
        # Connect signals
        self.connect_signals()
        
        # Start with status update
        self.update_status("Ready. Select a model and start the camera.")
        
        # Handle application exit
        qApp.aboutToQuit.connect(self.cleanup_resources)
        
    def setup_ui(self):
        """Set up the user interface"""
        # Central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create video display area
        self.visualization = VisualizationWidget()
        main_layout.addWidget(self.visualization)
        
        # Create transforms panel as a dock widget on the left
        transforms_dock = QDockWidget("Image Transforms")
        transforms_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable | 
                                  QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.transforms_panel = TransformsPanel()
        transforms_dock.setWidget(self.transforms_panel)
        
        # Add the transforms dock to the LEFT side
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, transforms_dock)
        
        # Set up control panel on the right
        control_dock = QDockWidget("Controls")
        control_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable | 
                               QDockWidget.DockWidgetFeature.DockWidgetMovable)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        # Model dropdown
        self.model_combo = QComboBox()
        model_options = [
            "FasterRCNN",
            "YOLOv8",
            "YOLOv8-Nano",
            "YOLOv8-Medium", 
            "YOLOv8-Large",
            "YOLOv8-Seg",
            "YOLOv8-Nano-Seg",
            "YOLOv8-Medium-Seg",
            "YOLOv8-Pose",
            "DINOv2"
        ]
        self.model_combo.addItems(model_options)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        
        # Model status label
        self.model_status_label = QLabel("Model: None")
        model_layout.addWidget(self.model_status_label)
        
        # Load model button
        self.load_button = QPushButton("Load Model")
        model_layout.addWidget(self.load_button)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        # Camera control group
        camera_group = QGroupBox("Camera Control")
        camera_layout = QVBoxLayout()
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        
        # Resolution selection
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        camera_layout.addWidget(QLabel("Resolution:"))
        camera_layout.addWidget(self.resolution_combo)
        
        # Start/stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        camera_layout.addLayout(button_layout)
        
        # Toggle display mode button
        self.toggle_display_button = QPushButton("Switch to Single Display")
        camera_layout.addWidget(self.toggle_display_button)
        
        # Add performance mode toggle
        self.performance_mode_check = QCheckBox("High Performance Mode")
        self.performance_mode_check.setChecked(self.high_performance_mode)
        self.performance_mode_check.setToolTip("Optimize for performance at the cost of some visual quality")
        camera_layout.addWidget(self.performance_mode_check)
        
        camera_group.setLayout(camera_layout)
        control_layout.addWidget(camera_group)
        
        # Detection parameters group
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        
        # Confidence threshold slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        param_layout.addLayout(conf_layout)
        
        # Visualization options
        self.show_boxes_check = QCheckBox("Show Boxes")
        self.show_boxes_check.setChecked(True)
        self.show_labels_check = QCheckBox("Show Labels")
        self.show_labels_check.setChecked(True)
        self.show_conf_check = QCheckBox("Show Confidence")
        self.show_conf_check.setChecked(True)
        
        param_layout.addWidget(self.show_boxes_check)
        param_layout.addWidget(self.show_labels_check)
        param_layout.addWidget(self.show_conf_check)
        
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)
        
        # Add YOLO-specific options group
        yolo_group = QGroupBox("YOLO Options")
        yolo_layout = QVBoxLayout()
        
        # IOU threshold
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IOU Threshold:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        yolo_layout.addLayout(iou_layout)
        
        # Max detections
        max_det_layout = QHBoxLayout()
        max_det_layout.addWidget(QLabel("Max Detections:"))
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 100)
        self.max_det_spin.setValue(20)
        max_det_layout.addWidget(self.max_det_spin)
        yolo_layout.addLayout(max_det_layout)
        
        yolo_group.setLayout(yolo_layout)
        control_layout.addWidget(yolo_group)
        
        # Add DINOv2-specific options group
        dino_group = QGroupBox("DINOv2 Options")
        dino_layout = QVBoxLayout()
        
        # Task selection
        dino_layout.addWidget(QLabel("Task:"))
        self.dino_task_combo = QComboBox()
        self.dino_task_combo.addItems(["Feature Visualization", "Segmentation", "Depth Estimation"])
        dino_layout.addWidget(self.dino_task_combo)
        
        # Feature visualization method
        dino_layout.addWidget(QLabel("Feature Reduction:"))
        self.dino_feature_method_combo = QComboBox()
        self.dino_feature_method_combo.addItems(["PCA", "t-SNE"])
        dino_layout.addWidget(self.dino_feature_method_combo)
        
        # Segmentation classes
        segment_layout = QHBoxLayout()
        segment_layout.addWidget(QLabel("Segment Classes:"))
        self.dino_segment_spin = QSpinBox()
        self.dino_segment_spin.setRange(2, 10)
        self.dino_segment_spin.setValue(5)
        segment_layout.addWidget(self.dino_segment_spin)
        dino_layout.addLayout(segment_layout)
        
        dino_group.setLayout(dino_layout)
        control_layout.addWidget(dino_group)
        
        # Add stretcher to push controls to the top
        control_layout.addStretch()
        
        # Set control widget
        control_dock.setWidget(control_widget)
        
        # Set central widget and add dock
        self.setCentralWidget(central_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)
        
    def connect_signals(self):
        """Set up signal/slot connections"""
        # Connect confidence slider to label
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        # Connect visualization options
        self.show_boxes_check.stateChanged.connect(self.update_visualization_options)
        self.show_labels_check.stateChanged.connect(self.update_visualization_options)
        self.show_conf_check.stateChanged.connect(self.update_visualization_options)
        
        # Connect camera buttons
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        
        # Connect toggle display button
        self.toggle_display_button.clicked.connect(self.toggle_display_mode)
        
        # Connect performance mode checkbox
        self.performance_mode_check.stateChanged.connect(self.toggle_performance_mode)
        
        # Connect model loading
        self.load_button.clicked.connect(self.load_model)
        
        # Connect resolution combo
        self.resolution_combo.currentIndexChanged.connect(self.update_resolution)
        
        # Connect YOLO-specific controls
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        self.max_det_spin.valueChanged.connect(self.update_max_detections)
        
        # Connect DINOv2-specific controls
        self.dino_task_combo.currentIndexChanged.connect(self.update_dino_task)
        self.dino_feature_method_combo.currentIndexChanged.connect(self.update_dino_feature_method)
        self.dino_segment_spin.valueChanged.connect(self.update_dino_segment_classes)
        
        # Show/hide model-specific controls based on model selection
        self.model_combo.currentTextChanged.connect(self.update_visible_controls)
        
        # Connect transforms panel signals
        self.transforms_panel.transformsChanged.connect(self.update_visualization)
        
        # Initialize visible controls
        self.update_visible_controls(self.model_combo.currentText())

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
            camera_index = int(self.camera_combo.currentText().split(' ')[1])
            
            self.camera_thread = CameraThread(self.model, width, height, camera_index)
            
            # Set optimization options
            self.camera_thread.set_adaptive_skipping(self.high_performance_mode)
            self.camera_thread.set_max_skip_frames(3 if self.high_performance_mode else 1)
            
            # Connect signals
            self.camera_thread.new_frame.connect(self.update_frame)
            self.camera_thread.error.connect(self.handle_camera_error)
            self.camera_thread.fps_update.connect(self.update_camera_fps)
            
            # Reset frame counter
            self.frame_count = 0
            
            # Start thread
            self.camera_thread.start()
            
            # Start FPS display timer
            self.fps_display_timer.start()
            
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
            
            # Stop FPS display timer
            self.fps_display_timer.stop()
            
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
            
        # Increment frame counter for FPS calculation
        self.frame_count += 1
            
        # Check if it's time for garbage collection (every 10 seconds)
        current_time = time.time()
        if current_time - self.last_gc_time > 10:
            # Run garbage collection periodically to avoid memory buildup
            gc.collect()
            self.last_gc_time = current_time
        
        try:
            # Store the current frame for later use
            self.current_frame = image
            
            # In high performance mode, skip transform processing for some frames
            should_transform = True
            if self.high_performance_mode:
                should_transform = (self.frame_count % self.transform_every_n_frames == 0)
            
            # Apply transforms to the image (if any and if not skipping)
            if should_transform:
                transformed_image = self.apply_transforms(image)
                # Cache the transformed image for reuse
                self.cached_transformed_image = transformed_image
            else:
                # Reuse cached transformed image if available
                transformed_image = getattr(self, 'cached_transformed_image', image)
            
            # Determine if we should update the original display
            # In dual mode with high performance, we can update the displays at different rates
            should_update_original = True
            should_update_transformed = True
            
            if self.high_performance_mode and self.visualization.dual_mode:
                # For dual mode with high performance, we can skip some updates
                if self.frame_count % 3 != 0:  # Skip some original frame updates
                    should_update_original = False
                if self.frame_count % 2 != 0:  # Skip some transformed frame updates
                    should_update_transformed = False
            
            # Update displays based on current mode
            if self.visualization.dual_mode:
                # In dual mode, update both displays selectively
                if should_update_original:
                    self.visualization.update_frame(image)
                if should_update_transformed:
                    self.visualization.update_transformed_frame(transformed_image)
            else:
                # In single mode, only update the visible display
                if self.visualization.video_label.isVisible() and should_update_original:
                    self.visualization.update_frame(image)
                elif self.visualization.transformed_label.isVisible() and should_update_transformed:
                    self.visualization.update_transformed_frame(transformed_image)
            
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
            elif model_name == "YOLOv8-Nano":
                self.model = YOLOv8NanoModel()
            elif model_name == "YOLOv8-Medium":
                self.model = YOLOv8MediumModel()
            elif model_name == "YOLOv8-Large":
                self.model = YOLOv8LargeModel()
            elif model_name == "YOLOv8-Seg":
                self.model = YOLOv8SegmentationModel()
            elif model_name == "YOLOv8-Nano-Seg":
                self.model = YOLOv8NanoSegmentationModel()
            elif model_name == "YOLOv8-Medium-Seg":
                self.model = YOLOv8MediumSegmentationModel()
            elif model_name == "YOLOv8-Pose":
                self.model = YOLOv8PoseModel()
            elif model_name == "DINOv2":
                self.model = DINOv2Model()
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
        # Clear both video labels
        self.visualization.clear()
        
        # Show message if provided
        if message:
            self.update_status(message)
            
    def update_iou_label(self, value):
        """Update the IOU threshold label and model parameter"""
        iou_value = value / 100.0
        self.iou_label.setText(f"{iou_value:.2f}")
        
        # If we have a YOLO model, update its IOU threshold
        if self.model is not None and hasattr(self.model, "iou_threshold"):
            self.model.iou_threshold = iou_value
            
    def update_max_detections(self, value):
        """Update the max detections parameter on the model"""
        # If we have a YOLO model, update its max detections
        if self.model is not None and hasattr(self.model, "max_detections"):
            self.model.max_detections = value
            
    def update_dino_task(self, index):
        """Update DINOv2 task"""
        if self.model is not None and isinstance(self.model, DINOv2Model):
            tasks = ['features', 'segmentation', 'depth']
            task = tasks[index]
            self.model.set_active_task(task)
            
    def update_dino_feature_method(self, index):
        """Update DINOv2 feature visualization method"""
        if self.model is not None and isinstance(self.model, DINOv2Model):
            methods = ['pca', 'tsne']
            method = methods[index]
            self.model.set_feature_dim_reduction(method)
            
    def update_dino_segment_classes(self, value):
        """Update DINOv2 segmentation classes"""
        if self.model is not None and isinstance(self.model, DINOv2Model):
            self.model.set_segment_classes(value)

    def update_visible_controls(self, model_name):
        """Show/hide controls based on model type"""
        # Determine model type
        is_yolo_model = "YOLO" in model_name
        is_dino_model = "DINOv2" in model_name
        
        # Find the control dock widget
        control_dock = None
        for dock in self.findChildren(QDockWidget):
            if dock.windowTitle() == "Controls":
                control_dock = dock
                break
                
        if control_dock:
            control_widget = control_dock.widget()
            
            # Update visibility of control groups
            for i in range(control_widget.layout().count()):
                item = control_widget.layout().itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QGroupBox):
                    group_box = item.widget()
                    
                    # Set visibility based on model type
                    if group_box.title() == "YOLO Options":
                        group_box.setVisible(is_yolo_model)
                    elif group_box.title() == "DINOv2 Options":
                        group_box.setVisible(is_dino_model)

    def apply_transforms(self, image):
        """Apply image transforms (if enabled) before processing with model"""
        # Check if transform panel exists
        if not hasattr(self, 'transforms_panel') or self.transforms_panel is None:
            return image
            
        # Check if any transform is actually enabled - skip everything if not
        if not self.transforms_panel.has_active_transforms():
            return image
        
        # Get current transform parameters
        current_params = self.transforms_panel.get_transform_params()
        
        # Use cached image if transform parameters haven't changed
        if (self.cached_transformed_image is not None and 
            self.last_transform_params == current_params):
            return self.cached_transformed_image
        
        # Store new parameters
        self.last_transform_params = current_params
            
        # Convert QImage to numpy for transformations - optimize this for speed
        try:
            # Get image dimensions
            width = image.width()
            height = image.height()
            
            # In high performance mode, downscale before processing
            scale_factor = 0.75 if self.high_performance_mode else 1.0
            
            # Memory efficient conversion to numpy array
            if scale_factor < 1.0:
                width_scaled = int(width * scale_factor)
                height_scaled = int(height * scale_factor)
                
                # Fast scaling with QImage
                scaled_image = image.scaled(
                    width_scaled, height_scaled,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                
                # Convert to numpy directly
                if scaled_image.format() == QImage.Format_RGB888:
                    ptr = scaled_image.constBits()
                    ptr.setsize(scaled_image.byteCount())
                    np_img = np.array(ptr).reshape(height_scaled, width_scaled, 3)
                else:
                    # Convert to RGB format
                    rgb_image = scaled_image.convertToFormat(QImage.Format_RGB888)
                    ptr = rgb_image.constBits()
                    ptr.setsize(rgb_image.byteCount())
                    np_img = np.array(ptr).reshape(height_scaled, width_scaled, 3)
            else:
                # Convert directly without scaling
                if image.format() == QImage.Format_RGB888:
                    ptr = image.constBits()
                    ptr.setsize(image.byteCount())
                    np_img = np.array(ptr).reshape(height, width, 3)
                else:
                    rgb_image = image.convertToFormat(QImage.Format_RGB888)
                    ptr = rgb_image.constBits()
                    ptr.setsize(rgb_image.byteCount())
                    np_img = np.array(ptr).reshape(height, width, 3)
            
            # Apply transformations
            transformed_np = self.transforms_panel.apply_transforms(np_img)
            
            # Make sure the result is contiguous for efficient conversion to QImage
            if not transformed_np.flags['C_CONTIGUOUS']:
                transformed_np = np.ascontiguousarray(transformed_np)
            
            # Convert back to QImage efficiently
            height, width, channels = transformed_np.shape
            bytes_per_line = channels * width
            transformed_image = QImage(
                transformed_np.data,
                width, height,
                bytes_per_line,
                QImage.Format_RGB888
            )
            
            # If scaled down, scale back up to original size
            if scale_factor < 1.0:
                transformed_image = transformed_image.scaled(
                    image.width(), image.height(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
            return transformed_image
            
        except Exception as e:
            print(f"Error applying transforms: {e}")
            import traceback
            traceback.print_exc()
            return image

    def update_visualization(self):
        """Update the visualization when transforms change"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.update_frame(self.current_frame)

    def toggle_display_mode(self):
        """Toggle between single and dual display modes"""
        # Call the visualization widget's toggle method
        self.visualization.toggle_display_mode()
        
        # Update button text according to current mode
        if self.visualization.dual_mode:
            self.toggle_display_button.setText("Switch to Single Display")
            self.update_status("Switched to dual display mode")
        else:
            self.toggle_display_button.setText("Switch to Dual Display")
            self.update_status("Switched to single display mode")
            
        # Force update of the display if there's a current frame
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.update_frame(self.current_frame)

    def toggle_performance_mode(self):
        """Toggle the performance mode"""
        self.high_performance_mode = self.performance_mode_check.isChecked()
        
        # Update camera thread settings if running
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.set_adaptive_skipping(self.high_performance_mode)
            self.camera_thread.set_max_skip_frames(3 if self.high_performance_mode else 0)
            
        # Update display refresh rate
        if hasattr(self, 'visualization'):
            self.visualization.set_max_display_fps(60 if self.high_performance_mode else 30)
            
        # Update status
        self.update_status(f"Performance mode: {'High' if self.high_performance_mode else 'Quality'}")
        
        # Clear cached transformed image to force recalculation
        self.cached_transformed_image = None

    def update_fps_display(self):
        """Update the FPS display"""
        self.current_fps = self.frame_count
        self.frame_count = 0
        self.update_status(f"FPS: {self.current_fps}")

    def update_camera_fps(self, fps):
        """Update the camera FPS display"""
        self.update_status(f"Camera FPS: {fps}")

def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern style
    
    window = ComputerVisionApp()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 