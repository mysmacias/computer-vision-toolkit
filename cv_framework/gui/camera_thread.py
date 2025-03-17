#!/usr/bin/env python
"""
Camera Thread Module for Computer Vision Toolkit
Handles webcam capture in a separate thread to keep the UI responsive
"""

import cv2
import time
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage

class CameraThread(QThread):
    """Thread for camera capture and processing"""
    
    # Signal for new processed frame
    new_frame = Signal(QImage)
    
    # Signal for error notifications
    error = Signal(str)
    
    def __init__(self, model=None, width=640, height=480, camera_index=0):
        """Initialize camera thread
        
        Args:
            model: Computer vision model to use for processing
            width: Camera width resolution
            height: Camera height resolution
            camera_index: Camera device index (default 0)
        """
        super().__init__()
        
        # Camera settings
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = 15  # Target FPS
        
        # Model and processing
        self.model = model
        self.running = False
        self.mutex = QMutex()
        self.capture = None
        self.frame_count = 0
        self.skip_frames = 1  # Process every other frame by default
        
        # Performance tracking
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.fps  # Minimum time between frames
    
    def run(self):
        """Thread main function"""
        self.mutex.lock()
        self.running = True
        self.mutex.unlock()
        
        # Try to open the camera
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            # Check if camera opened successfully
            if not self.capture.isOpened():
                self.error.emit(f"Failed to open camera #{self.camera_index}")
                return
            
            # Set camera resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Main capture loop
            while self.running:
                # Control timing to maintain consistent frame rate
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # Only capture if enough time has passed
                if elapsed >= self.frame_interval:
                    # Read frame
                    ret, frame = self.capture.read()
                    
                    # Check for successful capture
                    if not ret:
                        self.error.emit("Failed to capture frame")
                        # Try to reconnect
                        time.sleep(1.0)
                        continue
                    
                    # Update frame timestamp
                    self.last_frame_time = current_time
                    
                    # Increment frame counter
                    self.frame_count += 1
                    
                    # Skip frames if needed
                    if self.skip_frames > 0 and (self.frame_count % (self.skip_frames + 1)) != 0:
                        continue
                    
                    # Convert frame for display
                    try:
                        # Convert to RGB (OpenCV uses BGR)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Create QImage
                        h, w, ch = rgb_frame.shape
                        bytesPerLine = ch * w
                        qt_image = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        
                        # Process with model if available
                        if self.model is not None:
                            try:
                                result_image = self.model.process_frame(qt_image)
                                self.new_frame.emit(result_image)
                            except Exception as e:
                                print(f"Error processing frame: {e}")
                                self.new_frame.emit(qt_image)
                        else:
                            # If no model, just emit the frame
                            self.new_frame.emit(qt_image)
                            
                    except Exception as e:
                        print(f"Error converting frame: {e}")
                        continue
                else:
                    # Sleep a bit to reduce CPU usage
                    sleep_time = min(self.frame_interval - elapsed, 0.01)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except Exception as e:
            self.error.emit(f"Camera thread error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up resources
            if self.capture and self.capture.isOpened():
                self.capture.release()
                print("Camera released")
    
    def stop(self):
        """Stop the camera thread"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
        # Wait for thread to finish
        if not self.wait(2000):  # 2 second timeout
            print("Camera thread stop timeout, forcing quit")
    
    def set_resolution(self, width, height):
        """Set camera resolution"""
        self.width = width
        self.height = height
        
        # If capture is active, update settings
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_model(self, model):
        """Set the model to use for processing"""
        self.model = model
    
    def set_skip_frames(self, skip):
        """Set number of frames to skip (0 = process all frames)"""
        self.skip_frames = max(0, skip)
    
    def ensure_dimensions_compatible(self, frame, patch_size=None):
        """Ensure frame dimensions are compatible with model requirements
        
        Args:
            frame (numpy.ndarray): Input frame
            patch_size (int, optional): Patch size for models like DINOv2.
                                      If None, no special handling is done.
        
        Returns:
            numpy.ndarray: Resized frame if needed
        """
        if frame is None or patch_size is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # For models that require dimensions to be multiples of patch_size
        if patch_size > 1:
            new_h = ((h // patch_size) * patch_size)
            new_w = ((w // patch_size) * patch_size)
            
            # Only resize if dimensions need to change
            if h != new_h or w != new_w:
                frame = cv2.resize(frame, (new_w, new_h))
                print(f"Resized frame from {w}x{h} to {new_w}x{new_h} to be compatible with patch size {patch_size}")
                
        return frame 