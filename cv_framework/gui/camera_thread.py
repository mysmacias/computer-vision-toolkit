#!/usr/bin/env python
"""
Camera Thread Module for Computer Vision Toolkit
Handles webcam capture in a separate thread to keep the UI responsive
"""

import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import os  # Added for os.name check

class InferenceThread(threading.Thread):
    """Thread for running model inference separate from camera capture"""
    
    def __init__(self, model, max_queue_size=2):
        """Initialize inference thread
        
        Args:
            model: Computer vision model to use for processing
            max_queue_size: Maximum size of frame queue
        """
        super().__init__()
        self.model = model
        self.running = False
        self.daemon = True  # Thread will exit when main thread exits
        
        # Frame queues
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Performance tracking
        self.last_inference_time = 0
        self.inference_times = []  # Keep track of recent times for adaptive frame skipping
        
    def run(self):
        """Thread main function"""
        self.running = True
        
        while self.running:
            try:
                # Get frame from queue, blocking with timeout
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, meta = frame_data
                
                # Measure inference time
                start_time = time.time()
                
                # Process with model
                if self.model is not None:
                    try:
                        # Process frame directly with model
                        result_frame = self.model.process_frame_direct(frame)
                        
                        # Calculate inference time
                        inference_time = time.time() - start_time
                        self.last_inference_time = inference_time
                        
                        # Keep track of recent inference times (up to 5)
                        self.inference_times.append(inference_time)
                        if len(self.inference_times) > 5:
                            self.inference_times.pop(0)
                        
                        # Put result in output queue (non-blocking)
                        try:
                            self.output_queue.put_nowait((result_frame, meta))
                        except queue.Full:
                            # If queue is full, remove oldest item and add new one
                            try:
                                self.output_queue.get_nowait()
                                self.output_queue.put_nowait((result_frame, meta))
                            except:
                                pass
                    except Exception as e:
                        print(f"Error in inference: {e}")
                        # Put original frame in output queue on error
                        try:
                            self.output_queue.put_nowait((frame, meta))
                        except queue.Full:
                            pass
                else:
                    # If no model, pass through
                    self.output_queue.put_nowait((frame, meta))
                    
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                # No frames to process, just continue
                continue
            except Exception as e:
                print(f"Inference thread error: {e}")
                import traceback
                traceback.print_exc()
                
    def stop(self):
        """Stop the inference thread"""
        self.running = False
        # Clear queues
        self.clear_queues()
        
    def clear_queues(self):
        """Clear input and output queues"""
        # Clear input queue
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
                self.input_queue.task_done()
            except:
                pass
                
        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                pass
    
    def get_average_inference_time(self):
        """Get average inference time from recent frames"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)


class CameraThread(QThread):
    """Thread for camera capture and processing"""
    
    # Signal for new processed frame
    new_frame = Signal(QImage)
    
    # Signal for error notifications
    error = Signal(str)
    
    # Signal for performance metrics
    fps_update = Signal(float)
    
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
        self.target_fps = 30  # Target FPS
        
        # Model and processing
        self.model = model
        self.running = False
        self.mutex = QMutex()
        self.capture = None
        self.frame_count = 0
        self.processed_count = 0
        
        # Adaptive frame skipping
        self.min_skip_frames = 0  # Process every frame by default
        self.max_skip_frames = 5  # Maximum frames to skip
        self.current_skip = 0     # Current skip count
        self.adaptive_skipping = True  # Enable adaptive frame skipping
        
        # Frame buffers and inference thread
        self.inference_thread = None
        self.last_displayed_frame = None
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = 0
        self.current_fps = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.last_fps_update = 0
        self.last_frame_timestamp = 0
        
        # For ensuring smooth FPS at the capture level
        self.frame_interval = 1.0 / self.target_fps
        
        # OpenCV camera buffer parameters
        self.cv_api_preference = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
    def run(self):
        """Thread main function"""
        self.mutex.lock()
        self.running = True
        self.mutex.unlock()
        
        # Create and start inference thread if model is available
        if self.model is not None:
            self.inference_thread = InferenceThread(self.model)
            self.inference_thread.start()
        
        # Try to open the camera
        try:
            # Use DirectShow on Windows for better performance
            self.capture = cv2.VideoCapture(self.camera_index, self.cv_api_preference)
            
            # Check if camera opened successfully
            if not self.capture.isOpened():
                self.error.emit(f"Failed to open camera #{self.camera_index}")
                return
            
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Set smaller buffer size for lower latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Windows-specific camera optimizations
            if os.name == 'nt':
                # Additional DirectShow settings for Windows
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # Try to set lower resolution format (YUV instead of RGB) for faster capture
                self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                
                # Set hardware acceleration if available
                if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    # Try to enable hardware acceleration
                    for accel_mode in range(5):  # Try different acceleration modes
                        success = self.capture.set(cv2.CAP_PROP_HW_ACCELERATION, accel_mode)
                        if success:
                            print(f"Enabled hardware acceleration mode {accel_mode}")
                            break
            
            # Main capture loop
            while self.running:
                # Measure frame capture time
                loop_start = time.time()
                
                # Read frame
                ret, frame = self.capture.read()
                
                # Check for successful capture
                if not ret:
                    self.error.emit("Failed to capture frame")
                    # Try to reconnect
                    time.sleep(0.1)
                    continue
                
                # Increment frame counter
                self.frame_count += 1
                
                # Calculate FPS for camera capture
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # Update FPS calculation (for camera capture rate)
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    if self.current_fps == 0:
                        self.current_fps = instant_fps
                    else:
                        self.current_fps = self.fps_alpha * self.current_fps + (1.0 - self.fps_alpha) * instant_fps
                
                # Emit FPS update every second
                if current_time - self.last_fps_update > 1.0:
                    self.fps_update.emit(self.current_fps)
                    self.last_fps_update = current_time
                
                # Determine if we should process this frame
                should_process = True
                
                # Apply adaptive frame skipping if enabled
                if self.adaptive_skipping and self.inference_thread:
                    # Get average inference time
                    avg_inference_time = self.inference_thread.get_average_inference_time()
                    
                    if avg_inference_time > 0:
                        # Calculate how many frames to skip based on inference time
                        # Target: keep processing time less than 50% of frame interval
                        target_time = 0.5 * self.frame_interval
                        ratio = avg_inference_time / target_time
                        
                        # Update skip count (with limits)
                        if ratio > 1.5:  # Inference is too slow
                            self.current_skip = min(self.current_skip + 1, self.max_skip_frames)
                        elif ratio < 0.75:  # Inference is fast enough to process more frames
                            self.current_skip = max(self.current_skip - 1, self.min_skip_frames)
                    
                    # Apply frame skipping
                    should_process = (self.frame_count % (self.current_skip + 1) == 0)
                
                # Process frame if needed
                if should_process:
                    # Convert to RGB here (OpenCV uses BGR)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Use inference thread if available
                    if self.inference_thread and self.model:
                        # Save frame timestamp
                        frame_timestamp = current_time
                        
                        # Try to add frame to input queue (non-blocking)
                        try:
                            self.inference_thread.input_queue.put_nowait((rgb_frame, {"timestamp": frame_timestamp}))
                            self.processed_count += 1
                        except queue.Full:
                            # If input queue is full, skip this frame
                            pass
                            
                        # Try to get a processed frame from output queue (non-blocking)
                        try:
                            result_frame, meta = self.inference_thread.output_queue.get_nowait()
                            
                            # Store as last displayed frame
                            self.last_displayed_frame = result_frame
                            
                            # Convert to QImage efficiently
                            h, w, ch = result_frame.shape
                            bytes_per_line = ch * w
                            qt_image = QImage(result_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            
                            # Emit the frame
                            self.new_frame.emit(qt_image)
                        except queue.Empty:
                            # If no processed frame is available but we have a previous one, use it
                            if self.last_displayed_frame is not None:
                                # If more than 100ms old, create a new QImage to display
                                h, w, ch = self.last_displayed_frame.shape
                                bytes_per_line = ch * w
                                qt_image = QImage(self.last_displayed_frame.data.tobytes(), 
                                                 w, h, bytes_per_line, QImage.Format_RGB888)
                                self.new_frame.emit(qt_image)
                    else:
                        # Direct processing without inference thread
                        # Create QImage
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        
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
                
                # Calculate and enforce frame rate
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.frame_interval - loop_time)
                
                if sleep_time > 0:
                    # Sleep time available, we're running fast enough
                    time.sleep(sleep_time)
        
        except Exception as e:
            self.error.emit(f"Camera thread error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up resources
            if self.inference_thread:
                self.inference_thread.stop()
                
            if self.capture and self.capture.isOpened():
                self.capture.release()
                print("Camera released")
    
    def stop(self):
        """Stop the camera thread"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
        # Stop inference thread if running
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
        
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
        # Stop previous inference thread if exists
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
            
        self.model = model
        
        # Create new inference thread if model is provided
        if self.model is not None:
            self.inference_thread = InferenceThread(self.model)
            
            # Start thread if we're running
            if self.running:
                self.inference_thread.start()
    
    def set_skip_frames(self, skip):
        """Set minimum number of frames to skip (0 = process all frames)"""
        self.min_skip_frames = max(0, skip)
        self.current_skip = self.min_skip_frames
    
    def set_adaptive_skipping(self, enabled):
        """Enable or disable adaptive frame skipping"""
        self.adaptive_skipping = enabled
        
    def set_max_skip_frames(self, max_skip):
        """Set maximum number of frames to skip for adaptive skipping"""
        self.max_skip_frames = max(0, max_skip)
        
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