#!/usr/bin/env python
"""
Model Manager Module for Computer Vision Toolkit
Handles loading and managing different computer vision models
"""

import os
import sys
import torch
import cv2
import numpy as np
import time
from pathlib import Path

# Add parent directory to system path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from PySide6.QtGui import QImage, QPixmap
from utils.model_downloader import get_model_path, download_dinov2_model

class ModelInterface:
    """Base interface for all models"""
    
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.confidence_threshold = 0.5
        
    def load(self):
        """Load the model"""
        raise NotImplementedError("Subclasses must implement load()")
        
    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with the model
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with detections
        """
        # By default, use the transformed frame if provided
        process_frame = transformed_frame if transformed_frame is not None else frame
        
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement process_frame()")

    def process_frame_direct(self, frame_np):
        """Process a frame directly as numpy array with the model
        
        This is a more efficient processing path that avoids QImage conversions
        
        Args:
            frame_np (numpy.ndarray): Input frame as RGB numpy array
            
        Returns:
            numpy.ndarray: Processed frame with detections as RGB numpy array
        """
        # Default implementation converts to QImage and back
        # Subclasses should override this with a direct implementation
        try:
            # Convert numpy array to QImage
            h, w, ch = frame_np.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Process with QImage implementation
            result_qimage = self.process_frame(qt_image)
            
            # Convert back to numpy array
            return self.to_numpy(result_qimage)
        except Exception as e:
            print(f"Error in process_frame_direct: {str(e)}")
            return frame_np
        
    def set_threshold(self, threshold):
        """Set the confidence threshold
        
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = threshold
        
    def to_numpy(self, qimage):
        """Convert QImage to numpy array
        
        Args:
            qimage (QImage): Input QImage
            
        Returns:
            numpy.ndarray: Numpy array in RGB format
        """
        try:
            # More robust method to convert QImage to numpy array
            width = qimage.width()
            height = qimage.height()
            
            # Create numpy array directly using QImage.constBits
            ptr = qimage.constBits()
            
            # Different approach that works with memoryview objects
            # Create a buffer from the memory view
            buf = memoryview(ptr).tobytes()
            
            # Reshape the buffer to match the image dimensions
            if qimage.format() == QImage.Format_RGB32 or qimage.format() == QImage.Format_ARGB32:
                # 4 channels (BGRA or RGBA)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
                # Convert to RGB (drop alpha channel)
                return arr[:, :, 2::-1].copy()  # BGR to RGB
            elif qimage.format() == QImage.Format_RGB888:
                # 3 channels (RGB)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                return arr.copy()
            else:
                # Convert image to RGB888 format first
                rgb_image = qimage.convertToFormat(QImage.Format_RGB888)
                ptr = rgb_image.constBits()
                buf = memoryview(ptr).tobytes()
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                return arr.copy()
                
        except Exception as e:
            print(f"Error converting QImage to numpy array: {str(e)}")
            # Fallback method using QImage to QPixmap conversion
            # and then saving to a buffer
            import io
            buffer = io.BytesIO()
            qimage.save(buffer, format="PNG")
            buffer.seek(0)
            
            # Use OpenCV to read the image from the buffer
            import cv2
            nparr = np.frombuffer(buffer.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
    def to_qimage(self, arr):
        """Convert numpy array to QImage
        
        Args:
            arr (numpy.ndarray): Input numpy array in RGB format
            
        Returns:
            QImage: Converted QImage
        """
        # Ensure array is contiguous and has the right data type
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        
        # Create QImage from numpy array
        height, width, channels = arr.shape
        bytes_per_line = channels * width
        
        if channels == 3:  # RGB format
            return QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:  # RGBA format
            return QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")


class FasterRCNNModel(ModelInterface):
    """FasterRCNN model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45  # IoU threshold for NMS
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For smoothing FPS calculation
        self.current_fps = 0
        self.coco_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For smoothing FPS calculation
        self.current_fps = 0
        
        # Hardware acceleration settings
        self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        self.optimized_inference = True
        self.trace_model = False  # Whether to use torch.jit.trace for optimization
        self.traced_model = None  # Store the traced model
        
    def load(self):
        """Load the FasterRCNN model"""
        try:
            import torchvision
            from torchvision import models
            from torchvision.ops import nms
            
            # Try using jit compiled model if available
            if self.trace_model and torch.cuda.is_available():
                try:
                    jit_model_path = os.path.join(os.path.dirname(__file__), "fasterrcnn_jit.pt")
                    if os.path.exists(jit_model_path):
                        print(f"Loading JIT-compiled FasterRCNN model from {jit_model_path}")
                        self.traced_model = torch.jit.load(jit_model_path)
                        self.traced_model.eval().to(self.device)
                        print("Successfully loaded JIT-compiled model")
                        return True
                except Exception as e:
                    print(f"Failed to load JIT model: {e}")
                    self.traced_model = None
                    
            # Load a more efficient model for inference
            try:
                # For newer torchvision versions
                from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
                self.model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            except (ImportError, AttributeError):
                # For older torchvision versions
                print("Using legacy pretrained parameter")
                self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Optimize model
            if self.optimized_inference:
                # Use half-precision for faster inference
                if self.device == "cuda:0" and hasattr(torch, 'cuda'):
                    # Enable TF32 precision on Ampere or later GPUs
                    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    
                    # Enable torch.jit for faster execution
                    if self.trace_model and self.traced_model is None:
                        try:
                            # Create a sample input for tracing
                            dummy_input = torch.rand(1, 3, 640, 480).to(self.device)
                            # Trace the model
                            self.traced_model = torch.jit.trace(self.model, [dummy_input])
                            # Save the traced model
                            jit_model_path = os.path.join(os.path.dirname(__file__), "fasterrcnn_jit.pt")
                            torch.jit.save(self.traced_model, jit_model_path)
                            print(f"JIT model saved to {jit_model_path}")
                        except Exception as e:
                            print(f"Failed to trace model: {e}")
                            self.traced_model = None
            
            # Optimize for inference
            if self.device == "cuda:0":
                # Optimize for GPU inference if available
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    print("CUDA optimization enabled with AMP")
                else:
                    print("CUDA available but amp module not found")
            
            print("FasterRCNN model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading FasterRCNN model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def preprocess_tensor(self, image_np):
        """Preprocess numpy array to tensor for model input
        
        Args:
            image_np (numpy.ndarray): RGB image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        # Ensure the image is contiguous and float32
        if not image_np.flags['C_CONTIGUOUS']:
            image_np = np.ascontiguousarray(image_np)
            
        # Convert to tensor more efficiently by avoiding unnecessary copy
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        if self.device != "cpu":
            image_tensor = image_tensor.to(self.device, non_blocking=True)
            
        return image_tensor
            
    def process_frame_direct(self, image_np):
        """Process a frame directly with FasterRCNN
        
        Args:
            image_np (numpy.ndarray): RGB image as numpy array
            
        Returns:
            numpy.ndarray: Processed image with detections
        """
        if self.model is None:
            return image_np
            
        try:
            # Create a copy for drawing
            output_image = image_np.copy()
            
            # Resize image for faster inference
            height, width = image_np.shape[:2]
            max_size = 640  # Maximum size for faster inference
            
            # Calculate scaling factor
            scale = 1.0
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                resize_width = int(width * scale)
                resize_height = int(height * scale)
                
                # Resize image for inference (smaller is faster)
                small_image = cv2.resize(image_np, (resize_width, resize_height), 
                                        interpolation=cv2.INTER_LINEAR)
            else:
                small_image = image_np
            
            # Preprocess image to tensor
            image_tensor = self.preprocess_tensor(small_image)
            
            # Run inference with optimization for GPU
            with torch.no_grad():
                start_time = time.time()
                
                if self.use_amp and self.device == "cuda:0":
                    with torch.cuda.amp.autocast():
                        # Use traced model if available for faster inference
                        if self.traced_model is not None:
                            predictions = self.traced_model(image_tensor)
                        else:
                            predictions = self.model(image_tensor)
                else:
                    # Use traced model if available
                    if self.traced_model is not None:
                        predictions = self.traced_model(image_tensor)
                    else:
                        predictions = self.model(image_tensor)
                    
                current_time = time.time() - start_time
                
                # Smooth FPS calculation
                if self.last_inference_time > 0:
                    self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
                else:
                    self.current_fps = 1.0 / current_time
                self.last_inference_time = current_time
            
            # Process predictions - extract data from tensors
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            keep = scores >= self.confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Rescale boxes to original image size if needed
            if scale != 1.0:
                boxes = boxes / scale
            
            # Apply custom NMS per class for better filtering
            final_boxes = []
            final_scores = []
            final_labels = []
            
            # Process each class separately
            unique_labels = np.unique(labels)
            for class_id in unique_labels:
                # Get indices for this class
                class_indices = np.where(labels == class_id)[0]
                
                if len(class_indices) > 1:  # Only apply NMS if we have multiple detections of same class
                    class_boxes = boxes[class_indices]
                    class_scores = scores[class_indices]
                    
                    # Convert to torch tensors for NMS
                    boxes_tensor = torch.from_numpy(class_boxes).float()
                    scores_tensor = torch.from_numpy(class_scores).float()
                    
                    # Apply NMS using torchvision
                    from torchvision.ops import nms
                    keep_indices = nms(boxes_tensor, scores_tensor, self.iou_threshold)
                    keep_indices = keep_indices.cpu().numpy()
                    
                    # Keep the selected indices
                    final_boxes.extend(class_boxes[keep_indices])
                    final_scores.extend(class_scores[keep_indices])
                    final_labels.extend([class_id] * len(keep_indices))
                else:
                    # If only one detection, keep it
                    final_boxes.extend(boxes[class_indices])
                    final_scores.extend(scores[class_indices])
                    final_labels.extend(labels[class_indices])
            
            # Convert lists back to numpy arrays
            final_boxes = np.array(final_boxes) if final_boxes else np.empty((0, 4))
            final_scores = np.array(final_scores) if final_scores else np.empty(0)
            final_labels = np.array(final_labels) if final_labels else np.empty(0)
            
            # Draw boxes on the output image - optimized drawing
            for box, score, label in zip(final_boxes, final_scores, final_labels):
                # Get integer coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Get label name
                label_name = self.coco_names[label] if label < len(self.coco_names) else f"Class {label}"
                
                # Draw box efficiently - bright green for visibility
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw background for text
                text = f"{label_name}: {score:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                
                # Draw label and score in black text
                cv2.putText(output_image, text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw FPS and detection count
            detection_count = len(final_boxes)
            status_text = f"FPS: {self.current_fps:.1f} | Det: {detection_count}"
            
            # Draw background for status text
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return output_image
            
        except Exception as e:
            print(f"Error processing frame directly: {str(e)}")
            import traceback
            traceback.print_exc()
            return image_np

    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with FasterRCNN
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with detections
        """
        if self.model is None:
            return frame
            
        try:
            # Use transformed frame if provided, otherwise use original
            image_to_process = transformed_frame if transformed_frame is not None else frame
            
            # Convert QImage to numpy array - direct path for better performance
            image_np = self.to_numpy(image_to_process)
            
            # Process using the optimized direct implementation
            result_np = self.process_frame_direct(image_np)
            
            # Convert back to QImage
            return self.to_qimage(result_np)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame


class YOLOv8Model(ModelInterface):
    """YOLOv8 Object Detection model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8s.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8"
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.input_size = 640  # Default YOLO input size
        self.iou_threshold = 0.45
        self.max_detections = 20
        
    def load(self):
        """Load the YOLOv8 model"""
        try:
            # Attempt to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                print("Ultralytics not found. Please install with: pip install ultralytics")
                return False
                
            # Get model path (download if missing)
            try:
                model_path = get_model_path(self.model_filename)
                self.model_path = str(model_path)
                print(f"Using model at: {self.model_path}")
            except Exception as e:
                print(f"Error obtaining model path: {str(e)}")
                return False
                
            # Load the model
            self.model = YOLO(self.model_path)
            
            # Set model parameters for faster inference
            if hasattr(self.model, 'fuse') and callable(self.model.fuse):
                # Fuse conv and bn layers for faster inference
                self.model.fuse()
            
            print(f"{self.model_name} model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading {self.model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with YOLOv8
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with detections
        """
        if self.model is None:
            return frame
            
        try:
            # Use transformed frame if provided, otherwise use original
            image_to_process = transformed_frame if transformed_frame is not None else frame
            
            # Convert QImage to numpy array
            image_np = self.to_numpy(image_to_process)
            
            # Create a copy for output
            output_image = image_np.copy()
            
            # Get frame dimensions
            height, width = image_np.shape[:2]
            
            # Run inference with optimized settings
            start_time = time.time()
            results = self.model(image_np, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               max_det=self.max_detections,
                               verbose=False)
            
            # Calculate and smooth FPS
            current_time = time.time() - start_time
            if self.last_inference_time > 0:
                self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
            else:
                self.current_fps = 1.0 / current_time
            self.last_inference_time = current_time
            
            # Get detection results
            if len(results) > 0:
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    labels = results[0].boxes.cls.cpu().numpy().astype(int)
                    class_names = results[0].names
                    
                    # Draw boxes on the output image
                    for box, score, label_idx in zip(boxes, scores, labels):
                        # Convert box coordinates to integers
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Get label name
                        label_name = class_names[label_idx] if label_idx in class_names else f"Class {label_idx}"
                        
                        # Draw box - using a bright color
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Draw background for text
                        text = f"{label_name}: {score:.2f}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 255), -1)
                        
                        # Draw label and score in black text
                        cv2.putText(output_image, text, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                except Exception as e:
                    print(f"Error processing YOLO results: {e}")
                
                # Draw FPS and detection count
                detection_count = len(boxes) if 'boxes' in locals() else 0
                status_text = f"FPS: {self.current_fps:.1f} | Detections: {detection_count} | {self.model_name}"
            else:
                # Draw FPS if no detections
                status_text = f"FPS: {self.current_fps:.1f} | No detections | {self.model_name}"
            
            # Draw background for status text
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert back to QImage
            return self.to_qimage(output_image)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame


class YOLOv8SegmentationModel(ModelInterface):
    """YOLOv8 Instance Segmentation model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8s-seg.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Seg"
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.input_size = 640  # Default YOLO input size
        self.iou_threshold = 0.45
        self.max_detections = 10
        
    def load(self):
        """Load the YOLOv8 segmentation model"""
        try:
            # Attempt to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                print("Ultralytics not found. Please install with: pip install ultralytics")
                return False
                
            # Get model path (download if missing)
            try:
                model_path = get_model_path(self.model_filename)
                self.model_path = str(model_path)
                print(f"Using model at: {self.model_path}")
            except Exception as e:
                print(f"Error obtaining model path: {str(e)}")
                return False
                
            # Load the model
            self.model = YOLO(self.model_path)
            
            # Set model parameters for faster inference
            if hasattr(self.model, 'fuse') and callable(self.model.fuse):
                # Fuse conv and bn layers for faster inference
                self.model.fuse()
            
            print(f"{self.model_name} model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading {self.model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with YOLOv8 segmentation
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with detections and segmentation masks
        """
        if self.model is None:
            return frame
            
        try:
            # Use transformed frame if provided, otherwise use original
            image_to_process = transformed_frame if transformed_frame is not None else frame
            
            # Convert QImage to numpy array
            image_np = self.to_numpy(image_to_process)
            
            # Create a copy of the image for drawing
            output_image = image_np.copy()
            
            # Run inference with optimized settings
            start_time = time.time()
            results = self.model(image_np, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               max_det=self.max_detections,
                               verbose=False)
            
            # Calculate and smooth FPS
            current_time = time.time() - start_time
            if self.last_inference_time > 0:
                self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
            else:
                self.current_fps = 1.0 / current_time
            self.last_inference_time = current_time
            
            # Get detection results
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                try:
                    # Extract boxes, masks, scores, and class IDs
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    masks = results[0].masks.data.cpu().numpy()
                    class_names = results[0].names
                    
                    # Blend segmentation masks with semi-transparency - simplified for performance
                    if len(masks) > 0:
                        # Create an empty mask image
                        mask_img = np.zeros_like(image_np)
                        
                        # Color palette for different classes - pre-compute
                        colors = []
                        for i in range(max(10, len(class_ids) + 1)):
                            # Use HSV for better color diversity
                            hue = int(180 * i / max(10, len(class_ids) + 1))
                            # Convert HSV to BGR to RGB
                            color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                            colors.append((int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])))
                        
                        # Limit number of masks for performance
                        max_masks = min(5, len(masks))
                        
                        # Draw each mask with its class color
                        for i in range(max_masks):
                            mask = masks[i]
                            class_id = class_ids[i]
                            
                            # Resize mask to match image size
                            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
                            # Threshold mask to binary with simpler method
                            mask = (mask > 0.5).astype(np.uint8)
                            
                            # Get color for this class
                            color = colors[class_id % len(colors)]
                            
                            # Apply color to mask areas - optimized
                            color_mask = np.zeros_like(image_np)
                            color_mask[mask > 0] = color
                            
                            # Add this mask to the output image with alpha blending
                            alpha = 0.5
                            output_image = cv2.addWeighted(output_image, 1, color_mask, alpha, 0)
                        
                        # Draw bounding boxes and labels - only for masks that were drawn
                        for i in range(max_masks):
                            box = boxes[i]
                            score = scores[i]
                            class_id = class_ids[i]
                            
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Get label name
                            label_name = class_names[class_id] if class_id in class_names else f"Class {class_id}"
                            
                            # Draw box with class color
                            color = colors[class_id % len(colors)]
                            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw background for text
                            text = f"{label_name}: {score:.2f}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                            
                            # Draw label and score
                            cv2.putText(output_image, text, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Draw FPS and detection count
                    mask_count = len(masks)
                    status_text = f"FPS: {self.current_fps:.1f} | Masks: {mask_count} | {self.model_name}"
                except Exception as e:
                    # Error processing mask results
                    print(f"Error processing masks: {e}")
                    status_text = f"FPS: {self.current_fps:.1f} | Mask error | {self.model_name}"
            else:
                # No detections
                status_text = f"FPS: {self.current_fps:.1f} | No masks | {self.model_name}"
            
            # Draw background for status text
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
            
            # Convert back to QImage
            return self.to_qimage(output_image)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame


class YOLOv8NanoModel(YOLOv8Model):
    """YOLOv8 Nano model - smaller and faster"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8n.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Nano"


class YOLOv8MediumModel(YOLOv8Model):
    """YOLOv8 Medium model - balanced size and accuracy"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8m.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Medium"


class YOLOv8LargeModel(YOLOv8Model):
    """YOLOv8 Large model - high accuracy but slower"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8l.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Large"


class YOLOv8PoseModel(ModelInterface):
    """YOLOv8 Pose Estimation model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8s-pose.pt"  # Changed from nano to small for better performance
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Pose"
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.input_size = 640  # Default YOLO input size
        self.iou_threshold = 0.45
        self.max_detections = 20
        
    def load(self):
        """Load the YOLOv8 pose model"""
        try:
            # Attempt to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                print("Ultralytics not found. Please install with: pip install ultralytics")
                return False
                
            # Get model path (download if missing)
            try:
                model_path = get_model_path(self.model_filename)
                self.model_path = str(model_path)
                print(f"Using model at: {self.model_path}")
            except Exception as e:
                print(f"Error obtaining model path: {str(e)}")
                return False
                
            # Load the model
            self.model = YOLO(self.model_path)
            
            # Set model parameters for faster inference
            if hasattr(self.model, 'fuse') and callable(self.model.fuse):
                # Fuse conv and bn layers for faster inference
                self.model.fuse()
            
            print(f"{self.model_name} model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading {self.model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with YOLOv8 pose estimation
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with pose keypoints
        """
        if self.model is None:
            return frame
            
        try:
            # Use transformed frame if provided, otherwise use original
            image_to_process = transformed_frame if transformed_frame is not None else frame
            
            # Convert QImage to numpy array
            image_np = self.to_numpy(image_to_process)
            
            # Create a copy for output
            output_image = image_np.copy()
            
            # Run inference with optimized settings
            start_time = time.time()
            results = self.model(image_np, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               max_det=self.max_detections,
                               verbose=False)
            
            # Calculate and smooth FPS
            current_time = time.time() - start_time
            if self.last_inference_time > 0:
                self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
            else:
                self.current_fps = 1.0 / current_time
            self.last_inference_time = current_time
            
            # Get keypoints results
            if len(results) > 0:
                try:
                    # Extract keypoints data
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    if len(keypoints) > 0:
                        # Color definitions for different body parts
                        # Define connections for drawing skeleton lines
                        skeleton_connections = [
                            (5, 7), (7, 9),   # Right arm
                            (6, 8), (8, 10),  # Left arm
                            (5, 6),           # Shoulders
                            (5, 11), (6, 12), # Torso
                            (11, 13), (13, 15), # Right leg
                            (12, 14), (14, 16), # Left leg
                            (11, 12),         # Hips
                            (0, 1), (1, 3), (3, 5), # Face-Right side
                            (0, 2), (2, 4), (4, 6)  # Face-Left side
                        ]
                        
                        # Draw person count
                        person_count = len(keypoints)
                        
                        # Draw each person's keypoints and connections
                        for person_idx, kpts in enumerate(keypoints):
                            # Select a color based on person index
                            person_color = (0, 255 - (50 * person_idx) % 255, (80 * person_idx) % 255)
                            
                            # Draw keypoints
                            for idx, (x, y, conf) in enumerate(kpts):
                                if conf > self.confidence_threshold:
                                    # Different colors for different keypoint types
                                    if idx <= 4:  # Face keypoints
                                        color = (0, 255, 0)  # Green
                                    elif idx <= 10:  # Upper body
                                        color = (255, 0, 0)  # Blue
                                    else:  # Lower body
                                        color = (0, 0, 255)  # Red
                                    
                                    cv2.circle(output_image, (int(x), int(y)), 5, color, -1)
                            
                            # Draw skeleton connections
                            for connection in skeleton_connections:
                                idx1, idx2 = connection
                                if kpts[idx1, 2] > self.confidence_threshold and kpts[idx2, 2] > self.confidence_threshold:
                                    pt1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
                                    pt2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))
                                    cv2.line(output_image, pt1, pt2, person_color, 2)
                        
                        # Draw status text
                        status_text = f"FPS: {self.current_fps:.1f} | Persons: {person_count} | {self.model_name}"
                    else:
                        # No people detected
                        status_text = f"FPS: {self.current_fps:.1f} | No persons detected | {self.model_name}"
                except Exception as e:
                    print(f"Error processing pose results: {e}")
                    status_text = f"FPS: {self.current_fps:.1f} | Error processing poses | {self.model_name}"
            else:
                # No detections
                status_text = f"FPS: {self.current_fps:.1f} | No detections | {self.model_name}"
            
            # Draw background for status text
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert back to QImage
            return self.to_qimage(output_image)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame
            

# Add YOLO model variants for segmentation
class YOLOv8NanoSegmentationModel(YOLOv8SegmentationModel):
    """YOLOv8 Nano Segmentation model - smaller and faster"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8n-seg.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Nano-Seg"


class YOLOv8MediumSegmentationModel(YOLOv8SegmentationModel):
    """YOLOv8 Medium Segmentation model - balanced size and accuracy"""
    
    def __init__(self):
        super().__init__()
        self.model_filename = "yolov8m-seg.pt"
        self.model_path = str(parent_dir / self.model_filename)
        self.model_name = "YOLOv8-Medium-Seg"


class DINOv2Model(ModelInterface):
    """DINOv2 Vision Transformer implementation with multiple tasks"""
    
    def __init__(self):
        super().__init__()
        self.model_path = None  # DINOv2 is loaded from Torch Hub
        self.model_name = "DINOv2"
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.active_task = "features"  # Default task: 'features', 'segmentation', 'depth'
        self.feature_dim_reduction = "pca"  # 'pca' or 'tsne'
        self.segment_classes = 5  # Number of segments for unsupervised segmentation
        self.dino_variant = 'dinov2_vits14'  # Default to small model
        
    def load(self):
        """Load the DINOv2 model"""
        try:
            # Apply PyTorch version compatibility patch
            self._patch_interpolate()
            
            # Try to import the DINOv2 model from models directory
            sys.path.append(str(parent_dir))
            try:
                from models.dinov2_model import DINOv2Model as BaseDINOv2Model
            except ImportError:
                print("DINOv2 model implementation not found. Please ensure models/dinov2_model.py exists.")
                return False
            
            # Create the feature extractor based on run_dinov2.py implementation
            try:
                print(f"Loading DINOv2 model variant: {self.dino_variant}...")
                self.model = BaseDINOv2Model(model_name=self.dino_variant)
                
                # Try loading the model
                success = self.model.load_model()
                
                if not success:
                    print("Standard loading failed, attempting direct download...")
                    try:
                        # Try direct download if the regular loading failed
                        with torch.no_grad():
                            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.dino_variant)
                            dinov2_model = dinov2_model.to(self.model.device)
                            dinov2_model.eval()
                            
                            # If direct download worked, set it in our model wrapper
                            self.model.model = dinov2_model
                            print(f"Successfully downloaded {self.dino_variant} directly")
                            success = True
                    except Exception as direct_error:
                        print(f"Direct download failed: {direct_error}")
                        success = self.model.model == "simplified"  # Check if simplified model created
                
                if success:
                    print(f"{self.model_name} model loaded successfully")
                    return True
                else:
                    print("Failed to load DINOv2 model through any method")
                    return False
                
            except Exception as e:
                print(f"Error initializing DINOv2 model: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error loading DINOv2 model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _patch_interpolate(self):
        """
        Monkey patch the interpolate function to handle older PyTorch versions
        by removing the antialias parameter if it's not supported
        """
        try:
            orig_interpolate = torch.nn.functional.interpolate
            
            def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', 
                                   align_corners=None, recompute_scale_factor=None, 
                                   antialias=None):
                # Remove antialias parameter for older PyTorch versions
                if 'antialias' not in orig_interpolate.__code__.co_varnames:
                    return orig_interpolate(input, size, scale_factor, mode, 
                                           align_corners, recompute_scale_factor)
                else:
                    return orig_interpolate(input, size, scale_factor, mode, 
                                           align_corners, recompute_scale_factor, antialias)
                    
            # Replace the original interpolate function with our patched version
            torch.nn.functional.interpolate = patched_interpolate
            print("Applied PyTorch interpolate patch for compatibility")
        except Exception as e:
            print(f"Failed to apply interpolate patch: {e}")
            
    def set_active_task(self, task):
        """Set the active visualization task
        
        Args:
            task (str): Task name - 'features', 'segmentation', or 'depth'
        """
        valid_tasks = ['features', 'segmentation', 'depth']
        if task in valid_tasks:
            self.active_task = task
            # Update model's settings if applicable
            if hasattr(self.model, 'dim_reduction') and task == 'features':
                self.model.dim_reduction = self.feature_dim_reduction
            if hasattr(self.model, 'segment_classes') and task == 'segmentation':
                self.model.segment_classes = self.segment_classes
            print(f"Active task set to: {task}")
        else:
            print(f"Invalid task: {task}. Must be one of {valid_tasks}")
            
    def set_feature_dim_reduction(self, method):
        """Set the feature dimensionality reduction method
        
        Args:
            method (str): Method name - 'pca' or 'tsne'
        """
        valid_methods = ['pca', 'tsne']
        if method in valid_methods:
            self.feature_dim_reduction = method
            if hasattr(self.model, 'dim_reduction'):
                self.model.dim_reduction = method
            print(f"Feature dimensionality reduction set to: {method}")
        else:
            print(f"Invalid method: {method}. Must be one of {valid_methods}")
            
    def set_segment_classes(self, num_classes):
        """Set the number of segment classes for unsupervised segmentation
        
        Args:
            num_classes (int): Number of segment classes (2-10)
        """
        if 2 <= num_classes <= 10:
            self.segment_classes = num_classes
            if hasattr(self.model, 'segment_classes'):
                self.model.segment_classes = num_classes
            print(f"Segment classes set to: {num_classes}")
        else:
            print(f"Invalid number of segment classes: {num_classes}. Must be between 2 and 10")
        
    def process_frame(self, frame, transformed_frame=None):
        """Process a frame with DINOv2 based on the active task
        
        Args:
            frame (QImage): Original input frame
            transformed_frame (QImage, optional): Frame with transforms applied
                                               If None, the original frame is used
            
        Returns:
            QImage: Processed frame with visualizations
        """
        if self.model is None:
            return frame
            
        try:
            # Use transformed frame if provided, otherwise use original
            image_to_process = transformed_frame if transformed_frame is not None else frame
            
            # Convert QImage to numpy array
            image_np = self.to_numpy(image_to_process)
            
            # Create a copy for output (also used as fallback if processing fails)
            output_image = image_np.copy()
            
            try:
                # Run inference based on the active task
                start_time = time.time()
                
                # Ensure frame dimensions are compatible with patch size
                if hasattr(self.model, 'patch_size'):
                    patch_size = self.model.patch_size
                    h, w = image_np.shape[:2]
                    new_h = ((h // patch_size) * patch_size)
                    new_w = ((w // patch_size) * patch_size)
                    if h != new_h or w != new_w:
                        image_np = cv2.resize(image_np, (new_w, new_h))
                
                # Preprocess the frame for model input
                preprocessed_data = self.model.preprocess_frame(image_np)
                
                # Extract features
                features = self.model.extract_features(preprocessed_data)
                
                # Process based on selected task
                if self.active_task == 'features':
                    # Visualize features with PCA or t-SNE
                    visualization = self.model.visualize_features(features)
                    if visualization is not None:
                        output_image = visualization
                elif self.active_task == 'segmentation':
                    # Run unsupervised segmentation
                    predictions = self.model.segment_image(features)
                    visualization = self.model.visualize_predictions(image_np, predictions)
                    if visualization is not None:
                        output_image = visualization
                elif self.active_task == 'depth':
                    # Estimate depth
                    predictions = self.model.estimate_depth(features)
                    visualization = self.model.visualize_predictions(image_np, predictions)
                    if visualization is not None:
                        output_image = visualization
                
                # Calculate and smooth FPS
                current_time = time.time() - start_time
                if self.last_inference_time > 0:
                    self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
                else:
                    self.current_fps = 1.0 / current_time
                self.last_inference_time = current_time
                
                # Status message for successful processing
                task_name = self.active_task.capitalize()
                method_info = f" ({self.feature_dim_reduction})" if self.active_task == 'features' else ""
                status_text = f"FPS: {self.current_fps:.1f} | {task_name}{method_info} | {self.model_name}"
            except Exception as e:
                # If any processing fails, use the original image and show error
                print(f"Error in DINOv2 processing: {str(e)}")
                import traceback
                traceback.print_exc()
                status_text = f"Error: {str(e)[:30]}... | Task: {self.active_task} | {self.model_name}"
            
            # Draw background for status text (always do this)
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to QImage
            return self.to_qimage(output_image)
            
        except Exception as e:
            print(f"Critical error processing frame with DINOv2: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame


class ModelManager:
    """Manager class for handling different computer vision models"""
    
    def __init__(self):
        """Initialize the model manager"""
        self.models = {}
        
    def load_model(self, model_name):
        """Load a model by name
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            ModelInterface: Loaded model or None if loading failed
        """
        # Check if model is already loaded
        if model_name in self.models:
            print(f"Model {model_name} already loaded")
            return self.models[model_name]
            
        # Create appropriate model instance
        if model_name == "FasterRCNN":
            model = FasterRCNNModel()
        elif model_name == "YOLOv8":
            model = YOLOv8Model()
        elif model_name == "YOLOv8-Segmentation":
            model = YOLOv8SegmentationModel()
        elif model_name == "YOLOv8-Nano":
            model = YOLOv8NanoModel()
        elif model_name == "YOLOv8-Medium":
            model = YOLOv8MediumModel()
        elif model_name == "YOLOv8-Large":
            model = YOLOv8LargeModel()
        elif model_name == "YOLOv8-Pose":
            model = YOLOv8PoseModel()
        elif model_name == "DINOv2":
            model = DINOv2Model()
        else:
            print(f"Unknown model: {model_name}")
            return None
            
        # Load the model
        if model.load():
            # Store the loaded model
            self.models[model_name] = model
            return model
        else:
            print(f"Failed to load model: {model_name}")
            return None
            
    def get_model(self, model_name):
        """Get a loaded model by name
        
        Args:
            model_name (str): Name of the model to get
            
        Returns:
            ModelInterface: Model instance or None if not loaded
        """
        return self.models.get(model_name, None)
        
    def unload_model(self, model_name):
        """Unload a model by name
        
        Args:
            model_name (str): Name of the model to unload
            
        Returns:
            bool: True if model was unloaded, False otherwise
        """
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False 