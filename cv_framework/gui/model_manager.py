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

class ModelInterface:
    """Base interface for all models"""
    
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.confidence_threshold = 0.5
        
    def load(self):
        """Load the model"""
        raise NotImplementedError("Subclasses must implement load()")
        
    def process_frame(self, frame):
        """Process a frame with the model
        
        Args:
            frame (QImage): Input frame
            
        Returns:
            QImage: Processed frame with detections
        """
        raise NotImplementedError("Subclasses must implement process_frame()")
        
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
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        
    def load(self):
        """Load the FasterRCNN model"""
        try:
            import torchvision
            from torchvision import models
            
            # Determine device (CPU or CUDA)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
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
            
            # Optimize for inference
            if self.device == "cuda:0":
                # Optimize for GPU inference if available
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    print("CUDA optimization enabled")
                else:
                    print("CUDA available but amp module not found")
            
            print("FasterRCNN model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading FasterRCNN model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame):
        """Process a frame with FasterRCNN
        
        Args:
            frame (QImage): Input frame
            
        Returns:
            QImage: Processed frame with detections
        """
        if self.model is None:
            return frame
            
        try:
            # Convert QImage to numpy array
            image_np = self.to_numpy(frame)
            
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
                small_image = cv2.resize(image_np, (resize_width, resize_height))
            else:
                small_image = image_np
            
            # Preprocess image for PyTorch
            image_tensor = torch.from_numpy(small_image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Move tensor to correct device
            image_tensor = image_tensor.to(self.device)
            
            # Run inference with optimization for GPU
            with torch.no_grad():
                start_time = time.time()
                
                if self.device == "cuda:0" and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    with torch.cuda.amp.autocast():
                        predictions = self.model(image_tensor)
                else:
                    predictions = self.model(image_tensor)
                    
                current_time = time.time() - start_time
                
                # Smooth FPS calculation
                if self.last_inference_time > 0:
                    self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * (1.0 / current_time)
                else:
                    self.current_fps = 1.0 / current_time
                self.last_inference_time = current_time
            
            # Process predictions
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
            
            # Draw boxes on the output image
            for box, score, label in zip(boxes, scores, labels):
                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Get label name
                label_name = self.coco_names[label] if label < len(self.coco_names) else f"Class {label}"
                
                # Draw box - bright green for visibility
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw background for text
                text = f"{label_name}: {score:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                
                # Draw label and score in black text
                cv2.putText(output_image, text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw FPS and detection count
            detection_count = len(boxes)
            status_text = f"FPS: {self.current_fps:.1f} | Detections: {detection_count}"
            
            # Draw background for status text
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (5, 5), (5 + status_size[0] + 10, 5 + status_size[1] + 10), (0, 0, 0), -1)
            
            # Draw status text
            cv2.putText(output_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert back to QImage
            return self.to_qimage(output_image)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame


class YOLOv8Model(ModelInterface):
    """YOLOv8 model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_path = str(parent_dir / "yolov8s.pt")
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.input_size = 640  # Default YOLO input size
        
    def load(self):
        """Load the YOLOv8 model"""
        try:
            # Attempt to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                print("Ultralytics not found. Please install with: pip install ultralytics")
                return False
                
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
                
            # Load the model
            self.model = YOLO(self.model_path)
            
            # Set model parameters for faster inference
            if hasattr(self.model, 'fuse') and callable(self.model.fuse):
                # Fuse conv and bn layers for faster inference
                self.model.fuse()
            
            print("YOLOv8 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame):
        """Process a frame with YOLOv8
        
        Args:
            frame (QImage): Input frame
            
        Returns:
            QImage: Processed frame with detections
        """
        if self.model is None:
            return frame
            
        try:
            # Convert QImage to numpy array
            image_np = self.to_numpy(frame)
            
            # Create a copy for output
            output_image = image_np.copy()
            
            # Get frame dimensions
            height, width = image_np.shape[:2]
            
            # Run inference with optimized settings
            start_time = time.time()
            results = self.model(image_np, 
                               conf=self.confidence_threshold,
                               iou=0.45,  # Lower IOU threshold for faster NMS
                               max_det=20,  # Limit detections for speed
                               verbose=False)  # Disable verbose output
            
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
                status_text = f"FPS: {self.current_fps:.1f} | Detections: {detection_count} | YOLOv8"
            else:
                # Draw FPS if no detections
                status_text = f"FPS: {self.current_fps:.1f} | No detections | YOLOv8"
            
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
    """YOLOv8 segmentation model implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_path = str(parent_dir / "yolov8s-seg.pt")
        self.last_inference_time = 0
        self.fps_alpha = 0.9  # For FPS smoothing
        self.current_fps = 0
        self.input_size = 640  # Default YOLO input size
        
    def load(self):
        """Load the YOLOv8 segmentation model"""
        try:
            # Attempt to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                print("Ultralytics not found. Please install with: pip install ultralytics")
                return False
                
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
                
            # Load the model
            self.model = YOLO(self.model_path)
            
            # Set model parameters for faster inference
            if hasattr(self.model, 'fuse') and callable(self.model.fuse):
                # Fuse conv and bn layers for faster inference
                self.model.fuse()
            
            print("YOLOv8-Seg model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading YOLOv8-Seg model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def process_frame(self, frame):
        """Process a frame with YOLOv8 segmentation
        
        Args:
            frame (QImage): Input frame
            
        Returns:
            QImage: Processed frame with detections and segmentation masks
        """
        if self.model is None:
            return frame
            
        try:
            # Convert QImage to numpy array
            image_np = self.to_numpy(frame)
            
            # Create a copy of the image for drawing
            output_image = image_np.copy()
            
            # Run inference with optimized settings
            start_time = time.time()
            results = self.model(image_np, 
                               conf=self.confidence_threshold,
                               iou=0.45,  # Lower IOU threshold for faster NMS
                               max_det=10,  # Limit detections for speed
                               verbose=False)  # Disable verbose output
            
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
                    status_text = f"FPS: {self.current_fps:.1f} | Masks: {mask_count} | YOLOv8-Seg"
                except Exception as e:
                    # Error processing mask results
                    print(f"Error processing masks: {e}")
                    status_text = f"FPS: {self.current_fps:.1f} | Mask error | YOLOv8-Seg"
            else:
                # No detections
                status_text = f"FPS: {self.current_fps:.1f} | No masks | YOLOv8-Seg"
            
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