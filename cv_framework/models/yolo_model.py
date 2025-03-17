"""
YOLO model implementation for the computer vision framework.
"""

import os
import cv2
import numpy as np
import torch
from ..models.base_model import VisionModel
from ..utils.visualization import draw_bounding_box, draw_mask


class YOLOModel(VisionModel):
    """
    Implementation of YOLO models (v5, v8) for object detection, segmentation,
    and pose estimation.
    """
    
    def __init__(self, model_name='yolov8s', device=None):
        """
        Initialize the YOLO model.
        
        Args:
            model_name (str): Name of the YOLO model (e.g., 'yolov8s', 'yolov5m')
            device (str, optional): Device to run the model on ('cpu', 'cuda:0', etc.)
        """
        super().__init__(model_name, device)
        
        # YOLO-specific attributes
        self.has_segmentation = 'seg' in model_name
        self.has_pose = 'pose' in model_name
        self.model_version = 'v8' if model_name.startswith('yolov8') else 'v5'
        
        # Model configuration
        self.confidence_threshold = 0.4  # Default confidence threshold
        self.iou_threshold = 0.45  # Default IoU threshold for NMS
        
        # Color mapping for visualization
        self.class_colors = {}
        
    def load_model(self):
        """
        Load the YOLO model from Ultralytics.
        """
        try:
            # Set environment variables to avoid MKL threading issues
            import os
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            try:
                from ultralytics import YOLO
            except ImportError as e:
                print(f"Error: Ultralytics package not found: {e}")
                print("Please install it using: pip install ultralytics")
                return False
            
            model_path = f"{self.model_name}.pt"
            
            # First check if model exists in the current directory
            current_dir_path = os.path.join(os.getcwd(), model_path)
            framework_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
            
            # Printing paths for debugging
            print(f"Checking for model at current directory: {current_dir_path}")
            print(f"Checking for model at framework directory: {framework_dir_path}")
            
            # Check if model exists in various locations
            cached_model = None
            ultralytics_dir = os.path.expanduser("~/.cache/ultralytics/")
            
            if os.path.exists(current_dir_path):
                cached_model = current_dir_path
                print(f"Using model found in current directory: {current_dir_path}")
            elif os.path.exists(framework_dir_path):
                cached_model = framework_dir_path
                print(f"Using model found in framework directory: {framework_dir_path}")
            elif os.path.exists(os.path.join(ultralytics_dir, "models", model_path)):
                cached_model = os.path.join(ultralytics_dir, "models", model_path)
                print(f"Using cached model: {cached_model}")
            else:
                print(f"Model not found locally, will attempt to download: {model_path}")
            
            # Load the model with explicit error handling
            try:
                print(f"Loading {self.model_name} model...")
                if cached_model:
                    # Try loading from explicit path first
                    self.model = YOLO(cached_model)
                else:
                    # Fall back to letting ultralytics handle it
                    self.model = YOLO(self.model_name)
                
                print(f"Model loaded successfully: {self.model_name}")
                
                # Set model to appropriate device
                try:
                    self.model.to(self.device)
                    print(f"Model successfully moved to device: {self.device}")
                except Exception as dev_err:
                    print(f"Warning: Failed to move model to {self.device}: {dev_err}")
                    print("Continuing with default device")
                
                return True
            except Exception as load_err:
                print(f"Failed to load YOLO model: {load_err}")
                # If it failed, try once more with a different approach
                try:
                    print("Attempting alternative loading method...")
                    self.model = YOLO(self.model_name, task='detect')
                    print("Alternative loading succeeded")
                    return True
                except Exception as alt_err:
                    print(f"Alternative loading also failed: {alt_err}")
                    return False
            
        except Exception as e:
            import traceback
            print(f"Error loading YOLO model: {e}")
            print(traceback.format_exc())
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for YOLO model.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # YOLO from ultralytics handles preprocessing internally,
        # so we just return the frame as is
        return frame
    
    def predict(self, frame):
        """
        Run inference on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: YOLO results
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)
        return results
    
    def visualize_predictions(self, frame, results):
        """
        Visualize YOLO predictions on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            results: YOLO prediction results
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Initialize output frame
        output_frame = frame.copy()
        
        # Get first result (batch size is 1)
        result = results[0]
        
        # Process bounding boxes and class labels for detection
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name
                class_name = result.names[class_id]
                
                # Get color for this class (generate if it doesn't exist)
                if class_id not in self.class_colors:
                    self.class_colors[class_id] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                color = self.class_colors[class_id]
                
                # Draw bounding box
                draw_bounding_box(
                    output_frame, 
                    (x1, y1, x2, y2), 
                    label=class_name, 
                    score=conf, 
                    color=color
                )
        
        # Process segmentation masks if model supports it
        if self.has_segmentation and hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if len(masks) > 0:
                for i, mask_data in enumerate(masks):
                    # Get class ID associated with this mask
                    class_id = int(boxes[i].cls[0])
                    mask = mask_data.data.cpu().numpy()[0]
                    
                    # Get color for this class
                    color = self.class_colors.get(class_id, (0, 255, 0))
                    
                    # Draw mask
                    output_frame = draw_mask(
                        output_frame,
                        mask,
                        color=color,
                        alpha=0.5
                    )
        
        # Process keypoints for pose estimation if model supports it
        if self.has_pose and hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints
            if len(keypoints) > 0:
                # Implement pose visualization
                for kps in keypoints:
                    # Get keypoint coordinates
                    kp_array = kps.data.cpu().numpy()[0]
                    
                    # Draw keypoints and connections based on COCO keypoints format
                    # (This is a simplified version, a complete implementation would include the COCO skeleton)
                    for i, (x, y, conf) in enumerate(kp_array):
                        if conf > self.confidence_threshold:
                            # Draw keypoint
                            cv2.circle(output_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # Connect keypoints (simplified skeleton)
                    # A complete implementation would use the COCO keypoint connections
                    if kp_array.shape[0] >= 17:  # COCO format has 17 keypoints
                        # Example: connect some keypoints
                        connections = [
                            (5, 7), (7, 9),  # Left arm
                            (6, 8), (8, 10),  # Right arm
                            (11, 13), (13, 15),  # Left leg
                            (12, 14), (14, 16),  # Right leg
                            (5, 6), (5, 11), (6, 12),  # Torso
                            (1, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Head and neck
                        ]
                        
                        for p1, p2 in connections:
                            if kp_array[p1-1][2] > self.confidence_threshold and kp_array[p2-1][2] > self.confidence_threshold:
                                pt1 = (int(kp_array[p1-1][0]), int(kp_array[p1-1][1]))
                                pt2 = (int(kp_array[p2-1][0]), int(kp_array[p2-1][1]))
                                cv2.line(output_frame, pt1, pt2, (0, 255, 255), 2)
        
        return output_frame
    
    def set_confidence_threshold(self, threshold):
        """
        Set the confidence threshold for detection.
        
        Args:
            threshold (float): Confidence threshold
        """
        self.confidence_threshold = max(0.05, min(0.95, threshold))
        print(f"Confidence threshold set to: {self.confidence_threshold:.2f}")
    
    @staticmethod
    def list_available_models():
        """
        List all available YOLO models from Ultralytics.
        
        Returns:
            dict: Dictionary of available models grouped by category
        """
        models = {
            'YOLOv5 Detection': [
                'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
            ],
            'YOLOv8 Detection': [
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            ],
            'YOLOv8 Segmentation': [
                'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg'
            ],
            'YOLOv8 Pose': [
                'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose'
            ]
        }
        
        return models 