"""
SSD model implementation for the computer vision framework.
"""

import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from ..models.base_model import VisionModel
from ..utils.visualization import draw_bounding_box, generate_color_map


class SSDModel(VisionModel):
    """
    Implementation of SSD model for object detection.
    """
    
    def __init__(self, model_name='ssd300', device=None):
        """
        Initialize the SSD model.
        
        Args:
            model_name (str): Model name, used for display purposes
            device (str, optional): Device to run the model on ('cpu', 'cuda:0', etc.)
        """
        super().__init__(model_name, device)
        
        # Model configuration
        self.confidence_threshold = 0.5
        
        # COCO class names
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Generate colors for each class
        self.colors = generate_color_map(len(self.class_names))
    
    def load_model(self):
        """
        Load the SSD model from torchvision.
        """
        try:
            print("Loading SSD model...")
            self.model = ssd300_vgg16(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print("SSD model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading SSD model: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for SSD model.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = F.to_tensor(rgb_frame).to(self.device)
        
        return img_tensor
    
    def predict(self, img_tensor):
        """
        Run inference on a preprocessed tensor.
        
        Args:
            img_tensor (torch.Tensor): Preprocessed input tensor
            
        Returns:
            list: Prediction results
        """
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        return predictions
    
    def visualize_predictions(self, frame, predictions):
        """
        Visualize SSD predictions on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            predictions (list): Model predictions
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Initialize output frame
        output_frame = frame.copy()
        
        # Get the prediction for the first image (batch size is 1)
        pred = predictions[0]
        
        # Get boxes, labels, and scores
        boxes = pred['boxes'].cpu().numpy().astype(np.int32)
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Draw boxes and labels
        for box, label_id, score in zip(boxes, labels, scores):
            if score >= self.confidence_threshold:
                # Get class name
                label_name = self.class_names[label_id] if label_id < len(self.class_names) else f"Class {label_id}"
                
                # Get color for this class
                color = tuple(map(int, self.colors[label_id]))
                
                # Draw bounding box
                draw_bounding_box(
                    output_frame,
                    (box[0], box[1], box[2], box[3]),
                    label=label_name,
                    score=score,
                    color=color
                )
        
        return output_frame
    
    @staticmethod
    def get_available_models():
        """
        List all available SSD models.
        
        Returns:
            dict: Dictionary of available models
        """
        return {
            'SSD': [
                'ssd300'
            ]
        } 