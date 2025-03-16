"""
DETR (DEtection TRansformer) model implementation for the computer vision framework.
"""

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from ..models.base_model import VisionModel
from ..utils.visualization import draw_bounding_box, generate_color_map


class DETRModel(VisionModel):
    """
    Implementation of DETR (DEtection TRansformer) model for object detection.
    """
    
    def __init__(self, model_name='detr_resnet50', device=None):
        """
        Initialize the DETR model.
        
        Args:
            model_name (str): Model name, used for display purposes ('detr_resnet50' or 'detr_resnet101')
            device (str, optional): Device to run the model on ('cpu', 'cuda:0', etc.)
        """
        super().__init__(model_name, device)
        
        # Model configuration
        self.confidence_threshold = 0.7  # DETR often needs a higher threshold than other models
        
        # Model variant
        self.backbone = 'resnet101' if 'resnet101' in model_name else 'resnet50'
        
        # COCO class names
        self.class_names = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
        
        # Generate color map for visualization
        self.colors = generate_color_map(len(self.class_names))
        
        # Define transformation
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """
        Load the DETR model from the HuggingFace model hub.
        """
        try:
            print(f"Loading DETR model with {self.backbone} backbone...")
            model_name = f"detr_{self.backbone}"
            self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"DETR model loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for DETR model.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR format)
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply the transformations
        img_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def predict(self, img_tensor):
        """
        Run inference on a preprocessed tensor.
        
        Args:
            img_tensor (torch.Tensor): Preprocessed input tensor
            
        Returns:
            dict: Prediction results
        """
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        return outputs
    
    def visualize_predictions(self, frame, predictions):
        """
        Visualize DETR predictions on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            predictions (dict): Model predictions
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Initialize output frame
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]
        
        # Process predictions (DETR outputs a different format than other models)
        probas = predictions['pred_logits'].softmax(-1)[0, :, :-1]  # Remove no-object class
        boxes = predictions['pred_boxes'][0]
        
        # Keep only predictions with confidence above threshold
        keep = probas.max(-1).values > self.confidence_threshold
        
        # Get scores and labels for the kept predictions
        scores, labels = probas[keep].max(-1)
        
        # Convert from CPU to numpy arrays
        boxes = boxes[keep].cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Visualize the predictions
        for box, label_id, score in zip(boxes, labels, scores):
            # DETR boxes are in [cx, cy, w, h] format and normalized
            # Convert to [x1, y1, x2, y2] format in pixel coordinates
            cx, cy, box_w, box_h = box
            x1 = int((cx - box_w/2) * w)
            y1 = int((cy - box_h/2) * h)
            x2 = int((cx + box_w/2) * w)
            y2 = int((cy + box_h/2) * h)
            
            # Ensure box coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Get class name
            label_name = self.class_names[label_id] if label_id < len(self.class_names) else f"Class {label_id}"
            
            # Get color for this class
            color = tuple(map(int, self.colors[label_id]))
            
            # Draw bounding box
            draw_bounding_box(
                output_frame,
                (x1, y1, x2, y2),
                label=label_name,
                score=score,
                color=color
            )
        
        return output_frame
    
    @staticmethod
    def get_available_models():
        """
        List all available DETR models.
        
        Returns:
            dict: Dictionary of available models
        """
        return {
            'DETR': [
                'detr_resnet50',
                'detr_resnet101'
            ]
        } 