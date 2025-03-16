#!/usr/bin/env python
"""
Implementation of the Segment Anything Model (SAM) for the computer vision framework.
Supports both the original SAM model with downloadable weights and the Ultralytics-based SAM.
"""

import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from pathlib import Path
import threading
from collections import deque
import requests
import logging
from tqdm import tqdm
import random

from cv_framework.models.base_model import VisionModel

# Global variable for click coordinates
click_coords = []
processing_click = False

# Colors for visualization
COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Green (dark)
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
]

class SAMModel(VisionModel):
    """
    Segment Anything Model (SAM) for image segmentation.
    
    Attributes:
        model_name (str): Name of the model
        device (str): Device to run inference on
        model (object): The loaded model
        conf_threshold (float): Confidence threshold for predictions
    """
    
    def __init__(self, model_name="sam_vit_b", device=None):
        """
        Initialize SAM model.
        
        Args:
            model_name (str): Name of the model to use
            device (str, optional): Device to run on (cpu, cuda, etc.)
        """
        super().__init__(model_name, device)
        
        # Model-specific configuration
        self.conf_threshold = 0.3
        self.model = None
        self.predictor = None
        self.original_sam = False
        self.interactive_mode = False
        self.auto_mask_generation = True
        self.previous_masks = []
        self.prev_masks_colors = []
        self.is_ultralytics = model_name.startswith('yolo')
        
        # Click handling
        self.click_points = []
        self.click_labels = []  # 1 for positive, 0 for negative
        
        # UI settings
        self.point_radius = 5
        self.point_color_pos = (0, 255, 0)  # Green for positive
        self.point_color_neg = (255, 0, 0)  # Red for negative
        self.mask_alpha = 0.5
        self.random_color_masks = True
        self.masks_memory = deque(maxlen=5)  # Store last 5 sets of masks
        
        # Checkpoint path for original SAM
        self.checkpoint_path = None
    
    def load_model(self):
        """
        Load the SAM model based on model name.
        """
        if self.is_ultralytics:
            try:
                from ultralytics import YOLO, SAM
                print(f"Loading Ultralytics SAM model: {self.model_name}")
                self.model = YOLO(self.model_name)
                self.model_type = "ultralytics_sam"
                print("Ultralytics SAM model loaded successfully")
            except ImportError as e:
                print(f"Failed to import Ultralytics: {e}")
                print("Please install ultralytics with: pip install ultralytics")
                raise
        else:
            try:
                # First try checking if we can load the original SAM model
                # This needs some special handling since SAM has separate packages
                print(f"Loading original SAM model: {self.model_name}")
                
                # Determine checkpoint path
                model_type = self.model_name.replace("sam_", "")
                self.checkpoint_path = self._get_or_download_checkpoint(model_type)
                
                # Import and load models
                try:
                    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
                    
                    print(f"Loading SAM model from checkpoint: {self.checkpoint_path}")
                    self.original_sam = True
                    
                    # Determine model type
                    if "vit_h" in self.model_name:
                        model_type = "vit_h"
                    elif "vit_l" in self.model_name:
                        model_type = "vit_l"
                    elif "vit_b" in self.model_name:
                        model_type = "vit_b"
                    else:
                        model_type = "vit_b"  # Default
                    
                    # Load the model
                    sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
                    sam.to(self.device)
                    
                    # Initialize the predictor
                    self.predictor = SamPredictor(sam)
                    
                    # Initialize the automatic mask generator
                    self.mask_generator = SamAutomaticMaskGenerator(
                        model=sam,
                        points_per_side=32,
                        pred_iou_thresh=0.86,
                        stability_score_thresh=0.92,
                        crop_n_layers=1,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=100
                    )
                    
                    print("Original SAM model loaded successfully")
                except ImportError as e:
                    print(f"Failed to import segment_anything: {e}")
                    print("Please install segment_anything with: pip install segment-anything")
                    raise
                
            except Exception as e:
                print(f"Error loading SAM model: {e}")
                raise
    
    def _get_or_download_checkpoint(self, model_type):
        """
        Get checkpoint path or download it if not available.
        
        Args:
            model_type (str): SAM model type 
            
        Returns:
            str: Path to the checkpoint file
        """
        # Map model types to filenames and URLs
        checkpoint_info = {
            "vit_h": {
                "filename": "sam_vit_h_4b8939.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            },
            "vit_l": {
                "filename": "sam_vit_l_0b3195.pth", 
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
            },
            "vit_b": {
                "filename": "sam_vit_b_01ec64.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
        }
        
        # Default to vit_b if model_type is not recognized
        if model_type not in checkpoint_info:
            print(f"Unknown model type {model_type}, defaulting to vit_b")
            model_type = "vit_b"
        
        # Check common locations for the checkpoint file
        common_locations = [
            os.path.join(os.path.expanduser("~"), "models"),
            os.path.join(os.path.expanduser("~"), "Downloads"),
            os.path.join(os.getcwd(), "models"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
        ]
        
        filename = checkpoint_info[model_type]["filename"]
        
        # Try to find the checkpoint file
        for location in common_locations:
            if os.path.exists(os.path.join(location, filename)):
                return os.path.join(location, filename)
        
        # If not found, download the file
        print(f"Checkpoint file {filename} not found in common locations. Downloading...")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Download the file
        url = checkpoint_info[model_type]["url"]
        save_path = os.path.join(models_dir, filename)
        
        try:
            self._download_file(url, save_path)
            print(f"Downloaded {filename} to {save_path}")
            return save_path
        except Exception as e:
            print(f"Failed to download checkpoint file: {e}")
            raise
    
    def _download_file(self, url, save_path):
        """
        Download a file from URL with progress bar.
        
        Args:
            url (str): URL to download from
            save_path (str): Path to save the file to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(save_path, 'wb') as file, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)
        except Exception as e:
            # Clean up partial download if it failed
            if os.path.exists(save_path):
                os.remove(save_path)
            raise e
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # For original SAM, just return the frame as the predictor handles preprocessing
        if self.original_sam:
            return frame
        
        # For Ultralytics SAM, resize frame if needed
        if self.is_ultralytics:
            # No need to preprocess for YOLO
            return frame
        
        return frame
    
    def predict(self, frame):
        """
        Run prediction on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            dict: Prediction results
        """
        global click_coords, processing_click
        
        # Initialize model if not already done
        if self.model is None and not self.original_sam:
            self.load_model()
        
        if self.original_sam and self.predictor is None:
            self.load_model()
        
        # Copy the frame to avoid modifying the original
        org_frame = frame.copy()
        
        # Run inference
        if self.is_ultralytics:
            # For Ultralytics SAM model
            results = self.model(frame)
            
            # Extract masks
            masks = []
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                for i, mask in enumerate(results[0].masks.data):
                    mask_np = mask.cpu().numpy()
                    masks.append({
                        'segmentation': mask_np,
                        'area': mask_np.sum(),
                        'bbox': None,
                        'predicted_iou': 1.0,  # YOLO doesn't provide this
                        'stability_score': 1.0,  # YOLO doesn't provide this
                    })
            
            return {'masks': masks}
            
        elif self.original_sam:
            # For original SAM model
            
            # Process click coordinates if available
            if click_coords and not processing_click and self.interactive_mode:
                processing_click = True
                
                # Get the latest click coordinates
                x, y, is_positive = click_coords[-1]
                
                # Add the click to our lists
                self.click_points.append([x, y])
                self.click_labels.append(1 if is_positive else 0)
                
                # Set the image embedding if not already done
                if not hasattr(self.predictor, "is_image_set") or not self.predictor.is_image_set:
                    self.predictor.set_image(org_frame)
                    self.predictor.is_image_set = True  # Track if image is set
                
                # Get the masks from the predictor
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.click_points),
                    point_labels=np.array(self.click_labels),
                    multimask_output=True,
                    return_logits=True
                )
                
                # Convert to list of dictionaries for consistent format
                mask_dicts = []
                for i, mask in enumerate(masks):
                    mask_dicts.append({
                        'segmentation': mask,
                        'area': mask.sum(),
                        'bbox': None,
                        'predicted_iou': scores[i],
                        'stability_score': scores[i],
                    })
                
                # Store masks
                self.previous_masks = mask_dicts
                
                # Generate random colors for masks if needed
                if not self.prev_masks_colors or len(self.prev_masks_colors) != len(self.previous_masks):
                    self.prev_masks_colors = [self._get_random_color() for _ in range(len(self.previous_masks))]
                
                processing_click = False
                return {'masks': self.previous_masks}
            
            # Handle automatic mask generation
            if self.auto_mask_generation and not self.interactive_mode:
                try:
                    # Generate masks automatically
                    masks = self.mask_generator.generate(org_frame)
                    
                    # Convert to our standard format
                    for mask in masks:
                        mask['segmentation'] = mask.pop('segmentation')
                    
                    # Store masks
                    self.previous_masks = masks
                    
                    # Generate random colors for masks if needed
                    if not self.prev_masks_colors or len(self.prev_masks_colors) != len(self.previous_masks):
                        self.prev_masks_colors = [self._get_random_color() for _ in range(len(self.previous_masks))]
                    
                    return {'masks': masks}
                except Exception as e:
                    print(f"Error in automatic mask generation: {e}")
                    return {'masks': []}
            
            # Return previous masks if interactive mode but no new clicks
            if self.interactive_mode and self.previous_masks:
                return {'masks': self.previous_masks}
            
            # Default empty response
            return {'masks': []}
            
        # Default empty response if no model matched
        return {'masks': []}
    
    def _get_random_color(self):
        """
        Get a random color for mask visualization.
        
        Returns:
            tuple: RGB color tuple
        """
        if COLORS:
            return random.choice(COLORS)
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    def visualize_predictions(self, frame, prediction_results):
        """
        Visualize prediction results.
        
        Args:
            frame (numpy.ndarray): Input frame
            prediction_results (dict): Prediction results
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Get masks from prediction results
        masks = prediction_results.get('masks', [])
        
        # Draw the masks on the frame
        if masks:
            # Sort masks by area (largest first)
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # Generate random colors if needed
            if not self.prev_masks_colors or len(self.prev_masks_colors) != len(sorted_masks):
                self.prev_masks_colors = [self._get_random_color() for _ in range(len(sorted_masks))]
            
            # Create a mask overlay
            mask_overlay = np.zeros_like(viz_frame)
            
            # Draw each mask
            for i, mask_dict in enumerate(sorted_masks):
                # Get mask and color
                mask = mask_dict['segmentation']
                color = self.prev_masks_colors[i % len(self.prev_masks_colors)]
                
                # Handle different mask formats
                if isinstance(mask, dict) and 'data' in mask:
                    mask = mask['data']
                
                # Ensure mask is a binary numpy array
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure mask shape matches the frame
                if mask.shape != (viz_frame.shape[0], viz_frame.shape[1]):
                    # Resize mask to match frame
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (viz_frame.shape[1], viz_frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Apply mask
                mask_overlay[mask > 0.5] = color
            
            # Blend the mask overlay with the original frame
            cv2.addWeighted(
                viz_frame, 1.0,
                mask_overlay, self.mask_alpha,
                0, viz_frame
            )
            
            # Draw mask contours for clarity
            for i, mask_dict in enumerate(sorted_masks):
                mask = mask_dict['segmentation']
                color = self.prev_masks_colors[i % len(self.prev_masks_colors)]
                
                # Handle different mask formats
                if isinstance(mask, dict) and 'data' in mask:
                    mask = mask['data']
                
                # Ensure mask is a binary numpy array
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure mask shape matches the frame
                if mask.shape != (viz_frame.shape[0], viz_frame.shape[1]):
                    # Resize mask to match frame
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (viz_frame.shape[1], viz_frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Find contours
                mask_binary = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                cv2.drawContours(viz_frame, contours, -1, color, 2)
        
        # Draw click points in interactive mode
        if self.interactive_mode:
            for i, (point, label) in enumerate(zip(self.click_points, self.click_labels)):
                color = self.point_color_pos if label == 1 else self.point_color_neg
                cv2.circle(viz_frame, tuple(point), self.point_radius, color, -1)
        
        # Draw mode indicators and help text
        cv2.putText(
            viz_frame,
            f"Mode: {'Interactive' if self.interactive_mode else 'Automatic'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Draw controls help
        controls_text = "Controls: 'i' - Toggle interactive, 'c' - Clear points, 'a' - Auto mode"
        cv2.putText(
            viz_frame,
            controls_text,
            (10, viz_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Add click instructions if in interactive mode
        if self.interactive_mode:
            click_text = "Left-click: Add object, Right-click: Remove background"
            cv2.putText(
                viz_frame,
                click_text,
                (10, viz_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return viz_frame
    
    def process_frame(self, frame, frame_idx=0):
        """
        Process a frame using the SAM model.
        
        Args:
            frame (numpy.ndarray): Input frame
            frame_idx (int): Frame index
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        if frame is None:
            return None
        
        start_time = time.time()
        
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Run prediction
        prediction_results = self.predict(processed_frame)
        
        # Visualize results
        visualized_frame = self.visualize_predictions(frame, prediction_results)
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(
            visualized_frame,
            f"FPS: {fps:.1f}",
            (visualized_frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Display model info
        cv2.putText(
            visualized_frame,
            f"Model: {self.model_name}",
            (10, visualized_frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        return visualized_frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for interactive mode.
        
        Args:
            event (int): Mouse event type
            x (int): X coordinate
            y (int): Y coordinate
            flags (int): Event flags
            param: Additional parameters
        """
        global click_coords, processing_click
        
        # Only handle clicks in interactive mode
        if not self.interactive_mode:
            return
        
        # Handle left-click (positive point)
        if event == cv2.EVENT_LBUTTONDOWN:
            click_coords.append((x, y, True))  # True for positive
        
        # Handle right-click (negative point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_coords.append((x, y, False))  # False for negative
    
    def clear_points(self):
        """
        Clear all interactive points.
        """
        global click_coords
        click_coords = []
        self.click_points = []
        self.click_labels = []
    
    def toggle_interactive_mode(self):
        """
        Toggle between interactive and automatic modes.
        """
        self.interactive_mode = not self.interactive_mode
        if self.interactive_mode:
            self.clear_points()
            # Set predictor to None to force re-embedding of image
            if hasattr(self.predictor, "is_image_set"):
                self.predictor.is_image_set = False
    
    def run(self, camera_idx=0, input_source=None):
        """
        Run the model on a video stream or input source.
        
        Args:
            camera_idx (int): Camera index for cv2.VideoCapture
            input_source (str): Path to video file or None for camera
        """
        global click_coords, processing_click
        
        # Initialize the model if not already done
        if self.model is None and not self.original_sam:
            self.load_model()
        
        if self.original_sam and self.predictor is None:
            self.load_model()
        
        # Open video capture
        source = input_source if input_source else camera_idx
        cap = cv2.VideoCapture(source)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video source: {source}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")
        
        # Create window and set mouse callback
        window_name = f"SAM Model: {self.model_name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        frame_idx = 0
        
        try:
            while True:
                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    # If end of video, loop back to beginning for video files
                    if input_source and isinstance(input_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Process the frame
                processed_frame = self.process_frame(frame, frame_idx)
                
                # Display the processed frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                
                # Handle ESC key to exit
                if key == 27:  # ESC
                    break
                
                # Handle 'i' key to toggle interactive mode
                elif key == ord('i'):
                    self.toggle_interactive_mode()
                    print(f"Interactive mode: {self.interactive_mode}")
                
                # Handle 'c' key to clear points
                elif key == ord('c'):
                    self.clear_points()
                    print("Cleared all points")
                
                # Handle 'a' key to toggle auto mask generation
                elif key == ord('a'):
                    if self.interactive_mode:
                        self.interactive_mode = False
                    self.auto_mask_generation = True
                    print("Automatic mask generation enabled")
                
                frame_idx += 1
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
    
    @staticmethod
    def get_available_models():
        """
        Get a list of available SAM models.
        
        Returns:
            dict: Dictionary with model categories and names
        """
        # Define SAM models
        sam_models = {
            "Segment Anything Models (SAM)": [
                "sam_vit_b",
                "sam_vit_l", 
                "sam_vit_h",
                "yolov8n-seg",
                "yolov8s-seg",
                "yolov8m-seg",
                "yolov8l-seg",
                "yolov8x-seg"
            ]
        }
        
        return sam_models 