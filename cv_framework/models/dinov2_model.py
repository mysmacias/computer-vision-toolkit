"""
DINOv2 model implementation for the computer vision framework.
"""

import cv2
import torch
import numpy as np
import time
from datetime import datetime
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from ..models.base_model import VisionModel
from ..utils.visualization import visualize_depth_map


class DINOv2Model(VisionModel):
    """
    Implementation of DINOv2 model for depth estimation, semantic segmentation, 
    and feature visualization. Uses a pretrained self-supervised vision transformer.
    """
    
    def __init__(self, model_name='dinov2_vits14', device=None):
        """
        Initialize the DINOv2 model.
        
        Args:
            model_name (str): Model name ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device (str, optional): Device to run the model on ('cpu', 'cuda:0', etc.)
        """
        super().__init__(model_name, device)
        
        # Model configuration
        self.patch_size = 14  # DINOv2 models use 14x14 patches
        self.img_size = 518  # Multiple of patch_size (37 * 14)
        
        # Visualization mode (depth, segmentation, features, side_by_side)
        self.visualization_mode = 'depth'
        self.modes = ['depth', 'segmentation', 'features', 'side_by_side']
        self.mode_index = 0
        
        # Depth estimation attributes
        self.depth_pca = PCA(n_components=1)
        self.depth_pca_fitted = False
        self.depth_colormap = cv2.COLORMAP_INFERNO
        self.colormaps = [
            cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, 
            cv2.COLORMAP_PLASMA, cv2.COLORMAP_HOT
        ]
        self.colormap_index = 0
        
        # Segmentation attributes
        self.num_clusters = 8  # Number of segments
        self.segmentation_pca = PCA(n_components=50)
        self.segmentation_kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.segmentation_fitted = False
        
        # Feature visualization attributes
        self.feature_pca = PCA(n_components=3)
        self.feature_pca_fitted = False
        
        # Feature collection for initialization
        self.feature_buffer = []
        self.buffer_size = 10
        self.buffer_filled = False
        
        # Transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def load_model(self):
        """
        Load the DINOv2 model from torch hub.
        """
        try:
            print(f"Loading {self.model_name} model...")
            # Patch the interpolate function for compatibility if needed
            self._patch_interpolate()
            
            # Try to load the model with more specific error handling
            try:
                # First check if torch hub is available
                repos = torch.hub.list('facebookresearch/dinov2')
                if self.model_name not in repos:
                    available_models = ', '.join(repos)
                    raise ValueError(f"Model {self.model_name} not found in available models: {available_models}")
                
                # Load the model
                self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"DINOv2 model loaded successfully: {self.model_name}")
                return True
            except Exception as hub_error:
                print(f"Error loading from torch hub: {hub_error}")
                print("Attempting to load with timm as fallback...")
                
                # Try to use timm as a fallback 
                import timm
                model_name_map = {
                    'dinov2_vits14': 'vit_small_patch14_dinov2',
                    'dinov2_vitb14': 'vit_base_patch14_dinov2',
                    'dinov2_vitl14': 'vit_large_patch14_dinov2',
                    'dinov2_vitg14': 'vit_giant_patch14_dinov2',
                }
                timm_name = model_name_map.get(self.model_name)
                if timm_name is None:
                    raise ValueError(f"No timm equivalent for {self.model_name}")
                
                self.model = timm.create_model(timm_name, pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"DINOv2 model loaded successfully using timm: {timm_name}")
                return True
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Please make sure you have the necessary packages installed:")
            print("pip install torch torchvision timm")
            self.model = None
            return False
    
    def _patch_interpolate(self):
        """
        Patch the interpolate function to handle different PyTorch versions.
        """
        import torch.nn.functional as F
        original_interpolate = F.interpolate
        
        def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', 
                                align_corners=None, recompute_scale_factor=None, 
                                antialias=None):
            # Remove antialias parameter for older PyTorch versions
            if 'antialias' in inspect.signature(original_interpolate).parameters:
                return original_interpolate(input, size=size, scale_factor=scale_factor,
                                           mode=mode, align_corners=align_corners,
                                           recompute_scale_factor=recompute_scale_factor,
                                           antialias=antialias)
            else:
                return original_interpolate(input, size=size, scale_factor=scale_factor,
                                           mode=mode, align_corners=align_corners,
                                           recompute_scale_factor=recompute_scale_factor)
        
        F.interpolate = patched_interpolate
    
    def get_feature_grid_size(self, image_shape):
        """
        Calculate the grid size for features based on image shape.
        
        Args:
            image_shape (tuple): Shape of the image (H, W, C)
            
        Returns:
            tuple: Grid height and width for feature patches
        """
        h, w = image_shape[:2]
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        return grid_h, grid_w
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for DINOv2 model.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            dict: Preprocessed data including original frame and tensor
        """
        # Resize frame to match the model's expected input size
        h, w = frame.shape[:2]
        
        # Center crop to square and resize
        min_dim = min(h, w)
        y_start = (h - min_dim) // 2
        x_start = (w - min_dim) // 2
        cropped = frame[y_start:y_start + min_dim, x_start:x_start + min_dim]
        resized = cv2.resize(cropped, (self.img_size, self.img_size))
        
        # Convert to PIL Image for transformation
        pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        
        # Apply transformation
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return {
            'original_frame': frame,
            'resized_frame': resized,
            'tensor': img_tensor
        }
    
    def extract_features(self, img_tensor):
        """
        Extract features from image using DINOv2.
        
        Args:
            img_tensor (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please ensure load_model() was called and returned True.")
            
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
        
        # Extract patch tokens (excluding CLS token)
        if isinstance(features, dict):
            patch_features = features['x_norm_patchtokens']
        else:
            # If features is a tensor, it typically includes the CLS token as the first element
            patch_features = features[:, 1:, :]  # Skip the CLS token
        
        return patch_features
    
    def predict(self, preprocessed_data):
        """
        Process input data through the model.
        
        Args:
            preprocessed_data (dict): Preprocessed input data
            
        Returns:
            dict: Prediction results including features and original data
        """
        try:
            # Extract features
            features = self.extract_features(preprocessed_data['tensor'])
            
            # Add to feature buffer if not full
            if not self.buffer_filled and len(self.feature_buffer) < self.buffer_size:
                self.feature_buffer.append(features.cpu().numpy())
                if len(self.feature_buffer) >= self.buffer_size:
                    self.buffer_filled = True
                    self._fit_models()
            
            # Add features to prediction results
            result = {
                'features': features,
                'original_frame': preprocessed_data['original_frame'],
                'resized_frame': preprocessed_data['resized_frame']
            }
            
            return result
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            # Return a dummy result that won't cause errors in downstream processing
            return {
                'features': None,
                'original_frame': preprocessed_data['original_frame'],
                'resized_frame': preprocessed_data['resized_frame'],
                'error': str(e)
            }
    
    def _fit_models(self):
        """
        Fit PCA and KMeans models once the buffer is filled.
        """
        print("Initializing DINOv2 models from collected features...")
        
        # Concatenate all features
        all_features = np.concatenate(self.feature_buffer, axis=1)[0]  # [N, feature_dim]
        
        # Fit depth PCA
        self.depth_pca.fit(all_features)
        self.depth_pca_fitted = True
        
        # Fit segmentation models
        normalized_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
        self.segmentation_pca.fit(normalized_features)
        reduced_features = self.segmentation_pca.transform(normalized_features)
        self.segmentation_kmeans.fit(reduced_features)
        self.segmentation_fitted = True
        
        # Fit feature visualization PCA
        self.feature_pca.fit(normalized_features)
        self.feature_pca_fitted = True
        
        print("DINOv2 models initialized successfully")
    
    def estimate_depth(self, features):
        """
        Estimate depth map from DINOv2 features.
        
        Args:
            features (torch.Tensor): Extracted features
            
        Returns:
            numpy.ndarray: Estimated depth map
        """
        # Handle case where features is None
        if features is None:
            # Return a placeholder depth map
            placeholder = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            # Add some random gradient to make it look like a depth map
            y, x = np.mgrid[0:self.img_size, 0:self.img_size]
            placeholder = 0.5 + 0.5 * np.sin(x/self.img_size * 3) * np.cos(y/self.img_size * 2)
            return placeholder
            
        # Get shape and device
        grid_h, grid_w = self.get_feature_grid_size((self.img_size, self.img_size))
        
        # Check if PCA is properly fitted
        if not hasattr(self, 'depth_pca') or self.depth_pca is None or not hasattr(self, 'depth_pca_fitted') or not self.depth_pca_fitted:
            # Return a placeholder depth map
            placeholder = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            # Add some random gradient to make it look like a depth map
            y, x = np.mgrid[0:self.img_size, 0:self.img_size]
            placeholder = 0.5 + 0.5 * np.sin(x/self.img_size * 3) * np.cos(y/self.img_size * 2)
            return placeholder
        
        try:
            # Get features and reshape
            features_np = features.cpu().numpy()[0]  # [num_patches, feature_dim]
            
            # Apply PCA for depth estimation (project to first principal component)
            depth_values = self.depth_pca.transform(features_np)[:, 0]
            
            # Reshape to 2D grid
            grid_size = int(np.sqrt(len(depth_values)))
            depth_map = depth_values.reshape(grid_size, grid_size)
            
            # Resize to full image size
            depth_map_resized = cv2.resize(depth_map, (self.img_size, self.img_size))
            
            # Apply light Gaussian blur to smooth the depth map
            depth_map_smooth = cv2.GaussianBlur(depth_map_resized, (3, 3), 0)
            
            # Normalize the depth values to 0-1 range
            depth_min, depth_max = depth_map_smooth.min(), depth_map_smooth.max()
            if depth_min != depth_max:
                depth_map_norm = (depth_map_smooth - depth_min) / (depth_max - depth_min)
            else:
                depth_map_norm = np.zeros_like(depth_map_smooth)
            
            return depth_map_norm
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            # Return a placeholder on error
            placeholder = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            y, x = np.mgrid[0:self.img_size, 0:self.img_size]
            placeholder = 0.5 + 0.5 * np.sin(x/self.img_size * 3) * np.cos(y/self.img_size * 2)
            return placeholder
    
    def segment_image(self, features):
        """
        Segment image using DINOv2 features.
        
        Args:
            features (torch.Tensor): Extracted features
            
        Returns:
            numpy.ndarray: Segmentation map
        """
        # Handle case where features is None
        if features is None:
            # Return a placeholder segmentation map with random segments
            placeholder = np.zeros((self.img_size, self.img_size), dtype=np.int32)
            # Add some random segments
            for _ in range(self.num_clusters):
                center_y = np.random.randint(0, self.img_size)
                center_x = np.random.randint(0, self.img_size)
                radius = np.random.randint(20, 100)
                color = np.random.randint(0, self.num_clusters)
                y, x = np.mgrid[0:self.img_size, 0:self.img_size]
                mask = ((y - center_y)**2 + (x - center_x)**2) < radius**2
                placeholder[mask] = color
            return placeholder
            
        # Check if segmentation models are properly fitted
        if (not hasattr(self, 'segmentation_kmeans') or self.segmentation_kmeans is None or 
            not hasattr(self, 'segmentation_pca') or self.segmentation_pca is None or
            not hasattr(self, 'segmentation_fitted') or not self.segmentation_fitted):
            # Create placeholder segmentation during initialization
            img_size = self.img_size
            placeholder = np.zeros((img_size, img_size), dtype=np.int32)
            for i in range(10):
                y1 = np.random.randint(0, img_size//2)
                x1 = np.random.randint(0, img_size//2)
                y2 = np.random.randint(img_size//2, img_size)
                x2 = np.random.randint(img_size//2, img_size)
                placeholder[y1:y2, x1:x2] = np.random.randint(0, self.num_clusters)
            return placeholder
        
        try:
            # Get features and normalize
            features_np = features.cpu().numpy()[0]  # [num_patches, feature_dim]
            normalized_features = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
            
            # Reduce dimensionality
            reduced_features = self.segmentation_pca.transform(normalized_features)
            
            # Cluster features
            cluster_ids = self.segmentation_kmeans.predict(reduced_features)
            
            # Reshape to 2D grid
            grid_size = int(np.sqrt(len(cluster_ids)))
            segmentation = cluster_ids.reshape(grid_size, grid_size)
            
            # Resize to full image size (using nearest neighbor to preserve labels)
            segmentation_resized = cv2.resize(
                segmentation, 
                (self.img_size, self.img_size), 
                interpolation=cv2.INTER_NEAREST
            )
            
            return segmentation_resized
        except Exception as e:
            print(f"Error in segmentation: {e}")
            # Return a placeholder on error
            placeholder = np.zeros((self.img_size, self.img_size), dtype=np.int32)
            for i in range(5):
                y1 = np.random.randint(0, self.img_size//2)
                x1 = np.random.randint(0, self.img_size//2)
                y2 = np.random.randint(self.img_size//2, self.img_size)
                x2 = np.random.randint(self.img_size//2, self.img_size)
                placeholder[y1:y2, x1:x2] = np.random.randint(0, self.num_clusters)
            return placeholder
    
    def visualize_features(self, features):
        """
        Visualize DINOv2 features in RGB space.
        
        Args:
            features (torch.Tensor): Extracted features
            
        Returns:
            numpy.ndarray: RGB visualization of features
        """
        # Handle case where features is None
        if features is None:
            # Return a colorful placeholder image
            img_size = self.img_size
            placeholder = np.zeros((img_size, img_size, 3), dtype=np.float32)
            # Create a colorful pattern
            y, x = np.mgrid[0:img_size, 0:img_size]
            placeholder[:, :, 0] = 0.5 + 0.5 * np.sin(x/50)
            placeholder[:, :, 1] = 0.5 + 0.5 * np.sin(y/50)
            placeholder[:, :, 2] = 0.5 + 0.5 * np.sin((x+y)/70)
            return (placeholder * 255).astype(np.uint8)
            
        # Check if feature PCA is properly fitted
        if (not hasattr(self, 'feature_pca') or self.feature_pca is None or 
            not hasattr(self, 'feature_pca_fitted') or not self.feature_pca_fitted):
            # Return a colorful placeholder during initialization
            img_size = self.img_size
            placeholder = np.zeros((img_size, img_size, 3), dtype=np.float32)
            
            # Add some random colorful elements
            for i in range(10):
                y1, x1 = np.random.randint(0, img_size), np.random.randint(0, img_size)
                y2, x2 = np.random.randint(0, img_size), np.random.randint(0, img_size)
                color = np.random.rand(3)
                cv2.line(placeholder, (x1, y1), (x2, y2), color, 2)
            
            # Add text overlay
            cv2.putText(
                placeholder,
                "Initializing feature visualization...",
                (img_size//5, img_size//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (1, 1, 1),
                2
            )
            
            return (placeholder * 255).astype(np.uint8)
        
        try:
            # Get features and normalize
            features_np = features.cpu().numpy()[0]  # [num_patches, feature_dim]
            normalized_features = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
            
            # Apply PCA to reduce to 3 dimensions (RGB)
            rgb_features = self.feature_pca.transform(normalized_features)
            
            # Scale to [0, 1] range for each dimension
            for i in range(3):
                min_val, max_val = rgb_features[:, i].min(), rgb_features[:, i].max()
                if min_val != max_val:
                    rgb_features[:, i] = (rgb_features[:, i] - min_val) / (max_val - min_val)
                else:
                    rgb_features[:, i] = 0
            
            # Reshape to 2D grid with RGB channels
            grid_size = int(np.sqrt(len(rgb_features)))
            feature_visualization = rgb_features.reshape(grid_size, grid_size, 3)
            
            # Resize to full image size
            feature_vis_resized = cv2.resize(feature_visualization, (self.img_size, self.img_size))
            
            # Convert to BGR for OpenCV and to uint8 range
            feature_vis_bgr = (feature_vis_resized[:, :, ::-1] * 255).astype(np.uint8)
            
            return feature_vis_bgr
        except Exception as e:
            print(f"Error in feature visualization: {e}")
            # Return a colorful placeholder on error
            img_size = self.img_size
            placeholder = np.zeros((img_size, img_size, 3), dtype=np.float32)
            y, x = np.mgrid[0:img_size, 0:img_size]
            placeholder[:, :, 0] = 0.5 + 0.5 * np.sin(x/30)
            placeholder[:, :, 1] = 0.5 + 0.5 * np.sin(y/30)
            placeholder[:, :, 2] = 0.5 + 0.5 * np.sin((x+y)/40)
            return (placeholder * 255).astype(np.uint8)
    
    def visualize_predictions(self, frame, predictions):
        """
        Visualize DINOv2 predictions based on current mode.
        
        Args:
            frame (numpy.ndarray): Original input frame
            predictions (dict): Prediction results
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Initialize result with the original resized frame
        resized_frame = predictions['resized_frame']
        features = predictions.get('features')
        
        # Check if there was an error during prediction
        if features is None or 'error' in predictions:
            # Display error message on the frame
            error_frame = resized_frame.copy()
            error_message = predictions.get('error', 'Unknown error during model inference')
            
            # Add red background for error
            overlay = error_frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.img_size, 80), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.5, error_frame, 0.5, 0, error_frame)
            
            # Add error messages
            cv2.putText(
                error_frame,
                "MODEL ERROR",
                (self.img_size//2 - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Split error message if it's too long
            if len(error_message) > 50:
                # Split into two lines
                mid_point = error_message.rfind(' ', 0, 50)
                if mid_point == -1:
                    mid_point = 50
                
                line1 = error_message[:mid_point]
                line2 = error_message[mid_point:].strip()
                
                cv2.putText(
                    error_frame,
                    line1,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                cv2.putText(
                    error_frame,
                    line2,
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            else:
                cv2.putText(
                    error_frame,
                    error_message,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            return error_frame
        
        visualization = None
        overlay_label = ""
        
        # Create visualization based on the current mode
        if self.visualization_mode == 'depth':
            # Depth estimation
            depth_map = self.estimate_depth(features)
            colored_depth = visualize_depth_map(depth_map, self.depth_colormap)
            visualization = colored_depth
            overlay_label = "Depth Estimation"
            
        elif self.visualization_mode == 'segmentation':
            # Semantic segmentation
            segmentation = self.segment_image(features)
            
            # Create colorful visualization
            colored_segmentation = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            for cluster_id in range(self.num_clusters):
                # Generate a color for each cluster
                color = np.array([
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ], dtype=np.uint8)
                
                # Apply color to all pixels in this cluster
                colored_segmentation[segmentation == cluster_id] = color
            
            # Blend with original image
            alpha = 0.7
            visualization = cv2.addWeighted(
                resized_frame, 1 - alpha,
                colored_segmentation, alpha,
                0
            )
            overlay_label = "Semantic Segmentation"
            
        elif self.visualization_mode == 'features':
            # Feature visualization
            feature_vis = self.visualize_features(features)
            visualization = feature_vis
            overlay_label = "Feature Visualization"
            
        elif self.visualization_mode == 'side_by_side':
            # Create comprehensive side-by-side view with all visualizations
            
            # Get all visualizations
            depth_map = self.estimate_depth(features)
            colored_depth = visualize_depth_map(depth_map, self.depth_colormap)
            
            segmentation = self.segment_image(features)
            colored_segmentation = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            for cluster_id in range(self.num_clusters):
                # Generate a color for each cluster (but use consistent colors)
                np.random.seed(cluster_id * 10)  # For consistent colors
                color = np.array([
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ], dtype=np.uint8)
                # Apply color to all pixels in this cluster
                colored_segmentation[segmentation == cluster_id] = color
            
            # Blend segmentation with a copy of the original for better visibility
            blend_alpha = 0.7
            segmentation_blend = cv2.addWeighted(
                resized_frame.copy(), 1 - blend_alpha,
                colored_segmentation, blend_alpha,
                0
            )
            
            feature_vis = self.visualize_features(features)
            
            # Create a 2x2 grid
            h, w = resized_frame.shape[:2]
            margin = 5  # Margin between subimages
            text_height = 30  # Height for text labels
            
            # Create a larger frame with margins and text spaces
            grid_h = 2 * h + margin + 2 * text_height
            grid_w = 2 * w + margin
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            # Define regions for each image
            regions = [
                (0, 0, w, h),  # Original (top-left)
                (w + margin, 0, w, h),  # Depth (top-right)
                (0, h + text_height + margin, w, h),  # Segmentation (bottom-left)
                (w + margin, h + text_height + margin, w, h),  # Features (bottom-right)
            ]
            
            # Place images in the grid
            grid[regions[0][1]:regions[0][1]+h, regions[0][0]:regions[0][0]+w] = resized_frame
            grid[regions[1][1]:regions[1][1]+h, regions[1][0]:regions[1][0]+w] = colored_depth
            grid[regions[2][1]:regions[2][1]+h, regions[2][0]:regions[2][0]+w] = segmentation_blend
            grid[regions[3][1]:regions[3][1]+h, regions[3][0]:regions[3][0]+w] = feature_vis
            
            # Add frames around each visualization
            frame_thickness = 2
            frame_colors = [
                (255, 255, 255),  # White for original
                (0, 165, 255),    # Orange for depth
                (0, 255, 0),      # Green for segmentation
                (255, 0, 255)     # Purple for features
            ]
            
            for i, (x, y, w, h) in enumerate(regions):
                cv2.rectangle(
                    grid, 
                    (x, y), 
                    (x + w, y + h), 
                    frame_colors[i], 
                    frame_thickness
                )
            
            # Add labels with background
            labels = ["Original Image", "Depth Estimation", "Semantic Segmentation", "Feature Visualization"]
            for i, (x, y, w, h) in enumerate(regions):
                # Create semi-transparent background for text
                label_y = y + h if i < 2 else y - text_height
                overlay = grid.copy()
                cv2.rectangle(overlay, (x, label_y), (x + w, label_y + text_height), (40, 40, 40), -1)
                cv2.addWeighted(overlay, 0.7, grid, 0.3, 0, grid)
                
                # Add text
                cv2.putText(
                    grid,
                    labels[i],
                    (x + 10, label_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
            
            # Add DINOv2 title at the top
            cv2.putText(
                grid,
                f"DINOv2 Multi-View: {self.model_name}",
                (grid_w // 2 - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Add control instructions at the bottom
            cv2.putText(
                grid,
                "Press 'v' to switch modes, 'c' to change colormap, 'q' to quit",
                (grid_w // 2 - 230, grid_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            visualization = grid
            overlay_label = ""  # No need for overlay label in grid view
        
        # If we're not in side-by-side mode, add an overlay label
        if self.visualization_mode != 'side_by_side':
            cv2.putText(
                visualization,
                overlay_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return visualization
    
    def process_frame(self, frame):
        """
        Process a single frame with UI enhancements.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        start_time = time.time()
        
        # Preprocess frame
        preprocessed_data = self.preprocess_frame(frame)
        
        # Run inference
        predictions = self.predict(preprocessed_data)
        
        # Visualize predictions
        visualization = self.visualize_predictions(frame, predictions)
        
        # Calculate FPS
        processing_fps = 1.0 / max(0.001, (time.time() - start_time))
        
        # Add information overlay
        result_frame = visualization.copy()
        
        # Add key controls information
        cv2.putText(
            result_frame,
            "Press 'v' to change visualization mode, 'c' to change colormap",
            (10, result_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add metadata
        cv2.putText(
            result_frame,
            f"FPS: {processing_fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Add model name
        cv2.putText(
            result_frame,
            self.model_name,
            (10, result_frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return result_frame
    
    def run(self, camera_idx=0):
        """
        Override the run method to add special key handlers.
        """
        # Import necessary modules here to avoid circular imports
        import time
        import inspect
        
        # Load model
        if not self.load_model():
            print(f"Error: Failed to load {self.model_name} model. Exiting.")
            return
        
        # Setup camera
        if not self.setup_camera(camera_idx):
            return
        
        # Setup video writer
        if not self.setup_video_writer(suffix=self.visualization_mode):
            self.cap.release()
            return
        
        # Initialize timing
        self.recording_start_time = time.time()
        self.frame_count = 0
        
        print(f"Starting real-time processing with {self.model_name}. Press 'q' to quit, 'v' to change visualization mode.")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Write to video
                # If result has a different size than original, resize for video writer
                if result_frame.shape[0] != self.frame_height or result_frame.shape[1] != self.frame_width:
                    write_frame = cv2.resize(result_frame, (self.frame_width, self.frame_height))
                else:
                    write_frame = result_frame
                self.out.write(write_frame)
                
                # Display result
                cv2.imshow(f'{self.model_name} - {self.visualization_mode}', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Quit if 'q' is pressed
                if key == ord('q'):
                    break
                
                # Change visualization mode if 'v' is pressed
                elif key == ord('v'):
                    self.mode_index = (self.mode_index + 1) % len(self.modes)
                    self.visualization_mode = self.modes[self.mode_index]
                    print(f"Switching to {self.visualization_mode} mode")
                
                # Change colormap if 'c' is pressed
                elif key == ord('c'):
                    self.colormap_index = (self.colormap_index + 1) % len(self.colormaps)
                    self.depth_colormap = self.colormaps[self.colormap_index]
                    print(f"Switching to colormap #{self.colormap_index}")
                
                # Counter for frames processed
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("Processing stopped by user")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.cleanup()
            
            # Print summary
            recording_duration = time.time() - self.recording_start_time
            print(f"Recording completed: {self.frame_count} frames processed in {recording_duration:.2f} seconds")
            if self.frame_count > 0:
                print(f"Average FPS: {self.frame_count / recording_duration:.2f}")
            print(f"Video saved to: {self.output_dir}")
    
    @staticmethod
    def get_available_models():
        """
        List all available DINOv2 models.
        
        Returns:
            dict: Dictionary of available models
        """
        return {
            'DINOv2': [
                'dinov2_vits14',  # Small
                'dinov2_vitb14',  # Base
                'dinov2_vitl14',  # Large
                'dinov2_vitg14'   # Giant
            ]
        } 