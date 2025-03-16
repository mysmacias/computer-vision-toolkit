import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from PIL import Image
import torchvision.transforms as T
import warnings
import math
from sklearn.decomposition import PCA
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Patch for older PyTorch versions that don't have antialias parameter
def patch_interpolate():
    """
    Monkey patch the interpolate function to handle older PyTorch versions
    by removing the antialias parameter if it's not supported
    """
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

# Apply the patch at module level
patch_interpolate()

# DINOv2 feature extractor class
class DINOv2FeatureExtractor:
    def __init__(self, model_name='dinov2_vits14'):
        """Initialize the DINOv2 feature extractor with specified model size"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # For older PyTorch versions, suppress warnings
        warnings.filterwarnings("ignore", message=".*xFormers.*")
        
        try:
            # Try loading requested model variant
            print(f"Loading DINOv2 {model_name} model...")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.patch_size = 14  # Default patch size for DINOv2
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Trying DINOv2 ViT-S/14 model as fallback...")
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                self.model = self.model.to(self.device)
                self.model.eval()
                self.patch_size = 14
            except Exception as e:
                print(f"Error loading alternative model: {e}")
                print("Using torchvision's ViT model as fallback...")
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.patch_size = 16
        
        # Get image size divisible by patch size
        self.img_size = 518 - (518 % self.patch_size)
        print(f"Using image size: {self.img_size}")
        
        # Default transforms for preprocessing - use the calculated size
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        """Extract features from an image using DINOv2"""
        with torch.no_grad():
            if isinstance(image, np.ndarray):  # If OpenCV image (BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Apply transformations
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            if hasattr(self.model, 'get_intermediate_layers'):
                features = self.model.get_intermediate_layers(img_tensor, n=1)[0]
            elif hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(img_tensor)
                if isinstance(features, dict):
                    features = features['last_hidden_state'] if 'last_hidden_state' in features else list(features.values())[0]
            else:
                features = self.model(img_tensor)
                if isinstance(features, dict):
                    features = list(features.values())[0]
            
            # Return the features
            return features

    def extract_patch_features(self, image):
        """Extract patch tokens from an image, excluding the CLS token"""
        features = self.extract_features(image)
        
        # Handle different feature formats
        if features.dim() == 3 and features.shape[1] > 1:
            # Skip the CLS/distillation token (first token)
            patch_tokens = features[:, 1:, :]
            return patch_tokens
        else:
            # If no CLS token or unexpected format, return as is
            return features
            
    def get_feature_grid_size(self, image):
        """Calculate the feature grid dimensions based on the image and patch size"""
        if isinstance(image, np.ndarray):  # If OpenCV image
            h, w = image.shape[:2]
        elif isinstance(image, Image.Image):  # If PIL image
            w, h = image.size
        else:
            raise ValueError("Unsupported image type")
        
        # Calculate grid dimensions (after resize and center crop)
        h = w = self.img_size  # Square image after transforms
        
        # Calculate feature grid dimensions
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        
        return grid_h, grid_w

# Depth estimation model based on DINOv2 features
class DINOv2DepthEstimator:
    def __init__(self, feature_extractor):
        """Initialize depth estimator with a DINOv2 feature extractor"""
        self.feature_extractor = feature_extractor
        self.device = feature_extractor.device
        
        # Get feature dimension from the feature extractor's model
        # Default to 384 for ViT-S/14, but try to infer dynamically if possible
        self.feature_dim = 384  # Default for ViT-S
        if hasattr(feature_extractor.model, 'embed_dim'):
            self.feature_dim = feature_extractor.model.embed_dim
        
        # For a more meaningful depth estimation, we'll use PCA instead of random weights
        # PCA will find the principal components of variation in the feature space
        # This often correlates with depth in natural images
        self.pca = PCA(n_components=1)
        self.is_fitted = False
        self.feature_buffer = []
        self.buffer_size = 5  # Collect features from this many frames before fitting PCA
        
    def estimate_depth(self, image):
        """Estimate depth from an image using DINOv2 features"""
        # Get patch features from the image
        patch_features = self.feature_extractor.extract_patch_features(image)
        
        # Get the expected grid size
        grid_h, grid_w = self.feature_extractor.get_feature_grid_size(image)
        
        # Extract feature vectors
        with torch.no_grad():
            # Get feature shape and convert to numpy
            features_np = patch_features.squeeze(0).cpu().numpy()
            
            # If PCA not fitted yet, collect features
            if not self.is_fitted:
                self.feature_buffer.append(features_np)
                if len(self.feature_buffer) >= self.buffer_size:
                    print("Fitting PCA for depth estimation...")
                    # Concatenate all features
                    all_features = np.vstack(self.feature_buffer)
                    # Fit PCA
                    self.pca.fit(all_features)
                    self.is_fitted = True
                    self.feature_buffer = []
                    print("PCA fitted successfully!")
            
            # Apply PCA to get depth values (or use a placeholder if not fitted)
            if self.is_fitted:
                # Transform features to get primary component
                depth_values = self.pca.transform(features_np)
            else:
                # Use placeholder values based on feature norms until PCA is fitted
                print(f"Collecting features for PCA ({len(self.feature_buffer)}/{self.buffer_size})...")
                # Feature vector magnitudes can give rough depth estimate
                depth_values = np.linalg.norm(features_np, axis=1, keepdims=True)
            
            # Check if we have enough features for the expected grid
            expected_patches = grid_h * grid_w
            actual_patches = features_np.shape[0]
            
            if actual_patches < expected_patches:
                print(f"Warning: Not enough patches. Expected {expected_patches}, got {actual_patches}")
                # Pad with the mean value
                padding = np.full((expected_patches - actual_patches, 1), depth_values.mean())
                depth_values = np.vstack((depth_values, padding))
            elif actual_patches > expected_patches:
                # Truncate extra patches
                depth_values = depth_values[:expected_patches]
            
            # Reshape to grid
            depth_map = depth_values.reshape(grid_h, grid_w)
            
            # Resize to desired output size
            output_size = (self.feature_extractor.img_size, self.feature_extractor.img_size)
            depth_map = cv2.resize(depth_map, output_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply smoothing for better visualization
            depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
            
            # Normalize depth map for visualization
            depth_map = depth_map - depth_map.min()
            max_val = depth_map.max()
            if max_val > 0:
                depth_map = depth_map / max_val
            
            return depth_map

# Semantic segmentation model based on DINOv2 features
class DINOv2SemanticSegmenter:
    def __init__(self, feature_extractor, num_clusters=8):
        """Initialize semantic segmenter with a DINOv2 feature extractor"""
        self.feature_extractor = feature_extractor
        self.device = feature_extractor.device
        self.num_clusters = num_clusters
        
        # We'll use k-means for clustering features
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        
        # Flag to track if kmeans has been fitted
        self.is_fitted = False
        
        # Store a buffer of features to fit k-means initially
        self.feature_buffer = []
        self.buffer_size = 10
        
        # Color map for visualization - using a more distinct colormap
        self.color_map = plt.cm.get_cmap('viridis', num_clusters)
        
        # For dimensionality reduction before clustering
        # This makes clustering more effective and faster
        self.pca = PCA(n_components=50)  # Reduce to 50 dimensions
        self.pca_fitted = False
        
    def segment_image(self, image):
        """Perform semantic segmentation using DINOv2 features and k-means clustering"""
        # Get patch features from the image
        patch_features = self.feature_extractor.extract_patch_features(image)
        
        # Get the expected grid size
        grid_h, grid_w = self.feature_extractor.get_feature_grid_size(image)
        
        # Get features for clustering
        features = patch_features.squeeze(0).cpu().numpy()
        
        # Apply feature normalization - important for better clustering
        # L2 normalize each feature vector
        feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (feature_norms + 1e-8)  # avoid division by zero
        
        # Dimensionality reduction with PCA if we have enough data
        if not self.pca_fitted:
            self.feature_buffer.append(features_normalized)
            if len(self.feature_buffer) >= self.buffer_size:
                print("Fitting PCA for feature dimensionality reduction...")
                all_features = np.vstack(self.feature_buffer)
                self.pca.fit(all_features)
                self.pca_fitted = True
                # Now fit k-means on reduced features
                reduced_features = self.pca.transform(all_features)
                print("Fitting k-means on reduced features...")
                self.kmeans.fit(reduced_features)
                self.is_fitted = True
                self.feature_buffer = []
                print("Feature preprocessing and k-means initialization complete!")
        
        # If PCA is fitted, apply dimensionality reduction
        if self.pca_fitted:
            features_for_clustering = self.pca.transform(features_normalized)
            
            # If k-means is fitted, predict clusters
            if self.is_fitted:
                # Predict cluster for each patch
                clusters = self.kmeans.predict(features_for_clustering)
                
                # Check if we have enough features for the expected grid
                expected_patches = grid_h * grid_w
                actual_patches = features.shape[0]
                
                if actual_patches < expected_patches:
                    print(f"Warning: Not enough patches for segmentation. Expected {expected_patches}, got {actual_patches}")
                    # Pad with the most common cluster
                    if len(clusters) > 0:
                        from collections import Counter
                        most_common = Counter(clusters).most_common(1)[0][0]
                        padding = np.full(expected_patches - actual_patches, most_common)
                        clusters = np.concatenate((clusters, padding))
                    else:
                        clusters = np.zeros(expected_patches, dtype=np.int32)
                elif actual_patches > expected_patches:
                    # Truncate extra patches
                    clusters = clusters[:expected_patches]
                
                # Reshape to grid
                segmentation_map = clusters.reshape(grid_h, grid_w)
                
                # Apply median filtering to remove noise
                segmentation_map = cv2.medianBlur(segmentation_map.astype(np.uint8), 3)
                
                # Resize to original image size
                output_size = (self.feature_extractor.img_size, self.feature_extractor.img_size)
                segmentation_map = cv2.resize(
                    segmentation_map.astype(np.float32), 
                    output_size, 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Create colored segmentation map
                colored_segmentation = np.zeros((self.feature_extractor.img_size, self.feature_extractor.img_size, 3), dtype=np.float32)
                
                for cluster_idx in range(self.num_clusters):
                    # Get color for this cluster
                    color = np.array(self.color_map(cluster_idx)[:3])
                    # Apply color to all pixels in this cluster
                    colored_segmentation[segmentation_map == cluster_idx] = color
                
                return colored_segmentation
        
        # Return a placeholder during initialization
        print(f"Collecting features for segmentation ({len(self.feature_buffer)}/{self.buffer_size})...")
        img_size = self.feature_extractor.img_size
        placeholder = np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        # Generate a nicer placeholder visualization
        for i in range(10):
            y1, x1 = np.random.randint(0, img_size), np.random.randint(0, img_size)
            y2, x2 = np.random.randint(0, img_size), np.random.randint(0, img_size)
            color = np.random.rand(3)
            cv2.line(placeholder, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(
            placeholder,
            "Initializing segmentation...",
            (img_size//5, img_size//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (1, 1, 1),
            2
        )
        
        return placeholder

# Feature visualization based on DINOv2 features
class DINOv2FeatureVisualizer:
    def __init__(self, feature_extractor):
        """Initialize feature visualizer with a DINOv2 feature extractor"""
        self.feature_extractor = feature_extractor
        self.device = feature_extractor.device
        
        # For dimensionality reduction (will project high-dimensional features to 3D for RGB visualization)
        self.pca = PCA(n_components=3)
        self.is_fitted = False
        self.feature_buffer = []
        self.buffer_size = 5
    
    def visualize_features(self, image):
        """Create RGB visualization of DINOv2 features using PCA"""
        # Get patch features
        patch_features = self.feature_extractor.extract_patch_features(image)
        
        # Get the expected grid size
        grid_h, grid_w = self.feature_extractor.get_feature_grid_size(image)
        
        # Move to numpy for processing
        features = patch_features.squeeze(0).cpu().numpy()
        
        # Normalize features (important for better visualization)
        feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (feature_norms + 1e-8)
        
        # If PCA not fitted yet, collect features
        if not self.is_fitted:
            self.feature_buffer.append(features_normalized)
            if len(self.feature_buffer) >= self.buffer_size:
                print("Fitting PCA for feature visualization...")
                all_features = np.vstack(self.feature_buffer)
                self.pca.fit(all_features)
                self.is_fitted = True
                self.feature_buffer = []
                print("PCA fitted successfully for feature visualization!")
        
        # Project features to 3D space for RGB visualization
        if self.is_fitted:
            rgb_features = self.pca.transform(features_normalized)
            
            # Scale to [0, 1] range for RGB
            rgb_features = rgb_features - rgb_features.min(axis=0)
            rgb_features = rgb_features / (rgb_features.max(axis=0) + 1e-8)
            
            # Check if we have enough features for the expected grid
            expected_patches = grid_h * grid_w
            actual_patches = features.shape[0]
            
            if actual_patches < expected_patches:
                print(f"Warning: Not enough patches for visualization. Expected {expected_patches}, got {actual_patches}")
                # Pad with zeros
                padding = np.zeros((expected_patches - actual_patches, 3))
                rgb_features = np.vstack((rgb_features, padding))
            elif actual_patches > expected_patches:
                # Truncate extra patches
                rgb_features = rgb_features[:expected_patches]
            
            # Reshape to grid
            feature_map = rgb_features.reshape(grid_h, grid_w, 3)
            
            # Resize to desired output size
            output_size = (self.feature_extractor.img_size, self.feature_extractor.img_size)
            feature_map = cv2.resize(feature_map, output_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply light Gaussian blur for smoother visualization
            feature_map = cv2.GaussianBlur(feature_map, (3, 3), 0)
            
            return feature_map
        else:
            # Return a placeholder during initialization
            print(f"Collecting features for visualization ({len(self.feature_buffer)}/{self.buffer_size})...")
            
            # Create a more informative placeholder
            img_size = self.feature_extractor.img_size
            placeholder = np.zeros((img_size, img_size, 3), dtype=np.float32)
            
            # Add a gradient background
            y, x = np.mgrid[0:img_size, 0:img_size]
            placeholder[:,:,0] = x / img_size
            placeholder[:,:,1] = y / img_size
            placeholder[:,:,2] = 0.5
            
            cv2.putText(
                placeholder,
                "Initializing feature visualization...",
                (img_size//10, img_size//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (1, 1, 1),
                2
            )
            
            return placeholder

def main():
    """
    Main function to run real-time DINOv2 depth estimation and semantic segmentation
    on webcam feed, and save the processed video to a file
    """
    # Create DINOv2 feature extractor
    feature_extractor = DINOv2FeatureExtractor(model_name='dinov2_vits14')
    
    # Create depth estimator
    depth_estimator = DINOv2DepthEstimator(feature_extractor)
    
    # Create semantic segmenter
    segmenter = DINOv2SemanticSegmenter(feature_extractor, num_clusters=8)
    
    # Create feature visualizer
    feature_visualizer = DINOv2FeatureVisualizer(feature_extractor)
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get camera properties for display and video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Image size used for DINOv2 processing
    img_size = feature_extractor.img_size
    print(f"Using processing image size: {img_size}x{img_size}")
    
    # Create output directory if it doesn't exist
    output_dir = 'output_videos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"dinov2_perception_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Define visualization modes
    VIZ_MODES = {
        0: "DEPTH",
        1: "SEGMENTATION",
        2: "FEATURES",
        3: "SIDE_BY_SIDE"
    }
    current_viz_mode = 0
    
    # Generate colormaps for depth visualization
    depth_colormaps = [cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS]
    depth_colormap_names = ["Inferno", "Jet", "Turbo", "Viridis"]
    current_depth_colormap = 0
    
    print("Starting real-time DINOv2 perception. Press 'q' to quit.")
    print("Press 'v' to cycle through visualization modes (Depth, Segmentation, Features, Side-by-Side)")
    print("Press 'c' to cycle through depth colormaps")
    
    frame_count = 0
    recording_start_time = time.time()
    
    # For performance tracking
    processing_times = []
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Create a copy of the frame for saving
            frame_with_viz = frame.copy()
            
            # First, preprocess the frame to match DINOv2's expected size
            # Center-crop to square and resize to consistent size
            h, w = frame.shape[:2]
            size = min(h, w)
            x = (w - size) // 2
            y = (h - size) // 2
            cropped_frame = frame[y:y+size, x:x+size]
            processed_frame = cv2.resize(cropped_frame, (img_size, img_size))
            
            # Convert to PIL Image for processing
            pil_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            # Process the frame depending on the current visualization mode
            depth_map = None
            segmentation_map = None
            feature_map = None
            
            # Always process for the side-by-side view, or if in respective single mode
            if current_viz_mode in [0, 3]:  # DEPTH or SIDE_BY_SIDE
                # Estimate depth
                try:
                    depth_map = depth_estimator.estimate_depth(pil_image)
                    
                    # Convert depth map to colored visualization
                    colored_depth = cv2.applyColorMap(
                        (depth_map * 255).astype(np.uint8), 
                        depth_colormaps[current_depth_colormap]
                    )
                    
                    # Resize to match frame size
                    colored_depth = cv2.resize(colored_depth, (frame_width, frame_height))
                except Exception as e:
                    print(f"Error estimating depth: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a blank depth map on error
                    colored_depth = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        colored_depth,
                        "Depth estimation error",
                        (frame_width // 4, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            if current_viz_mode in [1, 3]:  # SEGMENTATION or SIDE_BY_SIDE
                # Perform semantic segmentation
                try:
                    segmentation_map = segmenter.segment_image(pil_image)
                    
                    # Convert to uint8 for display and resize to match frame size
                    colored_segmentation = (segmentation_map * 255).astype(np.uint8)
                    colored_segmentation = cv2.resize(
                        colored_segmentation, 
                        (frame_width, frame_height)
                    )
                except Exception as e:
                    print(f"Error performing segmentation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a blank segmentation map on error
                    colored_segmentation = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        colored_segmentation,
                        "Segmentation error",
                        (frame_width // 4, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            if current_viz_mode in [2, 3]:  # FEATURES or SIDE_BY_SIDE
                # Visualize features
                try:
                    feature_map = feature_visualizer.visualize_features(pil_image)
                    
                    # Convert to uint8 for display and resize to match frame size
                    colored_features = (feature_map * 255).astype(np.uint8)
                    colored_features = cv2.resize(
                        colored_features, 
                        (frame_width, frame_height)
                    )
                except Exception as e:
                    print(f"Error visualizing features: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a blank feature map on error
                    colored_features = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        colored_features,
                        "Feature visualization error",
                        (frame_width // 4, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            # Combine visualizations based on current mode
            if current_viz_mode == 0:  # DEPTH
                # Apply depth map as overlay
                alpha = 0.7
                frame_with_viz = cv2.addWeighted(
                    frame_with_viz, 1-alpha, colored_depth, alpha, 0
                )
            elif current_viz_mode == 1:  # SEGMENTATION
                # Apply segmentation map as overlay
                alpha = 0.7
                frame_with_viz = cv2.addWeighted(
                    frame_with_viz, 1-alpha, colored_segmentation, alpha, 0
                )
            elif current_viz_mode == 2:  # FEATURES
                # Apply feature map as overlay
                alpha = 0.7
                frame_with_viz = cv2.addWeighted(
                    frame_with_viz, 1-alpha, colored_features, alpha, 0
                )
            elif current_viz_mode == 3:  # SIDE_BY_SIDE
                # Create side by side display: [Original | Depth | Segmentation | Features]
                vis_width = frame_width // 4
                
                # Calculate appropriate height to maintain aspect ratio
                vis_height = int(frame_height * (vis_width / frame_width))
                
                # Resize the original frame and visualization maps to fit side by side
                original_resized = cv2.resize(frame, (vis_width, vis_height))
                depth_resized = cv2.resize(colored_depth, (vis_width, vis_height))
                segmentation_resized = cv2.resize(colored_segmentation, (vis_width, vis_height))
                features_resized = cv2.resize(colored_features, (vis_width, vis_height))
                
                # Combine horizontally
                frame_with_viz = np.hstack((original_resized, depth_resized, segmentation_resized, features_resized))
                
                # Create a header area
                header_height = 30
                header = np.zeros((header_height, frame_with_viz.shape[1], 3), dtype=np.uint8)
                
                # Add section titles
                cv2.putText(
                    header,
                    "Original",
                    (vis_width // 2 - 40, header_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                cv2.putText(
                    header,
                    "Depth",
                    (vis_width + vis_width // 2 - 30, header_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                cv2.putText(
                    header,
                    "Segmentation",
                    (2 * vis_width + vis_width // 2 - 50, header_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                cv2.putText(
                    header,
                    "Features",
                    (3 * vis_width + vis_width // 2 - 40, header_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Combine header with visualization
                frame_with_viz = np.vstack((header, frame_with_viz))
            
            # Calculate and display FPS
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            if len(processing_times) > 30:  # Keep only recent frames for FPS calculation
                processing_times.pop(0)
            
            # Calculate average FPS
            avg_processing_time = sum(processing_times) / len(processing_times)
            processing_fps = 1.0 / max(0.001, avg_processing_time)
            
            # Add recording indicator and FPS
            elapsed_time = time.time() - recording_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Skip overlay text for side-by-side view
            if current_viz_mode != 3:
                # Display recording time 
                cv2.putText(
                    frame_with_viz,
                    f"REC {time_text}",
                    (frame_width - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Red color
                    2
                )
                
                # Display FPS
                cv2.putText(
                    frame_with_viz,
                    f"FPS: {processing_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Red color
                    2
                )
                
                # Display model name and visualization mode
                cv2.putText(
                    frame_with_viz,
                    f"DINOv2 - Mode: {VIZ_MODES[current_viz_mode]}",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White color
                    2
                )
                
                # Display additional info based on current mode
                if current_viz_mode == 0:  # DEPTH
                    cv2.putText(
                        frame_with_viz,
                        f"Colormap: {depth_colormap_names[current_depth_colormap]} (Press 'c' to change)",
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),  # Light gray
                        1
                    )
                    
                    # Show depth estimation status
                    status_text = "PCA initialized" if depth_estimator.is_fitted else f"Initializing ({len(depth_estimator.feature_buffer)}/{depth_estimator.buffer_size})"
                    cv2.putText(
                        frame_with_viz,
                        f"Depth estimation: {status_text}",
                        (10, frame_height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )
                elif current_viz_mode == 1:  # SEGMENTATION
                    status_text = "Initialized" if segmenter.is_fitted else f"Initializing ({len(segmenter.feature_buffer)}/{segmenter.buffer_size})"
                    cv2.putText(
                        frame_with_viz,
                        f"Segmentation: {status_text}",
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),  # Light gray
                        1
                    )
                elif current_viz_mode == 2:  # FEATURES
                    status_text = "PCA initialized" if feature_visualizer.is_fitted else f"Initializing ({len(feature_visualizer.feature_buffer)}/{feature_visualizer.buffer_size})"
                    cv2.putText(
                        frame_with_viz,
                        f"Feature visualization: {status_text}",
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),  # Light gray
                        1
                    )
                
                # Display controls
                cv2.putText(
                    frame_with_viz,
                    "Controls: 'v' - change viz mode, 'c' - change colormap, 'q' - quit",
                    (10, frame_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),  # Light gray
                    1
                )
            
            # Write the frame to the output video
            if current_viz_mode == 3:
                # For side-by-side view, resize to match the original frame dimensions
                # This ensures the video writer gets correctly sized frames
                side_by_side_frame = cv2.resize(
                    frame_with_viz[header_height:], 
                    (frame_width, frame_height)
                )
                out.write(side_by_side_frame)
            else:
                out.write(frame_with_viz)
            
            # Display the result
            cv2.imshow('DINOv2 Perception', frame_with_viz)
            
            # Counter for frames processed
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Change colormap for depth visualization
                current_depth_colormap = (current_depth_colormap + 1) % len(depth_colormaps)
                print(f"Switched to depth colormap: {depth_colormap_names[current_depth_colormap]}")
            elif key == ord('v'):
                # Change visualization mode
                current_viz_mode = (current_viz_mode + 1) % len(VIZ_MODES)
                print(f"Switched to visualization mode: {VIZ_MODES[current_viz_mode]}")
    
    except KeyboardInterrupt:
        print("Processing stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            recording_duration = time.time() - recording_start_time
            print(f"Recording completed: {frame_count} frames processed in {recording_duration:.2f} seconds")
            if frame_count > 0:
                print(f"Average FPS: {frame_count / recording_duration:.2f}")
            else:
                print("Average FPS: 0.00")
            print(f"Video saved to: {output_filename}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Check if required libraries are available
    try:
        import torch
        import torchvision
        import sklearn
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"scikit-learn version: {sklearn.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install torch torchvision opencv-python numpy matplotlib pillow scikit-learn tqdm")
        exit(1)
    
    # Main execution
    try:
        print("Starting DINOv2 perception system...")
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf this is the first time running DINOv2, you might need additional dependencies:")
        print("pip install timm scikit-learn tqdm") 