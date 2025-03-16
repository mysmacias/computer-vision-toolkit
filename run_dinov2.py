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
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Trying DINOv2 ViT-S/14 model as fallback...")
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                self.model = self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading alternative model: {e}")
                print("Using torchvision's ViT model as fallback...")
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                self.model = self.model.to(self.device)
                self.model.eval()
        
        # Default transforms for preprocessing
        self.transform = T.Compose([
            T.Resize(518),  # Higher resolution for better results
            T.CenterCrop(518),
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
        
        # Linear regression layer for depth estimation (trained weights would be better)
        # This is a simple stand-in that should be replaced with a proper model
        self.depth_regressor = torch.nn.Linear(self.feature_dim, 1).to(self.device)
        
        # Initialize with random weights - ideally would load pretrained weights
        # This will produce noisy but structured depth maps
        torch.nn.init.normal_(self.depth_regressor.weight, std=0.01)
        
    def estimate_depth(self, image):
        """Estimate depth from an image using DINOv2 features"""
        # Get patch features from the image
        patch_features = self.feature_extractor.extract_patch_features(image)
        
        # Apply the depth regressor to estimate depth values
        with torch.no_grad():
            # Print debug info
            batch_size, num_patches, feature_dim = patch_features.shape
            print(f"Patch features shape: [{batch_size}, {num_patches}, {feature_dim}]")
            
            # Apply depth regression
            depth_values = self.depth_regressor(patch_features)
            
            # Calculate dimensions for reshaping
            # For non-square patch grids, we need to find the closest factors
            # that will accommodate all our patches
            h = int(math.sqrt(num_patches))
            
            # Find the best width given the height to fit all patches
            w = int(math.ceil(num_patches / h))
            
            print(f"Reshaping depth values to: [{batch_size}, {h}, {w}]")
            
            # Reshape to 2D grid, possibly with padding
            # First flatten the depth values
            depth_values_flat = depth_values.reshape(batch_size, -1)
            
            # Create a padded tensor if needed
            if h * w > num_patches:
                # Pad with zeros to make it fit the h*w shape
                padded_depth = torch.zeros(batch_size, h * w, device=self.device)
                padded_depth[:, :num_patches] = depth_values_flat
                depth_map = padded_depth.view(batch_size, h, w)
            else:
                # If no padding is needed (perfect square)
                depth_map = depth_values_flat.view(batch_size, h, w)
            
            # Apply basic post-processing
            depth_map = F.interpolate(
                depth_map.unsqueeze(1), 
                size=(518, 518), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            # Normalize depth values to 0-1 range for visualization
            normalized_depth = depth_map - depth_map.min()
            normalized_depth = normalized_depth / (normalized_depth.max() + 1e-8)
            
            return normalized_depth.cpu().numpy()[0]

# Semantic segmentation model based on DINOv2 features
class DINOv2SemanticSegmenter:
    def __init__(self, feature_extractor, num_clusters=8):
        """Initialize semantic segmenter with a DINOv2 feature extractor"""
        self.feature_extractor = feature_extractor
        self.device = feature_extractor.device
        self.num_clusters = num_clusters
        
        # We'll use simple k-means for clustering features
        # In production, one would use a learned clustering/segmentation head
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        
        # Flag to track if kmeans has been fitted
        self.is_fitted = False
        
        # Store a buffer of features to fit k-means initially
        self.feature_buffer = []
        self.buffer_size = 10
        
        # Color map for visualization
        self.color_map = plt.cm.get_cmap('tab10', num_clusters)
        
    def segment_image(self, image):
        """Perform semantic segmentation using DINOv2 features and k-means clustering"""
        # Get patch features from the image
        patch_features = self.feature_extractor.extract_patch_features(image)
        
        # Get features for clustering
        features_for_clustering = patch_features.squeeze(0).cpu().numpy()
        
        # If k-means isn't fitted yet, collect features
        if not self.is_fitted:
            self.feature_buffer.append(features_for_clustering)
            if len(self.feature_buffer) >= self.buffer_size:
                print("Fitting k-means for semantic segmentation...")
                # Concatenate all features
                all_features = np.vstack(self.feature_buffer)
                # Fit k-means on collected features
                self.kmeans.fit(all_features)
                # Set fitted flag
                self.is_fitted = True
                # Clear buffer
                self.feature_buffer = []
                print("K-means fitted successfully!")
        
        # If k-means is fitted, predict clusters
        if self.is_fitted:
            # Predict cluster for each patch
            clusters = self.kmeans.predict(features_for_clustering)
            
            # Reshape to 2D grid
            num_patches = features_for_clustering.shape[0]
            h = int(math.sqrt(num_patches))
            w = int(math.ceil(num_patches / h))
            
            # Create a padded array if needed
            if h * w > num_patches:
                segmentation_map = np.zeros((h, w), dtype=np.int32)
                for i in range(num_patches):
                    row = i // w
                    col = i % w
                    segmentation_map[row, col] = clusters[i]
            else:
                segmentation_map = clusters.reshape(h, w)
            
            # Resize to original image size
            segmentation_map = cv2.resize(
                segmentation_map.astype(np.float32), 
                (518, 518), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Create colored segmentation map
            colored_segmentation = np.zeros((518, 518, 3), dtype=np.float32)
            
            for cluster_idx in range(self.num_clusters):
                # Get color for this cluster
                color = np.array(self.color_map(cluster_idx)[:3])
                # Apply color to all pixels in this cluster
                colored_segmentation[segmentation_map == cluster_idx] = color
                
            return colored_segmentation
        else:
            # Return random colored noise until k-means is fitted
            print(f"Collecting features for k-means ({len(self.feature_buffer)}/{self.buffer_size})...")
            noise = np.random.rand(518, 518, 3)
            return noise

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
        2: "SIDE_BY_SIDE"
    }
    current_viz_mode = 0
    
    # Generate colormaps for depth visualization
    depth_colormaps = [cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS]
    depth_colormap_names = ["Inferno", "Jet", "Turbo", "Viridis"]
    current_depth_colormap = 0
    
    print("Starting real-time DINOv2 perception. Press 'q' to quit.")
    print("Press 'v' to cycle through visualization modes (Depth, Segmentation, Side-by-Side)")
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
            
            # Convert to PIL Image for processing
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process the frame depending on the current visualization mode
            depth_map = None
            segmentation_map = None
            
            # Always process for the side-by-side view, or if in respective single mode
            if current_viz_mode in [0, 2]:  # DEPTH or SIDE_BY_SIDE
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
            
            if current_viz_mode in [1, 2]:  # SEGMENTATION or SIDE_BY_SIDE
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
            elif current_viz_mode == 2:  # SIDE_BY_SIDE
                # Create side by side display: [Original | Depth | Segmentation]
                vis_width = frame_width // 3
                
                # Resize the original frame and visualization maps to fit side by side
                original_resized = cv2.resize(frame, (vis_width, frame_height))
                depth_resized = cv2.resize(colored_depth, (vis_width, frame_height))
                segmentation_resized = cv2.resize(colored_segmentation, (vis_width, frame_height))
                
                # Combine horizontally
                frame_with_viz = np.hstack((original_resized, depth_resized, segmentation_resized))
            
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
            if current_viz_mode != 2:
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
                
                # Display colormap info for depth visualization
                if current_viz_mode == 0:
                    cv2.putText(
                        frame_with_viz,
                        f"Colormap: {depth_colormap_names[current_depth_colormap]} (Press 'c' to change)",
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),  # Light gray
                        1
                    )
                
                # Display segmentation info
                if current_viz_mode == 1:
                    status_text = "K-means initialized" if segmenter.is_fitted else f"Initializing ({len(segmenter.feature_buffer)}/{segmenter.buffer_size})"
                    cv2.putText(
                        frame_with_viz,
                        f"Segmentation: {status_text}",
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
                    (10, frame_height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),  # Light gray
                    1
                )
            else:
                # For side-by-side view, add labels
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
                
                # Combine header with visualization
                frame_with_viz = np.vstack((header, frame_with_viz))
            
            # Write the frame to the output video
            if current_viz_mode == 2:
                # For side-by-side view, we need to create a version without the header
                out.write(frame_with_viz[header_height:])
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