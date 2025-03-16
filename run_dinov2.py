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

def main():
    """
    Main function to run real-time feature extraction with Meta's DINOv2 on webcam feed,
    visualize feature activations, and save the processed video to a file
    """
    # Step 1: Load the pre-trained DINOv2 model
    print("Loading pre-trained DINOv2 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # For older PyTorch versions, suppress warnings
    warnings.filterwarnings("ignore", message=".*xFormers.*")
    
    try:
        # Try loading the small model variant which is more compatible
        print("Loading DINOv2 ViT-S/14 model...")
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        patch_size = 14  # Default for ViT-S/14
        dinov2 = dinov2.to(device)
        dinov2.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        print("Trying DINOv2 ViT-B/14 model...")
        try:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            patch_size = 14  # Default for ViT-B/14
            dinov2 = dinov2.to(device)
            dinov2.eval()
        except Exception as e:
            print(f"Error loading alternative model: {e}")
            print("Using torchvision's ViT model as fallback...")
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            dinov2 = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            patch_size = 16  # Default for ViT-B/16
            dinov2 = dinov2.to(device)
            dinov2.eval()
    
    # Step 2: Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed
    
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
    output_filename = os.path.join(output_dir, f"dinov2_features_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Define transform for DINOv2
    transform = T.Compose([
        T.Resize(224),  # Resize to match ViT input size
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # For saliency map generation
    transform_raw = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    # Keep track of visualization mode
    VIZ_MODES = {
        0: "FEATURE_MAP",
        1: "ATTENTION",
        2: "SALIENCY"
    }
    current_viz_mode = 0
    
    def create_simple_saliency_map(model, img_tensor):
        """Generate a simple gradient-based saliency map"""
        img_tensor.requires_grad_(True)
        
        try:
            with torch.enable_grad():
                # Forward pass
                outputs = model(img_tensor.unsqueeze(0))
                
                if isinstance(outputs, dict):
                    outputs = outputs['logits'] if 'logits' in outputs else list(outputs.values())[0]
                    
                # Get predicted class
                pred_class = outputs.argmax(dim=1)
                
                # Backprop to input
                model.zero_grad()
                outputs[0, pred_class].backward()
                
                # Get gradients
                gradients = img_tensor.grad.abs()
                
                # Max over channels
                saliency, _ = torch.max(gradients, dim=0)
                
                # To numpy
                return saliency.detach().cpu().numpy()
                
        except Exception as e:
            print(f"Error creating saliency map: {e}")
            # Return a random saliency map
            return np.random.rand(224, 224)
    
    # Alternative feature visualization approach
    def extract_features_and_attentions(img_tensor, img_raw=None, mode="FEATURE_MAP"):
        try:
            # Return different visualizations based on mode
            if mode == "SALIENCY" and img_raw is not None:
                # Simple saliency map
                saliency = create_simple_saliency_map(dinov2, img_raw)
                return np.expand_dims(saliency, 0)
                
            # Get features for other modes
            with torch.no_grad():
                # Different strategies for feature extraction
                if hasattr(dinov2, 'get_intermediate_layers'):
                    # DINOv2 method
                    features = dinov2.get_intermediate_layers(img_tensor.unsqueeze(0), n=1)[0]
                elif hasattr(dinov2, 'forward_features'):
                    # Standard ViT forward_features approach
                    features = dinov2.forward_features(img_tensor.unsqueeze(0))
                    # Extract relevant tensor based on type
                    if isinstance(features, dict):
                        if 'last_hidden_state' in features:
                            features = features['last_hidden_state']
                        elif 'x' in features:
                            features = features['x']
                        else:
                            # Take the first value if keys are unknown
                            features = list(features.values())[0]
                else:
                    # Last resort: just run the model and use its output
                    features = dinov2(img_tensor.unsqueeze(0))
                    if isinstance(features, dict):
                        features = list(features.values())[0]
                
                # At this point, features should be a tensor we can work with
                
                # For attention map visualization
                if mode == "ATTENTION":
                    # If features have CLS token, use it to get attention
                    if features.dim() == 3 and features.shape[1] > 1:
                        cls_token = features[:, 0:1]
                        patch_tokens = features[:, 1:]
                        
                        # Calculate similarity as a form of attention
                        attn = torch.matmul(cls_token, patch_tokens.transpose(1, 2))
                        attn = torch.nn.functional.softmax(attn, dim=-1)
                        
                        # Get the attention map
                        attn_map = attn[0, 0].cpu().numpy()
                        
                        # Try to reshape attention to approximate spatial dimensions
                        n_patches = attn_map.shape[0]
                        h = w = int(math.sqrt(n_patches))
                        
                        if h * w == n_patches:  # Perfect square
                            attn_2d = attn_map.reshape(h, w)
                        else:
                            # Pad to next perfect square if needed
                            next_square = (h + 1) ** 2
                            padded = np.zeros(next_square)
                            padded[:n_patches] = attn_map
                            attn_2d = padded.reshape(h + 1, h + 1)
                            
                        return np.expand_dims(attn_2d, 0)
                
                # Default mode: Feature map
                # Average over feature dimension
                if features.dim() > 2:
                    feature_map = features.mean(dim=-1)
                else:
                    feature_map = features
                
                # Create a 2D feature map based on model architecture
                if feature_map.dim() == 2:  # [B, N]
                    # Skip CLS token if present (first token)
                    if feature_map.shape[1] > 1:
                        patches = feature_map[:, 1:] if feature_map.shape[1] > 1 else feature_map
                    else:
                        patches = feature_map
                    
                    # Convert to numpy
                    patches_np = patches.cpu().numpy()[0]
                    
                    # Get number of patches
                    n_patches = patches_np.shape[0]
                    
                    # Calculate grid dimensions based on patch count
                    # For a 224x224 image with patch_size=14, we get 16x16=256 patches
                    # For a 224x224 image with patch_size=16, we get 14x14=196 patches
                    grid_size = int(math.sqrt(n_patches))
                    
                    # If perfect square, reshape directly
                    if grid_size * grid_size == n_patches:
                        feature_map_2d = patches_np.reshape(grid_size, grid_size)
                    else:
                        # Otherwise, create a grid with the right aspect ratio
                        # and place the features into it
                        feature_map_2d = np.zeros((grid_size + 1, grid_size + 1))
                        for i in range(min(n_patches, (grid_size + 1) * (grid_size + 1))):
                            row = i // (grid_size + 1)
                            col = i % (grid_size + 1)
                            feature_map_2d[row, col] = patches_np[i]
                    
                    return np.expand_dims(feature_map_2d, 0)
                else:
                    # Already in spatial format
                    return feature_map.cpu().numpy()
                    
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            # Return a fallback feature map (random noise)
            return np.random.rand(1, 16, 16)
    
    print("Starting real-time DINOv2 feature extraction and recording. Press 'q' to quit.")
    print("Press 'v' to cycle through visualization modes (Feature Map, Attention, Saliency)")
    print("Press 'c' to cycle through colormaps")
    
    frame_count = 0
    recording_start_time = time.time()
    
    # Generate different color maps for visualization variety
    colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TURBO, cv2.COLORMAP_INFERNO]
    colormap_names = ["Jet", "Viridis", "Turbo", "Inferno"]
    current_colormap = 0
    
    # For logging - don't spam console with the same error
    last_error = None
    error_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # Step 3: Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Create a copy of the frame for saving
            frame_with_features = frame.copy()
            
            # Step 4: Transform the image for the model
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Create normalized input for the model
            img_tensor = transform(pil_image).to(device)
            
            # Create raw input for saliency map if needed
            img_raw = None
            if current_viz_mode == 2:  # Saliency mode
                img_raw = transform_raw(pil_image).to(device)
            
            # Step 5: Extract features based on current visualization mode
            feature_map = extract_features_and_attentions(
                img_tensor, img_raw, mode=VIZ_MODES[current_viz_mode]
            )
            
            # Resize feature map to match video frame using OpenCV
            resized_features = cv2.resize(
                feature_map[0], (frame_width, frame_height), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize feature map for visualization
            feature_min, feature_max = resized_features.min(), resized_features.max()
            normalized_features = (resized_features - feature_min) / (feature_max - feature_min + 1e-8)
            
            # Apply Gaussian blur to make visualization smoother
            normalized_features = cv2.GaussianBlur(normalized_features, (5, 5), 0)
            
            # Convert to heatmap using current colormap
            colormap = colormaps[current_colormap]
            heatmap = cv2.applyColorMap((normalized_features * 255).astype(np.uint8), colormap)
            
            # Overlay heatmap on original frame
            alpha = 0.7  # Transparency factor
            frame_with_features = cv2.addWeighted(frame_with_features, 1-alpha, heatmap, alpha, 0)
            
            # Calculate and display FPS
            processing_fps = 1.0 / max(0.001, time.time() - start_time)  # Prevent division by zero
            
            # Add recording indicator and FPS
            elapsed_time = time.time() - recording_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Display recording time
            cv2.putText(
                frame_with_features,
                f"REC {time_text}",
                (frame_width - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display FPS
            cv2.putText(
                frame_with_features,
                f"FPS: {processing_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display model name and visualization mode
            cv2.putText(
                frame_with_features,
                f"DINOv2 - Mode: {VIZ_MODES[current_viz_mode]}",
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White color
                2
            )
            
            # Display colormap name 
            cv2.putText(
                frame_with_features,
                f"Colormap: {colormap_names[current_colormap]} (Press 'c' to change)",
                (10, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),  # Light gray
                1
            )
            
            # Display controls
            cv2.putText(
                frame_with_features,
                "Controls: 'v' - change viz mode, 'q' - quit",
                (10, frame_height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),  # Light gray
                1
            )
            
            # Write the frame to the output video
            out.write(frame_with_features)
            
            # Display the result
            cv2.imshow('DINOv2 Vision Features', frame_with_features)
            
            # Counter for frames processed
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Change colormap on 'c' key press
                current_colormap = (current_colormap + 1) % len(colormaps)
                print(f"Switched to colormap: {colormap_names[current_colormap]}")
            elif key == ord('v'):
                # Change visualization mode on 'v' key press
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
            print(f"Average FPS: {frame_count / recording_duration:.2f}")
            print(f"Video saved to: {output_filename}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Check if required libraries are available
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install torch torchvision opencv-python numpy matplotlib pillow")
        exit(1)
    
    # Main execution
    try:
        print("Loading DINOv2 model...")
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf this is the first time running DINOv2, you might need additional dependencies:")
        print("pip install timm") 