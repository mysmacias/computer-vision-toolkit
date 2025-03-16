import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

def main():
    """
    Main function to run real-time segmentation with Meta's Segment Anything Model (SAM)
    on webcam feed and save the processed video to a file
    """
    # Step 1: Load the pre-trained SAM model
    print("Loading pre-trained Segment Anything Model (SAM)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Choose the model type - we'll use the lightweight mobile version for real-time performance
    model_type = "vit_b"  # Options: vit_h (largest), vit_l (large), vit_b (base/medium)
    
    # You need to download the model weights first
    # For vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    checkpoint_path = "sam_vit_b_01ec64.pth"
    
    # Check if the model file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please download the model weights from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return
    
    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Create the predictor for interactive segmentation
    predictor = SamPredictor(sam)
    
    # Create the automatic mask generator for automatic segmentation
    # Using lighter settings for real-time performance
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,  # Lower for better performance (default is 32)
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,  # Reduced for speed
        crop_n_points_downscale_factor=2,  # Reduced for speed
        min_mask_region_area=100,  # Smaller objects will be filtered out
    )
    
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
    output_filename = os.path.join(output_dir, f"sam_segmentation_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Initialize variables for point selection
    # For interactive prompt, we'll place points at the center for this demo
    input_point = np.array([[frame_width // 2, frame_height // 2]])
    input_label = np.array([1])  # 1 for foreground
    
    print("Starting real-time SAM segmentation and recording. Press 'q' to quit.")
    print("Click on the image to set a point for segmentation. Press 'a' for automatic segmentation.")
    
    frame_count = 0
    recording_start_time = time.time()
    
    # Variables for interactive mode
    point_mode = False
    auto_mode = True
    points = []
    labels = []
    
    # Color map for visualizing masks
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, labels, point_mode, auto_mode
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)  # Foreground point
            point_mode = True
            auto_mode = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)  # Background point
            point_mode = True
            auto_mode = False
    
    cv2.namedWindow('SAM Segmentation (Recording)')
    cv2.setMouseCallback('SAM Segmentation (Recording)', mouse_callback)
    
    try:
        # For keeping track of frame processing
        skip_frames = 0
        max_skip_frames = 5  # Process every 5 frames for better performance
        masks = None
        
        while True:
            start_time = time.time()
            
            # Step 3: Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Create a copy of the frame for saving
            frame_with_masks = frame.copy()
            
            # Skip frames for performance
            skip_frames += 1
            if skip_frames >= max_skip_frames or masks is None or point_mode:
                skip_frames = 0
                
                # Step 4: Prepare the image for SAM
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Step 5: Generate masks
                if auto_mode:
                    try:
                        # Automatic mask generation - this can be slower
                        print("Generating automatic masks...")
                        masks = mask_generator.generate(rgb_frame)
                        print(f"Generated {len(masks)} masks")
                    except Exception as e:
                        print(f"Error in automatic mask generation: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fall back to manual mode if automatic fails
                        auto_mode = False
                        if not points:  # Add a center point if none exists
                            points.append([frame_width // 2, frame_height // 2])
                            labels.append(1)
                            point_mode = True
                elif point_mode and points:
                    # Interactive point-based mask generation
                    # Set the image for the predictor
                    predictor.set_image(rgb_frame)
                    
                    input_points = np.array(points)
                    input_labels = np.array(labels)
                    masks, scores, logits = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True,
                    )
                    # Use the mask with highest score
                    best_idx = np.argmax(scores)
                    masks = [{"segmentation": masks[best_idx], "score": scores[best_idx]}]
                    point_mode = False
            
            # Step 6: Visualize masks
            if masks is not None:
                if auto_mode:
                    # Visualize all automatic masks with random colors
                    for i, mask_data in enumerate(masks):
                        mask = mask_data["segmentation"]
                        score = mask_data.get("score", 0.0)
                        
                        color = colors[i % len(colors)]
                        # Apply a color overlay for the mask
                        mask_area = np.zeros_like(frame_with_masks)
                        mask_area[:, :] = color
                        frame_with_masks = np.where(
                            np.expand_dims(mask, 2), 
                            cv2.addWeighted(frame_with_masks, 0.5, mask_area, 0.5, 0),
                            frame_with_masks
                        )
                        
                        # Draw contours around the mask
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(frame_with_masks, contours, -1, color.tolist(), 2)
                        
                        # Add score text
                        if i < 10:  # Only show scores for a few masks to avoid cluttering
                            # Find center of mask
                            M = cv2.moments(mask.astype(np.uint8))
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                # Display score
                                cv2.putText(
                                    frame_with_masks,
                                    f"{score:.2f}",
                                    (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1
                                )
                else:
                    # Single mask visualization for point mode
                    mask = masks[0]["segmentation"]
                    score = masks[0].get("score", 0.0)
                    
                    # Create a colored mask overlay
                    color = (0, 255, 0)  # Green color for the mask
                    colored_mask = np.zeros_like(frame_with_masks)
                    colored_mask[:, :] = color
                    # Apply mask with semi-transparency
                    frame_with_masks = np.where(
                        np.expand_dims(mask, 2), 
                        cv2.addWeighted(frame_with_masks, 0.7, colored_mask, 0.3, 0),
                        frame_with_masks
                    )
                    # Draw contours around the mask
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(frame_with_masks, contours, -1, color, 2)
                    
                    # Display confidence score
                    cv2.putText(
                        frame_with_masks,
                        f"Score: {score:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            
            # Draw input points
            for i, point in enumerate(points):
                color = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)  # Red for foreground, blue for background
                cv2.circle(frame_with_masks, tuple(point), 5, color, -1)
            
            # Calculate and display FPS
            processing_fps = 1.0 / max(0.001, (time.time() - start_time))
            
            # Add recording indicator and FPS
            elapsed_time = time.time() - recording_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Display recording time
            cv2.putText(
                frame_with_masks,
                f"REC {time_text}",
                (frame_width - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display FPS
            cv2.putText(
                frame_with_masks,
                f"FPS: {processing_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display model name and mode
            mode_text = "Auto" if auto_mode else "Interactive"
            cv2.putText(
                frame_with_masks,
                f"SAM ({mode_text})",
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White color
                2
            )
            
            # Add usage instructions
            cv2.putText(
                frame_with_masks,
                "Left-click: foreground point, Right-click: background, 'a': auto mode, 'c': clear points",
                (10, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # Write the frame to the output video
            out.write(frame_with_masks)
            
            # Display the result
            cv2.imshow('SAM Segmentation (Recording)', frame_with_masks)
            
            # Counter for frames processed
            frame_count += 1
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_mode = True
                points = []
                labels = []
            elif key == ord('c'):
                points = []
                labels = []
                auto_mode = False
    
    except KeyboardInterrupt:
        print("Processing stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
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

if __name__ == "__main__":
    # Check if required libraries are available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install torch torchvision opencv-python numpy matplotlib pillow segment-anything")
        exit(1)
    
    try:
        print("Checking for segment-anything package...")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("segment-anything package not found. Please install it with:")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    # Note: You need to download the model weights manually
    if not os.path.exists("sam_vit_b_01ec64.pth"):
        print("Model checkpoint not found. Please download the SAM ViT-B model weights from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("and place it in the current directory.")
        exit(1)
    
    main() 