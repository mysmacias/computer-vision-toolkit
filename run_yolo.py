import cv2
import torch
import numpy as np
import time
import os
import argparse
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import sys
import tqdm

# Define available YOLO models
YOLO_MODELS = {
    # YOLOv5 models
    "yolov5n": {"type": "v5", "task": "detect", "description": "YOLOv5 Nano - smallest and fastest"},
    "yolov5s": {"type": "v5", "task": "detect", "description": "YOLOv5 Small - good balance of speed and accuracy"},
    "yolov5m": {"type": "v5", "task": "detect", "description": "YOLOv5 Medium - more accurate but slower"},
    "yolov5l": {"type": "v5", "task": "detect", "description": "YOLOv5 Large - high accuracy, slower speed"},
    "yolov5x": {"type": "v5", "task": "detect", "description": "YOLOv5 XLarge - highest accuracy, slowest speed"},
    
    # YOLOv8 detection models
    "yolov8n": {"type": "v8", "task": "detect", "description": "YOLOv8 Nano - fastest"},
    "yolov8s": {"type": "v8", "task": "detect", "description": "YOLOv8 Small - balanced"},
    "yolov8m": {"type": "v8", "task": "detect", "description": "YOLOv8 Medium - better accuracy"},
    "yolov8l": {"type": "v8", "task": "detect", "description": "YOLOv8 Large - high accuracy"},
    "yolov8x": {"type": "v8", "task": "detect", "description": "YOLOv8 XLarge - highest accuracy"},
    
    # YOLOv8 segmentation models
    "yolov8n-seg": {"type": "v8", "task": "segment", "description": "YOLOv8 Nano with segmentation"},
    "yolov8s-seg": {"type": "v8", "task": "segment", "description": "YOLOv8 Small with segmentation"},
    "yolov8m-seg": {"type": "v8", "task": "segment", "description": "YOLOv8 Medium with segmentation"},
    "yolov8l-seg": {"type": "v8", "task": "segment", "description": "YOLOv8 Large with segmentation"},
    "yolov8x-seg": {"type": "v8", "task": "segment", "description": "YOLOv8 XLarge with segmentation"},
    
    # YOLOv8 pose models
    "yolov8n-pose": {"type": "v8", "task": "pose", "description": "YOLOv8 Nano with pose estimation"},
    "yolov8s-pose": {"type": "v8", "task": "pose", "description": "YOLOv8 Small with pose estimation"},
    "yolov8m-pose": {"type": "v8", "task": "pose", "description": "YOLOv8 Medium with pose estimation"},
    "yolov8l-pose": {"type": "v8", "task": "pose", "description": "YOLOv8 Large with pose estimation"},
    "yolov8x-pose": {"type": "v8", "task": "pose", "description": "YOLOv8 XLarge with pose estimation"},
}

def list_available_models():
    """Print a formatted list of available YOLO models"""
    print("\nAvailable YOLO Models:")
    print("=" * 80)
    print(f"{'Model Name':<15} {'Type':<8} {'Task':<10} {'Description':<40}")
    print("-" * 80)
    
    for model_name, info in YOLO_MODELS.items():
        print(f"{model_name:<15} {info['type']:<8} {info['task']:<10} {info['description']:<40}")
    print("=" * 80)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run YOLO object detection on webcam feed")
    parser.add_argument("-m", "--model", default="yolov8s", choices=YOLO_MODELS.keys(),
                       help="YOLO model to use (default: yolov8s)")
    parser.add_argument("-t", "--threshold", type=float, default=0.4,
                       help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("-d", "--device", default="",
                       help="Device to use (e.g., 'cpu', 'cuda:0', empty for automatic selection)")
    parser.add_argument("-c", "--camera", type=int, default=0,
                       help="Camera index to use (default: 0)")
    parser.add_argument("-l", "--list", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        exit(0)
        
    return args

def load_yolo_model(model_name, device=None):
    """
    Load a YOLO model with error handling and automatic downloading.
    
    Args:
        model_name (str): Name of the YOLO model to load (e.g., 'yolov8s')
        device (str, optional): Device to load the model on
        
    Returns:
        YOLO: Loaded YOLO model
        
    Raises:
        RuntimeError: If the model cannot be loaded or downloaded
    """
    model_path = f"{model_name}.pt"
    
    # Check if model exists in the current directory or ultralytics cache
    cached_model = None
    ultralytics_dir = os.path.expanduser("~/.cache/ultralytics/")
    if os.path.exists(model_path):
        cached_model = model_path
        print(f"Using model found in current directory: {model_path}")
    elif os.path.exists(os.path.join(ultralytics_dir, "models", model_path)):
        cached_model = os.path.join(ultralytics_dir, "models", model_path)
        print(f"Using cached model: {cached_model}")
    
    try:
        if cached_model:
            print(f"Loading {model_name} model...")
            model = YOLO(cached_model)
            print(f"Model loaded successfully.")
            return model
        else:
            print(f"Model {model_name}.pt not found. Attempting to download...")
            # Try to download the model
            model = YOLO(model_name)
            print(f"Model {model_name} downloaded and loaded successfully.")
            return model
            
    except Exception as e:
        print(f"\nError loading model {model_name}: {e}")
        
        # Check if this might be a connection issue
        if "Unable to download" in str(e) or "connect" in str(e).lower():
            print("\nNetwork error detected. Check your internet connection.")
            print("If you're behind a proxy, make sure it's properly configured.")
        
        # Check if this is a model name issue
        if "Model not found" in str(e) or "not found in the Ultralytics" in str(e):
            print(f"\nModel '{model_name}' does not exist or is not available for download.")
            print("Available models in Ultralytics:")
            for model_key in YOLO_MODELS.keys():
                print(f"  - {model_key}")
        
        # Raise an exception with helpful message
        print("\nWould you like to try a different model? (y/n)")
        response = input("> ")
        if response.lower() in ["y", "yes"]:
            print("\nAvailable models:")
            # Include all available models
            available_models = list(YOLO_MODELS.keys())
            for i, model_key in enumerate(available_models):
                print(f"{i+1}. {model_key} - {YOLO_MODELS[model_key]['description']}")
            
            print("\nEnter model number:")
            try:
                model_idx = int(input("> ")) - 1
                if 0 <= model_idx < len(available_models):
                    new_model_name = available_models[model_idx]
                    print(f"Trying to load {new_model_name} instead...")
                    return load_yolo_model(new_model_name, device)
                else:
                    print("Invalid selection. Exiting.")
            except ValueError:
                print("Invalid input. Exiting.")
        
        print("\nExiting due to model loading failure.")
        sys.exit(1)

def main():
    """
    Main function to run real-time object detection with YOLO on webcam feed
    and save the processed video to a file
    """
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model
    detection_threshold = args.threshold
    camera_idx = args.camera
    
    # Step 1: Load the specified YOLO model using the ultralytics API
    print(f"Loading YOLO model: {model_name}...")
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load the selected YOLO model with error handling
    model = load_yolo_model(model_name, device)
    
    # Get model info for display
    model_info = YOLO_MODELS[model_name]
    model_type = model_info["type"]
    model_task = model_info["task"]
    
    # Step 2: Initialize webcam
    print(f"Initializing webcam (index: {camera_idx})...")
    cap = cv2.VideoCapture(camera_idx)  # Use specified camera index
    
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
    
    # Generate output filename with timestamp and model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"{model_name}_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    print(f"Detection threshold: {detection_threshold}")
    print(f"Model: {model_name} ({model_type}, {model_task})")
    
    # Runtime variables
    frame_count = 0
    recording_start_time = time.time()
    current_model = model_name
    
    # Show available keyboard shortcuts
    print("\nKeyboard shortcuts:")
    print("  'q' - Quit")
    print("  '+' - Increase detection threshold")
    print("  '-' - Decrease detection threshold")
    print("  'm' - Show model selection menu")
    
    try:
        while True:
            start_time = time.time()
            
            # Step 3: Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Create a copy of the frame for showing and saving
            frame_with_detections = frame.copy()
            
            # Step 4 & 5: Run inference with YOLO
            results = model(frame, conf=detection_threshold)
            
            # Step 6 & 7: Process results and draw bounding boxes
            # The plot method handles different result types automatically
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            processing_time = time.time() - start_time
            processing_fps = 1.0 / max(0.001, processing_time)
            
            # Add recording indicator and FPS
            elapsed_time = time.time() - recording_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Display recording time
            cv2.putText(
                annotated_frame,
                f"REC {time_text}",
                (frame_width - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display FPS and threshold
            cv2.putText(
                annotated_frame,
                f"FPS: {processing_fps:.1f} | Conf: {detection_threshold:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display model info
            cv2.putText(
                annotated_frame,
                f"{current_model} ({model_type}, {model_task})",
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White color
                2
            )
            
            # Controls info
            cv2.putText(
                annotated_frame,
                "q: quit | +/-: threshold | m: model menu",
                (10, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),  # Light gray
                1
            )
            
            # Step 8: Write the frame to the output video
            out.write(annotated_frame)
            
            # Step 9: Display the result
            cv2.imshow('YOLO Detection (Recording)', annotated_frame)
            
            # Counter for frames processed
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Break the loop if 'q' is pressed
            if key == ord('q'):
                break
            # Increase threshold if '+' is pressed
            elif key == ord('+') or key == ord('='):
                detection_threshold = min(0.95, detection_threshold + 0.05)
                print(f"Increased threshold to {detection_threshold:.2f}")
            # Decrease threshold if '-' is pressed
            elif key == ord('-') or key == ord('_'):
                detection_threshold = max(0.05, detection_threshold - 0.05)
                print(f"Decreased threshold to {detection_threshold:.2f}")
            # Show model selection menu if 'm' is pressed
            elif key == ord('m'):
                # Pause capturing
                print("\nModel Selection Menu:")
                list_available_models()
                print("\nCurrent model: " + current_model)
                print("Enter model name to switch, or press Enter to continue with current model:")
                
                # Temporarily close the video window to avoid lockups
                cv2.destroyAllWindows()
                
                # Get user input
                new_model_name = input("> ").strip()
                
                # If user entered a valid model name, switch to it
                if new_model_name in YOLO_MODELS:
                    # Close current video writer
                    out.release()
                    
                    # Update model
                    model_name = new_model_name
                    current_model = model_name
                    
                    # Try to load the new model with error handling
                    try:
                        print(f"Switching to model: {model_name}")
                        model = load_yolo_model(model_name, device)
                        
                        # Update model info
                        model_info = YOLO_MODELS[model_name]
                        model_type = model_info["type"]
                        model_task = model_info["task"]
                        
                        # Create new video file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = os.path.join(output_dir, f"{model_name}_{timestamp}.mp4")
                        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                        
                        print(f"Recording to new file: {output_filename}")
                        
                        # Reset timing
                        recording_start_time = time.time()
                        frame_count = 0
                    except Exception as e:
                        print(f"Error switching models: {e}")
                        print("Continuing with current model: " + current_model)
                else:
                    print("Continuing with current model: " + current_model)
                
                # Re-open the window
                cv2.namedWindow('YOLO Detection (Recording)', cv2.WINDOW_NORMAL)
    
    except KeyboardInterrupt:
        print("Detection stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Step 10: Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        recording_duration = time.time() - recording_start_time
        print(f"Recording completed: {frame_count} frames processed in {recording_duration:.2f} seconds")
        if frame_count > 0:
            print(f"Average FPS: {frame_count / recording_duration:.2f}")
        print(f"Video saved to: {output_filename}")

if __name__ == "__main__":
    # Check if YOLO dependencies are available
    try:
        import torch
        from ultralytics import YOLO
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        try:
            print(f"Ultralytics version: {YOLO.__version__}")
        except AttributeError:
            print("Note: Unable to determine ultralytics version, but it's installed.")
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install torch ultralytics opencv-python numpy")
        exit(1)
    
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you encounter issues, try reinstalling ultralytics:")
        print("pip install -U ultralytics") 