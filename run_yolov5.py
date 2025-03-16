import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO  # Updated import for the new API

def main():
    """
    Main function to run real-time object detection with YOLOv5 on webcam feed
    and save the processed video to a file
    """
    # Step 1: Load the pre-trained YOLOv5 model using the ultralytics API
    print("Loading pre-trained YOLOv5 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load YOLOv5 model using the new API
    model = YOLO('yolov5s.pt')  # Load the official YOLOv5s model
    
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
    output_filename = os.path.join(output_dir, f"yolov5_detection_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Define threshold for detection confidence
    detection_threshold = 0.4
    
    print("Starting real-time detection and recording. Press 'q' to quit.")
    
    frame_count = 0
    recording_start_time = time.time()
    
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
            
            # Step 4 & 5: Run inference with YOLOv5 using the new API
            results = model(frame, conf=detection_threshold)
            
            # Step 6 & 7: Process results and draw bounding boxes
            # Using the new API results format which is different from torch hub
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            processing_fps = 1.0 / (time.time() - start_time)
            
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
            
            # Display FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {processing_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display model name
            cv2.putText(
                annotated_frame,
                "YOLOv5",
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White color
                2
            )
            
            # Step 8: Write the frame to the output video
            out.write(annotated_frame)
            
            # Step 9: Display the result
            cv2.imshow('YOLOv5 Object Detection (Recording)', annotated_frame)
            
            # Counter for frames processed
            frame_count += 1
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
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
        print(f"Average FPS: {frame_count / recording_duration:.2f}")
        print(f"Video saved to: {output_filename}")

if __name__ == "__main__":
    # Check if YOLOv5 dependencies are available
    try:
        import torch
        from ultralytics import YOLO
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Ultralytics version: {YOLO.__version__}")
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install torch ultralytics opencv-python numpy")
        exit(1)
    except AttributeError:
        print("Note: Unable to determine ultralytics version, but it's installed.")
    
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you encounter issues, try reinstalling ultralytics:")
        print("pip install -U ultralytics") 