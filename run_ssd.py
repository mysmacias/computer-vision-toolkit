import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
import time
import os
from datetime import datetime

def main():
    """
    Main function to run real-time object detection with SSD300 on webcam feed
    and save the processed video to a file
    """
    # Step 1: Load the pre-trained SSD model
    print("Loading pre-trained SSD300 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ssd300_vgg16(pretrained=True)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # COCO dataset class labels that the model was trained on
    COCO_INSTANCE_CATEGORY_NAMES = [
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
    output_filename = os.path.join(output_dir, f"ssd300_detection_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Define threshold for detection confidence
    detection_threshold = 0.5
    
    # Transform to convert OpenCV BGR image to RGB tensor for PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
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
            
            # Create a copy of the frame for saving
            frame_with_detections = frame.copy()
            
            # Step 4: Transform the image for the model
            img_tensor = transform(frame).to(device)
            
            # Step 5: Run inference
            with torch.no_grad():  # No need to track gradients for inference
                predictions = model([img_tensor])
            
            # Step 6: Process the predictions
            pred_boxes = predictions[0]['boxes'].cpu().numpy().astype(np.int32)
            pred_scores = predictions[0]['scores'].cpu().numpy()
            pred_labels = predictions[0]['labels'].cpu().numpy()
            
            # Step 7: Draw bounding boxes and labels on the frame
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score >= detection_threshold:
                    # Draw bounding box
                    cv2.rectangle(
                        frame_with_detections,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (0, 255, 0),  # Green color
                        2  # Line thickness
                    )
                    
                    # Draw label and score
                    label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                    label_text = f"{label_name}: {score:.2f}"
                    cv2.putText(
                        frame_with_detections,
                        label_text,
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # Font scale
                        (255, 255, 255),  # White color
                        2  # Line thickness
                    )
            
            # Calculate and display FPS
            processing_fps = 1.0 / (time.time() - start_time)
            
            # Add recording indicator and FPS
            elapsed_time = time.time() - recording_start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Display recording time
            cv2.putText(
                frame_with_detections,
                f"REC {time_text}",
                (frame_width - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Display FPS
            cv2.putText(
                frame_with_detections,
                f"FPS: {processing_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color
                2
            )
            
            # Step 8: Write the frame to the output video
            out.write(frame_with_detections)
            
            # Step 9: Display the result
            cv2.imshow('SSD300 Object Detection (Recording)', frame_with_detections)
            
            # Counter for frames processed
            frame_count += 1
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Detection stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
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
    # Check if PyTorch and required libraries are available
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
        print("pip install torch torchvision opencv-python numpy")
        exit(1)
    
    main() 