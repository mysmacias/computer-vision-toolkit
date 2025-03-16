import cv2
import torch
import numpy as np
import torchvision.transforms as T
import time
import os
from datetime import datetime

def main():
    """
    Main function to run real-time object detection with DETR (DEtection TRansformer) on webcam feed
    and save the processed video to a file
    """
    # Step 1: Load the pre-trained DETR model
    print("Loading pre-trained DETR model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load DETR model from torch hub
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # COCO class labels for DETR
    CLASSES = [
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
    output_filename = os.path.join(output_dir, f"detr_detection_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer.")
        cap.release()
        return
    
    print(f"Recording video to: {output_filename}")
    
    # Define threshold for detection confidence
    detection_threshold = 0.7  # DETR might need a higher threshold
    
    # Define the transformation for DETR
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Starting real-time detection and recording. Press 'q' to quit.")
    
    frame_count = 0
    recording_start_time = time.time()
    
    # Generate random colors for each class
    np.random.seed(42)  # for reproducible colors
    COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)
    
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
            # Convert BGR to RGB (DETR expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply the transformations
            img_tensor = transform(rgb_frame).unsqueeze(0).to(device)
            
            # Step 5: Run inference
            with torch.no_grad():
                outputs = model(img_tensor)
            
            # Step 6: Process the predictions
            # DETR output format is different from other detectors
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove no-object class
            boxes = outputs['pred_boxes'][0]
            
            # Keep only predictions with confidence above threshold
            keep = probas.max(-1).values > detection_threshold
            
            # Get scores and labels for the kept predictions
            scores, labels = probas[keep].max(-1)
            
            # Convert boxes to pixel coordinates
            boxes = boxes[keep].cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
            
            # Step 7: Draw bounding boxes and labels on the frame
            for box, label, score in zip(boxes, labels, scores):
                # DETR boxes are in format [cx, cy, w, h] and normalized
                # Convert to [x1, y1, x2, y2] format for drawing
                cx, cy, w, h = box
                x1 = int((cx - w/2) * frame_width)
                y1 = int((cy - h/2) * frame_height)
                x2 = int((cx + w/2) * frame_width)
                y2 = int((cy + h/2) * frame_height)
                
                # Ensure box coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
                # Draw bounding box with a class-specific color
                color = COLORS[label].tolist()
                cv2.rectangle(
                    frame_with_detections,
                    (x1, y1),
                    (x2, y2),
                    color,  # Use class-specific color
                    2  # Line thickness
                )
                
                # Get the class name and prepare label text
                label_id = int(label)
                class_name = CLASSES[label_id] if label_id < len(CLASSES) else f"Class {label_id}"
                label_text = f"{class_name}: {score:.2f}"
                
                # Create filled rectangle for label background
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    frame_with_detections,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    color,
                    -1  # Filled rectangle
                )
                
                # Draw label
                cv2.putText(
                    frame_with_detections,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (255, 255, 255),  # White color for text
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
            
            # Display model name
            cv2.putText(
                frame_with_detections,
                "DETR (DEtection TRansformer)",
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White color
                2
            )
            
            # Step 8: Write the frame to the output video
            out.write(frame_with_detections)
            
            # Step 9: Display the result
            cv2.imshow('DETR Object Detection (Recording)', frame_with_detections)
            
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
        print("pip install torch torchvision opencv-python numpy")
        exit(1)
    
    main() 