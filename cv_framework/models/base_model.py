"""
Base class for all vision models in the framework.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import torch


class VisionModel:
    """
    Base class for all vision models, providing common functionality for 
    camera handling, video recording, and visualization.
    """
    
    def __init__(self, model_name, device=None):
        """
        Initialize the vision model.
        
        Args:
            model_name (str): Name of the model
            device (str, optional): Device to run the model on ('cpu', 'cuda:0', etc.)
        """
        self.model_name = model_name
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Model-specific attributes (to be set by subclasses)
        self.model = None
        self.confidence_threshold = 0.5
        
        # Camera and video attributes
        self.cap = None
        self.out = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.recording_start_time = None
        self.frame_count = 0
        
        # Output directory
        self.output_dir = 'output_videos'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_model(self):
        """
        Load the model. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for model input. To be implemented by subclasses.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            Preprocessed input for model
        """
        raise NotImplementedError("Subclasses must implement preprocess_frame()")
    
    def predict(self, frame):
        """
        Run inference on a frame. To be implemented by subclasses.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def visualize_predictions(self, frame, predictions):
        """
        Draw predictions on a frame. To be implemented by subclasses.
        
        Args:
            frame (numpy.ndarray): Input frame
            predictions: Model predictions
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        raise NotImplementedError("Subclasses must implement visualize_predictions()")
    
    def setup_camera(self, camera_idx=0):
        """
        Initialize the camera.
        
        Args:
            camera_idx (int): Camera index
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Initializing camera (index: {camera_idx})...")
        
        # Try different camera indices if the first one fails
        for idx in [camera_idx, 0, 1]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                print(f"Successfully opened camera at index {idx}")
                break
        
        if not self.cap.isOpened():
            print("Error: Could not open camera. Please check if:")
            print("1. Your webcam is properly connected")
            print("2. You have the necessary permissions")
            print("3. If using WSL, make sure you have:")
            print("   - Installed v4l-utils: sudo apt install v4l-utils")
            print("   - Created symbolic link: sudo ln -s /dev/video0 /dev/video1")
            print("   - Updated WSL: wsl --update")
            return False
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        return True
    
    def setup_video_writer(self, suffix=None):
        """
        Initialize the video writer.
        
        Args:
            suffix (str, optional): Suffix to add to the filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            filename = f"{self.model_name}_{suffix}_{timestamp}.mp4"
        else:
            filename = f"{self.model_name}_{timestamp}.mp4"
        
        output_filename = os.path.join(self.output_dir, filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        self.out = cv2.VideoWriter(output_filename, fourcc, self.fps, 
                                  (self.frame_width, self.frame_height))
        
        if not self.out.isOpened():
            print("Error: Could not create video writer.")
            return False
        
        print(f"Recording video to: {output_filename}")
        return True
    
    def add_metadata_to_frame(self, frame, processing_fps):
        """
        Add metadata to frame (FPS, recording time, etc.)
        
        Args:
            frame (numpy.ndarray): Input frame
            processing_fps (float): Processing FPS
            
        Returns:
            numpy.ndarray: Frame with metadata
        """
        # Add recording time
        elapsed_time = time.time() - self.recording_start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        hours, minutes = divmod(minutes, 60)
        time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Display recording time
        cv2.putText(
            frame,
            f"REC {time_text}",
            (self.frame_width - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red color
            2
        )
        
        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {processing_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red color
            2
        )
        
        # Display model name
        cv2.putText(
            frame,
            self.model_name,
            (10, self.frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White color
            2
        )
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        start_time = time.time()
        
        # Preprocess frame
        inputs = self.preprocess_frame(frame)
        
        # Run inference
        predictions = self.predict(inputs)
        
        # Visualize predictions
        frame_with_predictions = self.visualize_predictions(frame.copy(), predictions)
        
        # Calculate FPS
        processing_fps = 1.0 / max(0.001, (time.time() - start_time))
        
        # Add metadata
        result_frame = self.add_metadata_to_frame(frame_with_predictions, processing_fps)
        
        return result_frame
    
    def run(self, camera_idx=0):
        """
        Run the model on a video stream.
        
        Args:
            camera_idx (int): Camera index
            
        Returns:
            None
        """
        # Load model
        self.load_model()
        
        # Setup camera
        if not self.setup_camera(camera_idx):
            return
        
        # Setup video writer
        if not self.setup_video_writer():
            self.cap.release()
            return
        
        # Initialize timing
        self.recording_start_time = time.time()
        self.frame_count = 0
        
        print(f"Starting real-time processing with {self.model_name}. Press 'q' to quit.")
        
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
                self.out.write(result_frame)
                
                # Display result
                cv2.imshow(f'{self.model_name} (Recording)', result_frame)
                
                # Counter for frames processed
                self.frame_count += 1
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Processing stopped by user")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            recording_duration = time.time() - self.recording_start_time
            print(f"Recording completed: {self.frame_count} frames processed in {recording_duration:.2f} seconds")
            if self.frame_count > 0:
                print(f"Average FPS: {self.frame_count / recording_duration:.2f}")
            print(f"Video saved to: {self.output_dir}")

    def set_confidence_threshold(self, threshold):
        """
        Set the confidence threshold for predictions.
        
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            None
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to: {self.confidence_threshold:.2f}")
    
    def cleanup(self):
        """
        Clean up resources.
        
        Returns:
            None
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        if self.out is not None and self.out.isOpened():
            self.out.release()
        
        cv2.destroyAllWindows() 