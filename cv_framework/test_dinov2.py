#!/usr/bin/env python3
"""
DINOv2 Test Script
-----------------
This script tests the DINOv2 model implementation in the Computer Vision Toolkit.
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path

# Add parent directory to system path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

def main():
    """Main test function"""
    print("Testing DINOv2 model...")
    
    # Import model classes
    try:
        from gui.model_manager import DINOv2Model
        from models.dinov2_model import DINOv2Model as BaseDINOv2Model
        print("Successfully imported model classes")
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        return False
    
    # Create and load model
    print("Creating DINOv2 model...")
    model = DINOv2Model()
    
    print("Loading DINOv2 model...")
    success = model.load()
    if not success:
        print("Failed to load DINOv2 model")
        return False
    
    print("DINOv2 model loaded successfully!")
    
    # Test with a dummy image
    print("Testing with a dummy image...")
    
    # Create a test image
    img_size = 640
    test_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Draw some shapes for better visualization
    cv2.rectangle(test_img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(test_img, (450, 450), 100, (0, 0, 255), -1)
    cv2.line(test_img, (0, 0), (img_size, img_size), (255, 0, 0), 5)
    
    # Convert to QImage for model processing
    from PySide6.QtGui import QImage
    h, w, c = test_img.shape
    q_img = QImage(test_img.data, w, h, w * c, QImage.Format_RGB888)
    
    # Process with each task
    for task in ["features", "segmentation", "depth"]:
        print(f"Testing '{task}' task...")
        
        # Set active task
        model.set_active_task(task)
        
        # Process image
        start_time = time.time()
        result_img = model.process_frame(q_img)
        processing_time = time.time() - start_time
        
        print(f"Processing took {processing_time:.2f} seconds")
        
        # Convert back to numpy array for saving
        result_width = result_img.width()
        result_height = result_img.height()
        result_bytes = result_img.constBits()
        
        # Create a buffer from the memory view
        buf = memoryview(result_bytes).tobytes()
        
        # Reshape the buffer to match the image dimensions
        if result_img.format() == QImage.Format_RGB888:
            # 3 channels (RGB)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(result_height, result_width, 3)
        else:
            # Convert to RGB888 format first
            rgb_image = result_img.convertToFormat(QImage.Format_RGB888)
            ptr = rgb_image.constBits()
            buf = memoryview(ptr).tobytes()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(result_height, result_width, 3)
        
        # Create output directory
        output_dir = current_dir / "output_images"
        output_dir.mkdir(exist_ok=True)
        
        # Save result
        output_file = output_dir / f"dinov2_{task}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {output_file}")
    
    print("DINOv2 testing completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 