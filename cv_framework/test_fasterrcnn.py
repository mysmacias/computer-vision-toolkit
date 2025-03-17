#!/usr/bin/env python
"""
Test script for loading and running a FasterRCNN model directly,
bypassing the Gradio interface for debugging.
"""

import os
import sys
import time
import torch
import traceback
import numpy as np
import cv2

# Set threading environment variables to avoid MKL conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def test_fasterrcnn_load():
    """Test loading FasterRCNN model directly."""
    print("\n=== Testing FasterRCNN Loading ===")
    try:
        import torchvision
        from torchvision import models
        print(f"Torchvision version: {torchvision.__version__}")
        
        # Try to load model with new API first
        try:
            from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
            print("Attempting to load model with new-style weights parameter...")
            model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            print("Successfully loaded model with new-style weights")
        except (ImportError, AttributeError):
            # Fall back to older style
            print("Attempting to load model with legacy pretrained parameter...")
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            print("Successfully loaded model with pretrained=True")
        
        # Test device compatibility
        if torch.cuda.is_available():
            try:
                print("Moving model to CUDA...")
                model = model.cuda()
                print("Successfully moved model to CUDA")
            except Exception as e:
                print(f"Error moving model to CUDA: {e}")
                print("Keeping model on CPU")
        else:
            print("CUDA not available, keeping model on CPU")
        
        # Set to evaluation mode
        model.eval()
        print("Model set to evaluation mode")
        
        # Print model parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters")
        
        print("FasterRCNN model loading test: SUCCESS")
        return model
    except Exception as e:
        print(f"Error loading FasterRCNN model: {e}")
        print(traceback.format_exc())
        print("FasterRCNN model loading test: FAILED")
        return None

def test_fasterrcnn_inference(model):
    """Test running inference with FasterRCNN model."""
    if model is None:
        print("Cannot run inference test: model is None")
        return False
    
    print("\n=== Testing FasterRCNN Inference ===")
    try:
        # Create a simple test image (640x480, 3 channels)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a rectangle to simulate an object
        cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
        
        print("Created test image")
        
        # Convert image to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to same device as model
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        print(f"Prepared tensor of shape {img_tensor.shape} on device {device}")
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            start_time = time.time()
            predictions = model(img_tensor)
            elapsed = time.time() - start_time
        
        print(f"Inference completed in {elapsed:.4f} seconds")
        
        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        print(f"Detected {len(boxes)} boxes")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if i < 5:  # Print only first 5 detections
                print(f"  Box {i}: {box}, Score: {score:.4f}, Label: {label}")
        
        print("FasterRCNN inference test: SUCCESS")
        return True
    except Exception as e:
        print(f"Error running inference: {e}")
        print(traceback.format_exc())
        print("FasterRCNN inference test: FAILED")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("FasterRCNN Model Test")
    print("=" * 60)
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current directory: {os.getcwd()}")
    
    # Test model loading
    model = test_fasterrcnn_load()
    
    # Test inference if model loaded successfully
    if model is not None:
        test_fasterrcnn_inference(model)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print("Model Loading: " + ("SUCCESS" if model is not None else "FAILED"))
    print("=" * 60)
    
    if model is None:
        print("\nTroubleshooting Tips:")
        print("1. Check your internet connection (model weights need to be downloaded)")
        print("2. Make sure torchvision is properly installed")
        print("3. Try running 'pip install --upgrade torchvision'")
        print("4. If CUDA errors occurred, try forcing CPU mode")

if __name__ == "__main__":
    main() 