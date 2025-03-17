#!/usr/bin/env python
"""
Test script for loading YOLO models directly, bypassing the Gradio interface.
This helps diagnose issues with model loading.
"""

import os
import sys
import time
import torch
import traceback

# Set threading environment variables to avoid MKL conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add parent directory to path so we can import the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_yolo_direct():
    """Test loading YOLO directly using ultralytics."""
    print("\n=== Testing YOLO Loading Directly ===")
    try:
        from ultralytics import YOLO
        print("Successfully imported ultralytics.YOLO")
        
        # Try to load a model
        model_name = "yolov8n"  # Use a small model for testing
        print(f"Attempting to load {model_name}...")
        
        # Check if model exists locally
        model_path = os.path.join(current_dir, f"{model_name}.pt")
        if os.path.exists(model_path):
            print(f"Found model weights at: {model_path}")
            model = YOLO(model_path)
        else:
            print(f"Model weights not found at {model_path}, will try downloading")
            model = YOLO(model_name)
            
        print(f"Successfully loaded {model_name}")
        return True
    except Exception as e:
        print(f"Error loading YOLO model directly: {e}")
        print(traceback.format_exc())
        return False

def test_yolo_cv_framework():
    """Test loading YOLO through the CV framework."""
    print("\n=== Testing YOLO Loading Through CV Framework ===")
    try:
        # Import the create_model function
        from cv_framework.run import create_model
        print("Successfully imported create_model")
        
        # Try to create a YOLO model
        model_name = "yolov8n"
        device = "cpu"  # Use CPU for testing to avoid potential CUDA issues
        print(f"Attempting to create {model_name} on {device}...")
        
        model = create_model(model_name, device)
        print(f"Successfully created model: {type(model)}")
        
        # Try to load the model
        print("Calling model.load_model()...")
        success = model.load_model()
        
        if success:
            print("Model loaded successfully")
            return True
        else:
            print("model.load_model() returned False")
            return False
    except Exception as e:
        print(f"Error loading YOLO model through framework: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("YOLO Model Loading Test")
    print("=" * 60)
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Sys path: {sys.path}")
    
    # Test direct loading
    direct_result = test_yolo_direct()
    
    # Test framework loading
    framework_result = test_yolo_cv_framework()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Direct YOLO loading: {'SUCCESS' if direct_result else 'FAILED'}")
    print(f"Framework YOLO loading: {'SUCCESS' if framework_result else 'FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main() 