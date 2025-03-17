#!/usr/bin/env python
"""
Test script for the Streamlit-based FasterRCNN implementation.
This is a minimal version to test the core functionality.
"""

import os
import sys
import time
import torch
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import traceback

# Set threading environment variables to avoid MKL conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def test_streamlit_basics():
    """Test basic Streamlit functionality"""
    st.title("Streamlit FasterRCNN Test")
    st.write("This is a simple test to verify Streamlit is working correctly.")
    
    # Test widgets
    st.sidebar.header("Controls")
    test_slider = st.sidebar.slider("Test Slider", 0, 100, 50)
    st.sidebar.write(f"Slider value: {test_slider}")
    
    # Test columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("Column 1")
        st.button("Test Button")
    with col2:
        st.write("Column 2")
        st.checkbox("Test Checkbox")
    
    # Test image display
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    st.image(test_image, caption="Test Image", use_column_width=True)
    
    return True

def test_fasterrcnn_load():
    """Test loading FasterRCNN model"""
    st.subheader("Testing FasterRCNN Model Loading")
    
    status = st.empty()
    status.info("Loading model...")
    
    try:
        import torchvision
        from torchvision import models
        st.write(f"Torchvision version: {torchvision.__version__}")
        
        # Try to load model with new API first
        try:
            from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
            status.info("Attempting to load model with new-style weights parameter...")
            model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            status.success("Successfully loaded model with new-style weights")
        except (ImportError, AttributeError):
            # Fall back to older style
            status.info("Attempting to load model with legacy pretrained parameter...")
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            status.success("Successfully loaded model with pretrained=True")
        
        # Test device compatibility
        if torch.cuda.is_available():
            try:
                status.info("Moving model to CUDA...")
                model = model.cuda()
                status.success("Successfully moved model to CUDA")
            except Exception as e:
                status.warning(f"Error moving model to CUDA: {e}")
                status.info("Keeping model on CPU")
        else:
            status.info("CUDA not available, keeping model on CPU")
        
        # Set to evaluation mode
        model.eval()
        
        # Print model parameters
        num_params = sum(p.numel() for p in model.parameters())
        st.write(f"Model has {num_params:,} parameters")
        device = next(model.parameters()).device
        st.write(f"Model is on device: {device}")
        
        status.success("FasterRCNN model loading test: SUCCESS")
        return model
    except Exception as e:
        status.error(f"Error loading FasterRCNN model: {str(e)}")
        st.code(traceback.format_exc())
        st.error("FasterRCNN model loading test: FAILED")
        return None

def test_model_inference(model):
    """Test model inference on a sample image"""
    if model is None:
        st.error("Cannot run inference: model is None")
        return
    
    st.subheader("Testing Model Inference")
    
    # Create a sample image
    st.write("Creating test image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a rectangle to simulate an object
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
    
    st.image(img, caption="Test Input Image", use_column_width=True)
    
    # Process image
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        # Convert to tensor
        status.info("Converting image to tensor...")
        progress_bar.progress(25)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to same device as model
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        status.info(f"Prepared tensor of shape {img_tensor.shape} on device {device}")
        progress_bar.progress(50)
        
        # Run inference
        status.info("Running inference...")
        with torch.no_grad():
            start_time = time.time()
            predictions = model(img_tensor)
            elapsed = time.time() - start_time
        
        progress_bar.progress(75)
        st.write(f"Inference completed in {elapsed:.4f} seconds")
        
        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        st.write(f"Detected {len(boxes)} boxes")
        
        # Draw boxes on image
        output_img = img.copy()
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score > 0.5:  # Only show high confidence predictions
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, f"Label: {label}, Score: {score:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        progress_bar.progress(100)
        status.success("Inference completed successfully")
        
        # Show output image
        st.image(output_img, caption="Output with detections", use_column_width=True)
        
        # Show detection details
        if len(boxes) > 0:
            st.write("Detection Details:")
            for i, (box, score, label) in enumerate(zip(boxes[:5], scores[:5], labels[:5])):
                st.write(f"Box {i}: {box}, Score: {score:.4f}, Label: {label}")
        else:
            st.info("No objects detected above threshold")
        
        return True
    except Exception as e:
        status.error(f"Error during inference: {str(e)}")
        st.code(traceback.format_exc())
        return False

def main():
    """Main test function"""
    st.set_page_config(
        page_title="Streamlit FasterRCNN Test",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Streamlit FasterRCNN Test Suite")
    st.write("This app tests the integration of Streamlit with FasterRCNN for object detection.")
    
    # System info
    st.sidebar.header("System Information")
    st.sidebar.write(f"Python version: {sys.version.split()[0]}")
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    st.sidebar.write(f"PyTorch version: {torch.__version__}")
    st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.sidebar.write(f"CUDA version: {torch.version.cuda}")
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test controls
    st.sidebar.header("Test Controls")
    run_ui_test = st.sidebar.checkbox("Run UI Test", value=True)
    run_model_test = st.sidebar.checkbox("Run Model Loading Test", value=True)
    run_inference_test = st.sidebar.checkbox("Run Inference Test", value=True)
    
    # Run tests
    if run_ui_test:
        st.header("1. UI Component Test")
        test_streamlit_basics()
        st.success("UI components test completed")
    
    if run_model_test:
        st.header("2. Model Loading Test")
        model = test_fasterrcnn_load()
        
        if run_inference_test and model is not None:
            st.header("3. Model Inference Test")
            test_model_inference(model)
    
    st.header("Test Summary")
    st.write("All tests have been completed. Check the results above for details.")

if __name__ == "__main__":
    main() 