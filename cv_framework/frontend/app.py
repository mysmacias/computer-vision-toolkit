#!/usr/bin/env python
"""
Streamlit frontend for the Computer Vision Framework.
This version focuses only on FasterRCNN for reliability and simplicity.
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import torch
import cv2
import traceback
from PIL import Image
import io

# Set environment variables to avoid threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add parent directory to path so we can import the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA Available: {CUDA_AVAILABLE}")

# Cache for loaded model
loaded_model = None

# Flag for model loading state - separate from session_state
# This will be used to detect discrepancies in state
MODEL_LOADED_FLAG = False

def load_fasterrcnn_model(device="cpu"):
    """Load the FasterRCNN model"""
    global loaded_model, MODEL_LOADED_FLAG
    
    # Debug output
    print(f"\n===== LOAD MODEL DEBUG =====")
    print(f"Current model status: loaded_model is {None if loaded_model is None else 'not None'}")
    print(f"Current MODEL_LOADED_FLAG: {MODEL_LOADED_FLAG}")
    print(f"Current session model_loaded: {st.session_state.get('model_loaded', False)}")
    print(f"Selected device: {device}")
    
    # If device is cuda but cuda is not available, default to cpu
    if device == 'cuda:0' and not CUDA_AVAILABLE:
        print("CUDA requested but not available. Using CPU instead.")
        device = 'cpu'
    
    if loaded_model is not None:
        print("Model already loaded")
        # Ensure session state is updated
        st.session_state['model_loaded'] = True
        MODEL_LOADED_FLAG = True
        print(f"Updated model_loaded to True (already loaded case)")
        return "Model already loaded", f"FasterRCNN model is already loaded on {device}"
    
    try:
        print(f"Loading FasterRCNN model on {device}...")
        
        # Use torchvision's pre-trained model
        import torchvision
        from torchvision import models
        import pkg_resources
        
        # Get torchvision version for compatibility
        torchvision_version = pkg_resources.get_distribution("torchvision").version
        print(f"Torchvision version: {torchvision_version}")
        
        # Handle different ways to load pretrained models based on torchvision version
        try:
            # Try the new approach first (for newer torchvision versions)
            from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
            model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            print("Loaded model with new-style weights parameter")
        except (ImportError, AttributeError):
            # Fall back to older approach for older torchvision versions
            print("Using legacy pretrained parameter")
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Move model to the right device
        try:
            if device == 'cuda:0':
                model = model.cuda()
            else:
                model = model.cpu()
        except RuntimeError as e:
            print(f"Error moving model to {device}: {e}")
            print("Falling back to CPU")
            model = model.cpu()
        
        # Set model to evaluation mode
        model.eval()
        
        # Store the model
        loaded_model = model
        
        # Verify that the model was loaded
        if loaded_model is None:
            raise ValueError("Model assignment failed - loaded_model is still None")
        
        # Update both session state and our flag
        st.session_state['model_loaded'] = True
        MODEL_LOADED_FLAG = True
        
        print(f"Updated session state model_loaded to: {st.session_state['model_loaded']}")
        print(f"Updated MODEL_LOADED_FLAG to: {MODEL_LOADED_FLAG}")
        print(f"Model ID: {id(loaded_model)}")
        
        # Get model info
        try:
            num_parameters = sum(p.numel() for p in model.parameters())
            model_info = f"FasterRCNN ResNet50 FPN\nParameters: {num_parameters:,}\nDevice: {str(next(model.parameters()).device)}"
        except:
            model_info = f"FasterRCNN ResNet50 FPN\nDevice: {device}"
        
        print("Successfully loaded FasterRCNN model")
        print(f"===== MODEL LOAD COMPLETE =====")
        return "Successfully loaded FasterRCNN model", model_info
    
    except Exception as e:
        error_message = f"Error loading model: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        st.session_state['model_loaded'] = False
        MODEL_LOADED_FLAG = False
        print(f"===== MODEL LOAD FAILED =====")
        return error_message, ""

def process_image(image, confidence_threshold=0.5):
    """Process an image with FasterRCNN model"""
    global loaded_model
    
    if image is None:
        return None
    
    if loaded_model is None:
        # Create an error message on the image
        error_img = image.copy() 
        cv2.putText(error_img, "Model not loaded", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return error_img
    
    try:
        # Make sure image is in the right format
        if len(image.shape) != 3 or image.shape[2] != 3:
            error_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(error_img, "Invalid image format", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return error_img
            
        # Convert from RGB (Streamlit) to the format expected by PyTorch
        try:
            # Use float32 for better precision and compatibility
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Check device of model's parameters to match tensor device
            device = next(loaded_model.parameters()).device
            image_tensor = image_tensor.to(device)
            
            print(f"Image tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
        except Exception as tensor_err:
            print(f"Error preparing image tensor: {tensor_err}")
            error_img = image.copy()
            cv2.putText(error_img, f"Tensor error: {str(tensor_err)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return error_img
        
        # Perform inference with error handling
        try:
            with torch.no_grad():
                start_time = time.time()
                predictions = loaded_model(image_tensor)
                elapsed = time.time() - start_time
                
            print(f"Prediction completed in {elapsed:.3f} seconds")
        except Exception as infer_err:
            print(f"Inference error: {infer_err}")
            print(traceback.format_exc())
            error_img = image.copy()
            cv2.putText(error_img, f"Inference error: {str(infer_err)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return error_img
            
        # Get predictions from the first image in batch
        try:
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            print(f"Detected {len(boxes)} objects before filtering")
            
            # Filter by confidence threshold
            keep_indices = scores >= confidence_threshold
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            
            print(f"Kept {len(boxes)} objects after filtering at threshold {confidence_threshold}")
        except Exception as pred_err:
            print(f"Error processing predictions: {pred_err}")
            error_img = image.copy()
            cv2.putText(error_img, f"Prediction error: {str(pred_err)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return error_img
        
        # COCO class names (simplified)
        coco_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Draw bounding boxes
        output_image = image.copy()
        for box, score, label_id in zip(boxes, scores, labels):
            try:
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image bounds
                h, w = output_image.shape[:2]
                x1, x2 = max(0, x1), min(w-1, x2)
                y1, y2 = max(0, y1), min(h-1, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid boxes
                
                # Get label name
                label_name = coco_names[label_id] if label_id < len(coco_names) else f"Class {label_id}"
                
                # Choose a color based on label_id (for consistency)
                color_id = label_id % 10
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                    (0, 0, 128), (128, 128, 0)
                ]
                color = colors[color_id]
                
                # Draw rectangle
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                text = f"{label_name}: {score:.2f}"
                cv2.putText(output_image, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as box_err:
                print(f"Error drawing box: {box_err}")
                continue
        
        # Add FPS info
        fps = 1.0 / elapsed
        cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_image
    
    except Exception as e:
        print(f"Error processing image: {e}")
        print(traceback.format_exc())
        
        # Return original image with error message
        error_img = image.copy() if image is not None else np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return error_img

def pil_to_numpy(pil_image):
    """Convert PIL Image to numpy array for processing"""
    return np.array(pil_image)

def get_webcam_video():
    """Open webcam and read a frame - used for direct camera access"""
    try:
        # Try to open the webcam
        cap = cv2.VideoCapture(0)
        
        # Check if successfully opened
        if not cap.isOpened():
            print("Failed to open webcam. Checking other camera indices...")
            
            # Try a few other common camera indices
            for idx in [1, 2, -1]:
                print(f"Trying camera index {idx}...")
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"Successfully opened camera with index {idx}")
                    break
            
            if not cap.isOpened():
                return None, "Unable to open webcam. Please check connection and permissions."
            
        # Try to read a frame
        ret, frame = cap.read()
        
        # Always release the camera
        cap.release()
        
        if not ret or frame is None:
            return None, "Failed to capture frame from webcam"
        
        # Basic frame validation
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None, "Received empty frame from webcam"
            
        # Convert BGR to RGB (expected by Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, None
        
    except Exception as e:
        print(f"Error accessing webcam: {e}")
        print(traceback.format_exc())
        return None, f"Error accessing webcam: {str(e)}"

def main():
    """Main function for the Streamlit app"""
    global loaded_model, MODEL_LOADED_FLAG
    
    # Debug output to console
    print("\n===== APP INITIALIZATION =====")
    print(f"Python ID of loaded_model: {id(loaded_model)}")
    print(f"loaded_model is None: {loaded_model is None}")
    print(f"MODEL_LOADED_FLAG: {MODEL_LOADED_FLAG}")
    
    # Initialize session state
    if 'frame_count' not in st.session_state:
        st.session_state['frame_count'] = 0
    if 'processing_active' not in st.session_state:
        st.session_state['processing_active'] = False
    if 'refresh_rate' not in st.session_state:
        st.session_state['refresh_rate'] = 1.0  # Make sure it's a float
    if 'last_frame' not in st.session_state:
        st.session_state['last_frame'] = None
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0  # Default to first tab
    if 'model_load_attempts' not in st.session_state:
        st.session_state['model_load_attempts'] = 0
    
    # Print session state
    print("Session state:")
    for key, value in st.session_state.items():
        if key not in ['last_frame', 'last_result']:  # Skip large objects
            print(f"  {key}: {value}")
    
    # Verify model status - do a thorough check of all indicators
    actual_model_loaded = loaded_model is not None
    session_says_loaded = st.session_state.get('model_loaded', False)
    flag_says_loaded = MODEL_LOADED_FLAG
    
    print(f"Model status check:")
    print(f"  Actual model loaded: {actual_model_loaded}")
    print(f"  Session state says loaded: {session_says_loaded}")
    print(f"  Flag says loaded: {flag_says_loaded}")
    
    # Fix inconsistencies
    if actual_model_loaded:
        # Model is actually loaded, make sure flags reflect that
        if not session_says_loaded:
            print("WARNING: Model is loaded but session state says it's not. Fixing...")
            st.session_state['model_loaded'] = True
        if not flag_says_loaded:
            print("WARNING: Model is loaded but MODEL_LOADED_FLAG is False. Fixing...")
            MODEL_LOADED_FLAG = True
    else:
        # Model is not loaded, make sure flags reflect that
        if session_says_loaded:
            print("WARNING: Model is not loaded but session state says it is. Fixing...")
            st.session_state['model_loaded'] = False
        if flag_says_loaded:
            print("WARNING: Model is not loaded but MODEL_LOADED_FLAG is True. Fixing...")
            MODEL_LOADED_FLAG = False
    
    print(f"Final model_loaded state: {st.session_state.get('model_loaded', False)}")
    print("===== APP INITIALIZATION COMPLETE =====\n")
    
    # Make sure refresh_rate is a float (handle potential list type issue)
    if isinstance(st.session_state.refresh_rate, list):
        st.session_state.refresh_rate = float(st.session_state.refresh_rate[0])
    elif not isinstance(st.session_state.refresh_rate, float):
        st.session_state.refresh_rate = float(st.session_state.refresh_rate)
    
    # Configure page
    st.set_page_config(
        page_title="FasterRCNN Object Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("FasterRCNN Object Detection Demo")
    st.markdown("A simplified demo focusing only on FasterRCNN for reliability")
    
    # Instructions
    with st.expander("📖 How to use this app", expanded=True):
        st.markdown("""
        ### Getting Started
        1. **Load the model** using the button in the sidebar
        2. Choose your preferred **device** (CPU or CUDA)
        3. Adjust the **confidence threshold** to filter detections
        4. Use one of the three tabs:
           - **Upload Image**: Process a static image file
           - **Single Camera Shot**: Take a photo using your webcam
           - **Live Camera Feed**: Process webcam video in real-time
        
        ### Troubleshooting
        - If you encounter webcam issues, try running `test_webcam.py` first
        - Lower the refresh rate if performance is slow
        - For better performance, use CUDA if available
        """)
    
    # Performance metrics in sidebar
    with st.sidebar:
        st.markdown("### System Info")
        sys_info_cols = st.columns(2)
        with sys_info_cols[0]:
            st.metric("PyTorch", f"{torch.__version__}")
            st.metric("CUDA", "Available" if CUDA_AVAILABLE else "Not available")
        with sys_info_cols[1]:
            st.metric("OpenCV", f"{cv2.__version__}")
            if CUDA_AVAILABLE and hasattr(torch.cuda, 'get_device_name'):
                st.metric("GPU", f"{torch.cuda.get_device_name(0)}")
        
        # Model status indicator
        st.markdown("### Model Status")
        # Check both dictionary access and attribute access to be safe
        model_loaded_dict = st.session_state.get('model_loaded', False)
        model_loaded_attr = getattr(st.session_state, 'model_loaded', False)
        is_model_loaded = model_loaded_dict or model_loaded_attr
        
        # Also directly check the model variable itself (most reliable)
        actual_model_loaded = loaded_model is not None
        
        # Display detailed status
        st.markdown("#### Session State:")
        if is_model_loaded:
            st.success("✅ Session shows model as loaded")
        else:
            st.warning("⚠️ Session shows model as NOT loaded")
        
        st.markdown("#### Actual Model:")
        if actual_model_loaded:
            st.success("✅ Model is actually loaded in memory")
        else:
            st.error("❌ No model loaded in memory")
        
        # Show attempts counter for debugging
        load_attempts = st.session_state.get('model_load_attempts', 0)
        if load_attempts > 0:
            st.info(f"Load button clicked {load_attempts} times")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Device selection
    device_options = ["cpu"]
    if CUDA_AVAILABLE:
        device_options.append("cuda:0")
    selected_device = st.sidebar.selectbox(
        "Select device",
        options=device_options,
        index=1 if "cuda:0" in device_options else 0
    )
    
    # Model loading button
    load_model_button = st.sidebar.button("Load Model")
    model_status = st.sidebar.empty()
    model_info = st.sidebar.empty()
    
    # Confidence threshold slider
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Create tabs for different input methods
    tab_names = ["Upload Image", "Single Camera Shot", "Live Camera Feed"]
    tabs = st.tabs(tab_names)
    
    # Function to handle tab selection
    def handle_tab_selection(tab_index):
        st.session_state.active_tab = tab_index
    
    # Tab 1: Upload Image
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Upload an image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            process_button = st.button("Process Image")
        
        with col2:
            output_placeholder = st.empty()
    
    # Tab 2: Single Camera Shot
    with tabs[1]:
        st.markdown("### Camera Input")
        camera_placeholder = st.empty()
        camera_file = camera_placeholder.camera_input("Take a picture")
        camera_output = st.empty()
    
    # Tab 3: Live Camera Feed
    with tabs[2]:
        st.markdown("### Live Camera Feed")
        
        # Controls column
        controls_col, _ = st.columns([1, 3])
        
        with controls_col:
            st.markdown("#### Controls")
            # Add start/stop buttons
            if not st.session_state.processing_active:
                if st.button("Start Live Processing"):
                    st.session_state.processing_active = True
                    st.experimental_rerun()
            else:
                if st.button("Stop Live Processing"):
                    st.session_state.processing_active = False
                    st.experimental_rerun()
            
            # Display status
            if st.session_state.processing_active:
                st.success("Live processing is active")
            else:
                st.info("Live processing is stopped")
                
            # Refresh rate control
            refresh_slider = st.slider(
                "Refresh Rate (seconds)", 
                min_value=0.1, 
                max_value=5.0, 
                value=float(st.session_state.refresh_rate),
                step=0.1,
                key="refresh_slider"
            )
            # Only update if it's changed
            if refresh_slider != st.session_state.refresh_rate:
                st.session_state.refresh_rate = float(refresh_slider)
            
            # Frame count display
            st.metric("Frames Processed", st.session_state.frame_count)
        
        # Display area for feed
        input_col, output_col = st.columns(2)
        
        with input_col:
            st.markdown("#### Camera Input")
            raw_feed = st.empty()
        
        with output_col:
            st.markdown("#### Detection Output")
            processed_feed = st.empty()
    
    # Handle model loading
    if load_model_button:
        # Increment load attempts counter to track how many times button was clicked
        st.session_state['model_load_attempts'] = st.session_state.get('model_load_attempts', 0) + 1
        
        print(f"\n===== MODEL LOAD BUTTON CLICKED (Attempt #{st.session_state['model_load_attempts']}) =====")
        print(f"Before loading: loaded_model is {None if loaded_model is None else 'not None'}")
        print(f"Before loading: MODEL_LOADED_FLAG = {MODEL_LOADED_FLAG}")
        print(f"Before loading: st.session_state['model_loaded'] = {st.session_state.get('model_loaded', False)}")
        
        # Call the loading function
        status_msg, info = load_fasterrcnn_model(selected_device)
        
        print(f"After loading: loaded_model is {None if loaded_model is None else 'not None'}")
        print(f"After loading: MODEL_LOADED_FLAG = {MODEL_LOADED_FLAG}")
        print(f"After loading: st.session_state['model_loaded'] = {st.session_state.get('model_loaded', False)}")
        
        # Update UI with status information
        model_status.markdown(f"**Status**: {status_msg}")
        model_info.text(info)
        
        # Add additional indicators for debugging
        if loaded_model is not None and st.session_state.get('model_loaded', False):
            print("Model appears to be properly loaded.")
            model_status.success(f"✅ Model successfully loaded on {selected_device}")
        else:
            print("Model loading appears to have failed.")
            model_status.error("❌ Model loading failed. Check console for details.")
        
        print(f"===== MODEL LOAD BUTTON HANDLING COMPLETE =====\n")
        
        # Force a rerun to refresh the UI with updated model status
        st.experimental_rerun()
        
        # Automatically start live processing and switch to live tab when model is loaded
        if st.session_state.model_loaded and not st.session_state.processing_active:
            st.session_state.processing_active = True
            # Switch to the live camera tab (index 2)
            st.session_state.active_tab = 2
    
    # Process uploaded image (Tab 1)
    if uploaded_file is not None and process_button:
        # Check if model is loaded
        if loaded_model is None:
            st.error("Please load the model first!")
        else:
            try:
                # Read and process image
                image = Image.open(uploaded_file)
                image_np = pil_to_numpy(image)
                
                # Process image
                with st.spinner("Processing image..."):
                    result = process_image(image_np, confidence)
                
                # Display result
                if result is not None:
                    output_placeholder.image(result, channels="RGB", caption="Detection Result")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.code(traceback.format_exc())
    
    # Process camera input (Tab 2)
    if camera_file is not None:
        # Check if model is loaded
        if loaded_model is None:
            camera_output.error("Please load the model first!")
        else:
            try:
                # Read and process image
                image = Image.open(camera_file)
                image_np = pil_to_numpy(image)
                
                # Process image
                with st.spinner("Processing camera image..."):
                    result = process_image(image_np, confidence)
                
                # Display result
                if result is not None:
                    camera_output.image(result, channels="RGB", caption="Camera Detection Result")
            except Exception as e:
                camera_output.error(f"Error processing camera image: {str(e)}")
                st.code(traceback.format_exc())
    
    # Process live camera feed (Tab 3)
    if st.session_state.processing_active:
        # Check if model is loaded using session state
        if not st.session_state.model_loaded or loaded_model is None:
            processed_feed.error("Please load the model first!")
            st.error("Model not loaded. Please go back and click 'Load Model' in the sidebar.")
            st.session_state.processing_active = False
        else:
            try:
                # Get frame from webcam
                frame, error = get_webcam_video()
                
                if frame is not None:
                    # Store the raw frame
                    st.session_state.last_frame = frame
                    
                    # Display raw frame
                    raw_feed.image(frame, channels="RGB", caption="Raw Camera Feed")
                    
                    # Process frame
                    with st.spinner("Processing live feed..."):
                        result = process_image(frame, confidence)
                        st.session_state.last_result = result
                        st.session_state.frame_count += 1
                    
                    # Display result and status
                    if result is not None:
                        processed_feed.image(result, channels="RGB", caption="Live Detection")
                        # Display current FPS based on refresh rate
                        st.sidebar.metric("Current FPS", f"{1.0/max(0.1, st.session_state.refresh_rate):.1f}")
                else:
                    # Handle camera errors with specific guidance
                    camera_error = error if error else "Unknown webcam error"
                    raw_feed.error(f"Webcam error: {camera_error}")
                    processed_feed.error(f"Webcam error: {camera_error}")
                    
                    # Show troubleshooting tips
                    st.error("""
                    Webcam access issues can be caused by:
                    - Another application using the camera
                    - Webcam permissions not granted
                    - Disconnected or malfunctioning camera
                    - OS-level restrictions
                    
                    Try running the test_webcam.py script to diagnose camera issues.
                    """)
                    
                    # Display last result if available
                    if st.session_state.last_result is not None:
                        processed_feed.image(st.session_state.last_result, channels="RGB", 
                                         caption="Last Result (webcam error)")
                    
                    # Pause a bit longer on errors to prevent UI flooding
                    time.sleep(1.0)
                    
                    # Give a chance to stop processing without repeatedly showing errors
                    stop_on_error = st.button("Stop Processing (Camera Error)")
                    if stop_on_error:
                        st.session_state.processing_active = False
                        st.experimental_rerun()
                
                # Add a delay based on refresh rate
                if not error:  # Only sleep if there was no error (we already slept on error)
                    time.sleep(st.session_state.refresh_rate)
                
                # Rerun the app to refresh
                st.experimental_rerun()
                
            except Exception as e:
                # Detailed error handling
                error_msg = str(e)
                tb_str = traceback.format_exc()
                processed_feed.error(f"Error in live processing: {error_msg}")
                
                # Show more details in an expander
                with st.expander("See error details"):
                    st.code(tb_str)
                    
                # Prevent immediate restarting on serious errors
                time.sleep(2.0)
                
                # Check if we've had too many consecutive errors
                if 'error_count' not in st.session_state:
                    st.session_state.error_count = 0
                    
                st.session_state.error_count += 1
                
                # Stop processing after too many consecutive errors
                if st.session_state.error_count > 5:
                    st.warning("Stopping live processing due to multiple consecutive errors.")
                    st.session_state.processing_active = False
                    st.session_state.error_count = 0
    
    # Display last frame if we stopped processing
    elif st.session_state.last_result is not None and not st.session_state.processing_active:
        # Display the last raw frame
        if st.session_state.last_frame is not None:
            raw_feed.image(st.session_state.last_frame, channels="RGB", caption="Last Raw Frame")
        
        # Display the last processed frame
        processed_feed.image(st.session_state.last_result, channels="RGB", caption="Last Result (processing stopped)")

if __name__ == "__main__":
    main() 