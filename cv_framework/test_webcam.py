#!/usr/bin/env python
"""
Simple test script to verify webcam access with OpenCV.
Run this script to confirm your webcam can be accessed before using the Streamlit app.
"""

import cv2
import time
import numpy as np

def test_webcam():
    """Test webcam access using OpenCV"""
    print("Attempting to access webcam...")
    
    # Try to open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Please check your camera connection.")
        return False
    
    print("Webcam opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Could not read frame from webcam.")
        cap.release()
        return False
    
    print(f"Successfully read frame with shape: {frame.shape}")
    
    # Display basic frame info
    height, width, channels = frame.shape
    print(f"Frame resolution: {width}x{height}, Channels: {channels}")
    
    # Save a test image
    test_filename = "webcam_test.jpg"
    cv2.imwrite(test_filename, frame)
    print(f"Test image saved as {test_filename}")
    
    # Try to show the frame (this may not work in all environments)
    try:
        print("Displaying frame... (press any key to continue)")
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(3000)  # Wait for 3 seconds or key press
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Couldn't display frame on screen: {e}")
        print("This is normal if running in a headless environment.")
    
    # Try to capture a few frames to test streaming
    print("Testing continuous capture (5 frames)...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: OK")
        else:
            print(f"Frame {i+1}: Failed to capture")
        time.sleep(0.1)
    
    # Release the webcam
    cap.release()
    print("Webcam released. Test completed successfully!")
    return True

def test_webcam_properties():
    """Test and display webcam properties"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not access webcam to get properties.")
        return
    
    # List of common properties to check
    props = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_AUTOFOCUS, "Autofocus"),
        (cv2.CAP_PROP_FOCUS, "Focus"),
    ]
    
    print("\nWebcam Properties:")
    print("-----------------")
    for prop_id, prop_name in props:
        prop_value = cap.get(prop_id)
        print(f"{prop_name}: {prop_value}")
    
    cap.release()
    print("-----------------")

def main():
    """Main function"""
    print("=" * 50)
    print("OpenCV Webcam Test")
    print("=" * 50)
    
    # Print OpenCV version for reference
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Try different camera indices if the default one doesn't work
    camera_index = 0
    max_attempts = 3
    
    for i in range(max_attempts):
        print(f"\nAttempting to access camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"Success! Camera index {camera_index} is working.")
            cap.release()
            break
        else:
            print(f"Failed to open camera index {camera_index}")
            camera_index += 1
            
            if i == max_attempts - 1:
                print("\nERROR: Could not find a working camera.")
                print("Please check your camera connection and permissions.")
                return
    
    # Run main test
    success = test_webcam()
    
    if success:
        # If successful, show camera properties
        test_webcam_properties()
        
        print("\nWebcam test completed successfully!")
        print("Your webcam should work with the Streamlit app's live processing feature.")
    else:
        print("\nWebcam test failed.")
        print("Please check your camera connection and try again.")
        print("The Streamlit app's live processing feature may not work.")

if __name__ == "__main__":
    main() 