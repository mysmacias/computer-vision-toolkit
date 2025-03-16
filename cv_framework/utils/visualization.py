"""
Visualization utilities for computer vision models.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_bounding_box(frame, box, label=None, score=None, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box with label and confidence score.
    
    Args:
        frame (numpy.ndarray): Frame to draw on
        box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        label (str, optional): Class label
        score (float, optional): Confidence score
        color (tuple, optional): RGB color for the box
        thickness (int, optional): Line thickness
        
    Returns:
        numpy.ndarray: Frame with bounding box
    """
    x1, y1, x2, y2 = box
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label and score if provided
    if label or score is not None:
        # Prepare text
        text = ""
        if label:
            text += label
        if score is not None:
            if text:
                text += f": {score:.2f}"
            else:
                text += f"{score:.2f}"
        
        # Determine text size for background rectangle
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            2
        )
    
    return frame


def draw_mask(frame, mask, color=(0, 255, 0), alpha=0.5, draw_contour=True, contour_color=None):
    """
    Draw a segmentation mask on a frame.
    
    Args:
        frame (numpy.ndarray): Frame to draw on
        mask (numpy.ndarray): Binary mask (same shape as frame height/width)
        color (tuple, optional): RGB color for the mask
        alpha (float, optional): Transparency value (0-1)
        draw_contour (bool, optional): Whether to draw the contour of the mask
        contour_color (tuple, optional): RGB color for the contour (defaults to mask color)
        
    Returns:
        numpy.ndarray: Frame with mask overlay
    """
    # Create a colored mask overlay
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :] = color
    
    # Apply mask with transparency
    frame_with_mask = np.where(
        np.expand_dims(mask, 2),
        cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0),
        frame
    )
    
    # Draw contours if requested
    if draw_contour:
        if contour_color is None:
            contour_color = color
            
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame_with_mask, contours, -1, contour_color, 2)
    
    return frame_with_mask


def generate_color_map(num_classes, random_seed=42):
    """
    Generate a colormap for visualizing classes.
    
    Args:
        num_classes (int): Number of classes
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Array of RGB colors
    """
    np.random.seed(random_seed)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image while preserving aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image
        width (int, optional): Target width
        height (int, optional): Target height
        inter (int, optional): Interpolation method
        
    Returns:
        numpy.ndarray: Resized image
    """
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)


def create_side_by_side(frame, processed_frame, label1="Original", label2="Processed"):
    """
    Create a side-by-side comparison of original and processed frames.
    
    Args:
        frame (numpy.ndarray): Original frame
        processed_frame (numpy.ndarray): Processed frame
        label1 (str, optional): Label for original frame
        label2 (str, optional): Label for processed frame
        
    Returns:
        numpy.ndarray: Side-by-side comparison image
    """
    # Make sure both frames have the same height
    h1, w1 = frame.shape[:2]
    h2, w2 = processed_frame.shape[:2]
    
    # Use the maximum height
    h = max(h1, h2)
    
    # Resize frames if necessary
    if h1 != h:
        frame = resize_with_aspect_ratio(frame, height=h)
        h1, w1 = frame.shape[:2]
    
    if h2 != h:
        processed_frame = resize_with_aspect_ratio(processed_frame, height=h)
        h2, w2 = processed_frame.shape[:2]
    
    # Create the side-by-side image
    side_by_side = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    
    # Copy the frames
    side_by_side[:, :w1] = frame
    side_by_side[:, w1:w1+w2] = processed_frame
    
    # Add labels
    cv2.putText(
        side_by_side,
        label1,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    cv2.putText(
        side_by_side,
        label2,
        (w1 + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return side_by_side


def visualize_depth_map(depth_map, colormap=cv2.COLORMAP_INFERNO):
    """
    Visualize a depth map.
    
    Args:
        depth_map (numpy.ndarray): Depth map
        colormap (int, optional): OpenCV colormap
        
    Returns:
        numpy.ndarray: Colored depth map visualization
    """
    # Normalize depth map to 0-255
    if depth_map.min() != depth_map.max():
        normalized_depth = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    else:
        normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)
    
    # Apply colormap
    colored_depth = cv2.applyColorMap(normalized_depth, colormap)
    
    return colored_depth 