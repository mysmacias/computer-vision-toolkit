# DINOv2 Perception System

## Overview

This document explains the DINOv2 Perception System, a real-time computer vision application that leverages DINOv2, a powerful self-supervised vision transformer model from Facebook Research. The system provides three different visual perception capabilities:

1. **Depth Estimation** - Perceives relative depth in the scene
2. **Semantic Segmentation** - Groups pixels into meaningful segments/regions
3. **Feature Visualization** - Visualizes learned visual features from the model

All three capabilities are derived solely from DINOv2's pretrained features without any additional training, demonstrating the rich visual representations captured by the model.

## System Architecture

The system consists of four main components:

1. **Feature Extractor** - Extracts patch features from images using DINOv2
2. **Depth Estimator** - Converts features to depth maps using PCA
3. **Semantic Segmenter** - Groups similar features using dimensionality reduction and k-means clustering
4. **Feature Visualizer** - Projects high-dimensional features to RGB space for visualization

The main function provides real-time processing from a webcam feed, visualization, and video recording capabilities.

## Components in Detail

### 1. DINOv2 Feature Extractor

```python
class DINOv2FeatureExtractor:
    # ...
```

The feature extractor is the foundation of the perception system:

- **Purpose**: Loads the DINOv2 model and extracts patch features from images
- **Key Methods**:
  - `extract_features()`: Extracts features from an image using DINOv2
  - `extract_patch_features()`: Extracts patch tokens, excluding the CLS token
  - `get_feature_grid_size()`: Calculates the feature grid dimensions

DINOv2 processes images as a grid of non-overlapping patches (typically 14×14 pixels). Each patch is processed independently through the transformer architecture, resulting in a feature vector per patch that captures semantic information.

The extractor calculates the appropriate image size based on the patch size to ensure proper alignment of features with the visual content.

### 2. Depth Estimation

```python
class DINOv2DepthEstimator:
    # ...
```

The depth estimator leverages the correlation between feature variation and scene depth:

- **Purpose**: Generate relative depth maps from DINOv2 features
- **Method**: Uses Principal Component Analysis (PCA) to find the primary dimension of variation in the feature space
- **Key Insights**:
  - No explicit training for depth estimation is performed
  - The primary component of variation in feature space often correlates with depth
  - Features are processed in a grid layout matching the original image spatial structure

The depth estimation process:
1. Extract patch features from the image
2. Apply PCA to reduce features to a single value per patch
3. Reshape values to the original grid layout
4. Resize, smooth, and normalize for visualization

### 3. Semantic Segmentation

```python
class DINOv2SemanticSegmenter:
    # ...
```

The semantic segmenter groups similar regions in the image:

- **Purpose**: Segment the image into semantically similar regions
- **Method**: Uses dimensionality reduction (PCA) and k-means clustering on feature vectors
- **Process**:
  1. Extract patch features
  2. Normalize features (L2 normalization)
  3. Reduce dimensionality with PCA (to 50 dimensions)
  4. Apply k-means clustering (default: 8 clusters)
  5. Reshape cluster assignments to the image grid
  6. Apply color mapping for visualization

The segmentation doesn't have predefined class labels (unlike models trained on datasets like COCO), but it groups visually similar content, often aligning with object boundaries.

### 4. Feature Visualization

```python
class DINOv2FeatureVisualizer:
    # ...
```

The feature visualizer makes the abstract feature space visible:

- **Purpose**: Visualize high-dimensional features in RGB color space
- **Method**: Uses PCA to project features from hundreds of dimensions to just 3 (RGB)
- **Process**:
  1. Extract patch features
  2. Normalize the features
  3. Apply PCA to reduce to 3 dimensions
  4. Scale to [0,1] range for RGB values
  5. Reshape to original grid
  6. Apply light smoothing for visualization

Colors in the visualization represent semantic similarity - similar colors indicate similar features, even across different images.

### 5. Main Loop & UI

```python
def main():
    # ...
```

The main function ties everything together:

- **Initialization**: Sets up the model, webcam, and output video file
- **Processing Loop**:
  1. Captures frame from webcam
  2. Preprocesses the frame (center-crop to square, resize)
  3. Processes through the different perception systems based on current visualization mode
  4. Overlays visualization on the original frame or creates side-by-side view
  5. Adds UI elements (mode, FPS, status, controls)
  6. Writes to output video and displays
- **Interaction**:
  - 'v' key: Cycles through visualization modes
  - 'c' key: Changes depth colormaps
  - 'q' key: Quits the application

## Technical Implementation Details

### Patch Grid Management

A critical aspect of the system is maintaining proper spatial alignment between the extracted features and the original image:

- **Grid Size Calculation**: The feature grid size is calculated based on the input image size and patch size
- **Consistency**: All visualizations use the same grid size for proper alignment
- **Padding/Truncation**: If the number of features doesn't match the expected grid size, padding or truncation is applied

### Initialization Process

Each perception component has an initialization process:

1. **Collection Phase**: Features are collected from multiple frames
2. **Fitting Phase**: PCA or k-means is fitted using the collected features
3. **Application Phase**: The fitted models are applied to new frames

During initialization, placeholder visualizations provide feedback to the user.

## Running the System

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- PIL (Pillow)
- matplotlib

### Execution

```bash
python run_dinov2.py
```

On first run, the system will download the DINOv2 model (approximately 200MB). The system will then initialize each perception component, which may take a few seconds as it collects features for calibration.

### Output

The system creates an output video file in the `output_videos` directory, named with a timestamp. This video captures all visualizations shown during execution.

## Extending the System

The modular design allows for easy extension:

- **Different Models**: Replace DINOv2 with other transformer models
- **Additional Visualizations**: Add new perception capabilities by creating new processor classes
- **Customization**: Adjust parameters like cluster count, PCA dimensions, etc. to fine-tune performance

## Conclusion

The DINOv2 Perception System demonstrates how a single self-supervised model can enable multiple perception tasks without task-specific training. It showcases the rich feature representations captured by modern vision transformers and provides a platform for exploring these representations in real-time. 