# Computer Vision Toolkit

A collection of scripts for running real-time inference with state-of-the-art computer vision models using webcam input.

## Supported Models

- **Object Detection**:
  - Faster R-CNN (`run_cam.py`)
  - SSD (`run_ssd.py`)
  - RetinaNet (`run_retinanet.py`) 
  - DETR - DEtection TRansformer (`run_detr.py`)
  - YOLOv5 (`run_yolov5.py`)

- **Segmentation**:
  - Segment Anything Model (SAM) (`run_sam.py`)

- **Feature Visualization**:
  - DINOv2 - Vision Transformer Features (`run_dinov2.py`)

## Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA (recommended for faster inference)
- OpenCV
- Various model-specific dependencies

## Setup

1. Create a conda environment:
```bash
conda create -n cam_inf python=3.9
conda activate cam_inf
```

2. Install common dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib pillow
```

3. Install model-specific dependencies:
```bash
# For YOLOv5
pip install ultralytics

# For Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

4. Download model weights:
   - SAM model weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## Usage

Run any of the scripts to start real-time inference:

```bash
python run_yolov5.py   # For YOLOv5 object detection
python run_sam.py      # For SAM segmentation
python run_dinov2.py   # For DINOv2 feature visualization
```

### Common Controls:
- Press `q` to quit
- Specific scripts may have additional keyboard shortcuts (see on-screen instructions)

## Output

All processed videos are saved to the `output_videos` directory with timestamps in the filename.

## Project Structure

- `run_*.py` - Main scripts for different models
- `output_videos/` - Directory for saved video outputs
- `improvement_recommendations.md` - Future development roadmap

## Future Improvements

See `improvement_recommendations.md` for detailed recommended improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 