"""
Model downloader utility
Provides functions to automatically download models when needed
"""
import os
import sys
import requests
import torch
from pathlib import Path
from tqdm import tqdm

# Dictionary of model URLs
MODEL_URLS = {
    # YOLO models
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    
    # YOLO segmentation models
    'yolov8n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt',
    'yolov8s-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt',
    'yolov8m-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt',
    'yolov8l-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt',
    'yolov8x-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt',
    
    # YOLO pose models
    'yolov8n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    'yolov8s-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt',
    'yolov8m-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt',
    'yolov8l-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt',
    'yolov8x-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt',
}

def get_model_path(model_name, model_dir=None):
    """
    Get the path to a model file, downloading it if it doesn't exist
    
    Args:
        model_name (str): Name of the model file (e.g., 'yolov8s.pt')
        model_dir (str, optional): Directory to save/look for the model. 
                                  If None, uses the script's directory
    
    Returns:
        Path: Path to the model file
    """
    # Determine model directory
    if model_dir is None:
        # Use the root directory of the project
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        model_dir = script_dir.parent  # cv_framework directory
    else:
        model_dir = Path(model_dir)
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to the model file
    model_path = model_dir / model_name
    
    # If model file doesn't exist, download it
    if not model_path.exists():
        download_model(model_name, model_path)
    
    return model_path

def download_model(model_name, model_path):
    """
    Download a model from its URL
    
    Args:
        model_name (str): Name of the model file
        model_path (Path): Path to save the model file
    """
    # Check if model is in our URL dictionary
    if model_name not in MODEL_URLS:
        raise ValueError(f"Model {model_name} not found in URL dictionary")
    
    # Get URL
    url = MODEL_URLS[model_name]
    
    # Download with progress bar
    print(f"Downloading {model_name} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(model_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print(f"Downloaded {model_name} to {model_path}")
    except Exception as e:
        # Remove partially downloaded file
        if model_path.exists():
            model_path.unlink()
        print(f"Error downloading {model_name}: {str(e)}")
        raise

def download_dinov2_model(model_name):
    """
    Download a DINOv2 model using torch hub
    
    Args:
        model_name (str): Name of the DINOv2 model
        
    Returns:
        model: The loaded model
    """
    try:
        print(f"Downloading {model_name} from torch hub...")
        model = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=True)
        print(f"Successfully downloaded {model_name}")
        return model
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")
        return None 