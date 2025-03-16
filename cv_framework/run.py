#!/usr/bin/env python
"""
Main entry point for running computer vision models within the framework.
"""

import os
import argparse
import sys
import importlib
import inspect
from datetime import datetime

# Add parent directory to path so we can import the framework
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def list_available_models():
    """
    List all available models in the framework.
    """
    from cv_framework.models.yolo_model import YOLOModel
    from cv_framework.models.faster_rcnn_model import FasterRCNNModel
    from cv_framework.models.ssd_model import SSDModel
    from cv_framework.models.detr_model import DETRModel
    from cv_framework.models.dinov2_model import DINOv2Model
    from cv_framework.models.sam_model import SAMModel
    
    # Collect models from all model classes
    all_models = {}
    
    # Add YOLO models
    yolo_models = YOLOModel.list_available_models()
    all_models.update(yolo_models)
    
    # Add Faster R-CNN models
    rcnn_models = FasterRCNNModel.get_available_models()
    all_models.update(rcnn_models)
    
    # Add SSD models
    ssd_models = SSDModel.get_available_models()
    all_models.update(ssd_models)
    
    # Add DETR models
    detr_models = DETRModel.get_available_models()
    all_models.update(detr_models)
    
    # Add DINOv2 models
    dinov2_models = DINOv2Model.get_available_models()
    all_models.update(dinov2_models)
    
    # Add SAM models
    sam_models = SAMModel.get_available_models()
    all_models.update(sam_models)
    
    # Print available models
    print("\n=== Available Models ===")
    for category, models in all_models.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")
    
    return all_models


def create_model(model_name, device=None):
    """
    Create a model instance based on model name.
    
    Args:
        model_name (str): Name of the model
        device (str, optional): Device to run on
        
    Returns:
        VisionModel: Model instance
    """
    # Determine which class to use based on model name
    if model_name.startswith('yolo'):
        from cv_framework.models.yolo_model import YOLOModel
        return YOLOModel(model_name=model_name, device=device)
    elif model_name.startswith('faster') or model_name == 'rcnn':
        from cv_framework.models.faster_rcnn_model import FasterRCNNModel
        return FasterRCNNModel(model_name=model_name, device=device)
    elif model_name.startswith('ssd'):
        from cv_framework.models.ssd_model import SSDModel
        return SSDModel(model_name=model_name, device=device)
    elif model_name.startswith('detr'):
        from cv_framework.models.detr_model import DETRModel
        return DETRModel(model_name=model_name, device=device)
    elif model_name.startswith('dinov2'):
        from cv_framework.models.dinov2_model import DINOv2Model
        return DINOv2Model(model_name=model_name, device=device)
    elif model_name.startswith('sam') or (model_name.startswith('yolo') and model_name.endswith('seg')):
        from cv_framework.models.sam_model import SAMModel
        return SAMModel(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run computer vision models')
    
    parser.add_argument('-m', '--model', type=str, default='yolov8s',
                        help='Model to use (e.g., yolov8s, faster_rcnn, ssd300, detr_resnet50, dinov2_vits14, sam_vit_b)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera index to use')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Device to run on (e.g., cpu, cuda:0)')
    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help='Confidence threshold (0.0 to 1.0)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List available models and exit')
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    # Parse arguments
    args = parse_args()
    
    # List models if requested
    if args.list:
        list_available_models()
        return
    
    try:
        # Create model
        print(f"Creating model: {args.model}")
        model = create_model(args.model, args.device)
        
        # Set confidence threshold if provided
        if args.threshold is not None:
            model.set_confidence_threshold(args.threshold)
        
        # Run model
        model.run(camera_idx=args.camera)
        
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 