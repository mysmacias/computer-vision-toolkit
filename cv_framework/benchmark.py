#!/usr/bin/env python
"""
Benchmark utility for measuring performance of computer vision models.
"""

import os
import argparse
import time
import sys
import torch
import cv2
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path so we can import the framework
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def get_model_list():
    """
    Get a list of available model types with representative models.
    
    Returns:
        list: List of model names for benchmarking
    """
    model_list = [
        # YOLO models - various sizes
        'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l',
        
        # Segmentation model
        'yolov8s-seg',
        
        # Pose model
        'yolov8s-pose',
        
        # Faster R-CNN
        'fasterrcnn_resnet50_fpn',
        
        # SSD
        'ssd300',
        
        # DETR
        'detr_resnet50',
        
        # DINOv2
        'dinov2_vits14'
    ]
    
    return model_list


def load_model(model_name, device):
    """
    Load a model for benchmarking.
    
    Args:
        model_name (str): Name of the model
        device (str): Device to run on
        
    Returns:
        tuple: Model instance and preprocessing function
    """
    from cv_framework.run import create_model
    
    model = create_model(model_name, device)
    model.load_model()
    
    return model


def benchmark_model(model, num_iterations=100, input_size=(640, 640)):
    """
    Benchmark a model's inference speed and memory usage.
    
    Args:
        model: Model instance
        num_iterations (int): Number of inference iterations
        input_size (tuple): Input image size (width, height)
        
    Returns:
        dict: Benchmark results
    """
    # Create a dummy input frame
    frame = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        preprocessed = model.preprocess_frame(frame)
        _ = model.predict(preprocessed)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Track memory usage
    memory_start = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_start = torch.cuda.memory_allocated()
    
    # Measure inference time
    start_time = time.time()
    
    for _ in range(num_iterations):
        preprocessed = model.preprocess_frame(frame)
        _ = model.predict(preprocessed)
        
        # Synchronize GPU after each iteration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    # Get peak memory usage
    memory_peak = 0
    if torch.cuda.is_available():
        memory_peak = torch.cuda.max_memory_allocated() - memory_start
    
    return {
        'model_name': model.model_name,
        'avg_inference_time': avg_time,
        'fps': fps,
        'memory_peak': memory_peak / (1024 * 1024),  # Convert to MB
        'num_iterations': num_iterations
    }


def run_benchmarks(model_names, device=None, iterations=100, input_size=(640, 640)):
    """
    Run benchmarks on multiple models.
    
    Args:
        model_names (list): List of model names
        device (str): Device to run on
        iterations (int): Number of iterations
        input_size (tuple): Input size (width, height)
        
    Returns:
        list: Benchmark results for all models
    """
    results = []
    
    for model_name in model_names:
        print(f"Benchmarking {model_name}...")
        
        try:
            # Load model
            model = load_model(model_name, device)
            
            # Run benchmark
            result = benchmark_model(model, iterations, input_size)
            
            # Add to results
            results.append(result)
            
            print(f"  - FPS: {result['fps']:.2f}")
            print(f"  - Avg Time: {result['avg_inference_time'] * 1000:.2f} ms")
            if torch.cuda.is_available():
                print(f"  - Peak Memory: {result['memory_peak']:.2f} MB")
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_report(results, output_dir='benchmark_results'):
    """
    Generate a benchmark report.
    
    Args:
        results (list): Benchmark results
        output_dir (str): Output directory for report
        
    Returns:
        str: Path to the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.txt")
    chart_file = os.path.join(output_dir, f"benchmark_chart_{timestamp}.png")
    
    # Prepare data for table
    table_data = []
    for result in results:
        table_data.append([
            result['model_name'],
            f"{result['fps']:.2f}",
            f"{result['avg_inference_time'] * 1000:.2f} ms",
            f"{result['memory_peak']:.2f} MB" if torch.cuda.is_available() else "N/A"
        ])
    
    # Generate table
    headers = ["Model", "FPS", "Avg. Inference Time", "Peak Memory (MB)"]
    table = tabulate(table_data, headers=headers, tablefmt='grid')
    
    # Write report
    with open(report_file, 'w') as f:
        f.write("# Computer Vision Model Benchmark Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Iterations per model: {results[0]['num_iterations']}\n\n")
        f.write(table)
    
    # Generate chart
    plt.figure(figsize=(12, 8))
    
    # FPS chart
    model_names = [r['model_name'] for r in results]
    fps_values = [r['fps'] for r in results]
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(model_names, fps_values, color='skyblue')
    plt.title('Model Performance Comparison (FPS)')
    plt.ylabel('Frames Per Second')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Inference time chart
    inference_times = [r['avg_inference_time'] * 1000 for r in results]
    
    plt.subplot(2, 1, 2)
    bars = plt.bar(model_names, inference_times, color='salmon')
    plt.title('Model Inference Time')
    plt.ylabel('Average Inference Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(chart_file)
    
    return report_file, chart_file


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Benchmark computer vision models')
    
    parser.add_argument('-m', '--models', nargs='+', default=None,
                        help='Models to benchmark (default: all models)')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Device to run on (e.g., cpu, cuda:0)')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations for benchmarking (default: 100)')
    parser.add_argument('-s', '--input-size', type=int, nargs=2, default=[640, 640],
                        help='Input size for benchmarking (width height, default: 640 640)')
    parser.add_argument('-o', '--output-dir', type=str, default='benchmark_results',
                        help='Output directory for reports (default: benchmark_results)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List available models and exit')
    
    return parser.parse_args()


def main():
    """
    Main entry point for benchmarking.
    """
    # Parse arguments
    args = parse_args()
    
    # Get available models
    available_models = get_model_list()
    
    # List models if requested
    if args.list:
        print("\nAvailable models for benchmarking:")
        for model in available_models:
            print(f"  - {model}")
        return
    
    # Use specific models if specified, otherwise use all
    model_names = args.models if args.models else available_models
    
    # Check if models are valid
    for model_name in model_names:
        if model_name not in available_models:
            print(f"Warning: Model '{model_name}' not in available models list. It may still work if supported.")
    
    # Run benchmarks
    print(f"\nRunning benchmarks with {args.iterations} iterations per model...")
    results = run_benchmarks(
        model_names,
        device=args.device,
        iterations=args.iterations,
        input_size=tuple(args.input_size)
    )
    
    # Generate report
    if results:
        report_path, chart_path = generate_report(results, args.output_dir)
        print(f"\nBenchmark report saved to: {report_path}")
        print(f"Benchmark chart saved to: {chart_path}")
    else:
        print("\nNo benchmark results to report.")


if __name__ == "__main__":
    main() 