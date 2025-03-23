#!/usr/bin/env python
"""
Detection Analysis Script

This script analyzes the CSV output from run_cam.py to generate
visualizations and insights into object detection performance.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime

def print_available_styles():
    """Print available matplotlib styles to help with debugging"""
    available_styles = plt.style.available
    print("\nAvailable matplotlib styles:")
    for style in sorted(available_styles):
        print(f"- {style}")
    print("\n")

# Set up nicer plotting styles
try:
    # Try newest style naming convention first
    plt.style.use('seaborn')
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
except:
    try:
        # Try newer style naming next
        plt.style.use('seaborn-whitegrid')
        mpl.rcParams['grid.alpha'] = 0.3
    except:
        try:
            # Try older v0_8 naming convention
            plt.style.use('seaborn-v0_8-whitegrid')
            mpl.rcParams['grid.alpha'] = 0.3
        except:
            # Fallback to default style with grid
            try:
                print_available_styles()
                plt.style.use('default')
            except:
                # Last resort, don't use any style
                pass
            mpl.rcParams['axes.grid'] = True
            mpl.rcParams['grid.alpha'] = 0.3
            print("Note: Using default matplotlib style with grid as seaborn styles weren't available.")

sns.set_palette("viridis")
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['axes.axisbelow'] = True  # grid lines below the data

def load_latest_csv(directory='detection_results'):
    """Load the most recent CSV file from the specified directory"""
    try:
        # Verify directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' not found.")
            return None
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in '{directory}'.")
            return None
        
        # Get the most recent file
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"Analyzing: {latest_file}")
        
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(latest_file)
        
        # Basic validation
        if df.empty:
            print("Warning: CSV file is empty.")
            return None
        
        return df, latest_file
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_specific_csv(filepath):
    """Load a specific CSV file"""
    try:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            return None
        
        print(f"Analyzing: {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            print("Warning: CSV file is empty.")
            return None
        
        return df, filepath
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def apply_plot_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to plot axes"""
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def analyze_detections(df, filepath, output_dir='analysis_results'):
    """Analyze detection data and create visualizations"""
    
    # Generate timestamp for output files and folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped subfolder
    timestamped_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    if not os.path.exists(timestamped_dir):
        os.makedirs(timestamped_dir)
    
    # Basic statistics
    total_frames = df['frame_number'].nunique()
    total_detections = len(df)
    unique_classes = df['class_name'].nunique()
    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0
    avg_confidence = df['confidence'].astype(float).mean()
    avg_processing_time = df['processing_time'].astype(float).mean()
    
    print("\n=== DETECTION SUMMARY ===")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Unique object classes: {unique_classes}")
    print(f"Average detections per frame: {avg_detections_per_frame:.2f}")
    print(f"Average confidence score: {avg_confidence:.4f}")
    print(f"Average processing time: {avg_processing_time:.4f} seconds")
    
    # Convert data types for analysis
    df['confidence'] = df['confidence'].astype(float)
    df['processing_time'] = df['processing_time'].astype(float)
    df['timestamp'] = df['timestamp'].astype(float)
    
    # Calculate FPS per frame
    df['fps'] = 1.0 / df['processing_time']
    
    # Calculate bounding box metrics for all visualizations
    df['box_width'] = df['x_max'] - df['x_min']
    df['box_height'] = df['y_max'] - df['y_min']
    df['box_area'] = df['box_width'] * df['box_height']
    
    # Calculate box center coordinates
    df['box_center_x'] = (df['x_min'] + df['x_max']) / 2
    df['box_center_y'] = (df['y_min'] + df['y_max']) / 2
    
    # Calculate aspect ratio
    df['aspect_ratio'] = df['box_width'] / df['box_height']
    
    # Get top classes for consistent color schemes
    class_counts = df['class_name'].value_counts()
    top_classes = class_counts.head(15).index.tolist()
    
    # Create a custom color palette for classes
    class_palette = sns.color_palette("husl", len(top_classes))
    class_color_dict = dict(zip(top_classes, class_palette))
    
    # Store all visualization paths
    visual_paths = {}
    
    # 1. Class distribution
    plt.figure(figsize=(12, 8))
    class_counts_top20 = class_counts.head(20)  # Top 20 for readability
    
    # Define color gradient based on count
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_counts_top20)))
    
    ax = sns.barplot(x=class_counts_top20.values, y=class_counts_top20.index, palette=colors)
    apply_plot_style(ax, 'Distribution of Detected Object Classes (Top 20)', 'Count', 'Object Class')
    
    # Add count labels to bars
    for i, count in enumerate(class_counts_top20.values):
        ax.text(count + 0.5, i, f"{count}", va='center', fontweight='bold')
    
    plt.tight_layout()
    class_dist_path = os.path.join(timestamped_dir, f'class_distribution.png')
    plt.savefig(class_dist_path, bbox_inches='tight')
    print(f"Saved class distribution chart to: {class_dist_path}")
    visual_paths['class_distribution'] = class_dist_path
    
    # 2. Confidence distribution
    plt.figure(figsize=(10, 6))
    # Create a nicer histogram with KDE
    ax = sns.histplot(df['confidence'], bins=20, kde=True, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
    apply_plot_style(ax, 'Distribution of Confidence Scores', 'Confidence Score', 'Count')
    
    # Add vertical line for mean and median confidence
    mean_conf = df['confidence'].mean()
    median_conf = df['confidence'].median()
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Mean: {mean_conf:.3f}')
    ax.axvline(median_conf, color='blue', linestyle='-.', linewidth=2, alpha=0.7,
               label=f'Median: {median_conf:.3f}')
    
    # Add detection threshold line
    ax.axvline(0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Detection Threshold: 0.5')
    
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    conf_dist_path = os.path.join(timestamped_dir, f'confidence_distribution.png')
    plt.savefig(conf_dist_path, bbox_inches='tight')
    print(f"Saved confidence distribution chart to: {conf_dist_path}")
    visual_paths['confidence_distribution'] = conf_dist_path
    
    # 3. Processing time / FPS over time
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x='frame_number', y='fps', data=df.drop_duplicates('frame_number').sort_values('frame_number'), 
                 color='dodgerblue', linewidth=2)
    apply_plot_style(ax, 'FPS Over Time', 'Frame Number', 'Frames Per Second (FPS)')
    
    # Add horizontal lines for min, max and mean FPS
    mean_fps = df['fps'].mean()
    ax.axhline(mean_fps, color='red', linestyle='--', linewidth=1.5, 
              label=f'Mean: {mean_fps:.2f} FPS')
    
    # Add shaded area for 1 std deviation
    std_fps = df['fps'].std()
    ax.axhline(mean_fps + std_fps, color='darkgray', linestyle=':', linewidth=1)
    ax.axhline(mean_fps - std_fps, color='darkgray', linestyle=':', linewidth=1)
    ax.fill_between(df.drop_duplicates('frame_number').sort_values('frame_number')['frame_number'], 
                   mean_fps - std_fps, mean_fps + std_fps, color='blue', alpha=0.1,
                   label=f'±1 Std Dev: {std_fps:.2f}')
    
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    fps_path = os.path.join(timestamped_dir, f'fps_over_time.png')
    plt.savefig(fps_path, bbox_inches='tight')
    print(f"Saved FPS chart to: {fps_path}")
    visual_paths['fps_over_time'] = fps_path
    
    # 4. Average confidence by class
    plt.figure(figsize=(12, 8))
    conf_by_class = df.groupby('class_name')['confidence'].mean().sort_values(ascending=False)
    conf_by_class_top20 = conf_by_class.head(20)  # Top 20 for readability
    
    # Use a color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(conf_by_class_top20)))
    
    ax = sns.barplot(x=conf_by_class_top20.values, y=conf_by_class_top20.index, palette=colors)
    apply_plot_style(ax, 'Average Confidence by Object Class (Top 20)', 'Average Confidence', 'Object Class')
    
    # Add confidence value labels to bars
    for i, conf in enumerate(conf_by_class_top20.values):
        ax.text(conf + 0.01, i, f"{conf:.3f}", va='center', fontweight='bold')
    
    # Add detection threshold line
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5,
              label='Detection Threshold: 0.5')
    
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    conf_by_class_path = os.path.join(timestamped_dir, f'confidence_by_class.png')
    plt.savefig(conf_by_class_path, bbox_inches='tight')
    print(f"Saved confidence by class chart to: {conf_by_class_path}")
    visual_paths['confidence_by_class'] = conf_by_class_path
    
    # 5. Number of detections over time
    detection_counts = df.groupby('frame_number').size().reset_index(name='count')
    
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x='frame_number', y='count', data=detection_counts, 
                 color='darkgreen', linewidth=2)
    apply_plot_style(ax, 'Number of Detections Over Time', 'Frame Number', 'Number of Detections')
    
    # Add rolling average for trend
    window = min(30, len(detection_counts) // 5)  # Use 1/5 of data points or 30, whichever is smaller
    if window > 1:
        detection_counts['rolling_avg'] = detection_counts['count'].rolling(window=window, center=True).mean()
        sns.lineplot(x='frame_number', y='rolling_avg', data=detection_counts, 
                    color='red', linewidth=2, label=f'Moving Average (window={window})')
    
    # Add mean line
    mean_detections = detection_counts['count'].mean()
    ax.axhline(mean_detections, color='blue', linestyle='--', linewidth=1.5, 
              label=f'Mean: {mean_detections:.2f} detections')
    
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    detections_time_path = os.path.join(timestamped_dir, f'detections_over_time.png')
    plt.savefig(detections_time_path, bbox_inches='tight')
    print(f"Saved detections over time chart to: {detections_time_path}")
    visual_paths['detections_over_time'] = detections_time_path
    
    # 6. Class presence over time (line plot)
    # Create a pivot table for class presence
    frames = sorted(df['frame_number'].unique())
    
    # We'll sample frames to avoid too many data points
    sample_step = max(1, len(frames) // 200)  # Aim for ~200 frames in the plot for readability
    sampled_frames = frames[::sample_step]
    
    # Create a dataframe with class counts per frame
    presence_data = []
    for frame in sampled_frames:
        frame_df = df[df['frame_number'] == frame]
        class_counts_frame = Counter(frame_df['class_name'])
        
        # Add data for each class
        for cls in top_classes:
            presence_data.append({
                'frame_number': frame,
                'class_name': cls,
                'count': class_counts_frame.get(cls, 0)
            })
    
    # Convert to DataFrame for plotting
    presence_df = pd.DataFrame(presence_data)
    
    # Create the line plot
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(x='frame_number', y='count', hue='class_name', data=presence_df, 
                 marker='o', markersize=4, palette=class_color_dict)
    apply_plot_style(ax, 'Object Class Presence Over Time', 'Frame Number', 'Number of Detections')
    
    # Better legend placement and style
    ax.legend(title='Object Class', bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    class_presence_path = os.path.join(timestamped_dir, f'class_presence_over_time.png')
    plt.savefig(class_presence_path, bbox_inches='tight')
    print(f"Saved class presence over time chart to: {class_presence_path}")
    visual_paths['class_presence_over_time'] = class_presence_path
    
    # 7. Class confidence over time
    # Create a dataframe with confidence scores per frame and class
    confidence_data = []
    for frame in sampled_frames:
        frame_df = df[df['frame_number'] == frame]
        # For each class, calculate average confidence in this frame (if present)
        for cls in top_classes:
            class_frame_df = frame_df[frame_df['class_name'] == cls]
            if not class_frame_df.empty:
                avg_conf = class_frame_df['confidence'].mean()
                confidence_data.append({
                    'frame_number': frame,
                    'class_name': cls,
                    'confidence': avg_conf
                })
            # Skip frames/classes with no detections (no confidence score to show)
    
    # Convert to DataFrame for plotting
    if confidence_data:  # Only create plot if we have data
        confidence_df = pd.DataFrame(confidence_data)
        
        # Create the line plot
        plt.figure(figsize=(14, 8))
        ax = sns.lineplot(x='frame_number', y='confidence', hue='class_name', 
                     data=confidence_df, marker='o', markersize=4, palette=class_color_dict)
        apply_plot_style(ax, 'Object Class Confidence Over Time', 'Frame Number', 'Confidence Score')
        
        # Add detection threshold line
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='Detection Threshold: 0.5')
        
        # Set y-axis limits with a bit of padding
        ax.set_ylim(0.5, 1.05)
        
        # Better legend placement and style
        handles, labels = ax.get_legend_handles_labels()
        # Remove the detection threshold from class colors in legend
        threshold_idx = labels.index('Detection Threshold: 0.5') if 'Detection Threshold: 0.5' in labels else -1
        if threshold_idx >= 0:
            handles.pop(threshold_idx)
            labels.pop(threshold_idx)
            # Add it back at the end
            handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, alpha=0.7))
            labels.append('Detection Threshold: 0.5')
        
        ax.legend(handles, labels, title='Object Class', bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=True, facecolor='white', edgecolor='lightgray')
        
        plt.tight_layout()
        confidence_time_path = os.path.join(timestamped_dir, f'confidence_over_time.png')
        plt.savefig(confidence_time_path, bbox_inches='tight')
        print(f"Saved class confidence over time chart to: {confidence_time_path}")
        visual_paths['confidence_over_time'] = confidence_time_path
    else:
        confidence_time_path = None
        print("No confidence data available to create confidence over time chart.")
    
    # 8. Bounding box size analysis
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='box_width', y='box_height', hue='class_name', 
                     data=df.sample(min(1000, len(df))), alpha=0.7, palette=class_color_dict)
    apply_plot_style(ax, 'Bounding Box Dimensions by Object Class', 'Width (pixels)', 'Height (pixels)')
    
    # Better legend placement and style
    ax.legend(title='Object Class', bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    bbox_dims_path = os.path.join(timestamped_dir, f'bbox_dimensions.png')
    plt.savefig(bbox_dims_path, bbox_inches='tight')
    print(f"Saved bounding box dimensions chart to: {bbox_dims_path}")
    visual_paths['bbox_dimensions'] = bbox_dims_path
    
    # 9. Box area vs. confidence
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='box_area', y='confidence', hue='class_name', 
                     data=df.sample(min(1000, len(df))), alpha=0.7, palette=class_color_dict)
    apply_plot_style(ax, 'Confidence vs. Bounding Box Area', 'Bounding Box Area (pixels²)', 'Confidence Score')
    
    # Add detection threshold line
    ax.axhline(0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5,
              label='Detection Threshold: 0.5')
    
    # Better legend placement and style
    ax.legend(title='Object Class', bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, facecolor='white', edgecolor='lightgray')
    
    plt.tight_layout()
    area_conf_path = os.path.join(timestamped_dir, f'area_vs_confidence.png')
    plt.savefig(area_conf_path, bbox_inches='tight')
    print(f"Saved area vs. confidence chart to: {area_conf_path}")
    visual_paths['area_vs_confidence'] = area_conf_path
    
    # 10. NEW: Detection heatmap - where in the frame objects are detected
    plt.figure(figsize=(12, 10))
    ax = sns.kdeplot(x='box_center_x', y='box_center_y', data=df, 
                 cmap="YlOrRd", fill=True, levels=15, 
                 alpha=0.7, thresh=0.05)
    apply_plot_style(ax, 'Detection Density Heatmap', 'X Position (pixels)', 'Y Position (pixels)')
    
    # Add a colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.01)
    cbar.set_label('Detection Density')
    
    # Draw image frame outline using the max width and height as boundaries
    if len(df) > 0:
        max_x = df[['x_min', 'x_max']].values.max()
        max_y = df[['y_min', 'y_max']].values.max()
        
        # Draw rectangle representing the frame
        frame_rect = plt.Rectangle((0, 0), max_x, max_y, fill=False, 
                                  edgecolor='black', linestyle='--')
        ax.add_patch(frame_rect)
        
        # Set axes limits with a bit of padding
        ax.set_xlim(-max_x*0.05, max_x*1.05)
        ax.set_ylim(-max_y*0.05, max_y*1.05)
        
        # Invert y-axis to match image coordinates (0,0 at top-left)
        ax.invert_yaxis()
    
    plt.tight_layout()
    heatmap_path = os.path.join(timestamped_dir, f'detection_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    print(f"Saved detection heatmap to: {heatmap_path}")
    visual_paths['detection_heatmap'] = heatmap_path
    
    # 11. NEW: Class-specific heatmaps for top 4 classes
    top4_classes = class_counts.head(4).index.tolist()
    
    plt.figure(figsize=(15, 12))
    for i, cls in enumerate(top4_classes):
        class_df = df[df['class_name'] == cls]
        if len(class_df) > 10:  # Only plot if we have enough data
            ax = plt.subplot(2, 2, i+1)
            sns.kdeplot(x='box_center_x', y='box_center_y', data=class_df, 
                      cmap="YlOrRd", fill=True, levels=15, 
                      alpha=0.7, thresh=0.05, ax=ax)
            apply_plot_style(ax, f'"{cls}" Detection Heatmap', 'X Position (pixels)', 'Y Position (pixels)')
            
            # Draw image frame outline
            if len(class_df) > 0:
                max_x = df[['x_min', 'x_max']].values.max()
                max_y = df[['y_min', 'y_max']].values.max()
                
                # Draw rectangle representing the frame
                frame_rect = plt.Rectangle((0, 0), max_x, max_y, fill=False, 
                                          edgecolor='black', linestyle='--')
                ax.add_patch(frame_rect)
                
                # Set axes limits with a bit of padding
                ax.set_xlim(-max_x*0.05, max_x*1.05)
                ax.set_ylim(-max_y*0.05, max_y*1.05)
                
                # Invert y-axis to match image coordinates (0,0 at top-left)
                ax.invert_yaxis()
    
    plt.tight_layout()
    class_heatmap_path = os.path.join(timestamped_dir, f'class_heatmaps.png')
    plt.savefig(class_heatmap_path, bbox_inches='tight')
    print(f"Saved class-specific heatmaps to: {class_heatmap_path}")
    visual_paths['class_heatmaps'] = class_heatmap_path
    
    # 12. NEW: Correlation matrix of detection metrics
    # Select numeric columns for correlation analysis
    numeric_columns = ['confidence', 'box_width', 'box_height', 'box_area', 'aspect_ratio', 'fps']
    corr_df = df[numeric_columns].copy()
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # For upper triangle mask
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                 mask=mask, vmin=-1, vmax=1, linewidths=0.5, 
                 cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix of Detection Metrics", fontweight='bold', pad=15)
    
    plt.tight_layout()
    corr_matrix_path = os.path.join(timestamped_dir, f'correlation_matrix.png')
    plt.savefig(corr_matrix_path, bbox_inches='tight')
    print(f"Saved correlation matrix to: {corr_matrix_path}")
    visual_paths['correlation_matrix'] = corr_matrix_path
    
    # 13. NEW: Aspect ratio distribution by class
    plt.figure(figsize=(12, 8))
    
    # Filter extreme aspect ratios for better visualization
    aspect_df = df[(df['aspect_ratio'] > 0.1) & (df['aspect_ratio'] < 10)]
    
    ax = sns.boxplot(x='class_name', y='aspect_ratio', data=aspect_df, 
                 palette=class_color_dict, order=top_classes)
    apply_plot_style(ax, 'Bounding Box Aspect Ratio by Class', 'Object Class', 'Aspect Ratio (width/height)')
    
    # Add horizontal line at aspect ratio = 1 (square)
    ax.axhline(1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
              label='Square (1:1)')
    
    # Add legend
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    aspect_ratio_path = os.path.join(timestamped_dir, f'aspect_ratio_by_class.png')
    plt.savefig(aspect_ratio_path, bbox_inches='tight')
    print(f"Saved aspect ratio distribution to: {aspect_ratio_path}")
    visual_paths['aspect_ratio_by_class'] = aspect_ratio_path
    
    # Generate a comprehensive report
    report_path = os.path.join(timestamped_dir, f'detection_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("===== FASTER R-CNN DETECTION ANALYSIS REPORT =====\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data source: {os.path.basename(filepath)}\n\n")
        
        f.write("--- SUMMARY STATISTICS ---\n")
        f.write(f"Total frames analyzed: {total_frames}\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Unique object classes: {unique_classes}\n")
        f.write(f"Average detections per frame: {avg_detections_per_frame:.2f}\n")
        f.write(f"Average confidence score: {avg_confidence:.4f}\n")
        f.write(f"Average processing time: {avg_processing_time:.4f} seconds\n")
        f.write(f"Average FPS: {1/avg_processing_time:.2f}\n\n")
        
        f.write("--- CLASS FREQUENCY ---\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count} detections ({count/total_detections*100:.2f}%)\n")
        
        f.write("\n--- DETECTION PERFORMANCE ---\n")
        f.write(f"Min confidence: {df['confidence'].min():.4f}\n")
        f.write(f"Max confidence: {df['confidence'].max():.4f}\n")
        f.write(f"Median confidence: {df['confidence'].median():.4f}\n")
        f.write(f"Min FPS: {df['fps'].min():.2f}\n")
        f.write(f"Max FPS: {df['fps'].max():.2f}\n")
        f.write(f"Median FPS: {df['fps'].median():.2f}\n")
        
        f.write("\n--- GENERATED VISUALIZATIONS ---\n")
        f.write(f"1. Class distribution: {os.path.basename(visual_paths.get('class_distribution', 'N/A'))}\n")
        f.write(f"2. Confidence distribution: {os.path.basename(visual_paths.get('confidence_distribution', 'N/A'))}\n")
        f.write(f"3. FPS over time: {os.path.basename(visual_paths.get('fps_over_time', 'N/A'))}\n")
        f.write(f"4. Confidence by class: {os.path.basename(visual_paths.get('confidence_by_class', 'N/A'))}\n")
        f.write(f"5. Detections over time: {os.path.basename(visual_paths.get('detections_over_time', 'N/A'))}\n")
        f.write(f"6. Class presence over time: {os.path.basename(visual_paths.get('class_presence_over_time', 'N/A'))}\n")
        
        # List remaining visualizations based on what was generated
        item_num = 7
        for key, path in visual_paths.items():
            if key not in ['class_distribution', 'confidence_distribution', 'fps_over_time', 
                          'confidence_by_class', 'detections_over_time', 'class_presence_over_time']:
                f.write(f"{item_num}. {key.replace('_', ' ').title()}: {os.path.basename(path)}\n")
                item_num += 1
    
    print(f"\nGenerated comprehensive report: {report_path}")
    print(f"\nAll analysis results saved to: {timestamped_dir}/")
    
    # Add report path to visual paths
    visual_paths['report'] = report_path
    
    return visual_paths

def main():
    """Main function to run the analysis"""
    print("FasterRCNN Detection Analysis Tool")
    print("==================================")
    
    # Check if required packages are installed
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install the required packages using:")
        print("pip install pandas matplotlib seaborn")
        sys.exit(1)
    
    # Set up command-line arguments
    if len(sys.argv) > 1:
        # If a specific file is provided as argument
        data, filepath = load_specific_csv(sys.argv[1])
    else:
        # Otherwise use the most recent file
        data, filepath = load_latest_csv()
    
    if data is None:
        print("No valid data found for analysis. Exiting.")
        sys.exit(1)
    
    # Run the analysis - now passing the filepath parameter
    results = analyze_detections(data, filepath)
    
    # Print instructions for viewing results
    print("\nTo open any generated image, you can use:")
    print(f"  - Your system's default image viewer on the files in the output folder")
    
    # Ask if user wants to open the report
    import webbrowser
    answer = input("\nWould you like to open the report now? (y/n): ")
    if answer.lower() in ['y', 'yes']:
        try:
            # Try to open the report file with the default text editor
            webbrowser.open(results['report'])
        except Exception as e:
            print(f"Could not open the report automatically: {e}")
            print(f"Please open manually: {results['report']}")

if __name__ == "__main__":
    main() 