# Computer Vision Framework - Gradio Frontend

A sleek, user-friendly web interface for the Computer Vision Framework using Gradio.

## Features

- **Intuitive Interface**: Clean, modern UI for interacting with computer vision models
- **Live Camera Processing**: Process camera feeds in real-time with various models
- **Image Upload**: Upload and process static images
- **Model Selection**: Easily switch between different model types
- **Parameter Adjustment**: Fine-tune model parameters with interactive controls
- **Benchmarking**: Compare performance across different models with visual charts
- **Cross-platform**: Works on Windows, macOS, and Linux

## Installation

### Requirements

- Python 3.7+
- Our Computer Vision Framework
- Dependencies listed in `requirements.txt`

### Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the Computer Vision Framework is properly installed:

```bash
# From the root directory (where cv_framework is located)
pip install -e .
```

## Usage

### Running the Frontend

To launch the frontend, run:

```bash
# From the frontend directory
python launch.py

# Or from anywhere
python -m cv_framework.frontend.launch
```

Once the server is running, you'll see a message like:
```
Running on local URL: http://localhost:7860
```

Open this URL in your web browser to access the interface. If the message shows a different IP (like 0.0.0.0), use http://localhost:7860 instead.

### Using the Interface

1. **Live Camera**:
   - Select a model category and specific model
   - Set confidence threshold and device
   - Click "Load Model"
   - Allow camera access to see real-time processing

2. **Image Processing**:
   - Upload an image
   - Select a model and adjust parameters
   - Click "Process Image" to see results

3. **Benchmarking**:
   - Select models to benchmark
   - Set iterations and input size
   - Run the benchmark and view comparative results

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Run `python launch.py` which will attempt to install missing dependencies
   - If that fails, manually install with `pip install -r requirements.txt`

2. **Camera Access**:
   - Ensure your browser has permission to access your camera
   - Try a different camera if available by using the camera dropdown

3. **Model Loading Errors**:
   - Check the error message in the status box
   - Ensure model weights are properly downloaded
   - Try a different model if one fails to load

### Getting Help

If you encounter issues:
1. Check the error message in the terminal or status box
2. Refer to the Computer Vision Framework documentation
3. File an issue on the project repository with detailed error information

## Development

### Directory Structure

- `app.py`: Main Gradio application code
- `launch.py`: Launcher script with dependency checking
- `requirements.txt`: Frontend-specific dependencies

### Adding New Features

To extend the frontend:

1. For new UI components, modify the appropriate tab creation function in `app.py`
2. For new functionality, add the necessary processing functions
3. Make sure to connect UI elements to backend functions with Gradio event handlers

## License

This frontend is part of the Computer Vision Framework project and is distributed under the same license terms. 