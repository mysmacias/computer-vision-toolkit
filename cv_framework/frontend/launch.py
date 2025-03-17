#!/usr/bin/env python
"""
Launcher script for the Streamlit-based FasterRCNN demo.
"""

import os
import sys
import importlib
import subprocess
import traceback

# Set environment variables to avoid threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "streamlit",
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "Pillow"
    ]
    
    # Map package names to import names where they differ
    import_map = {
        "opencv-python": "cv2",
        "Pillow": "PIL"
    }
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Get the correct import name
            import_name = import_map.get(package, package.replace("-", "_"))
            
            # Try to import the package
            importlib.import_module(import_name)
            print(f"✓ {package} is installed")
            
            # Check version for important packages
            if package in ["torch", "torchvision", "streamlit"]:
                version = importlib.import_module(import_name).__version__
                print(f"  Version: {version}")
                
        except ImportError:
            # Special case for opencv-python which might be installed as opencv
            if package == "opencv-python":
                try:
                    importlib.import_module("cv2")
                    print(f"✓ {package} is installed (as opencv)")
                    continue
                except ImportError:
                    pass
            
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies."""
    print(f"Installing missing dependencies: {', '.join(packages)}")
    try:
        if packages:
            # Install the packages
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print("\nPlease install the required packages manually:")
        for pkg in packages:
            print(f"  pip install {pkg}")
        return False

def launch_streamlit():
    """Launch the Streamlit UI."""
    try:
        # Try to preload torch and check CUDA status to catch potential issues early
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                
                # Try a small tensor operation on GPU to check if CUDA is working properly
                test_tensor = torch.zeros(1).cuda()
                print("CUDA test successful")
        except Exception as e:
            print(f"Warning: Error during PyTorch/CUDA test: {e}")
            print("Continuing with CPU...")
        
        # Check if app.py exists
        app_path = os.path.join(os.path.dirname(__file__), "app.py")
        if not os.path.exists(app_path):
            print("Error: app.py not found in the current directory")
            return
        
        print("Launching Streamlit app...")
        
        # Run the streamlit command
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501"]
        subprocess.run(streamlit_cmd)
    except Exception as e:
        print(f"Error launching Streamlit UI: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        
        # Check for common errors and provide troubleshooting steps
        error_str = str(e).lower()
        
        if "cuda" in error_str or "gpu" in error_str:
            print("\nTroubleshooting CUDA issues:")
            print("1. Try running with CPU only")
            print("2. Update your GPU drivers")
            print("3. Check PyTorch is installed with CUDA support")
        
        elif "import" in error_str or "no module" in error_str:
            print("\nTroubleshooting import errors:")
            print("1. Make sure all dependencies are installed")
            print("2. Check your Python environment")
        
        elif "streamlit" in error_str:
            print("\nTroubleshooting Streamlit issues:")
            print("1. Try updating Streamlit: pip install --upgrade streamlit")
            print("2. Check for port conflicts (default port is 8501)")
            print("3. Try running streamlit directly: streamlit run app.py")

def main():
    """Main function to check dependencies and launch the UI."""
    print("=" * 50)
    print("FasterRCNN Streamlit Demo Launcher")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        success = install_dependencies(missing)
        if not success:
            print("Cannot proceed without required dependencies.")
            return
    
    print("\nLaunching Streamlit UI...")
    launch_streamlit()

if __name__ == "__main__":
    main() 