# DINOv2 Perception System

A real-time computer vision application leveraging Facebook Research's DINOv2 model to perform multiple perception tasks without task-specific training:

- **Depth Estimation**: Perceives relative depth in a scene
- **Semantic Segmentation**: Groups pixels into meaningful regions
- **Feature Visualization**: Visualizes the model's learned features in RGB space

## Features

- Real-time processing from webcam feed
- Multiple visualization modes
- Video recording with timestamp
- Intuitive keyboard controls
- Automatic initialization and calibration

## Quick Start

```bash
# Install dependencies
pip install torch torchvision opencv-python numpy scikit-learn pillow matplotlib

# Run the application
python run_dinov2.py
```

## Controls

- `v`: Cycle through visualization modes (Depth, Segmentation, Features, Side-by-Side)
- `c`: Change depth colormaps
- `q`: Quit the application

## System Requirements

- Python 3.x
- Webcam
- CUDA-compatible GPU recommended for faster processing

## WSL Setup

To run this project in Windows Subsystem for Linux (WSL):

1. Install WSL and Ubuntu:
```bash
wsl --install
```

2. Install Python and pip in WSL:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

3. Install CUDA toolkit in WSL:
```bash
# Create installation directory
mkdir ~/cuda-install
cd ~/cuda-install

# Download and setup CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/7fa2af80.pub /etc/apt/trusted.gpg.d/

# Install required dependencies
sudo add-apt-repository universe
wget http://archive.ubuntu.com/ubuntu/pool/main/n/ncurses/libtinfo5_6.2-0ubuntu2_amd64.deb
sudo dpkg -i libtinfo5_6.2-0ubuntu2_amd64.deb

# Install CUDA
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

4. Access your project in WSL:
```bash
cd /mnt/c/Users/mysma/projects/computer-vision-toolkit
```

5. Optional: Add a convenient alias to your `~/.bashrc`:
```bash
echo 'alias cvkit="cd /mnt/c/Users/mysma/projects/computer-vision-toolkit"' >> ~/.bashrc
source ~/.bashrc
```

Then you can simply type `cvkit` to navigate to your project directory.

## Documentation

For detailed explanation of how the system works, see [DINOv2_Perception.md](DINOv2_Perception.md). 