#!/usr/bin/env python3
"""
Computer Vision Toolkit - GUI Launcher
-------------------------------------
This module launches the GUI application for the Computer Vision Toolkit.
"""

import sys
import os
import traceback
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

# Configure path for imports
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

try:
    # Import the main application
    from cv_framework.gui.main import ComputerVisionApp
    
    # Check for models
    model_files = [
        parent_dir / "yolov8s.pt",
        parent_dir / "yolov8s-seg.pt",
        parent_dir / "fasterrcnn_resnet50_fpn.pt"
    ]
    
    missing_models = [str(m) for m in model_files if not m.exists()]
    
    def main():
        """Main entry point for the application"""
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Computer Vision Toolkit")
        
        # Set stylesheet for modern look
        app.setStyle("Fusion")
        
        # Set high DPI scaling
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Create and show the main window
        window = ComputerVisionApp()
        
        # Show model warning if needed
        if missing_models:
            QMessageBox.warning(
                window,
                "Missing Model Files",
                f"The following model files are missing:\n{chr(10).join(missing_models)}\n\n"
                "The application will run, but you won't be able to use these models "
                "until they are downloaded."
            )
        
        window.show()
        
        # Run the event loop
        return app.exec()
    
    if __name__ == "__main__":
        sys.exit(main())

except Exception as e:
    # Handle any unexpected exceptions
    error_msg = f"Critical error: {str(e)}\n\n{traceback.format_exc()}"
    print(error_msg)
    
    # Try to show a GUI error if possible
    try:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Critical Error", error_msg)
    except:
        pass  # If GUI fails, we already printed to console
    
    sys.exit(1) 