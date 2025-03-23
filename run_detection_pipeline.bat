@echo off
echo ====================================
echo FasterRCNN Detection Pipeline Runner
echo ====================================
echo.

echo [1/2] Running webcam detection (press 'q' to stop recording)...
python run_cam.py

echo.
echo [2/2] Analyzing detection results...
python analyze_detections.py

echo.
echo Pipeline complete!
echo Results are saved in the 'detection_results' and 'analysis_results' folders.
echo.

pause 