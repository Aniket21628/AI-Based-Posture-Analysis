# Real-Time Posture Analysis

This project implements a real-time posture analysis system using MediaPipe for human detection and a CNN for classifying posture as good or bad. The system uses a webcam feed to detect and analyze posture, displaying results with bounding boxes and confidence levels.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

## Features
- Human detection using MediaPipe
- Posture classification using CNN with softmax layer
- Real-time display with OpenCV
- Bounding box with color-coded posture status (green for good, red for bad)
- Confidence level display

## Dataset
The model is trained on a dataset from Roboflow: [Sitting Posture Dataset](https://universe.roboflow.com/ikornproject/sitting-posture-rofqf)

## Notes
Optimized for potential deployment on Raspberry Pi 5.
