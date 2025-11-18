import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.3)

# Load CNN Model
def load_model():
    try:
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path='posture_model.tflite')
        interpreter.allocate_tensors()
        print("TensorFlow Lite model loaded successfully.")
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Function to download dataset (placeholder for actual implementation)
def download_dataset():
    url = "https://universe.roboflow.com/ikornproject/sitting-posture-rofqf"
    # Implement dataset download logic here
    print(f"Dataset download from {url} is not yet implemented.")

# Function to preprocess image for CNN
def preprocess_image(image):
    if image.shape[0] == 0 or image.shape[1] == 0:
        return None
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize to FLOAT32 as expected by the model
    return np.expand_dims(image, axis=0).astype(np.float32)

# Function to extract human region for CNN input
def extract_human_region(frame, landmarks, h, w):
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = max(0, min(x_min, x - 20))  # Add padding
        y_min = max(0, min(y_min, y - 20))
        x_max = min(w, max(x_max, x + 20))
        y_max = min(h, max(y_max, y + 20))
    
    # Extract region with human
    region = frame[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return None, (x_min, y_min, x_max, y_max)
    return region, (x_min, y_min, x_max, y_max)

# Main loop for real-time posture analysis
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set window to full screen
    cv2.namedWindow("Real-Time Posture Analysis", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Real-Time Posture Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            region, bbox = extract_human_region(frame, results.pose_landmarks, h, w)
            posture = "Unknown"
            confidence = 0.0
            color = (128, 128, 128)  # Gray for unknown

            if region is not None and model is not None:
                processed_region = preprocess_image(region)
                if processed_region is not None:
                    try:
                        # Set input tensor
                        input_details = model.get_input_details()
                        output_details = model.get_output_details()
                        model.set_tensor(input_details[0]['index'], processed_region)
                        # Run inference
                        model.invoke()
                        # Get output
                        prediction = model.get_tensor(output_details[0]['index'])
                        print(f"Raw prediction output: {prediction}")  # Debug output
                        # Apply softmax normalization if raw outputs are not probabilities
                        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
                        confidence = np.max(prediction)
                        posture_idx = np.argmax(prediction)
                        posture = "Good" if posture_idx == 0 else "Bad"  # Reversing the mapping
                        color = (0, 255, 0) if posture == "Good" else (0, 0, 255)
                        # Apply a threshold to avoid overconfidence
                        if confidence < 0.6:
                            posture = "Unknown"
                            color = (128, 128, 128)  # Gray for unknown
                    except Exception as e:
                        print(f"Prediction error: {e}")

            x_min, y_min, x_max, y_max = bbox
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Display posture and confidence
            cv2.putText(frame, f"Posture: {posture}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        else:
            h, w, _ = frame.shape
            cv2.putText(frame, "No human detected", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, "Confidence: 0.00", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Real-Time Posture Analysis", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Real-Time Posture Analysis", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed or 'q' pressed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
