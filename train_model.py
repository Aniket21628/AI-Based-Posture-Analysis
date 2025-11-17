import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image
import json

# Function to load dataset from local folders with COCO annotations
def load_dataset():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found. Exiting.")
        return None, None

    images = []
    labels = []
    class_dirs = ['train', 'valid', 'test']
    for split in class_dirs:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            print(f"Split directory {split_path} not found. Skipping.")
            continue

        # Load COCO annotations
        anno_file = os.path.join(split_path, "_annotations.coco.json")
        if not os.path.exists(anno_file):
            print(f"Annotations file {anno_file} not found. Skipping.")
            continue

        with open(anno_file, 'r') as f:
            coco_data = json.load(f)

        # Print category information for debugging
        if 'categories' in coco_data:
            print(f"Categories in {split} dataset: {coco_data['categories']}")

        # Create a mapping of image ID to filename
        img_map = {img['id']: img['file_name'] for img in coco_data['images']}
        # Create a mapping of image ID to label (category_id 1=Bad, 2=Good)
        anno_map = {}
        for anno in coco_data['annotations']:
            img_id = anno['image_id']
            # Corrected mapping based on dataset categories
            label = 1 if anno['category_id'] == 2 else 0  # 2=Good, 1=Bad
            anno_map[img_id] = label

        # Load images based on annotations
        for img_id, filename in img_map.items():
            img_path = os.path.join(split_path, filename)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    img = np.array(img) / 255.0
                    images.append(img)
                    labels.append(anno_map.get(img_id, 0))  # Default to 0 (Bad) if not annotated
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"Image {img_path} not found. Skipping.")

    if not images:
        print("No images found in dataset. Check structure.")
        return None, None

    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)

# Build CNN model with softmax layer
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # Softmax layer for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main training function
def train():
    images, labels = load_dataset()
    if images is None or labels is None:
        print("Dataset loading failed. Exiting.")
        return
    model = build_model()
    model.summary()
    model.fit(images, labels, epochs=10, validation_split=0.3, batch_size=32)  # Increased validation split to 0.3
    model.save('posture_model.h5')
    print("Model training completed and saved as 'posture_model.h5'")

if __name__ == "__main__":
    train()
