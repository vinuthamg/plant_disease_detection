import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cv2
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def preprocess_image(img):
    # Convert to RGB and enhance features
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Larger size for better accuracy
    return img.flatten()

def load_dataset(folder):
    images = []
    labels = []
    print("Loading dataset...")
    for disease_class in os.listdir(folder):
        print(f"Processing {disease_class} images...")
        path = os.path.join(folder, disease_class)
        if os.path.isdir(path):
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        processed_img = preprocess_image(img)
                        images.append(processed_img)
                        labels.append(disease_class)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def train_model():
    # Dataset path
    train_dir = "dataset/train"
    
    # Load and prepare data
    print("Starting training process...")
    X, y = load_dataset(train_dir)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("-" * 50)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    y_pred = model.predict(X_test)
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Save model and encoder
    print("\nSaving model and encoder...")
    with open('plant_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model()