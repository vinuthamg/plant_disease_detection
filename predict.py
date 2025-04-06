import cv2
import numpy as np
import pickle
from tkinter import Tk, filedialog
import os

DISEASE_INFO = {
    'Tomato_healthy': {  # Updated to match exact folder names
        'name': 'Healthy Plant',
        'symptoms': 'No disease symptoms',
        'treatment': 'Continue regular maintenance'
    },
    'Tomato_Early_blight': {  # Updated to match exact folder names
        'name': 'Early Blight Disease',
        'symptoms': 'Dark brown spots with concentric rings',
        'treatment': 'Remove infected leaves, apply fungicide, improve air circulation'
    },
    'Tomato_Late_blight': {  # Updated to match exact folder names
        'name': 'Late Blight Disease',
        'symptoms': 'Dark water-soaked spots, whitish fungal growth',
        'treatment': 'Apply copper-based fungicide, remove infected plants'
    },
    'Tomato_Leaf_Mold': {  # Updated to match exact folder names
        'name': 'Leaf Mold',
        'symptoms': 'Yellow spots on upper leaf surface, gray mold underneath',
        'treatment': 'Improve ventilation, reduce humidity, apply fungicide'
    }
}

def preprocess_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not load image")
    
    # Convert to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = cv2.resize(img, (400, 400))
    processed_img = cv2.resize(img, (128, 128))
    
    return processed_img.flatten(), display_img

def predict_disease(image_path):
    try:
        # Load model and encoder
        with open('plant_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Process image
        img_processed, display_img = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict([img_processed])[0]
        probabilities = model.predict_proba([img_processed])[0]
        
        # Get disease name and information
        disease_name = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        print("\nDebug - Original predicted disease name:", disease_name)
        
        # Map the predicted name to the correct dictionary key
        name_mapping = {
            'late blight': 'Tomato_Late_blight',
            'late_blight': 'Tomato_Late_blight',
            'Late_blight': 'Tomato_Late_blight',
            'early blight': 'Tomato_Early_blight',
            'early_blight': 'Tomato_Early_blight',
            'Early_blight': 'Tomato_Early_blight',
            'healthy': 'Tomato_healthy',
            'Healthy': 'Tomato_healthy',
            'leaf mold': 'Tomato_Leaf_Mold',
            'leaf_mold': 'Tomato_Leaf_Mold',
            'Leaf_Mold': 'Tomato_Leaf_Mold'
        }
        
        # Get the correct disease key (convert to lowercase for matching)
        disease_name_lower = disease_name.lower().replace(' ', '_')
        modified_name = name_mapping.get(disease_name_lower, name_mapping.get(disease_name, f"Tomato_{disease_name}"))
        print("Debug - Modified disease name:", modified_name)
        
        # Get disease information
        disease_info = DISEASE_INFO[modified_name]
        
        # Display results on image
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        y_pos = 30
        lines = [
            f"Disease: {disease_info['name']}",
            f"Symptoms: {disease_info['symptoms']}",
            f"Treatment: {disease_info['treatment']}"
        ]
        
        for line in lines:
            cv2.putText(display_img, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        # Show image
        cv2.imshow('Plant Disease Detection', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return disease_info, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0

def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Plant Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    return file_path

if __name__ == "__main__":
    print("Plant Disease Detection System")
    print("=" * 30)
    
    image_path = select_image()
    if image_path:
        disease_info, confidence = predict_disease(image_path)
        if disease_info:
            print("\nPredicted Disease:")
            print("=" * 50)
            print(f"Disease: {disease_info['name']}")
            print(f"Symptoms: {disease_info['symptoms']}")
            print(f"Treatment: {disease_info['treatment']}")
            print("=" * 50)
    else:
        print("No image selected!")