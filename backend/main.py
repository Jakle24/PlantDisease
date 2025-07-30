# main.py: Backend for Plant Disease Detection (Hopefully Finalised Version!)
# ---------------------------------------------------------------
# Now loads pre-trained model (no retraining), starts Flask API. Saves the exaserbatingly long waiting times. 

import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
MODEL_PATH = "plant_disease_model.h5"
IMG_SIZE = (224, 224)

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run training first.")

print("✅ Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Hardcoded class names (based on training set)
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]
class_indices = {i: name for i, name in enumerate(class_names)}

# Flask API setup
app = Flask(__name__)

def prepare_image(file):
    img = load_img(file, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    x = prepare_image(file)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    return jsonify({
        "disease": class_indices[idx],
        "confidence": round(float(preds[idx]), 4)
    })

if __name__ == "__main__":
    print("🚀 Starting Flask API at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
