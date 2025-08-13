# main.py: Backend for Plant Disease Detection + Gamification
# -----------------------------------------------------------

import os
import json
import numpy as np
from datetime import date, timedelta
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
MODEL_PATH = "plant_disease_model.keras"  # Model file from training
IMG_SIZE = (224, 224)
USER_DATA_FILE = "userdata.json"

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Trained model not found at {MODEL_PATH}. Please run maintrain.py first.")

print("✅ Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# -----------------------------------------------------------
# Load class names
# -----------------------------------------------------------
if os.path.exists("class_names.json"):
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
else:
    raise FileNotFoundError("❌ class_names.json not found. Please export from training.")

class_indices = {i: name for i, name in enumerate(class_names)}

# -----------------------------------------------------------
# Gamification data handling
# -----------------------------------------------------------
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    # Default user profile
    return {"xp": 0, "streak": 0, "last_scan": None, "badges": []}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------------------------------------
# Flask app
# -----------------------------------------------------------
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
    
    # Load gamification data
    user_data = load_user_data()

    # Add XP
    user_data["xp"] += 10  # +10 XP per scan

    # Update streak
    today = date.today()
    if user_data["last_scan"] == str(today - timedelta(days=1)):
        user_data["streak"] += 1
    elif user_data["last_scan"] != str(today):
        user_data["streak"] = 1
    user_data["last_scan"] = str(today)

    # Award badges
    if user_data["xp"] >= 100 and "Green Thumb" not in user_data["badges"]:
        user_data["badges"].append("Green Thumb")
    if user_data["streak"] >= 7 and "One Week Wonder" not in user_data["badges"]:
        user_data["badges"].append("One Week Wonder")

    save_user_data(user_data)

    return jsonify({
        "disease": class_indices[idx],
        "confidence": round(float(preds[idx]), 4),
        "xp": user_data["xp"],
        "streak": user_data["streak"],
        "badges": user_data["badges"]
    })

@app.route("/profile", methods=["GET"])
def profile():
    return jsonify(load_user_data())

# -----------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask API at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
