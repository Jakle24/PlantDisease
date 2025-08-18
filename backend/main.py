# backend/main.py
import os
import json
import numpy as np
from io import BytesIO
from datetime import date, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import traceback
import logging

# configure logging once (top of file)
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("plantd")


app = Flask(__name__)
CORS(app)  # dev: allow all origins

MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = (224, 224)
USER_DATA_FILE = "userdata.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

if os.path.exists("class_names.json"):
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
else:
    raise FileNotFoundError("class_names.json not found.")

class_indices = {i: name for i, name in enumerate(class_names)}

if os.path.exists("disease_facts.json"):
    with open("disease_facts.json", "r") as f:
        disease_facts = json.load(f)
else:
    disease_facts = {}

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"xp": 0, "streak": 0, "last_scan": None, "badges": []}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def prepare_image(file_storage):
    try:
        # read bytes and wrap in BytesIO so load_img always receives a proper file-like object
        file_storage.stream.seek(0)
        data = file_storage.read()
        bio = BytesIO(data)
        bio.seek(0)
        img = load_img(bio, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        return np.expand_dims(arr, 0)
    except Exception:
        raise

@app.before_request
def log_request():
    print("Incoming request:", request.method, request.path)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Request received at /predict")
        print("Request content type:", request.content_type)
        print("Request files keys:", list(request.files.keys()))

        file = request.files.get("file")
        if not file:
            print("No file in request.files -> returning 400")
            return jsonify({"error": "No file provided"}), 400

        print(f"Received file: {file.filename}, content_type={file.content_type}")

        try:
            x = prepare_image(file)
        except Exception as e:
            print("❌ Error processing image:", e)
            traceback.print_exc()
            return jsonify({"error": "Invalid image or could not process file"}), 400

        try:
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
        except Exception as e:
            print("❌ Error during model.predict:", e)
            traceback.print_exc()
            return jsonify({"error": "Model prediction failed"}), 500

        # gamification logic
        user_data = load_user_data()
        today = date.today()
        last_scan = user_data.get("last_scan")
        if last_scan == str(today - timedelta(days=1)):
            user_data["streak"] += 1
        elif last_scan != str(today):
            user_data["streak"] = 1

        user_data["xp"] += 10
        user_data["last_scan"] = str(today)

        if user_data["xp"] >= 100 and "Green Thumb" not in user_data["badges"]:
            user_data["badges"].append("Green Thumb")
        if user_data["streak"] >= 7 and "One Week Wonder" not in user_data["badges"]:
            user_data["badges"].append("One Week Wonder")

        save_user_data(user_data)

        # Return confidence as percentage (rounded 2 dp)
        return jsonify({
            "disease": class_indices[idx],
            "confidence": round(float(preds[idx]) * 100, 2),  # percentage
            "xp": user_data["xp"],
            "streak": user_data["streak"],
            "badges": user_data["badges"],
            "fact": disease_facts.get(class_indices[idx], "No fact available for this disease yet.")
        })

    except Exception as e:
        print("Unhandled exception in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/profile", methods=["GET"])
def profile():
    return jsonify(load_user_data())

if __name__ == "__main__":
    logger.info("Starting Flask API at http://localhost:5000")
    app.run(host="localhost", port=5000, debug=True)
    @app.before_request
    def log_request():
        logger.debug("Incoming request: %s %s", request.method, request.path)

# backend/main.py
# This file contains the Flask API for the Plant Disease Detection application.