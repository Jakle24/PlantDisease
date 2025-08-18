# backend/main.py
import os
import json
import sqlite3
import logging
from io import BytesIO
from datetime import date, timedelta
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import traceback

# ---------------------------
# Logging config
# ---------------------------
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("plantd")

# ---------------------------
# Flask + CORS
# ---------------------------
app = Flask(__name__)
CORS(app)  # dev: allow all origins; restrict in production

# ---------------------------
# Config
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
FACTS_PATH = BASE_DIR / "disease_facts.json"
DB_PATH = BASE_DIR / "userdata.db"
IMG_SIZE = (224, 224)

# ---------------------------
# DB helpers (SQLite, lightweight)
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        xp INTEGER DEFAULT 0,
        streak INTEGER DEFAULT 0,
        last_scan TEXT,
        badges TEXT DEFAULT '[]'
    )
    """)
    c.execute("INSERT OR IGNORE INTO users(user_id, xp, streak, last_scan, badges) VALUES (?, ?, ?, ?, ?)",
              ("default", 0, 0, None, "[]"))
    conn.commit()
    conn.close()
    logger.info("Initialized sqlite DB at %s", DB_PATH)

def load_user_data(user_id="default"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT xp, streak, last_scan, badges FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        xp, streak, last_scan, badges_json = row
        try:
            badges = json.loads(badges_json)
        except Exception:
            badges = []
        return {"xp": xp or 0, "streak": streak or 0, "last_scan": last_scan, "badges": badges}
    return {"xp": 0, "streak": 0, "last_scan": None, "badges": []}

def save_user_data(data, user_id="default"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    badges_json = json.dumps(data.get("badges", []))
    c.execute("""
        INSERT INTO users(user_id, xp, streak, last_scan, badges)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            xp=excluded.xp,
            streak=excluded.streak,
            last_scan=excluded.last_scan,
            badges=excluded.badges
    """, (user_id, data.get("xp", 0), data.get("streak", 0), data.get("last_scan"), badges_json))
    conn.commit()
    conn.close()

# init DB early
init_db()

# ---------------------------
# Model loading
# ---------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please place model file here.")

logger.info("Loading trained model from %s ...", MODEL_PATH)
model = load_model(str(MODEL_PATH))
logger.info("Model loaded successfully.")

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError("class_names.json not found. Please export from training.")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
class_indices = {i: name for i, name in enumerate(class_names)}

if FACTS_PATH.exists():
    with open(FACTS_PATH, "r") as f:
        disease_facts = json.load(f)
else:
    disease_facts = {}

# ---------------------------
# Image prep helper (BytesIO safe)
# ---------------------------
def prepare_image(file_storage):
    try:
        # Ensure pointer at start, read bytes, wrap in BytesIO
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
    logger.debug("Incoming request: %s %s", request.method, request.path)

# ---------------------------
# Routes
# ---------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Request received at /predict - content_type: %s", request.content_type)
        logger.debug("Request files keys: %s", list(request.files.keys()))

        file = request.files.get("file")
        if not file:
            logger.warning("No file in request.files -> returning 400")
            return jsonify({"error": "No file provided"}), 400

        logger.info("Received file: %s (%s)", file.filename, file.content_type)

        try:
            x = prepare_image(file)
        except Exception as e:
            logger.error("Error processing image: %s", e)
            traceback.print_exc()
            return jsonify({"error": "Invalid image or could not process file"}), 400

        try:
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
        except Exception as e:
            logger.error("Error during model.predict: %s", e)
            traceback.print_exc()
            return jsonify({"error": "Model prediction failed"}), 500

        # gamification
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

        return jsonify({
            "disease": class_indices[idx],
            "confidence": round(float(preds[idx]) * 100, 2),  # percent
            "xp": user_data["xp"],
            "streak": user_data["streak"],
            "badges": user_data["badges"],
            "fact": disease_facts.get(class_indices[idx], "No fact available for this disease yet.")
        })

    except Exception as e:
        logger.exception("Unhandled exception in /predict:")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/profile", methods=["GET"])
def profile():
    return jsonify(load_user_data())

# ---------------------------
if __name__ == "__main__":
    logger.info("Starting Flask API at http://localhost:5000 (dev server)")
    # For dev on Windows, use python main.py or waitress.
    app.run(host="localhost", port=5000, debug=True)
# For production, consider using a WSGI server like Gunicorn or Waitress.
# Example: waitress-serve --port=5000 main:app