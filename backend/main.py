import os
import json
import sqlite3
import logging
import traceback
from io import BytesIO
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model



# ---------------------------
# Logging config
# ---------------------------
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("plantd")

def log_prediction(user_data, disease, confidence):
    logger.info(
        "[PREDICTION] Disease: %s | Confidence: %.2f%% | XP: %d | Streak: %d | Badges: %s",
        disease,
        confidence,
        user_data["xp"],
        user_data["streak"],
        user_data["badges"]
    )


# ---------------------------
# Flask + CORS
# ---------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})


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
    """
    Robust image loader: read bytes, try PIL.Image.open, convert to RGB,
    resize to IMG_SIZE and return a Keras-ready tensor.
    Saves a debug copy on failure to backend/debug_upload.jpg or debug_upload_failed.bin.
    """
    try:
        file_storage.stream.seek(0)
        data = file_storage.read()
        if not data or len(data) == 0:
            raise ValueError("Uploaded file is empty (0 bytes)")

        MAX_BYTES = 10 * 1024 * 1024
        if len(data) > MAX_BYTES:
            raise ValueError(f"Uploaded file too large ({len(data)} bytes)")

        bio = BytesIO(data)
        bio.seek(0)

        try:
            pil_img = Image.open(bio)
        except UnidentifiedImageError as e:
            # save raw bytes for inspection then re-raise a helpful error
            try:
                dbg_path = Path(__file__).resolve().parent / "debug_upload_failed.bin"
                with open(dbg_path, "wb") as fh:
                    fh.write(data)
                logger.warning("Saved debug upload to %s", dbg_path)
            except Exception:
                logger.exception("Failed to save debug upload")
            raise ValueError("Could not identify image format (PIL.UnidentifiedImageError)") from e

        pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize(IMG_SIZE)

        arr = np.asarray(pil_img) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[2] != 3:
            arr = arr[..., :3]

        return np.expand_dims(arr, 0).astype(np.float32)

    except Exception:
        # save raw bytes for inspection (best-effort)
        try:
            dbg_path = Path(__file__).resolve().parent / "debug_upload_failed.bin"
            with open(dbg_path, "wb") as fh:
                fh.write(data if 'data' in locals() else b"")
            logger.debug("Wrote debug upload to %s", dbg_path)
        except Exception:
            logger.exception("Failed to write debug upload")
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

        logger.debug("Last scan: %s | Today: %s", last_scan, today)
        logger.debug("Updated streak: %d | Updated XP: %d", user_data["streak"], user_data["xp"])
        logger.debug("Badges: %s", user_data["badges"])


        save_user_data(user_data)
        log_prediction(user_data, class_indices[idx], float(preds[idx]) * 100)

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