# main.py: Backend for Plant Disease Detection
# ------------------------------------------------
# 1. Download and prepare dataset via KaggleHub
# 2. Build and train MobileNetV2 transfer learning model
# 3. Save model for serving in API

import os
import kagglehub
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Step 1: Download dataset
print("Downloading PlantVillage dataset...")
DATA_PATH = kagglehub.dataset_download("emmarex/plantdisease")
print(f"Dataset available at {DATA_PATH}")

# Step 2: Build tf.data pipelines using image_dataset_from_directory
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")

# Normalize data
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Step 3: Define model with Transfer Learning
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    pooling='avg'
)
base_model.trainable = False

model = Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train initial classifier head
print("Training classifier head...")
history1 = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# Step 5: Fine-tune last layers of base
print("Fine-tuning base model...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# Step 6: Save trained model
MODEL_PATH = 'plant_disease_model.h5'
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Step 7: Create Flask API for serving predictions
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
# Load the trained model
model = load_model(MODEL_PATH)
# Build class index mapping
class_indices = {v: k for k, v in enumerate(class_names)}

# Helper to preprocess incoming image files
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
    idx = np.argmax(preds)
    return jsonify({
        "disease": class_indices[idx],
        "confidence": float(preds[idx])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
