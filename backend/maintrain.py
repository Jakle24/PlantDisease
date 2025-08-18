# maintrain.py — Train Plant Disease Detection Model from local dataset
# ---------------------------------------------------------------------
# 1. Loads dataset from backend/dataset/
# 2. Trains MobileNetV2 with transfer learning
# 3. Saves model as .keras + class_names.json

import os
import json
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all domains for now


# Step 1: Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset folder not found at {DATA_PATH}")

# Step 2: Load datasets
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

# Step 3: Save class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Detected {num_classes} classes: {class_names}")

with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("💾 Saved class_names.json")

# Step 4: Normalization & prefetch
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Step 5: Build model
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

# Step 6: Train classifier head
print("🚀 Training classifier head...")
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Step 7: Fine-tune base model
print("🔧 Fine-tuning base model...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Step 8: Save trained model
MODEL_PATH = "plant_disease_model.keras"
model.save(MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
