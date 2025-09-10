# evaluate_model.py
# -----------------------------------------------------------
# Evaluate trained plant disease detection model
# -----------------------------------------------------------

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
TEST_DIR = r"C:\Users\Student\Documents\Module Dev Containers\PlantDisease\backend\dataset\test\PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------------------------------------
# Load model and class names
# -----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")

print("Loading trained model...")
model = load_model(MODEL_PATH)

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError("class_names.json not found.")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} classes.")

# -----------------------------------------------------------
# Data generator for test set
# -----------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------------------------------------
# Predictions
# -----------------------------------------------------------
print("Running predictions on test set...")
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# -----------------------------------------------------------
# Classification report
# -----------------------------------------------------------
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)

# Save to file
with open("evaluation_report.txt", "w") as f:
    f.write(report)

# -----------------------------------------------------------
# Confusion matrix
# -----------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nEvaluation complete. Results saved to evaluation_report.txt and confusion_matrix.png")
