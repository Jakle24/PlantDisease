from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = load_model("plant_disease_model.h5")
class_indices = {v:k for k,v in train_data.class_indices.items()}

def prepare_image(file):
    img = load_img(file, target_size=(224, 224))
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
