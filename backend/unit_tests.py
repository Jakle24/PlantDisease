# unit_tests.py
import json
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "plant_disease_model.keras"
CLASS_FILE = "class_names.json"

def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"

def test_classnames_exists():
    assert os.path.exists(CLASS_FILE), f"{CLASS_FILE} not found"
    with open(CLASS_FILE) as f:
        cs = json.load(f)
    assert isinstance(cs, list) and len(cs) > 0

def test_model_shape():
    model = load_model(MODEL_PATH)
    # model.output_shape -> (None, num_classes)
    out_shape = model.output_shape
    assert isinstance(out_shape, tuple) or (isinstance(out_shape, list) and len(out_shape) > 0)
    # if tuple:
    if isinstance(out_shape, tuple):
        assert len(out_shape) == 2
    print("Model output shape:", out_shape)

if __name__ == "__main__":
    test_model_exists()
    test_classnames_exists()
    test_model_shape()
    print("All quick checks passed.")
