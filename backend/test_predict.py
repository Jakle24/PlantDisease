# test_predict.py
import requests
import os

API_URL = "http://localhost:5000/predict"
TEST_IMAGES_DIR = "../test_images"  # put 5–10 sample leaf photos here

def test_image(path):
    with open(path, "rb") as f:
        files = {"file": f}
        resp = requests.post(API_URL, files=files)
    if resp.status_code == 200:
        data = resp.json()
        print(f"{os.path.basename(path)} → {data['disease']} ({data['confidence']:.2f})")
    else:
        print(f"{os.path.basename(path)} → ERROR {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    for fname in os.listdir(TEST_IMAGES_DIR):
        if fname.lower().endswith((".jpg", ".png")):
            test_image(os.path.join(TEST_IMAGES_DIR, fname))
