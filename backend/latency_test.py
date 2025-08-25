# latency_test.py
import time
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_sample_images(folder, img_size=(224,224), max_images=100):
    import os, glob
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)[:max_images]
    images = []
    for p in paths:
        img = load_img(p, target_size=img_size)
        arr = img_to_array(img)/255.0
        images.append(arr)
    return np.array(images), paths

def main(args):
    model = load_model(args.model)
    X, paths = load_sample_images(args.sample_folder, img_size=tuple(args.img_size), max_images=args.max_images)
    if X.shape[0] == 0:
        print("No sample images found.")
        return
    # warmup
    for _ in range(5):
        model.predict(X[:1])
    times = []
    for i in range(X.shape[0]):
        t0 = time.time()
        model.predict(X[i:i+1])
        t1 = time.time()
        times.append((t1-t0)*1000.0)  # ms
    arr = np.array(times)
    results = {
        "count": int(len(times)),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr,95)),
        "p99_ms": float(np.percentile(arr,99))
    }
    import json, os
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.outdir + "/latency_summary.json","w") as f:
        json.dump(results,f,indent=4)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="plant_disease_model.keras")
    parser.add_argument("--sample_folder", required=True, help="Folder with sample images to time")
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--outdir", default="eval_results")
    args = parser.parse_args()
    main(args)
