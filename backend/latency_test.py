import time, os, json, argparse, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

def main(args):
    model = load_model(args.model)

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        image_size=tuple(args.img_size),
        batch_size=1,
        shuffle=True
    ).take(args.max_images)
    ds = ds.map(lambda x, y: (x/255.0, y))

    times = []
    # Warmup
    for images, _ in ds.take(5):
        model.predict(images, verbose=0)

    for images, _ in ds:
        start = time.time()
        _ = model.predict(images, verbose=0)
        times.append((time.time()-start)*1000)

    arr = np.array(times)
    results = {
        "count": len(times),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr,95)),
        "p99_ms": float(np.percentile(arr,99))
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir,"latency_summary.json"),"w") as f:
        json.dump(results,f,indent=4)
    print(results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True, help="Folder with class subfolders")
    p.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--outdir", default="eval_results")
    args = p.parse_args()
    main(args)
