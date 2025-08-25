# api_batch_test.py
import os, time, argparse, csv, requests, glob, json
from tqdm import tqdm

def main(args):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        files.extend(glob.glob(os.path.join(args.image_folder, ext)))
    files = sorted(files)[:args.max_images]
    if not files:
        print("No images found.")
        return
    out_rows = []
    for path in tqdm(files):
        with open(path, "rb") as f:
            t0 = time.time()
            try:
                r = requests.post(args.url, files={"file": f}, timeout=args.timeout)
                dt = time.time() - t0
                status = r.status_code
                try:
                    data = r.json()
                except Exception as e:
                    data = {"error": "invalid_json", "text": r.text[:200]}
            except Exception as e:
                dt = time.time() - t0
                status = "error"
                data = {"error": str(e)}
        out_rows.append({
            "file": os.path.basename(path),
            "status": status,
            "time_s": dt,
            "response": json.dumps(data)
        })
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "api_batch_results.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["file","status","time_s","response"])
        writer.writeheader()
        writer.writerows(out_rows)
    print("Saved:", csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--outdir", default="api_test_results")
    args = parser.parse_args()
    main(args)
