import argparse, os, requests, json

def main(args):
    results = []
    for root, _, files in os.walk(args.data_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg",".png",".jpeg")): continue
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                r = requests.post(args.url, files={"file": f})
                try:
                    results.append({"file": fname, **r.json()})
                except Exception:
                    results.append({"file": fname, "error": r.text})
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir,"api_results.json")
    with open(outpath,"w") as f: json.dump(results,f,indent=2)
    print(f"Saved API results to {outpath}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--data_dir", required=True, help="Folder of test images")
    p.add_argument("--outdir", default="eval_results")
    args = p.parse_args()
    main(args)
