import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_ds_from_dir(data_dir, img_size=(224,224), batch_size=32):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    return ds, ds.class_names

def predict_dataset(model, ds):
    y_true, y_pred, y_probs = [], [], []

    for X_batch, y_batch in ds:
        preds = model.predict(X_batch, verbose=0)
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label',
           title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    print(f"✅ Loading model: {args.model}")
    model = load_model(args.model)

    print(f"📂 Loading dataset from: {args.data_dir}")
    ds, class_names = load_ds_from_dir(args.data_dir, tuple(args.img_size), args.batch_size)
    ds = ds.map(lambda x, y: (x/255.0, y))  # normalize

    print("🔎 Running predictions...")
    y_true, y_pred, y_probs = predict_dataset(model, ds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save text report
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    # Save JSON summary
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w") as f:
        json.dump({
            "accuracy": float(acc),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "num_samples": int(len(y_true)),
            "num_classes": int(len(class_names))
        }, f, indent=4)

    # Save CSV predictions
    df = pd.DataFrame({
        "y_true": [class_names[i] for i in y_true],
        "y_pred": [class_names[i] for i in y_pred],
    })
    df.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    # Confusion matrix plot
    plot_confusion(cm, class_names, os.path.join(args.outdir, "confusion_matrix.png"))

    print(f"🎉 Results saved to {args.outdir}")
    print(f"- Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to test data folder")
    parser.add_argument("--model", default="plant_disease_model.keras")
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", default="eval_results")
    args = parser.parse_args()
    main(args)
