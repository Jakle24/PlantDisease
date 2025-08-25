# evaluate_model.py
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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_ds_from_dir(data_dir, img_size=(224,224), batch_size=32):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    class_names = ds.class_names
    return ds, class_names

def predict_dataset(model, ds):
    # Build arrays of filenames and labels (we'll iterate manually)
    images = []
    labels = []
    filepaths = []
    for batch, (X, y) in enumerate(ds):
        for i in range(X.shape[0]):
            images.append(X[i].numpy())
            labels.append(int(y[i].numpy()))
        # if dataset returns file path metadata change accordingly
    X_all = np.stack(images, axis=0)
    preds = model.predict(X_all, verbose=1)
    return preds, np.array(labels)

def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label',
           title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate
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
    model = load_model(args.model)
    ds, class_names = load_ds_from_dir(args.data_dir, img_size=tuple(args.img_size), batch_size=args.batch_size)
    # Normalize if you used Rescaling in training
    ds = ds.map(lambda x, y: (x / 255.0, y))

    preds, y_true = predict_dataset(model, ds)
    y_pred = np.argmax(preds, axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}, Macro Recall: {recall:.4f}, Macro F1: {f1:.4f}")
    print("\nClassification Report:\n", report)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    # Confusion matrix plot
    plot_confusion(cm, class_names, os.path.join(args.outdir, "confusion_matrix.png"))

    # Per-image CSV
    # Re-create filenames: image_dataset_from_directory doesn't give filenames by default.
    # We'll iterate through the directory structure in the same sorted order
    filenames = []
    for root, dirs, files in os.walk(args.data_dir):
        # nothing
        pass
    # Simpler: rely on dataset.unbatch to get file paths not available; instead we accept not saving filenames.
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    df.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    # Top-k accuracy
    top1 = np.mean(y_pred == y_true)
    top3 = np.mean([y_true[i] in np.argsort(preds[i])[-3:] for i in range(len(y_true))])
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w") as f:
        json.dump({
            "accuracy": float(acc),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "top1": float(top1),
            "top3": float(top3),
            "num_samples": int(len(y_true)),
            "num_classes": int(len(class_names))
        }, f, indent=4)

    print(f"Saved results to {args.outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Keras model on image directory")
    parser.add_argument("--data_dir", required=True, help="Path to test data folder (one subfolder per class)")
    parser.add_argument("--model", default="plant_disease_model.keras", help="Path to model file")
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", default="eval_results")
    args = parser.parse_args()
    main(args)
