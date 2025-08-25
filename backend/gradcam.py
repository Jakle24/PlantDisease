# gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import os

MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = (224,224)
OUTDIR = "gradcam_out"
os.makedirs(OUTDIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
# pick the last convolutional layer name dynamically
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name:
        last_conv_layer_name = layer.name
        break
else:
    raise RuntimeError("No conv layer found in model")

print("Using conv layer:", last_conv_layer_name)

def make_gradcam(img_path, outname):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)/255.0
    x = np.expand_dims(x, 0)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    # overlay
    import cv2
    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig, IMG_SIZE)
    heatmap = cv2.resize(heatmap.numpy(), IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(OUTDIR, outname)
    cv2.imwrite(save_path, overlay)
    print("Saved:", save_path)

if __name__ == "__main__":
    import glob
    sample_images = glob.glob("test_images/*.*")[:10]  # point at your test images
    for i, p in enumerate(sample_images):
        make_gradcam(p, f"gradcam_{i}.jpg")
