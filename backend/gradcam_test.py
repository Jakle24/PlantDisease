import argparse, os, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)/255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, out_path, alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    heatmap = np.uint8(255*heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:,:3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(np.expand_dims(jet_heatmap,0), (img.shape[0], img.shape[1]))[0].numpy()
    superimposed_img = jet_heatmap*alpha + img
    plt.imshow(superimposed_img.astype("uint8"))
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main(args):
    model = load_model(args.model)
    img_array = get_img_array(args.image, size=tuple(args.img_size))
    heatmap = make_gradcam_heatmap(model, img_array, args.layer)
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "gradcam.png")
    save_gradcam(args.image, heatmap, outpath)
    print(f"Saved Grad-CAM to {outpath}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--layer", default="Conv_1")
    p.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    p.add_argument("--outdir", default="gradcam_results")
    args = p.parse_args()
    main(args)
