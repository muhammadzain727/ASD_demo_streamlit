import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import gdown
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
load_dotenv()
# MODEL_PATH = "best_model_fold_5.h5"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME")

# 🔽 Google Drive DIRECT download link
# Example original link:
# https://drive.google.com/file/d/1AbCdEfGhIjKlMnOP/view?usp=sharing
# FILE ID = 1AbCdEfGhIjKlMnOP
MODEL_URL = os.environ.get("MODEL_URL")

IMG_SIZE = (299, 299)

if not os.path.exists(MODEL_FILENAME):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
else:
    print("✅ Model already exists. Skipping download.")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = tf.keras.models.load_model(MODEL_FILENAME)
print("🧠 Model loaded successfully")

# model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

def predict_autism(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = model.predict(img_array)[0][0]
    label = "Autistic" if prob >= 0.5 else "Non-Autistic"

    return label, float(prob)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="mixed10"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img

    return superimposed_img.astype(np.uint8)


def explain_prediction(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    label, prob = predict_autism(img_path)
    heatmap = make_gradcam_heatmap(img_array, model)
    cam_image = overlay_gradcam(img_path, heatmap)

    return label, prob, cam_image


