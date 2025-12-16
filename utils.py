import os
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from dotenv import load_dotenv
load_dotenv()
MODEL_FILENAME = os.environ.get("MODEL_FILENAME")
REPO_ID = os.environ.get("REPO_ID")
IMG_SIZE = (299, 299)

# ---------------- DOWNLOAD MODEL ----------------
MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME
)

# ---------------- LOAD MODEL ----------------
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ---------------- PREDICTION ----------------
def predict_autism(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = model.predict(img_array, verbose=0)[0][0]
    label = "Autistic" if prob >= 0.5 else "Non-Autistic"
    return label, float(prob)

# ---------------- GRAD-CAM ----------------
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

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cam_image = heatmap * alpha + img
    return cam_image.astype(np.uint8)

# ---------------- FULL PIPELINE ----------------
def explain_prediction(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    label, prob = predict_autism(img_path)
    heatmap = make_gradcam_heatmap(img_array, model)
    cam_image = overlay_gradcam(img_path, heatmap)

    return label, prob, cam_image
