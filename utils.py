import os
import gdown
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ---------------- CONFIG ----------------
MODEL_FILENAME = "best_model_fold_5.h5"
FILE_ID = "1ioAte2SuXX_X31id_Bxs5lDem_d8yq1u"
IMG_SIZE = (299, 299)

# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists(MODEL_FILENAME):
    print("Downloading model from Google Drive...")
    gdown.download(
        id=FILE_ID,
        output=MODEL_FILENAME,
        quiet=False
    )

# ---------------- VERIFY FILE ----------------
file_size_mb = os.path.getsize(MODEL_FILENAME) / (1024 * 1024)
print(f"Model file size: {file_size_mb:.2f} MB")

if file_size_mb < 10:
    raise RuntimeError(
        "Downloaded file is not a valid model. "
        "Check Google Drive sharing permissions."
    )

# ---------------- LOAD MODEL ----------------
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(
    MODEL_FILENAME,
    compile=False
)

print("Model loaded successfully")

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
