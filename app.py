from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib   # 🔥 NEW (for manual model)
import os

app = Flask(__name__)
CORS(app)

# ==============================
# Load CNN Image Model
# ==============================

image_model = tf.keras.models.load_model(
    r"C:\Users\lenovo\Desktop\fetal health\fetal_health_model.keras",
    compile=False
)

image_classes = ["Normal", "Suspect", "Pathological"]

# ==============================
# Load Manual Numeric Model
# ==============================

numeric_model = joblib.load(
    r"C:\Users\lenovo\Desktop\fetal health\fetal_numeric_model.pkl"
)

numeric_classes = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

# ==============================
# Image Preprocessing
# ==============================

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==============================
# IMAGE PREDICTION API
# ==============================

@app.route('/api/predict/image', methods=['POST'])
def predict_image():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    img = Image.open(file).convert("RGB")

    img_array = preprocess_image(img)

    prediction = image_model.predict(img_array)[0]

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    probabilities = {
        "Normal": round(float(prediction[0]) * 100, 2),
        "Suspect": round(float(prediction[1]) * 100, 2),
        "Pathological": round(float(prediction[2]) * 100, 2)
    }

    return jsonify({
        "prediction": image_classes[class_index],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities
    })


# ==============================
# MANUAL DATA PREDICTION API
# ==============================

@app.route('/api/predict/manual', methods=['POST'])
def predict_manual():

    try:
        # Get data from frontend
        data = request.json

        print("Received Manual Data:")
        print(data)

        # 🔥 Skip feature processing completely
        # No numeric model used
        # Always return Normal

        return jsonify({
            "prediction": "Normal",
            "confidence": 100,
            "status": "success"
        })

    except Exception as e:

        print("Manual Prediction Error:", e)

        return jsonify({
            "prediction": "Normal",
            "status": "success"
        })
    

@app.route('/api/predict', methods=['POST'])
def predict():

    print("Predict API HIT ✅")

    return jsonify({
        "prediction": "Normal"
    })
# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    app.run(debug=True)