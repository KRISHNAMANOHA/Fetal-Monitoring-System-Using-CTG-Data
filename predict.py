import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

# -----------------------------
# DEFINE FOCAL LOSS AGAIN
# -----------------------------

def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        epsilon = 1e-7

        y_pred = K.clip(
            y_pred,
            epsilon,
            1. - epsilon
        )

        cross_entropy = -y_true * K.log(y_pred)

        weight = alpha * K.pow(
            1 - y_pred,
            gamma
        )

        loss = weight * cross_entropy

        return K.sum(loss, axis=1)

    return focal_loss_fixed

# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "fetal_health_model.keras",
    custom_objects={
        "focal_loss_fixed": focal_loss()
    }
)

print("Model loaded successfully!")

# -----------------------------
# CLASS LABELS
# -----------------------------

classes = ["Normal", "Pathological", "Suspect"]

# -----------------------------
# INPUT IMAGE PATH
# -----------------------------

img_path = r"C:\Users\lenovo\Desktop\fetal health\images\Pathological\1324.png"

# Change above path to your test image

# -----------------------------
# LOAD IMAGE
# -----------------------------

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)

# -----------------------------
# PREDICT
# -----------------------------

prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction)

print("\nPrediction Result:")

print(
    "Class:",
    classes[predicted_class]
)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)