import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow.keras.backend as K

print("Loading model...")

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
# LOAD MODEL (WITH CUSTOM LOSS)
# -----------------------------

model = tf.keras.models.load_model(
    "fetal_health_model.keras",
    custom_objects={
        "focal_loss_fixed": focal_loss()
    }
)

val_dir = r"C:\Users\lenovo\Desktop\fetal health\images\val"

classes = ["Normal", "Pathological", "Suspect"]

y_true = []
y_pred = []

print("Evaluating validation images...")

for label, class_name in enumerate(classes):

    class_folder = os.path.join(
        val_dir,
        class_name
    )

    for img_name in os.listdir(class_folder):

        img_path = os.path.join(
            class_folder,
            img_name
        )

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

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction
        )

        y_true.append(label)

        y_pred.append(predicted_class)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes
    )
)