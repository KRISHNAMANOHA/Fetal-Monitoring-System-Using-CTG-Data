import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("Training with Focal Loss...")

train_dir = r"C:\Users\lenovo\Desktop\fetal health\images\train"
val_dir   = r"C:\Users\lenovo\Desktop\fetal health\images\val"

# -----------------------------
# FOCAL LOSS (important)
# -----------------------------

def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred)

        weight = alpha * K.pow(1 - y_pred, gamma)

        loss = weight * cross_entropy

        return K.sum(loss, axis=1)

    return focal_loss_fixed

# -----------------------------
# DATA GENERATORS
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=18,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    color_mode='rgb'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    color_mode='rgb'
)

print("Class mapping:")
print(train_generator.class_indices)
print("Total training samples per class:")

for i, class_name in enumerate(train_generator.class_indices):
    print(
        class_name,
        np.sum(train_generator.classes == i)
    )
# -----------------------------
# COMPUTE CLASS WEIGHTS
# -----------------------------

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(
    enumerate(class_weights)
)

print("Class Weights:", class_weights)
# -----------------------------
# MODEL
# -----------------------------

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze most layers

for layer in base_model.layers[:-20]:
    layer.trainable = False

# Unfreeze last layers

for layer in base_model.layers[-40:]:
    layer.trainable = True

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(3, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=outputs
)

# -----------------------------
# COMPILE (FOCAL LOSS HERE)
# -----------------------------

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss=focal_loss(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=60,
    callbacks=[early_stop],
    class_weight=class_weights
)

model.save("fetal_health_model.keras")

print("Training completed!")