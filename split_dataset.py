import os
import shutil
import random

# 🔧 CHANGE THIS PATH if needed
base_dir = r"C:\Users\lenovo\Desktop\fetal health\images"

# Output folders
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Split ratio
split_ratio = 0.8   # 80% train, 20% validation

# Classes
classes = ["Normal", "Suspect", "Pathological"]

print("Starting dataset split...")

for cls in classes:

    class_path = os.path.join(base_dir, cls)

    # Skip if class folder not found
    if not os.path.exists(class_path):
        print(f"Skipping missing class: {cls}")
        continue

    # Create output folders
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    images = os.listdir(class_path)

    # Keep only image files
    images = [
        img for img in images
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"\nClass: {cls}")
    print(f"Total: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")

    # Copy training images
    for img in train_images:

        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, cls, img)

        shutil.copy2(src, dst)

    # Copy validation images
    for img in val_images:

        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, cls, img)

        shutil.copy2(src, dst)

print("\nDataset split completed successfully!")

print("\nFinal structure:")
print("images/")
print("   train/")
print("   val/")