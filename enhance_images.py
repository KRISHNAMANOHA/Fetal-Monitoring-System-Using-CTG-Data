import os
import cv2

base_dir = r"C:\Users\lenovo\Desktop\fetal health\images"

folders = ["train", "val"]

classes = ["Normal", "Suspect", "Pathological"]

print("Enhancing CTG images...")

for folder in folders:

    for cls in classes:

        path = os.path.join(base_dir, folder, cls)

        for img_name in os.listdir(path):

            if img_name.lower().endswith(".png"):

                img_path = os.path.join(path, img_name)

                img = cv2.imread(img_path)

                # Convert to grayscale
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                # Increase contrast
                enhanced = cv2.equalizeHist(gray)

                # Resize again (clean size)
                enhanced = cv2.resize(
                    enhanced,
                    (224,224)
                )

                # Save back
                cv2.imwrite(
                    img_path,
                    enhanced
                )

print("Image enhancement completed!")