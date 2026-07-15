import os
import shutil
import pandas as pd
import re

# dataset path
csv_path = r"C:\Users\lenovo\Desktop\fetal health\data\fetal_health.csv"

# images folder
image_folder = r"C:\Users\lenovo\Desktop\fetal health\images"

# read dataset
df = pd.read_csv(csv_path)

# get image files
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# function to extract numbers from filename
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# sort images using extracted number
images.sort(key=extract_number)

for i, image in enumerate(images):

    label = df.iloc[i]["fetal_health"]

    if label == 1:
        folder = "Normal"
    elif label == 2:
        folder = "Suspect"
    else:
        folder = "Pathological"

    src = os.path.join(image_folder, image)
    dst = os.path.join(image_folder, folder, image)

    shutil.move(src, dst)

print("Images organized successfully!")