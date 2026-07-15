import os
import shutil
import random

print("Oversampling Pathological images...")

pathological_train = r"C:\Users\lenovo\Desktop\fetal health\images\train\Pathological"

files = [
    f for f in os.listdir(pathological_train)
    if f.lower().endswith(".png")
]

current_count = len(files)

print("Current Pathological count:", current_count)

target_count = 180   # match Suspect approx

while len(files) < target_count:

    file = random.choice(files)

    src = os.path.join(pathological_train, file)

    new_name = f"copy_{len(files)}_{file}"

    dst = os.path.join(
        pathological_train,
        new_name
    )

    shutil.copy(src, dst)

    files.append(new_name)

print("New Pathological count:", len(files))
print("Oversampling completed!")