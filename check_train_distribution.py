import os

train_dir = r"C:\Users\lenovo\Desktop\fetal health\images\train"

classes = ["Normal", "Suspect", "Pathological"]

print("Train Set Distribution:\n")

for cls in classes:

    path = os.path.join(train_dir, cls)

    if os.path.exists(path):

        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith((".png",".jpg",".jpeg"))
        ])

        print(f"{cls}: {count}")

    else:

        print(f"{cls}: folder missing")