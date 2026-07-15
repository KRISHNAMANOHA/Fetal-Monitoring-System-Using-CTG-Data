import os

val_dir = r"C:\Users\lenovo\Desktop\fetal health\images\val"

classes = ["Normal", "Suspect", "Pathological"]

print("Validation Set Distribution:\n")

for cls in classes:

    path = os.path.join(val_dir, cls)

    if os.path.exists(path):

        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith((".png",".jpg",".jpeg"))
        ])

        print(f"{cls}: {count}")

    else:

        print(f"{cls}: folder missing")