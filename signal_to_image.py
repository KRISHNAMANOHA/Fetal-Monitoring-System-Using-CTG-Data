import os
import matplotlib.pyplot as plt
from preprocessing import preprocess_signal


# -------------------------------------------------
# 1. GENERATE CTG IMAGE FOR A SINGLE RECORD
# -------------------------------------------------

def generate_ctg_image(df, save_path):
    """
    Convert CTG signals into CTG graph image
    """

    fhr = df["FHR"]
    uc = df["UC"]

    plt.figure(figsize=(12,6))

    # Top graph → Fetal Heart Rate
    plt.subplot(2,1,1)
    plt.plot(fhr)
    plt.title("Fetal Heart Rate")
    plt.ylabel("FHR")

    # Bottom graph → Uterine Contractions
    plt.subplot(2,1,2)
    plt.plot(uc)
    plt.title("Uterine Contractions")
    plt.ylabel("UC")
    plt.xlabel("Time")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()



# -------------------------------------------------
# 2. GENERATE IMAGES FOR ALL RECORDS
# -------------------------------------------------

def generate_all_ctg_images(dataset_folder, image_folder):

    """
    Generate CTG images for every record in dataset
    """

    os.makedirs(image_folder, exist_ok=True)

    for file in os.listdir(dataset_folder):

        # Each record has a .hea file
        if file.endswith(".hea"):

            record_name = file.replace(".hea", "")

            record_path = os.path.join(dataset_folder, record_name)

            print("Processing record:", record_name)

            try:
                df = preprocess_signal(record_path)

                save_path = os.path.join(image_folder, record_name + ".png")

                generate_ctg_image(df, save_path)

                print("Saved:", save_path)

            except Exception as e:
                print("Error processing", record_name, ":", e)



# -------------------------------------------------
# 3. RUN FILE DIRECTLY
# -------------------------------------------------

if __name__ == "__main__":

    dataset_folder =r"C:\Users\lenovo\Desktop\fetal health\data\ctu-chb-intrapartum-cardiotocography-database-1.0.0"

    image_folder =r"C:\Users\lenovo\Desktop\fetal health\images"

    generate_all_ctg_images(dataset_folder, image_folder)

    print("All CTG images generated successfully")