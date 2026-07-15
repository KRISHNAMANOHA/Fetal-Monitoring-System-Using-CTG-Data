import wfdb
import numpy as np
import pandas as pd


# ----------------------------------------------------
# 1. LOAD CTG SIGNAL FROM PHYSIONET DATASET
# ----------------------------------------------------

def load_ctg_signal(record_path):
    """
    Load CTG signal using WFDB
    Extract FHR and UC signals
    """

    record = wfdb.rdrecord(record_path)

    signal = record.p_signal

    fhr = signal[:, 0]   # Fetal Heart Rate
    uc = signal[:, 1]    # Uterine Contractions

    df = pd.DataFrame({
        "FHR": fhr,
        "UC": uc
    })

    return df


# ----------------------------------------------------
# 2. HANDLE MISSING VALUES
# ----------------------------------------------------

def handle_missing_values(df):

    df["FHR"] = df["FHR"].replace(0, np.nan)
    df["UC"] = df["UC"].replace(0, np.nan)

    df["FHR"] = df["FHR"].interpolate()
    df["UC"] = df["UC"].interpolate()

    df = df.dropna()

    return df


# ----------------------------------------------------
# 3. SMOOTH SIGNAL (NOISE REMOVAL)
# ----------------------------------------------------

def smooth_signal(df, window_size=5):

    df["FHR"] = df["FHR"].rolling(window=window_size).mean()
    df["UC"] = df["UC"].rolling(window=window_size).mean()

    df = df.dropna()

    return df


# ----------------------------------------------------
# 4. REMOVE OUTLIERS
# ----------------------------------------------------

def remove_outliers(df):

    # normal fetal heart rate range
    df = df[(df["FHR"] > 80) & (df["FHR"] < 200)]

    return df


# ----------------------------------------------------
# 5. NORMALIZE SIGNALS
# ----------------------------------------------------

def normalize_signal(df):

    df["FHR"] = (df["FHR"] - df["FHR"].min()) / (df["FHR"].max() - df["FHR"].min())

    df["UC"] = (df["UC"] - df["UC"].min()) / (df["UC"].max() - df["UC"].min())

    return df


# ----------------------------------------------------
# 6. COMPLETE PREPROCESSING PIPELINE
# ----------------------------------------------------

def preprocess_signal(record_path):

    df = load_ctg_signal(record_path)

    df = handle_missing_values(df)

    df = smooth_signal(df)

    df = remove_outliers(df)

    df = normalize_signal(df)

    return df


# ----------------------------------------------------
# 7. TEST PREPROCESSING
# ----------------------------------------------------

if __name__ == "__main__":

    # Example record path
    record_path =r"C:\Users\LENOVO\Desktop\fetalcare\data\ctu-chb-intrapartum-cardiotocography-database-1.0.0\1001"

    data = preprocess_signal(record_path)

    print("Preprocessed CTG Data:")
    print(data.head())