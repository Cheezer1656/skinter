import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

BASE_DIR = "."

DATA_DIR = BASE_DIR + "/data"

CSV_PATH = DATA_DIR + "/HAM10000_metadata.csv"

IMAGE_DIR_1 = DATA_DIR + "/HAM10000_images_part_1"

IMAGE_DIR_2 = DATA_DIR + "/HAM10000_images_part_2"

df = pd.read_csv(CSV_PATH)

df["dx"].value_counts().plot(kind="bar")
plt.title("HAM10000 Disease Classes")
plt.xlabel("Class")
plt.ylabel("Images")
plt.show()

image_paths = {}

for folder in [IMAGE_DIR_1, IMAGE_DIR_2]:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            image_id = filename.replace(".jpg", "")
            image_paths[image_id] = os.path.join(folder, filename)

print("Total images found:", len(image_paths))

df["path"] = df["image_id"].map(image_paths)

df.head()

df["path"].isna().sum()

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["dx"])

encoder.classes_

from sklearn.model_selection import GroupShuffleSplit

print(f"Total images before split: {len(df)}")

# 1. Split off the Training set (80%) and a temporary set (20%) for Val/Test
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss1.split(df, df["label"], groups=df["lesion_id"]))

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# 2. Split the temporary set in half to get Validation (10%) and Test (10%) sets
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, temp_df["label"], groups=temp_df["lesion_id"]))

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print(f"Train set size: {len(train_df)}")
print(f"Val set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

PROCESSED_DIR = "processed"

train_df.to_csv(PROCESSED_DIR + "/train.csv", index=False)
val_df.to_csv(PROCESSED_DIR + "/val.csv", index=False)
test_df.to_csv(PROCESSED_DIR + "/test.csv", index=False)
