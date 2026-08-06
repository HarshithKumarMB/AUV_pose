'''import joblib
import numpy as np

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("obstacle_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =====================================================
# USER INPUT
# =====================================================

x = float(input("Enter X: "))
y = float(input("Enter Y: "))
z = float(input("Enter Z: "))

sample = np.array([[x, y, z]])

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(sample)

obstacle_name = label_encoder.inverse_transform(prediction)

print("\nPredicted Obstacle:")
print(obstacle_name[0],obstacle_name)'''

#######
'''

import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("obstacle_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("/home/harshith/Downloads/estimated_3d_obstacle_coordinates_1m.csv")

# Input
x = float(input("Enter X: "))
y = float(input("Enter Y: "))
z = float(input("Enter Z: "))

sample = pd.DataFrame([[x, y, z]], columns=["x", "y", "z"])

# Predict obstacle
prediction = model.predict(sample)
obstacle = label_encoder.inverse_transform(prediction)[0]

print(f"\nPredicted Obstacle: {obstacle}")

# Only rows of predicted obstacle
subset = df[df["obstacle_type"] == obstacle].copy()

# Compute distance
subset["distance"] = np.sqrt(
    (subset["x"] - x) ** 2 +
    (subset["y"] - y) ** 2 +
    (subset["z"] - z) ** 2
)

nearest = subset.loc[subset["distance"].idxmin()]

print("\nNearest matching obstacle:")
print(f"Type: {nearest['obstacle_type']}")
print(f"X: {nearest['x']}")
print(f"Y: {nearest['y']}")
print(f"Z: {nearest['z']}")
print(f"Distance: {nearest['distance']:.2f}")'''

##########

import pandas as pd
import numpy as np
import joblib
import glob
import os

# Load model
model = joblib.load("obstacle_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Original obstacle dataset
map_df1 = pd.read_csv(
    "/home/harshith/Downloads/estimated_3d_obstacle_coordinates_1m.csv",
    low_memory=False
)

map_df2 = pd.read_csv(
    "/home/harshith/Downloads/seabed_coordinates_1m_labeled.csv",
    low_memory=False
)

map_df = pd.concat([map_df1, map_df2], ignore_index=True)
obstacle_maps = {
    obstacle: group.reset_index(drop=True)
    for obstacle, group in map_df.groupby("obstacle_type")
}

# Remove duplicate headers accidentally embedded in CSVs
map_df = map_df[map_df["obstacle_type"] != "obstacle_type"]

# Convert coordinates to numeric
for col in ["x", "y", "z"]:
    map_df[col] = pd.to_numeric(map_df[col], errors="coerce")

# Drop bad rows
map_df = map_df.dropna(subset=["x", "y", "z", "obstacle_type"])

'''# New coordinates to classify
input_df = pd.read_csv("/home/harshith/sims/HoloOcean-release/pointclouds/pc_501.csv")

# Predict obstacle type
predictions = model.predict(input_df[["x", "y", "z"]])
input_df["predicted_obstacle"] = label_encoder.inverse_transform(predictions)

# Find nearest coordinate of the predicted obstacle
nearest_x = []
nearest_y = []
nearest_z = []
nearest_distance = []

for _, row in input_df.iterrows():

    obstacle = row["predicted_obstacle"]

    subset = map_df[map_df["obstacle_type"] == obstacle]

    distances = np.sqrt(
        (subset["x"] - row["x"])**2 +
        (subset["y"] - row["y"])**2 +
        (subset["z"] - row["z"])**2
    )

    idx = distances.idxmin()

    nearest_x.append(map_df.loc[idx, "x"])
    nearest_y.append(map_df.loc[idx, "y"])
    nearest_z.append(map_df.loc[idx, "z"])
    nearest_distance.append(distances.loc[idx])

input_df["nearest_x"] = nearest_x
input_df["nearest_y"] = nearest_y
input_df["nearest_z"] = nearest_z
input_df["distance"] = nearest_distance

input_df.to_csv("predictions.csv", index=False)

print(input_df.head())'''

# --------------------------
# Input folder
# --------------------------
input_folder = "/home/harshith/sims/HoloOcean-release/pointclouds"

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

print(f"Found {len(csv_files)} files")

for csv_file in csv_files:

    print(f"Processing {csv_file}")

    input_df = pd.read_csv(csv_file)

    predictions = model.predict(
        input_df[["x", "y", "z"]]
    )

    input_df["predicted_obstacle"] = (
        label_encoder.inverse_transform(predictions)
    )

    nearest_x = []
    nearest_y = []
    nearest_z = []
    nearest_distance = []

    for _, row in input_df.iterrows():

        obstacle = row["predicted_obstacle"]

        if obstacle not in obstacle_maps:
            nearest_x.append(np.nan)
            nearest_y.append(np.nan)
            nearest_z.append(np.nan)
            nearest_distance.append(np.nan)
            continue

        subset = obstacle_maps[obstacle]

        distances = np.sqrt(
            (subset["x"] - row["x"])**2 +
            (subset["y"] - row["y"])**2 +
            (subset["z"] - row["z"])**2
        )

        idx = distances.idxmin()

        nearest_x.append(subset.loc[idx, "x"])
        nearest_y.append(subset.loc[idx, "y"])
        nearest_z.append(subset.loc[idx, "z"])
        nearest_distance.append(distances.loc[idx])

    input_df["nearest_x"] = nearest_x
    input_df["nearest_y"] = nearest_y
    input_df["nearest_z"] = nearest_z
    input_df["distance"] = nearest_distance

    output_file = os.path.join(
        input_folder,
        f"predicted_{os.path.basename(csv_file)}"
    )

    filtered_df = input_df[input_df["distance"] < 10]
    output_file = os.path.join(
    input_folder,
    f"predicted_{os.path.basename(csv_file)}")
    filtered_df.to_csv(output_file, index=False)
    print(f"Saved {len(filtered_df)} of {len(input_df)} rows to {output_file}")

    print(f"Saved: {output_file}")