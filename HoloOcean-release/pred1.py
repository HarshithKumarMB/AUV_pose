import pandas as pd
import numpy as np
import joblib
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# =====================================================
# Load model
# =====================================================

model = joblib.load("obstacle_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =====================================================
# Load obstacle maps
# =====================================================

map_df1 = pd.read_csv(
    "/home/harshith/Downloads/estimated_3d_obstacle_coordinates_1m.csv",
    low_memory=False
)

map_df2 = pd.read_csv(
    "/home/harshith/Downloads/seabed_coordinates_1m_labeled.csv",
    low_memory=False
)

map_df = pd.concat([map_df1, map_df2], ignore_index=True)

# Remove duplicate headers
map_df = map_df[map_df["obstacle_type"] != "obstacle_type"]

# Convert coordinates
for col in ["x", "y", "z"]:
    map_df[col] = pd.to_numeric(map_df[col], errors="coerce")

map_df = map_df.dropna(
    subset=["x", "y", "z", "obstacle_type"]
)

# Group by obstacle type
obstacle_maps = {
    obstacle: group.reset_index(drop=True)
    for obstacle, group in map_df.groupby("obstacle_type")
}

# =====================================================
# Input folder
# =====================================================

input_folder = "/home/harshith/sims/HoloOcean-release/pointclouds"

csv_files = glob.glob(
    os.path.join(input_folder, "*.csv")
)

print(f"Found {len(csv_files)} files")

# =====================================================
# Live plot setup
# =====================================================

plt.ion()  # Interactive mode ON

fig, ax = plt.subplots(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')

'''ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Live Calculated XYZ Points")

calculated_scatter = None'''
calc_points_x = []
calc_points_y = []
#calc_points_z = []

# =====================================================
# Process files
# =====================================================



for csv_file in csv_files:

    print(f"\nProcessing {csv_file}")

    input_df = pd.read_csv(csv_file)

    # -------------------------------------------------
    # Predict obstacle class
    # -------------------------------------------------

    predictions = model.predict(
        input_df[["x", "y", "z"]]
    )

    input_df["predicted_obstacle"] = (
        label_encoder.inverse_transform(predictions)
    )

    # -------------------------------------------------
    # Find nearest obstacle
    # -------------------------------------------------

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

    # -------------------------------------------------
# Calculate XYZ from range and angle
# + Live plotting
# -------------------------------------------------

    calc_x_list = []
    calc_y_list = []
    calc_z_list = []

    for idx, row in input_df.iterrows():

        theta = row["theta"]

        calc_x = row["x"]

        calc_y = (
            row["nearest_y"]
            - (row["r"] * np.sin(theta))
        )

        calc_z = (
            row["nearest_z"]
            + (row["r"] * np.cos(theta))
        )

        calc_x_list.append(calc_x)
        calc_y_list.append(calc_y)
        calc_z_list.append(calc_z)

        # Store points for plotting
        calc_points_x.append(calc_x)
        calc_points_y.append(calc_y)

        # Update every 10 points
        if idx % 10 == 0:

            ax.clear()

            ax.scatter(
                calc_points_x,
                calc_points_y,
                c="green",
                s=10,
                label="Calculated Points"
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Live Calculated X-Y Coordinates")
            ax.grid(True)
            ax.legend()

            plt.draw()
            plt.pause(0.001)

    input_df["calc_x"] = calc_x_list
    input_df["calc_y"] = calc_y_list
    input_df["calc_z"] = calc_z_list
    # -------------------------------------------------
    # Filter
    # -------------------------------------------------

filtered_df = input_df[input_df["distance"] < 5].copy()

    # -------------------------------------------------
    # Print comparison
    # -------------------------------------------------

print(
    filtered_df[
        [
                "calc_x",
                "calc_y",
                "calc_z",
                "nearest_x",
                "nearest_y",
                "nearest_z",
                "distance"
        ]
        ]
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

output_file = os.path.join(
        input_folder,
        f"predicted_{os.path.basename(csv_file)}"
    )

filtered_df.to_csv(
        output_file,
        index=False
    )

print(
        f"Saved {len(filtered_df)} rows to {output_file}"
    )

    # =================================================
# Y-Z Plot
# =================================================

if len(filtered_df) > 0:

    plt.figure(figsize=(10, 8))

    # Measured points
    plt.scatter(
        filtered_df["y"],
        filtered_df["z"],
        c="blue",
        s=15,
        label="Measured"
    )

    # Nearest obstacle points
    plt.scatter(
        filtered_df["nearest_y"],
        filtered_df["nearest_z"],
        c="red",
        s=15,
        label="Nearest Obstacle"
    )

    # Draw lines connecting matched points
    for _, row in filtered_df.iterrows():

        plt.plot(
            [row["y"], row["nearest_y"]],
            [row["z"], row["nearest_z"]],
            color="gray",
            alpha=0.3
        )

    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title(
        f"Measured vs Nearest Obstacle (Y-Z Plane)\n{os.path.basename(csv_file)}"
    )
    plt.legend()
    plt.grid(True)

    plt.axis("equal")

    plt.tight_layout()
    plt.show()

# =====================================================
# Save final accumulated plot
# =====================================================

final_plot = os.path.join(
    input_folder,
    "all_calculated_xy_points.png"
)

ax.clear()

ax.scatter(
    calc_points_x,
    calc_points_y,
    c="green",
    s=10,
    label="Calculated Points"
)




print("X coordinates: ", calc_points_x)
print("Y coordinates: ", calc_points_y)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(" X-Y Coordinates from Sonar calculations")
ax.grid(True)
ax.legend()
ax.set_aspect('equal', adjustable='box')

fig.savefig(
    final_plot,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved final plot to {final_plot}")

plt.ioff()
plt.show()

diff_values = (
    ((df["y"] + df["r"] * np.sin(df["theta"])) - df["calc_y"])
    - ((df["z"] - df["r"] * np.cos(df["theta"])) - df["calc_z"])
)

fig_diff, ax_diff = plt.subplots(figsize=(10, 6))

ax_diff.scatter(
    range(len(diff_values)),
    diff_values,
    c="red",
    s=10,
    label="Difference"
)

ax_diff.set_xlabel("Point Number")
ax_diff.set_ylabel("Difference")
ax_diff.set_title("Difference vs Point Number")
ax_diff.grid(True)
ax_diff.legend()

fig_diff.savefig(
    os.path.join(input_folder, "difference_vs_point_number.png"),
    dpi=300,
    bbox_inches="tight"
)