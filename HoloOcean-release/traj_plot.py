import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
# Load CSV
df = pd.read_csv("wp_c.csv")

# Create figure
plt.figure(figsize=(10, 8))

# Ground truth trajectory
plt.plot(
    df["x"],
    df["z"],
    label="Ground Truth",
    linewidth=3,
    color="blue"
)

# Dead reckoning trajectory
'''plt.plot(
    df["dr_x"],
    df["dr_z"],
    label="Dead Reckoning",
    linewidth=2,
    color="red",
    linestyle="--"
)
'''
# EKF trajectory
plt.plot(
    df["ekf_x"],
    df["ekf_z"],
    label="EKF",
    linewidth=2,
    color="green"
)

# Mark start and end
plt.scatter(
    df["x"].iloc[0],
    df["z"].iloc[0],
    color="black",
    s=100,
    marker="o",
    label="Start"
)

plt.scatter(
    df["x"].iloc[-1],
    df["z"].iloc[-1],
    color="purple",
    s=100,
    marker="x",
    label="End"
)

plt.xlabel("X Position (m)")
plt.ylabel("Z Position (m)")
plt.title("Trajectory Comparison")
plt.grid(True)
plt.axis("equal")
plt.legend()

# Save image
plt.savefig("trajectory_comparison_z_ekf.png", dpi=300, bbox_inches="tight")

# Optional display
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(
    range(len(df)),
    ((df["x"] - df["ekf_x"])**2 + (df["y"] - df["ekf_y"])**2 + (df["z"] - df["ekf_z"])**2)**0.5,
    label="Error",
    linewidth=3,
    color="blue"
)
plt.xlabel("Time Step")
plt.ylabel("Position Error (m)")
plt.title("Position Error Over Time (EKF)")
plt.grid(True)
plt.legend()
plt.savefig("trajectory_error_z_ekf.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved trajectory_comparison_z_ekf.png")