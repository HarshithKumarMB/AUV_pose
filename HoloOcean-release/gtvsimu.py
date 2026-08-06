'''import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/home/harshith/sims/HoloOcean-release/dead_reckoning_log.csv")

# Plot step vs error
plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["error"], linewidth=1.5)

plt.xlabel("Step")
plt.ylabel("Error")
plt.title("Step vs Error")
plt.grid(True)

plt.show()'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("/home/harshith/sims/HoloOcean-release/dead_reckoning_log.csv")

# Compute delta_imu
df["delta_imu"] = (df["gt_z"] - (-df["dr_z"] / 2.5))#np.sqrt(
    #(df["gt_x"] - (df["dr_x"] / 2.5))**2 +
    #(df["gt_y"] - (-df["dr_y"] / 2.5))**2 +
    #(df["gt_z"] - (-df["dr_z"] / 2.5))#**2
#)

# Plot step vs delta_imu
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["delta_imu"], color="blue", linewidth=1.5)

plt.xlabel("Step")
plt.ylabel("Delta IMU")
plt.title("Step vs Delta IMU")
plt.grid(True)
plt.tight_layout()

plt.show()