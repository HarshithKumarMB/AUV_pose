import holoocean
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# ===============================
# CREATE OUTPUT FOLDERS
# ===============================
os.makedirs("sidescan_images", exist_ok=True)
os.makedirs("pointclouds", exist_ok=True)

# ===============================
# REAL-TIME PLOT SETUP
# ===============================
plt.ion()
fig, ax = plt.subplots()

# ===============================
# SCENARIO
# ===============================
scenario = {
    "name": "testing",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "agents": [
        {
            "agent_name": "rov",
            "agent_type": "BlueROV2",
            "location": [1.0, 2.0, -3.0],
            "control_scheme": 0,
            "sensors": [
                {"sensor_name": "imu_1", "sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 20},
                {"sensor_name": "imu_2", "sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 30},
                {"sensor_name": "pose", "sensor_type": "PoseSensor"},
                {"sensor_name": "orient", "sensor_type": "OrientationSensor"},

                # ✅ SIDESCAN ONLY
                {
                    "sensor_name": "sidescan",
                    "sensor_type": "SidescanSonar",
                    "socket": "IMUSocket",
                    "rotation": [0, -90, 0],
                    "Hz": 10,
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": 70.0,
                        "RangeBins": 256,   # length of scanline
                        "AzimuthBins": 256,
                        "AddNoise": True
                    }
                }
            ]
        }
    ]
}

env = holoocean.make(scenario_cfg=scenario)

# ===============================
# SONAR CONFIG
# ===============================
cfg = scenario["agents"][0]["sensors"][-1]["configuration"]

bins = cfg["RangeBins"]
ranges = np.linspace(cfg["RangeMin"], cfg["RangeMax"], bins)

# angles (for point cloud approx)
angles = np.linspace(-np.pi/2, np.pi/2, bins)

# ===============================
# WATERFALL BUFFER
# ===============================
waterfall = []
max_rows = 3000

# ===============================
# WAYPOINTS
# ===============================
waypoints = [
    [1.0, 2.0, -10.0],
    [5.0, 6.0, -10.0],
    [-1.0, -2.0, -10.0],
    [11.0, 12.0, -10.0]
]

current_wp = 0
command = np.zeros(8)

# ===============================
# IMU LOGGING
# ===============================
with open("imu_log3.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "step",
        "imu1_ax","imu1_ay","imu1_az",
        "imu1_gx","imu1_gy","imu1_gz",
        "imu2_ax","imu2_ay","imu2_az",
        "imu2_gx","imu2_gy","imu2_gz"
    ])

    # ===============================
    # MAIN LOOP
    # ===============================
    for step in range(4000):

        state = env.step(command)

        # -------------------------------
        # POSITION
        # -------------------------------
        pose = state["pose"]

        position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])

        target = np.array(waypoints[current_wp])
        error = target - position
        dist = np.linalg.norm(error)

        print(f"Step {step} | Dist {dist:.2f}")

        # waypoint logic
        if dist < 0.5:
            print(f"✅ Reached waypoint {current_wp}")
            current_wp += 1

            if current_wp >= len(waypoints):
                print("✅ All waypoints completed!")
                break
            continue

        # simple control
        command = np.array([
            error[2], error[2], error[2], error[2],
            error[0] + error[1],
            error[0] - error[1],
            error[1],
            -error[1]
        ])

        # -------------------------------
        # IMU LOGGING
        # -------------------------------
        imu1 = state["imu_1"]
        imu2 = state["imu_2"]

        acc1, gyro1 = imu1
        acc2, gyro2 = imu2

        writer.writerow([
            step,
            acc1[0], acc1[1], acc1[2],
            gyro1[0], gyro1[1], gyro1[2],
            acc2[0], acc2[1], acc2[2],
            gyro2[0], gyro2[1], gyro2[2]
        ])

        # -------------------------------
        # ✅ SIDESCAN PROCESSING
        # -------------------------------
        if "sidescan" in state:
            raw = np.array(state["sidescan"], dtype=np.float32)

            # ✅ normalize
            mn, mx = raw.min(), raw.max()
            if mx > mn:
                vis = (raw - mn) / (mx - mn)
            else:
                vis = np.zeros_like(raw)

            # ✅ convert to image row
            vis = (vis * 255).astype(np.uint8)

            # ✅ build waterfall
            waterfall.append(vis)
            if len(waterfall) > max_rows:
                waterfall.pop(0)

            img = np.array(waterfall)

            # ✅ visualize
            ax.clear()
            ax.imshow(img, cmap="gray", aspect='auto')
            ax.set_title("Sidescan Waterfall")
            ax.set_xlabel("Beam Index")
            ax.set_ylabel("Time (Frames)")
            plt.pause(0.001)

            # ✅ save image occasionally
            if step % 3 == 0:
                plt.imsave(f"sidescan_images/sidescan_{step}.png", img, cmap="gray")

            # -------------------------------
            # ✅ SIMPLE POINT CLOUD (APPROX)
            # -------------------------------
            points = []
            threshold = 200  # because uint8 now

            for i in range(bins):
                intensity = vis[i]

                if intensity > threshold:
                    r = ranges[i]
                    theta = angles[i]

                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    z = 0  # flat assumption

                    world_point = position + np.array([x, y, z])
                    points.append(world_point)

            # ✅ save point cloud
            if len(points) > 0:
                with open(f"pointclouds/pc_{step}.csv", "w", newline="") as pcfile:
                    pc_writer = csv.writer(pcfile)
                    pc_writer.writerow(["x", "y", "z"])
                    pc_writer.writerows(points)

# ===============================
# CLEANUP
# ===============================
env.close()
plt.ioff()
plt.show()

print("✅ Finished!")