import holoocean
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
#from mpl_toolkits.mplot3d import Axes3D

# ===============================
# CREATE OUTPUT FOLDERS
# ===============================
os.makedirs("sidescan_images", exist_ok=True)
os.makedirs("pointclouds", exist_ok=True)
def skew(w):
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
# ===============================
# REAL-TIME PLOT SETUP
# ===============================
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ===============================
# SCENARIO
# ===============================
scenario = {
    "name": "testing",
    "world": "Dam",
    "package_name": "Ocean",
    "agents": [
        {
            "agent_name": "rov",
            "agent_type": "BlueROV2",
            "location": [1.0, 2.0, -6.0],
            "control_scheme": 0,
            "sensors": [
                {"sensor_name": "imu_1", "sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 20},
                {"sensor_name": "imu_2", "sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 30},
                {"sensor_name": "pose", "sensor_type": "PoseSensor"},
                {"sensor_name": "orient", "sensor_type": "OrientationSensor"},

                {
                    "sensor_name": "sidescan",
                    "sensor_type": "SidescanSonar",
                    "socket": "IMUSocket",
                    "rotation": [0, -90, 0],
                    "Hz": 10,
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": 70.0,
                        "RangeBins": 256,
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

angles = np.linspace(-np.pi/4, np.pi/4, bins)

# ===============================
# WATERFALL BUFFER
# ===============================
waterfall = []
max_rows = 3000

# ===============================
# GLOBAL MAP STORAGE
# ===============================
global_points = []
global_intensity = []

# ===============================
# WAYPOINTS
# ===============================
waypoints = [
    [-5.0, -6.0, -20.0],
    [-11.0, -12.0, -40.0],
    [-21.0, -20.0, -60.0],
    [-31.0, -25.0, -60.0],
    [-35.0, -35.0, -60.0]
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
    for step in range(6000):

        state = env.step(command)

        # -------------------------------
        # POSITION + ROTATION
        # -------------------------------
        pose = state["pose"]

        position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])

        R = np.array([
            [pose[0][0], pose[0][1], pose[0][2]],
            [pose[1][0], pose[1][1], pose[1][2]],
            [pose[2][0], pose[2][1], pose[2][2]]
        ])

        target = np.array(waypoints[current_wp])
        error = target - position
        dist = np.linalg.norm(error)

        print(f"Step {step} | Dist {dist:.2f}")

        if dist < 0.5:
            print(f"Reached waypoint {current_wp}")
            current_wp += 1

            if current_wp >= len(waypoints):
                print(" All waypoints completed!")
                break
            continue

        # -------------------------------
        # CONTROL
        # -------------------------------
        
        e_z = error[2]
        err_p = error[0] + error[1]
        err_n = error[0] - error[1]
        e_y = error[1]

        command = np.array([
            e_z, e_z, e_z, e_z,
            err_p,
            err_n,
            e_y,
            -e_y
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
        # SIDESCAN PROCESSING
        # -------------------------------
        if "sidescan" in state:
            raw = np.array(state["sidescan"], dtype=np.float32)

            # normalize
            mn, mx = raw.min(), raw.max()
            vis = (raw - mn) / (mx - mn) if mx > mn else np.zeros_like(raw)
            vis = (vis * 255).astype(np.uint8)

            # -------- WATERFALL --------
            waterfall.append(vis)
            if len(waterfall) > max_rows:
                waterfall.pop(0)

            img = np.array(waterfall)

            ax1.clear()
            ax1.imshow(img, cmap="gray", aspect='auto')
            ax1.set_title("Waterfall")
            ax1.set_xlabel("Beam")
            ax1.set_ylabel("Time")

            points = []
            threshold = 200  # because uint8 now

            for i in range(bins):
                intensity = vis[i]

                if intensity > threshold:
                    r = ranges[i]
                    theta = angles[i]

                    z = -r * np.cos(theta)
                    y = r * np.sin(theta)
                    x = 0  # flat assumption

                    world_point = position + np.array([x, y, z])
                    points.append([world_point[0], world_point[1], world_point[2], r,theta])

            #  save point cloud
            if len(points) > 0:
                with open(f"pointclouds/pc_{step}.csv", "w", newline="") as pcfile:
                    pc_writer = csv.writer(pcfile)
                    pc_writer.writerow(["x", "y", "z", "r", "theta"])
                    pc_writer.writerows(points)


            # -------- XY MAPPING --------
            # ===============================
            # GLOBAL STORAGE
            # ===============================
            xy_data = []
            if "sidescan" in state:
                raw = np.array(state["sidescan"], dtype=np.float32)
                # normalize (keep continuous values)
                mn, mx = raw.min(), raw.max()
                vis = (raw - mn) / (mx - mn) if mx > mn else np.zeros_like(raw)
                # -------------------------------
                # GET POSE
                # -------------------------------
                pose = state["pose"]
                position = np.array([
                    pose[0][3],
                    pose[1][3],
                    pose[2][3]
                ])
                R = np.array([
                    [pose[0][0], pose[0][1], pose[0][2]],
                    [pose[1][0], pose[1][1], pose[1][2]],
                    [pose[2][0], pose[2][1], pose[2][2]]
                ])
                # -------------------------------
                # MAP EVERY SONAR SAMPLE
                # -------------------------------
                for i in range(bins):
                    intensity = vis[i]   #  keep ALL values
                    r = ranges[i]
                    theta = angles[i]
                    # local coordinates
                    local_point = np.array([
                        r * np.cos(theta),
                        r * np.sin(theta),
                        0
                    ])
                    # transform → GLOBAL frame
                    world_point = position + R @ local_point
                    # store mapping
                    xy_data.append([
                        world_point[0],
                        world_point[1],
                        intensity
                        ])

    # -------------------------------
    # PLOT EVERYTHING
    # -------------------------------
                if len(xy_data) > 0:
                    data = np.array(xy_data)
                    ax2.clear()
                    sc = ax2.scatter(
                        data[:, 0],   # X
                        data[:, 1],   # Y
                        c=data[:, 2], # intensity
                        cmap='gray',
                        s=2
                        )
                    ax2.set_title("Full Sonar XY Mapping (No Threshold)")
                    ax2.set_xlabel("X (m)")
                    ax2.set_ylabel("Y (m)")
                    ax2.axis("equal")
                    plt.pause(0.001)

# ===============================
# CLEANUP
# ===============================
env.close()
plt.ioff()
plt.show()

print("✅ Finished!")

