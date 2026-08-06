import holoocean
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# ============================================
# ROTATION MATRIX → QUATERNION
# ============================================
def rotmat_to_quat(R):
    q = np.zeros(4)
    t = np.trace(R)

    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2,1] - R[1,2]) / S
        q[2] = (R[0,2] - R[2,0]) / S
        q[3] = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0,1] + R[1,0]) / S
            q[3] = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / S
            q[1] = (R[0,1] + R[1,0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
            q[0] = (R[1,0] - R[0,1]) / S
            q[1] = (R[0,2] + R[2,0]) / S
            q[2] = (R[1,2] + R[2,1]) / S
            q[3] = 0.25 * S

    return q / np.linalg.norm(q)

# ============================================
# FOLDER
# ============================================
os.makedirs("sonar_images3", exist_ok=True)

# ============================================
# SCENARIO
# ============================================


scenario = {
    "name": "testing",
    "world": "Dam",
    "package_name": "Ocean",
    "world_config": {
    "meshes": [
        {
            "name": "pool_floor",
            "type": "box",
            "scale": [30, 5, 0.2],
            "location": [0, 0, -5]
        },
        {
            "name": "wall_1",
            "type": "box",
            "scale": [30, 0.2, 5],
            "location": [0, -2.5, -2.5]
        },
        {
            "name": "wall_2",
            "type": "box",
            "scale": [30, 0.2, 5],
            "location": [0, 2.5, -2.5]
        },
        {
            "name": "wall_3",
            "type": "box",
            "scale": [0.2, 5, 5],
            "location": [-15, 0, -2.5]
        },
        {
            "name": "wall_4",
            "type": "box",
            "scale": [0.2, 5, 5],
            "location": [15, 0, -2.5]
        }
    ]
},
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
                {
                    "sensor_name": "sonar",
                    "sensor_type": "ImagingSonar",
                    "socket": "IMUSocket",
                    "rotation": [0, -90, 0],
                    "Hz": 30,
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": 100.0,
                        "RangeBins": 256,
                        "AzimuthBins": 256,
                        "Azimuth": 90,
                        "AddNoise": True
                    }
                }
            ]
        }
    ]
}

env = holoocean.make(scenario_cfg=scenario)

# ============================================
# SONAR SETUP
# ============================================
sonar_config = scenario["agents"][0]["sensors"][-1]["configuration"]

azi = sonar_config["Azimuth"]
minR = sonar_config["RangeMin"]
maxR = sonar_config["RangeMax"]
binsR = sonar_config["RangeBins"]
binsA = sonar_config["AzimuthBins"]

plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi/2)
ax.set_thetamax(azi/2)

theta = np.linspace(-azi/2, azi/2, binsA) * np.pi / 180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)

plot = ax.pcolormesh(T, R, np.zeros_like(T), cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()

# ============================================
# WAYPOINTS
# ============================================
waypoints = [
    [1.0, 2.0, -3.0],
    [5.0, 6.0, -3.0],
    [-1.0, -2.0, -3.0],
    [11.0, 12.0, -3.0]
]

current_wp = 0
kp = 1.5

command = np.zeros(8)

# ============================================
# CSV LOGGING
# ============================================
with open("imu_log3.csv", "w", newline="") as file:
    writer = csv.writer(file)

    #  Added quaternion columns
    writer.writerow([
        "step",
        "qw","qx","qy","qz",

        "imu1_ax","imu1_ay","imu1_az",
        "imu1_gx","imu1_gy","imu1_gz",
        "imu2_ax","imu2_ay","imu2_az",
        "imu2_gx","imu2_gy","imu2_gz"
    ])

    # ============================================
    # MAIN LOOP
    # ============================================
    for step in range(2000):

        state = env.step(command)

        # ============================================
        # ORIENTATION
        # ============================================
        orient = np.array(state["orient"])
        q = rotmat_to_quat(orient)

        print("Quaternion:", q)

        # ============================================
        # POSITION
        # ============================================
        pose = state["pose"]

        position = np.array([pose[0][3], pose[1][3], pose[2][3]])
        target = np.array(waypoints[current_wp])

        error = target - position
        dist = np.linalg.norm(error)

        print("Error:", error)

        if dist < 0.5:
            print(f" Reached waypoint {current_wp}")
            current_wp += 1

            if current_wp >= len(waypoints):
                print("All waypoints completed!")
                command = np.zeros(8)
                break
            continue

        # simple control
        command = np.array([0,0,0,0,error[0]+error[1],error[0]-error[1],error[1],-error[1]])

        # ============================================
        # IMU DATA
        # ============================================
        imu1 = state["imu_1"]
        imu2 = state["imu_2"]

        acc1 = imu1[0]
        gyro1 = imu1[1]
        acc2 = imu2[0]
        gyro2 = imu2[1]

        # ============================================
        # WRITE CSV
        # ============================================
        writer.writerow([
            step,

            q[0], q[1], q[2], q[3],   #  quaternion

            acc1[0], acc1[1], acc1[2],
            gyro1[0], gyro1[1], gyro1[2],
            acc2[0], acc2[1], acc2[2],
            gyro2[0], gyro2[1], gyro2[2]
        ])

        # ============================================
        # SONAR
        # ============================================
        if "sonar" in state:
            sonar = state["sonar"]

            plot.set_array(sonar.ravel())
            fig.canvas.draw()
            fig.canvas.flush_events()

            filename = f"sonar_images3/sonar_{step}.png"
            plt.imsave(filename, sonar, cmap="gray")

# ============================================
# CLEANUP
# ============================================
env.close()
print("Finished!")
plt.ioff()
plt.show()