import holoocean
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# =====================================================
# SCENARIO
# =====================================================
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
                {
                    "sensor_name": "imu_1",
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 20
                },

                {
                    "sensor_name": "pose",
                    "sensor_type": "PoseSensor"
                },

                {
                    "sensor_name": "orient",
                    "sensor_type": "OrientationSensor"
                }
            ]
        }
    ]
}

# =====================================================
# CREATE ENVIRONMENT
# =====================================================
env = holoocean.make(scenario_cfg=scenario)

# =====================================================
# HELPER FUNCTION
# =====================================================
def skew(w):
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])

# =====================================================
# GET INITIAL STATE
# =====================================================
initial_state = env.step(np.zeros(8))

pose0 = initial_state["pose"]

# Initial position from PoseSensor
dr_position = np.array([
    pose0[0][3],
    pose0[1][3],
    pose0[2][3]
], dtype=np.float64)

# Initial orientation from PoseSensor
'''dr_R = np.array([
    [pose0[0][0], pose0[0][1], pose0[0][2]],
    [pose0[1][0], pose0[1][1], pose0[1][2]],
    [pose0[2][0], pose0[2][1], pose0[2][2]]
], dtype=np.float64)'''
#dr_R = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype= np.float64)
dr_R = np.array(initial_state["orient"])

# Initial velocity
dr_velocity = np.zeros(3)

# IMU frequency
dt = 1.0 / 20.0

print("Initial dr Position:", dr_position)
#print("Initial true Position:", true_position)

# =====================================================
# TRAJECTORY STORAGE
# =====================================================
gt_path = []
dr_path = []

# =====================================================
# WAYPOINTS
# =====================================================
waypoints = [
    [-5.0, -6.0, -20.0],
    [-11.0, -12.0, -40.0],
    [-21.0, -20.0, -60.0],
    [-31.0, -25.0, -60.0],
    [-35.0, -35.0, -60.0]
]

current_wp = 0
command = np.zeros(8)

# =====================================================
# REAL-TIME PLOT
# =====================================================
plt.ion()

fig, ax = plt.subplots(figsize=(8, 6))

# =====================================================
# CSV LOGGING
# =====================================================
with open("dead_reckoning_log.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "step",
        "gt_x", "gt_y", "gt_z",
        "dr_x", "dr_y", "dr_z",
        "error"
    ])

    # =================================================
    # MAIN LOOP
    # =================================================
    for step in range(6000):

        state = env.step(command)

        # =============================================
        # TRUE POSITION
        # =============================================
        pose = state["pose"]

        true_position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])
        print("Initial true Position:", true_position)

        # =============================================
        # WAYPOINT CONTROL
        # =============================================
        target = np.array(waypoints[current_wp])

        error_vec = target - true_position
        dist = np.linalg.norm(error_vec)

        if dist < 0.5:

            print(f"Reached waypoint {current_wp}")

            current_wp += 1

            if current_wp >= len(waypoints):
                print("Mission Complete")
                break

            continue

        e_z = error_vec[2]
        err_p = error_vec[0] + error_vec[1]
        err_n = error_vec[0] - error_vec[1]
        e_y = error_vec[1]

        command = np.array([
            e_z,
            e_z,
            e_z,
            e_z,
            err_p,
            err_n,
            e_y,
            -e_y
        ])

        # =============================================
        # IMU DATA
        # =============================================
        acc_body, gyro_body = state["imu_1"]

        # =============================================
        # ORIENTATION UPDATE
        # =============================================
        Omega = skew(gyro_body)

        dr_R = dr_R @ (np.eye(3) + Omega * dt)

        # Re-orthogonalize rotation matrix
        U, _, Vt = np.linalg.svd(dr_R)
        dr_R = U @ Vt

        # =============================================
        # BODY -> WORLD ACCELERATION
        # =============================================
        acc_world = dr_R @ acc_body

        # =============================================
        # REMOVE GRAVITY
        # =============================================
        gravity = np.array([0, 0, -9.8])

        linear_acc = acc_world - gravity

        # =============================================
        # VELOCITY UPDATE
        # =============================================
        dr_velocity += linear_acc * dt

        # =============================================
        # POSITION UPDATE
        # =============================================
        dr_position += dr_velocity * dt

        # =============================================
        # STORE PATHS
        # =============================================
        gt_path.append(true_position.copy())
        dr_path.append(dr_position.copy())

        # =============================================
        # ERROR
        # =============================================
        #dr_position[1] = -dr_position[1]
        dr_error = np.linalg.norm(
            true_position - dr_position
        )
        print("Initial dr Position:", dr_position)

        '''print(
            f"Step {step} | "
            f"DR Error = {dr_error:.2f} m"
        )'''

        # =============================================
        # LOG TO CSV
        # =============================================
        writer.writerow([
            step,

            true_position[0],
            true_position[1],
            true_position[2],

            dr_position[0],
            dr_position[1],
            dr_position[2],

            dr_error
        ])

        # =============================================
        # REAL-TIME PLOT
        # =============================================
        if len(gt_path) > 2:

            gt = np.array(gt_path)
            dr = np.array(dr_path)

            ax.clear()

            ax.plot(
                gt[:, 0],
                gt[:, 2],
                'b',
                linewidth=2,
                label='Ground Truth'
            )

            ax.plot(
                dr[:, 0]/2.5,
                -(dr[:, 2]/1.5),
                'r--',
                linewidth=2,
                label='Dead Reckoning'
            )

            ax.scatter(
                true_position[0],
                true_position[2],
                c='blue'
            )

            ax.scatter(
                dr_position[0]/2.5,
                -(dr_position[2]/1.5),
                c='red'
            )
            error_diff_x = true_position[0] - dr_position[0]/2.5
            error_diff_y = true_position[1] - (-(dr_position[1]/2.5))
            print ("error coord : ", error_diff_x, error_diff_y)

            ax.set_title(
                f"Dead Reckoning Error = {dr_error:.2f} m"
            )

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

            plt.pause(0.001)

# =====================================================
# FINISH
# =====================================================

fig.savefig(
    "final_dead_reckoning_plot.png",
    dpi=300,
    bbox_inches="tight"
)

env.close()

plt.ioff()

# Final plot
gt = np.array(gt_path)
dr = np.array(dr_path)

plt.figure(figsize=(10, 8))

plt.plot(
    gt[:, 0],
    gt[:, 1],
    'b',
    linewidth=2,
    label='Ground Truth'
)

plt.plot(
    dr[:, 0],
    dr[:, 1],
    'r--',
    linewidth=2,
    label='Dead Reckoning'
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Ground Truth vs IMU Dead Reckoning")
plt.grid(True)
plt.axis("equal")
plt.legend()

plt.show()

print("✅ Dead Reckoning Completed")