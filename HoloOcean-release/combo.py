import holoocean
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import glob


# =====================================================
# HELPERS
# =====================================================

def skew(w):
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])


def setup_folders():
    os.makedirs("sidescan_images", exist_ok=True)
    os.makedirs("pointclouds", exist_ok=True)


# =====================================================
# SCENARIO
# =====================================================

scenario = {
    "name": "testing",
    "world": "Dam",
    "package_name": "Ocean",
    "agents": [{
        "agent_name": "rov",
        "agent_type": "BlueROV2",
        "location": [1.0, 2.0, -6.0],
        "control_scheme": 0,
        "sensors": [
            {
                "sensor_name": "imu_1",
                "sensor_type": "IMUSensor",
                "socket": "IMUSocket",
                "Hz": 20,
                "AddNoise": True,
                "AccelBiasSigma": 0.01,
                "GyroBiasSigma": 0.01
            },
            {
                "sensor_name": "imu_2",
                "sensor_type": "IMUSensor",
                "socket": "IMUSocket",
                "Hz": 30,
                "AddNoise": True,
                "AccelBiasSigma": 0.01,
                "GyroBiasSigma": 0.01
            },
            {
                "sensor_name": "pose",
                "sensor_type": "PoseSensor"
            },
            {
                "sensor_name": "orient",
                "sensor_type": "OrientationSensor"
            },
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
            }}
            
        ]
    }]
}

setup_folders()
with open("dead_reckoning_log.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "step",
        "gt_x", "gt_y", "gt_z",
        "dr_x", "dr_y", "dr_z",
        "error"
    ])

'''
class EKF:
    def __init__(self):

        self.x = np.zeros((6,1))

        self.P = np.eye(6) * 1.0

        self.Q = np.diag([
            0.05,0.05,0.05,
            0.1,0.1,0.1
        ])

        self.R = np.diag([
            2.0,2.0,2.0
        ])

    def predict(self, acc, dt):

        F = np.array([
            [1,0,0,dt,0,0],
            [0,1,0,0,dt,0],
            [0,0,1,0,0,dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])

        B = np.array([
            [0.5*dt**2,0,0],
            [0,0.5*dt**2,0],
            [0,0,0.5*dt**2],
            [dt,0,0],
            [0,dt,0],
            [0,0,dt]
        ])

        u = acc.reshape(3,1)

        self.x = F @ self.x + B @ u

        self.P = F @ self.P @ F.T + self.Q

    def update(self, position_meas):

        H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ])

        z = position_meas.reshape(3,1)

        y = z - H @ self.x

        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(6)

        self.P = (I - K @ H) @ self.P'''

#ekf = EKF()


def waypoint():
    global state, command
    current_wp = 0
    #command = np.zeros(8)
    for step in range(6000):
        state = env.step(command)
        pose0 = state["pose"]
        print(f"Step: {step}, Current Waypoint: {current_wp}")
        dr_position = np.array([pose0[0][3], pose0[1][3], pose0[2][3]], dtype=np.float64)
        dr2_position = np.array([pose0[0][3], pose0[1][3], pose0[2][3]], dtype=np.float64)
        dr_R = np.array(initial_state["orient"])
        true_position = np.array([state["pose"][0][3], state["pose"][1][3], state["pose"][2][3]], dtype=np.float64)
        target = np.array(waypoints[current_wp])
        error_vec = target - true_position
        dist = np.linalg.norm(error_vec)
        if dist < 0.5:
            current_wp += 1
            print(f"Reached waypoint {current_wp}")
            if current_wp >= len(waypoints):
                print("All waypoints reached.")
                break
            continue
        error_z = error_vec[2]
        error_pos = error_vec[0] + error_vec[1]
        error_neg = error_vec[0] - error_vec[1]
        error_y = error_vec[1]
        command = np.array([error_z, error_z, error_z, error_z,error_pos, error_neg, error_y, -error_y])
        gt_path.append(true_position.copy())
        print("True Position:", true_position)
        #print("Dead Reckoning Position:", dr_position)
        deadreck = imu_read(dr_position, dr_R)
        dr_error = np.linalg.norm(true_position - deadreck)
        #writer.writerow([step, *true_position, *deadreck, dr_error])

def imu_read(dr_p, dr_r):
    #print("Initial dead reckoning Position:", dr_p)
    global dr_position, dr_velocity, dr2_position, dr2_velocity, imu_pose2
    acc_body2, gyro_body2 = state["imu_2"]
    omega2 = skew(gyro_body2)
    dr2_R = dr_r @ (np.eye(3) + omega2 * dt2)
    U2, _, Vt2 = np.linalg.svd(dr2_R)
    dr2_R = U2 @ Vt2
    acc_world2 = dr2_R @ acc_body2
    gravity2 = np.array([0, 0, -9.81])
    linear_acc2 = acc_world2 - gravity2
    dr2_velocity += (linear_acc2 * dt2)
    dr2_position += (dr2_velocity * dt2)
    imu_pose2 = np.array([dr2_position[0], -dr2_position[1], -dr2_position[2]])
    #print("Updated dead reckoning Position:", imu_pose2)
    dr2_path.append(dr2_position.copy())
    acc_body, gyro_body = state["imu_1"]
    omega = skew(gyro_body)
    dr_R = dr_r @ (np.eye(3) + omega * dt)
    U, _, Vt = np.linalg.svd(dr_R)
    dr_R = U @ Vt
    acc_world = dr_R @ acc_body
    gravity = np.array([0, 0, -9.81])
    linear_acc = acc_world - gravity
    dr_velocity += (linear_acc * dt)
    dr_position += (dr_velocity * dt)/2.5
    imu_pose = np.array([dr_position[0], -dr_position[1], -dr_position[2]])
    print("Updated dead reckoning Position:", imu_pose)
    print("2nd IMU readings:", imu_pose2)
    dr_path.append(dr_position.copy())
    dr2_path.append(dr2_position.copy())
    sonar_read(imu_pose)
    return dr_position



def sonar_read(imu_pose):
    global dr2_position, dr2_velocity, imu_pose2
    #print("Initial dead reckoning Position:", dr_position)
    if "sidescan" in state:
        sidescan_data = state["sidescan"]
        raw = np.array(state["sidescan"], dtype=np.float32)
        if (dr_position[2]<-63.0):
            imu_correct(imu_pose[0], -imu_pose[1], 60.0)
        mn, mx = raw.min(), raw.max()
        vis = (raw - mn) / (mx - mn) if mx > mn else np.zeros_like(raw)
        vis = (vis * 255).astype(np.uint8)
        waterfall.append(sidescan_data)
        if len(waterfall) > max_rows:
            waterfall.pop(0)
        sonar_points = []
        sonar_intensity = []
        for i in range(bins):
            intensity = vis[i]
            if intensity > 250:
                r = ranges[i]
                angle = angles[i]
                x_local = 0
                y_local = r * np.sin(angle)
                z_local = -r * np.cos(angle)
                local_point = np.array([x_local, y_local, z_local])
                global_point = imu_pose + local_point
                ref_point = imu_pose2 + local_point
                #dist2 = np.linalg.norm(global_point - dr2_position)
                sonar_points.append(global_point)
                sonar_intensity.append(sidescan_data[i])
                dist = np.linalg.norm(global_point - imu_pose)
                offset = np.linalg.norm(imu_pose - imu_pose2)
                if (dr_position[2]<-63.0):
                    imu_correct(imu_pose[0], -imu_pose[1], 60.0)
                if offset > 3.0 or offset < -3.0:
                    print("pose mismatch")
                    imu_correct(imu_pose2[0], imu_pose2[1], imu_pose2[2])
                elif dist < 20.0:
                    obs = pred(global_point[0], global_point[1], global_point[2])
                    calc_x = obs[0]
                    calc_y = obs[1] - (r * np.sin(angle))
                    calc_z = obs[2] + (r * np.cos(angle))
                    print(f"Calculated obstacle coordinates: ({calc_x}, {calc_y}, {calc_z})")
                    if obs[3] <10.0:
                        updated_pose = imu_correct(calc_x, calc_y, calc_z)
                global_points.extend(sonar_points)
                global_intensity.extend(sonar_intensity)
        


def pred(in_x, in_y, in_z):
    #print(f"Input coordinates: ({in_x}, {in_y}, {in_z})")
    model = joblib.load("obstacle_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    map_df1 = pd.read_csv("/home/harshith/Downloads/estimated_3d_obstacle_coordinates_1m.csv", low_memory=False)
    map_df2 = pd.read_csv("/home/harshith/Downloads/seabed_coordinates_1m_labeled.csv", low_memory=False)
    map_df = pd.concat([map_df1, map_df2], ignore_index=True)
    obstacle_maps = {obstacle: group.reset_index(drop=True) for obstacle, group in map_df.groupby("obstacle_type")}
    prediction = model.predict(pd.DataFrame([[in_x, in_y, in_z]], columns=["x", "y", "z"]))
    obstacle = label_encoder.inverse_transform(prediction)[0]
    #print(f"Predicted obstacle type: {obstacle}")
    subset = obstacle_maps.get(obstacle)
    if subset is not None:
        probability = model.predict_proba(pd.DataFrame([[in_x, in_y, in_z]], columns=["x", "y", "z"]))
        print(f"Prediction probabilities for each obstacle type: {dict(zip(label_encoder.classes_, probability[0]))}")
        distances = np.linalg.norm(subset[["x", "y", "z"]].values - np.array([in_x, in_y, in_z]), axis=1)
        idx = np.argmin(distances)
        nearest_x = subset.iloc[idx]["x"]
        nearest_y = subset.iloc[idx]["y"]
        nearest_z = subset.iloc[idx]["z"]
        nearest_distance = distances[idx]
        #print(f"Nearest matching obstacle coordinates: ({nearest_x}, {nearest_y}, {nearest_z}) with distance: {nearest_distance:.2f}")
        return [nearest_x, nearest_y, nearest_z, nearest_distance]


def imu_correct(calc_x, calc_y, calc_z):
    global dr_position, dr2_position
    correction_error = np.array([calc_x, calc_y, calc_z]) - dr_position
    #dr_position1 = ekf.update(np.array([calc_x, calc_y, calc_z]))
    dr_position1 = np.array([calc_x, calc_y, calc_z])
    if dr_position1[2] < -62.0:
        dr_position1[2] = 60.0
        dr2_position[2] = 60.0
    print(f"Dead reckoning position corrected to: ({dr_position1[0]}, {dr_position1[1]}, {dr_position1[2]})")
    dr_position = np.array([calc_x, -calc_y, -calc_z])
    return dr_position



gt_path =[]
dr_path = []
dr2_path = []

command = np.zeros(8)
waypoints = [
    [-6.0,  -4.0,  -20.0],
    [-14.0, -10.0, -40.0],
    [-22.0, -18.0, -60.0],
    [-30.0, -28.0, -60.0],
    [-40.0, -15.0, -60.0]
]
global_points =[]
global_intensity = []
deadreck = []
dt = 1.0 / 20.0
dt2 = 1.0 / 30.0
dr_velocity = np.zeros(3)
dr2_velocity = np.zeros(3)
dr_position = np.zeros(3)
dr2_position = np.zeros(3)
env = holoocean.make(scenario_cfg=scenario)
initial_state = env.step(np.zeros(8))
cfg = scenario["agents"][0]["sensors"][-1]["configuration"]
bins = cfg["RangeBins"]
ranges = np.linspace(cfg["RangeMin"], cfg["RangeMax"], bins)
angles = np.linspace(-np.pi/4, np.pi/4, bins)
waterfall = []
max_rows = 3000
command = np.zeros(8)
waypoint()
gt_path = np.array(gt_path)
dr_path = np.array(dr_path)
dr2_path = np.array(dr2_path)
dr_path_corrected = np.column_stack([
    dr_path[:, 0],
    -dr_path[:, 1],
    -dr_path[:, 2]
])

errors = np.linalg.norm(gt_path - dr_path_corrected, axis=1)

plt.figure(figsize=(12, 8))
plt.plot(gt_path[:, 0], gt_path[:, 1], label="Ground Truth Path", color='blue')
plt.plot(dr_path[:, 0], -dr_path[:, 1], label="Dead Reckoning Path", color='orange')
plt.scatter(gt_path[-1,0], gt_path[-1,1], color='black', s=100, label='End')
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Ground Truth vs Dead Reckoning Path with correction")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("corrected_dead_reckoning_plot.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(errors, 'r', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Position Error (m)")
plt.title("Corrected Dead Reckoning Error")
plt.grid(True)

plt.savefig(
    "corrected_dead_reckoning_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()