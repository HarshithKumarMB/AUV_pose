import holoocean
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pickle
import torch
import gpytorch
import numpy as np
with open("svgp_bathymetry.pkl", "rb") as f:
    bathy_model = pickle.load(f)

# ====================================================
# SCENARIO
# ====================================================

scenario = {
    "name": "testing",
    "world": "Dam",
    "package_name": "Ocean",
    "agents": [
        {
            "agent_name": "rov",
            "agent_type": "BlueROV2",
            "location": [-20.0, -10.0, 0.0],
            "control_scheme": 0,
            "sensors": [
                {
                    "sensor_name": "pose",
                    "sensor_type": "PoseSensor"
                },
                {
                    "sensor_name": "depthsensor",
                    "sensor_type": "DepthSensor",
                    "socket": "IMUSocket",
                    "Hz": 30
                },
                {
                    "sensor_name": "imu_1",
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 30,
                    "AddNoise": True,
                    "AccelBiasSigma": 0.01,
                    "GyroBiasSigma": 0.01
                },
                {
                    "sensor_name": "singlebeam",
                    "sensor_type": "SinglebeamSonar",
                    "rotation": [0, -90, 0],
                    "socket": "IMUSocket",
                    "Hz": 30,
                    "configuration": {
                        "OpeningAngle": 10.0,
                        "RangeMin": 0.5,
                        "RangeMax": 100.0,
                        "RangeBins": 256
                    }
                }
            ]
        }
    ]
}

# ====================================================
# EKF
# State:
# [x y z vx vy vz]^T
# ====================================================

class EKF:

    def __init__(self):
        print("Initializing EKF...")

        self.x = np.zeros((6,1))

        self.P = np.eye(6)

        self.R = np.diag([
            0.5,
            0.5,
            0.5
        ])
        self.filtered_x = []
        self.filtered_P = []

        self.predicted_x = []
        self.predicted_P = []

        self.F_history = []

    def predict(self, acc, dt):
        print(f"Predicting with dt={dt:.3f} seconds")

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

        sigma_a = 0.5

        Q = sigma_a**2 * (B @ B.T)

        acc = acc.reshape(3,1)

        self.x = F @ self.x + B @ acc

        self.P = F @ self.P @ F.T + Q

        self.filtered_x.append(self.x.copy())
        self.filtered_P.append(self.P.copy())
        self.F_history.append(F.copy())

    def update(self, position_meas):
        print("Updating with position measurement")

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

        self.P = (I - K @ H) @ self.P
        self.filtered_x.append(self.x.copy())
        self.filtered_P.append(self.P.copy())

        return self.x

    def rts_smoother(self):
        print("Running RTS smoother...")
        n_filt = len(self.filtered_x)
        n_pred = len(self.predicted_x)
        print(f"Number of filtered states: {n_filt}")
        print(f"Number of predicted states: {n_pred}")
        #N = len(self.filtered_x)
        N = min(n_filt, n_pred)
        smoothed_x = [None] * N
        smoothed_P = [None] * N
        smoothed_x[-1] = self.filtered_x[-1]
        smoothed_P[-1] = self.filtered_P[-1]
        for k in range(N-2, -1, -1):
            Pk = self.filtered_P[k]
            Pk1_pred = self.predicted_P[k+1]
            Fk = self.F_history[k]
            Ck = Pk @ Fk.T @ np.linalg.inv(Pk1_pred)
            smoothed_x[k] = (self.filtered_x[k] + Ck @ (smoothed_x[k+1]-self.predicted_x[k+1]))
            smoothed_P[k] = (self.filtered_P[k]+Ck @ (smoothed_P[k+1]-self.predicted_P[k+1]) @ Ck.T)
        return smoothed_x, smoothed_P

# ====================================================
# RTS smoother
# ====================================================


# ====================================================
# ENVIRONMENT
# ====================================================

env = holoocean.make(scenario_cfg=scenario)

# ====================================================
# ENVIRONMENT
# ====================================================


# Recreate the model class
class SVGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points):

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
        )

        variational_strategy = (
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x
        )

# Load checkpoint
with open("svgp_bathymetry.pkl", "rb") as f:
    checkpoint = pickle.load(f)

model = SVGPModel(checkpoint["inducing_points"])
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.load_state_dict(checkpoint["model_state_dict"])
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

x_scaler = checkpoint["x_scaler"]
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

model.eval()
likelihood.eval()



# ====================================================
# SONAR CONFIG
# ====================================================

range_min = 0.5
range_max = 100.0
range_bins = 256

ranges = np.linspace(range_min, range_max, range_bins)

# ====================================================
# WAYPOINTS
# ====================================================

waypoints = [
    [-20.0, -10.0, 0.0],
    [-10.0, -5.0, 0.0],
    [-20.0, -5.0, 0.0],
    [-30.0, -5.0, 0.0],
    [-30.0, -10.0, 0.0],
    [-30.0, -15.0, 0.0],
    [-20.0, -15.0, 0.0],
    [-10.0, -15.0, 0.0],
    [-20.0, -10.0, 0.0]
]

# ====================================================
# INITIALIZATION
# ====================================================

dt = 1.0 / 30.0

ekf = EKF()

# initialize filter near start position
ekf.x[0] = -20.0
ekf.x[1] = -10.0
ekf.x[2] = 0.0

dr_p = np.array([-20.0, -10.0, 0.0])
dr_vel = np.array([0.0, 0.0, 0.0])

current_wp = 0
command = np.zeros(8)

# ====================================================
# CSV OUTPUT
# ====================================================

with open("wp_c.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "ax", "ay", "az",
        "sonar_depth",
        "x", "y", "z",
        "ekf_x", "ekf_y", "ekf_z",
        "ekf_vx", "ekf_vy", "ekf_vz",
        "dr_x", "dr_y", "dr_z",
        "dr_vx", "dr_vy", "dr_vz"
    ])

    step = 0

    while True:

        state = env.step(command)

        # ============================================
        # GROUND TRUTH POSITION
        # ============================================

        pose = state["pose"]
        depth_sensor = state["depthsensor"]
        deep = depth_sensor[0]


        position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])

                # ============================================
        # SONAR
        # ============================================

        sonar_depth = np.nan

        if "singlebeam" in state:

            sonar = np.asarray(state["singlebeam"])

            if sonar.ndim == 1:

                idx = np.argmax(sonar)

                sonar_depth = ranges[idx]

            elif sonar.ndim == 0:

                sonar_depth = float(sonar)

        # ============================================
        # IMU
        # ============================================

        imu = state["imu_1"]

        imu_read = np.array([
            imu[0][0],
            -imu[0][1],
            imu[0][2] + 9.81
        ])

        # ============================================
        # DEAD RECKONING
        # ============================================

        dr_vel += np.array([
            imu_read[0] * dt,
            imu_read[1] * dt,
            imu_read[2] * dt
        ])

        dr_p += dr_vel * dt


        points = np.array([
            [dr_p[0], dr_p[1]]
        ], dtype=np.float32)
        points_scaled = x_scaler.transform(points)
        x = torch.tensor(points_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred = likelihood(model(x))
        depth = pred.mean.numpy()
        depth = depth * y_std + y_mean
        print(depth[0], sonar_depth)
        dr_p[2] = (sonar_depth + depth[0])
        dr1_p = dr_p.copy()
        dr1_p[2] = deep

        # ============================================
        # EKF
        # ============================================

        ekf.predict(imu_read, dt)

        ekf_state = ekf.update(dr_p)
        ekf_state = ekf.update(dr1_p)
        #ekf_state = ekf.x

        ekf_x = ekf_state[0,0]
        ekf_y = ekf_state[1,0]
        ekf_z = ekf_state[2,0]

        ekf_vx = ekf_state[3,0]
        ekf_vy = ekf_state[4,0]
        ekf_vz = ekf_state[5,0]

        # ============================================
        # WAYPOINT CONTROL
        # ============================================

        target = np.array(waypoints[current_wp])

        estimated_position = np.array([
            ekf_x,
            ekf_y,
            ekf_z
        ])

        error = target - position

        dist = np.linalg.norm(error)

        print(
            f"Step {step} | "
            f"WP {current_wp} | "
            f"Distance {dist:.2f} | "
            #f"EKF ({ekf_x:.2f}, {ekf_y:.2f}, {ekf_z:.2f})"
        )

        if dist < 0.2:

            print(f"Reached waypoint {current_wp}")

            current_wp += 1

            if current_wp >= len(waypoints):
                print("All waypoints completed")
                break

            continue

        e_z = error[2]

        err_p = error[0] + error[1]
        err_n = error[0] - error[1]

        e_y = error[1]

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

        command = np.clip(command, -20, 20)



        # ============================================
        # SAVE CSV
        # ============================================

        writer.writerow([
            imu_read[0],
            imu_read[1],
            imu_read[2],
            sonar_depth,

            position[0],
            position[1],
            position[2],

            ekf_x,
            ekf_y,
            ekf_z,

            ekf_vx,
            ekf_vy,
            ekf_vz,

            dr_p[0],
            dr_p[1],
            dr_p[2],

            dr_vel[0],
            dr_vel[1],
            dr_vel[2]
        ])

        step += 1



print("Finished. Data saved to wp_c.csv")

'''smoothed_x, smoothed_P = ekf.rts_smoother()

sx = []
sy = []
sz = []

for s in smoothed_x:
    sx.append(s[0,0])
    sy.append(s[1,0])
    sz.append(s[2,0])

plt.plot(
    sx,
    sy,
    color="magenta",
    linewidth=3,
    label="RTS Smoothed"
)

smooth_df = pd.DataFrame({
    "rts_x": sx,
    "rts_y": sy,
    "rts_z": sz
})

smooth_df.to_csv(
    "rts_smoothed.csv",
    index=False
)'''
# Load CSV
df = pd.read_csv("wp_c.csv")

# Create figure
plt.figure(figsize=(10, 8))

# Ground truth trajectory
plt.plot(
    df["x"],
    df["y"],
    label="Ground Truth",
    linewidth=3,
    color="blue"
)

# Dead reckoning trajectory
plt.plot(
    df["dr_x"],
    df["dr_y"],
    label="Dead Reckoning",
    linewidth=2,
    color="red",
    linestyle="--"
)

# EKF trajectory
plt.plot(
    df["ekf_x"],
    df["ekf_y"],
    label="EKF",
    linewidth=2,
    color="green"
)

# Mark start and end
plt.scatter(
    df["x"].iloc[0],
    df["y"].iloc[0],
    color="black",
    s=100,
    marker="o",
    label="Start"
)

plt.scatter(
    df["x"].iloc[-1],
    df["y"].iloc[-1],
    color="purple",
    s=100,
    marker="x",
    label="End"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Trajectory Comparison")
plt.grid(True)
plt.axis("equal")
plt.legend()

# Save image
plt.savefig("trajectory_comparison_y.png", dpi=300, bbox_inches="tight")

# Optional display
plt.show()

print("Saved trajectory_comparison_y.png")
