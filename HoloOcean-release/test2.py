import holoocean
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

#  create folder
os.makedirs("sonar_images3", exist_ok=True)

#  -------- SCENARIO --------
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
                {
                    "sensor_name": "sonar",
                    "sensor_type": "ImagingSonar",
                    "socket": "IMUSocket",
                    "rotation": [0, -90, 0],
                    "Hz": 10,
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": 50.0,
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

#  -------- SONAR PLOT SETUP --------
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
z = np.zeros_like(T)

plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()

#  -------- WAYPOINTS --------
waypoints = [
    [1.0, 2.0, -3.0],
    [5.0, 6.0, -3.0],
    [-1.0, -2.0, -3.0],
    [11.0, 12.0, -3.0]
]

current_wp = 0
kp = 1.5
max_thrust = 10

command = np.zeros(8)

#  -------- LOGGING --------
with open("imu_log3.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "step",
        "imu1_ax","imu1_ay","imu1_az",
        "imu1_gx","imu1_gy","imu1_gz",
        "imu2_ax","imu2_ay","imu2_az",
        "imu2_gx","imu2_gy","imu2_gz"
    ])

    #  -------- MAIN LOOP --------
    for step in range(2000):

        state = env.step(command)

        #  -------- POSITION --------
        pose = state["pose"]
        orient = state["orient"]

        print("orient:", orient)

        # extract translation from 3x4 pose matrix
        position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])

        target = np.array(waypoints[current_wp])

        error = target - position
        #print(error[1])
        dist = np.linalg.norm(error)
        print(error)

        #print(f"Step {step} | WP {current_wp} | Pos {position} | Dist {dist:.2f}")

        #  checkpoint reached
        if dist < 0.5:
            print(f"✅ Reached waypoint {current_wp}")

            current_wp += 1

            if current_wp >= len(waypoints):
                print(" All waypoints completed!")
                command = np.zeros(8)
                #command = np.array([0,0,0,0,0,0,0,0])
                break
            continue

        #  motion control (simple proportional)
        direction = error / (dist + 1e-6)
        control = kp * direction
        
        command = np.array([0,0,0,0,error[0]+error[1],error[0]-error[1],error[1],-error[1]])
        '''

        command = np.zeros(8)

        # forward/back (x)
        command[4] = np.clip(control[0]*max_thrust, -max_thrust, max_thrust)
        command[5] = np.clip(control[0]*max_thrust, -max_thrust, max_thrust)

        # sideways (y)
        command[0] = np.clip(control[1]*max_thrust, -max_thrust, max_thrust)
        command[1] = np.clip(-control[1]*max_thrust, -max_thrust, max_thrust)

        # vertical (z)
        command[6] = np.clip(control[2]*max_thrust, -max_thrust, max_thrust)
        command[7] = np.clip(control[2]*max_thrust, -max_thrust, max_thrust)'''

        #  -------- IMU LOGGING --------
        imu1 = state["imu_1"]
        imu2 = state["imu_2"]

        acc1 = imu1[0]
        gyro1 = imu1[1]

        acc2 = imu2[0]
        gyro2 = imu2[1]

        writer.writerow([
            step,
            acc1[0], acc1[1], acc1[2],
            gyro1[0], gyro1[1], gyro1[2],
            acc2[0], acc2[1], acc2[2],
            gyro2[0], gyro2[1], gyro2[2]
        ])

        #  -------- SONAR --------
        if "sonar" in state:
            sonar = state["sonar"]

            plot.set_array(sonar.ravel())
            fig.canvas.draw()
            fig.canvas.flush_events()

            filename = f"sonar_images3/sonar_{step}.png"
            plt.imsave(filename, sonar, cmap="gray")

#  cleanup
env.close()

print(" Finished!")
plt.ioff()
plt.show()