import holoocean
import numpy as np
import csv

scenario = {
    "name": "testing",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "agents": [
        {
            "agent_name": "rov",
            "agent_type": "BlueROV2",
            "location": [1.0, 2.0, -10.0],
            "control_scheme": 0,
            "sensors": [
                {
                    "sensor_name": "imu_1",
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 20,
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True
                    }
                },

                {
                    "sensor_name": "depth",
                    "sensor_type": "DepthSensor",
                    "socket": "DepthSocket",
                    "Hz": 30,
                    "configuration": {
                        "DepthSigma": 0.1,
                        "ReturnBias": True,
                        "AddNoise": True
                    }
                }
                {
                    "sensor_name": "imu_2",
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 30,
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True
                    }
                },
                 {
                    "sensor_name": "sonar",
                    "sensor_type": "ImagingSonar",
                    "socket": "SonarSocket",
                    "rotation": [0, -90, 0],
                    "Hz": 10,
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": 50.0,
                        "RangeBins": 512,
                        "AzimuthBins": 512,
                        "AddNoise": True
                    }
                }]

                }
            ]
        }
    

env = holoocean.make(scenario_cfg=scenario)

command = np.array([10,10,10,10,0,0,0,0])

with open("imu_logx.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time",
        "imu1_ax", "imu1_ay", "imu1_az",
        "imu1_gx", "imu1_gy", "imu1_gz",
        "imu2_ax", "imu2_ay", "imu2_az",
        "imu2_gx", "imu2_gy", "imu2_gz"
    ])






    for step in range(2000):
        state = env.step(command)
        print(state.keys())

        imu1 = state["imu_1"]
        imu2 = state["imu_2"]
        if(step%3 == 0):
            son = state["sonar"]

        print("IMU 1:", imu1)
        print("IMU 2:", imu2)
        print(son)
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

env.close()
print("Finished!")