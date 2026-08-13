import holoocean
import numpy as np
import csv

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
            "location": [0.0, 0.0, 0.0],
            "control_scheme": 0,
            "sensors": [
                {
                    "sensor_name": "pose",
                    "sensor_type": "PoseSensor"
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

env = holoocean.make(scenario_cfg=scenario)

# ====================================================
# SONAR CONFIG
# ====================================================

cfg = scenario["agents"][0]["sensors"][1]["configuration"]

range_min = cfg["RangeMin"]
range_max = cfg["RangeMax"]
range_bins = cfg["RangeBins"]

ranges = np.linspace(range_min, range_max, range_bins)

# ====================================================
# WAYPOINTS
# ====================================================

waypoints = [
    [0.0, -5.0, 0.0],
    [0.0, -10.0, 0.0],
    [0.0, -15.0, 0.0],
    [0.0, -20.0, 0.0],
    [-2.0, -20.0, 0.0],
    [-2.0, -15.0, 0.0],
    [-2.0, -10.0, 0.0],
    [-2.0, -5.0, 0.0],
    [-2.0, 0.0, 0.0],
    [-4.0, 0.0, 0.0],
    [-4.0, -5.0, 0.0],
    [-4.0, -10.0, 0.0],
    [-4.0, -15.0, 0.0],
    [-4.0, -20.0, 0.0],
    [-6.0, -20.0, 0.0],
    [-6.0, -15.0, 0.0],
    [-6.0, -10.0, 0.0],
    [-6.0, -5.0, 0.0],
    [-6.0, 0.0, 0.0],
    [-8.0, 0.0, 0.0],
    [-8.0, -5.0, 0.0],
    [-8.0, -10.0, 0.0],
    [-8.0, -15.0, 0.0],
    [-8.0, -20.0, 0.0],
    [-10.0, -20.0, 0.0],
    [-10.0, -15.0, 0.0],
    [-10.0, -10.0, 0.0],
    [-10.0, -5.0, 0.0],
    [-10.0, 0.0, 0.0],
    [-12.0, 0.0, 0.0],
    [-12.0, -5.0, 0.0],
    [-12.0, -10.0, 0.0],
    [-12.0, -15.0, 0.0],
    [-12.0, -20.0, 0.0],
    [-14.0, -20.0, 0.0],
    [-14.0, -15.0, 0.0],
    [-14.0, -10.0, 0.0],
    [-14.0, -5.0, 0.0],
    [-14.0, 0.0, 0.0],
    [-16.0, 0.0, 0.0],
    [-16.0, -5.0, 0.0],
    [-16.0, -10.0, 0.0],
    [-16.0, -15.0, 0.0],
    [-16.0, -20.0, 0.0],
    [-18.0, -20.0, 0.0],
    [-18.0, -15.0, 0.0],
    [-18.0, -10.0, 0.0],
    [-18.0, -5.0, 0.0],
    [-18.0, 0.0, 0.0],
    [-20.0, 0.0, 0.0],
    [-20.0, -5.0, 0.0],
    [-20.0, -10.0, 0.0],
    [-20.0, -15.0, 0.0],
    [-20.0, -20.0, 0.0],
    [-22.0, -20.0, 0.0],
    [-22.0, -15.0, 0.0],
    [-22.0, -10.0, 0.0],
    [-22.0, -5.0, 0.0],
    [-22.0, 0.0, 0.0],
    [-24.0, 0.0, 0.0],
    [-24.0, -5.0, 0.0],
    [-24.0, -10.0, 0.0],
    [-24.0, -15.0, 0.0],
    [-24.0, -20.0, 0.0],
    [-26.0, -20.0, 0.0],
    [-26.0, -15.0, 0.0],
    [-26.0, -10.0, 0.0],
    [-26.0, -5.0, 0.0],
    [-26.0, 0.0, 0.0],
    [-28.0, 0.0, 0.0],
    [-28.0, -5.0, 0.0],
    [-28.0, -10.0, 0.0],
    [-28.0, -15.0, 0.0],
    [-28.0, -20.0, 0.0],
    [-30.0, -20.0, 0.0],
    [-30.0, -15.0, 0.0],
    [-30.0, -10.0, 0.0],
    [-30.0, -5.0, 0.0],
    [-30.0, 0.0, 0.0],
    [-32.0, 0.0, 0.0],
    [-32.0, -5.0, 0.0],
    [-32.0, -10.0, 0.0],
    [-32.0, -15.0, 0.0],
    [-32.0, -20.0, 0.0],
    [-34.0, -20.0, 0.0],
    [-34.0, -15.0, 0.0],
    [-34.0, -10.0, 0.0],
    [-34.0, -5.0, 0.0],
    [-34.0, 0.0, 0.0],
    [-36.0, 0.0, 0.0],
    [-36.0, -5.0, 0.0],
    [-36.0, -10.0, 0.0],
    [-36.0, -15.0, 0.0],
    [-36.0, -20.0, 0.0],
    [-38.0, -20.0, 0.0],
    [-38.0, -15.0, 0.0],
    [-38.0, -10.0, 0.0],
    [-38.0, -5.0, 0.0],
    [-38.0, 0.0, 0.0],
    [-40.0, 0.0, 0.0],
    [-40.0, -5.0, 0.0],
    [-40.0, -10.0, 0.0],
    [-40.0, -15.0, 0.0],
    [-40.0, -20.0, 0.0]
]

print(waypoints)

current_wp = 0

# BlueROV2 scheme 0 = 8 thrusters
command = np.zeros(8)

# ====================================================
# CSV OUTPUT
# ====================================================

with open("map1.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["x", "y", "sonar_depth"])
    step = 0

    while True:

        state = env.step(command)

        # ------------------------------------------------
        # POSE
        # ------------------------------------------------

        pose = state["pose"]

        position = np.array([
            pose[0][3],
            pose[1][3],
            pose[2][3]
        ])

        target = np.array(waypoints[current_wp])

        error = target - position

        dist = np.linalg.norm(error)

        print(
            f"Step {step} | "
            f"WP {current_wp} | "
            f"Distance {dist:.2f}"
            #f"Soanr Depth {state['singlebeam']}"
        )

        # ------------------------------------------------
        # WAYPOINT CHECK
        # ------------------------------------------------

        if dist < 0.5:

            print(f"Reached waypoint {current_wp}")

            current_wp += 1

            if current_wp >= len(waypoints):

                print("All waypoints completed")
                break

            continue

        # ------------------------------------------------
        # SIMPLE WAYPOINT CONTROLLER
        # same style as your reference
        # ------------------------------------------------

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

        # ------------------------------------------------
        # SINGLE BEAM SONAR
        # ------------------------------------------------

        sonar_depth = np.nan

        if "singlebeam" in state:

            sonar = np.asarray(state["singlebeam"])

            #
            # Most HoloOcean releases return a 256-bin profile.
            # We estimate bottom return from strongest bin.
            #
            if sonar.ndim == 1:

                idx = np.argmax(sonar)

                sonar_depth = ranges[idx]

            #
            # Some versions return a direct range.
            #
            elif sonar.ndim == 0:

                sonar_depth = float(sonar)

        # ------------------------------------------------
        # SAVE MAP
        # ------------------------------------------------

        writer.writerow([
            position[0],
            position[1],
            sonar_depth
        ])

        step += 1

print("Finished. Data saved to map1.csv")