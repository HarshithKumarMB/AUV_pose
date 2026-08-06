import holoocean
import numpy as np
import uuid

import open3d as o3d
from pynput import keyboard


def visualize_point_cloud(pc, name, zoom):
    # Visualize the point cloud

    # Create a coordinate frame with a specified size
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10.0, origin=[0, 0, 0]
    )

    # Create window and geometry
    # Set the window size explicitly using width and height (e.g., 800x600 pixels)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=800, height=600)

    # Add the point cloud and coordinate axis to the visualizer
    vis.add_geometry(pc)
    vis.add_geometry(axis)

    # Set the view control (camera parameters)
    ctr = vis.get_view_control()

    # Set camera position so you're looking out along the X-axis,
    # with Y to the left and Z pointing up (standard coordinate frame)
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    ctr.set_front([-1, 0, 1])  # Camera facing along -X direction
    ctr.set_up([0, 0, 1])  # Z is up

    # Set zoom level for the camera view
    ctr.set_zoom(zoom)

    # Run the visualizer
    vis.run()

    # Close the visualizer window after use
    vis.destroy_window()


ticks_per_sec = 10
pressed_keys = list()


# Key to exit and plot
def on_press(key):
    global pressed_keys
    if hasattr(key, "char"):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))


listener = keyboard.Listener(on_press=on_press)
listener.start()


# point = [-10, 10, 0.25]
# point = [-10, 7, 0.25]
point = [-6.5, -30, 5.0]

config = {
    "name": "SurfaceNavigator",
    "package_name": "TestWorlds",
    "world": "TestWorld",
    "main_agent": "turtle0",
    "ticks_per_sec": 30,
    "agents": [
        {
            "agent_name": "turtle0",
            "agent_type": "TurtleAgent",
            "sensors": [
                {
                    "sensor_type": "RaycastLidar",
                    "configuration": {
                        "socket": "CameraSocket",
                        "Channels": 128,  # Number of lasers
                        "Range": 200,  # Max distance each laser can measure
                        "PointsPerSecond": 140000,  # Number of points per second
                        "RotationFrequency": 30,  # Lidar rotation frequency in Hz
                        "UpperFovLimit": 90,  # Upper field of view limit (degrees above horizontal)
                        "LowerFovLimit": -90,  # Lower field of view limit (degrees below horizontal)
                        "HorizontalFov": 360.0,  # Horizontal field of view (degrees)
                        "AtmospAttenRate": 0.4,  # Atmospheric attenuation rate
                        "RandomSeed": 42,  # Seed for random number generation
                        "DropOffGenRate": 0.45,  # General drop-off rate
                        "DropOffIntensityLimit": 0.8,  # Intensity value below which drop-off starts
                        "DropOffAtZeroIntensity": 0.4,  # Drop-off rate at zero intensity
                        "ShowDebugPoints": False,  # Show laser hit points in simulator for debugging
                        "NoiseStdDev": 0.1,  # Standard deviation of measurement noise in centimeters
                    },
                },
                {
                    "sensor_type": "RaycastSemanticLidar",
                    "configuration": {
                        "socket": "CameraSocket",
                        "Channels": 128,  # Number of lasers
                        "Range": 200,  # Max distance each laser can measure
                        "PointsPerSecond": 140000,  # Number of points per second
                        "RotationFrequency": 30,  # Lidar rotation frequency in Hz
                        "UpperFovLimit": 90,  # Upper field of view limit (degrees above horizontal)
                        "LowerFovLimit": -90,  # Lower field of view limit (degrees below horizontal)
                        "HorizontalFov": 360.0,  # Horizontal field of view (degrees)
                        "AtmospAttenRate": 0.4,  # Atmospheric attenuation rate
                        "RandomSeed": 42,  # Seed for random number generation
                        "DropOffGenRate": 0.45,  # General drop-off rate
                        "DropOffIntensityLimit": 0.8,  # Intensity value below which drop-off starts
                        "DropOffAtZeroIntensity": 0.4,  # Drop-off rate at zero intensity
                        "ShowDebugPoints": False,  # Show laser hit points in simulator for debugging
                        "NoiseStdDev": 0.1,  # Standard deviation of measurement noise in centimeters
                    },
                },
            ],
            "control_scheme": 1,
            "location": [-6.5, -30, 5.0],
            "rotation": [0, 0, 0],
        }
    ],
}


# plot = False
plot = True
ticks = 0
pcl = o3d.geometry.PointCloud()
tick_count = 0
binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

# Start simulation
# with holoocean.make(scenario_cfg=config, start_world=True, show_viewport=True) as env:

with holoocean.environments.HoloOceanEnvironment(
    scenario=config,
    binary_path=binary_path,
    show_viewport=False,
    verbose=True,
    uuid=str(uuid.uuid4()),
) as env:
    # env.spawn_prop("box", )
    while True:
        # Send waypoint to holoocean
        state = env.tick(1)
        tick_count += 1
        lidar = state["RaycastLidar"]
        # Convert lidar data to point cloud
        print("Lidar data shape: ", lidar.shape)

        # Save the lidar data as .npy
        np.save("expected_lidar.npy", lidar)

        semantic_lidar = state["RaycastSemanticLidar"]
        # Convert lidar data to point cloud
        print("Semantic lidar data shape: ", semantic_lidar.shape)

        # Save the lidar data as .npy
        np.save("expected_semantic_lidar.npy", semantic_lidar)
        break
