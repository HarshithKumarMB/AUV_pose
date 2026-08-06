import holoocean
import copy
import uuid
import pytest
import numpy as np


@pytest.fixture
def cfg1():
    base_config = {
        "name": "test_raycast_lidar_sensor",
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
                    }
                ],
                "control_scheme": 1,
                "location": [-6.5, -30, 5.0],
                "rotation": [0, 0, 0],
            }
        ],
    }
    return base_config


def test_raycast_lidar_sensor(cfg1):
    """Test basic functionality of RaycastLidar sensor. Compares one scan to the expected data."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg1,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick(1)

        actual = state["RaycastLidar"]

        expected = np.load("tests/sensors/baseline_data/expected_lidar.npy")
        # Verify that the x, y, and z is in the expected range 200
        assert np.all(actual[:, 0] >= -200) and np.all(actual[:, 0] <= 200), (
            "X values are not in the range [-200, 200]!"
        )
        assert np.all(actual[:, 1] >= -200) and np.all(actual[:, 1] <= 200), (
            "Y values are not in the range [-200, 200]!"
        )
        assert np.all(actual[:, 2] >= -200) and np.all(actual[:, 2] <= 200), (
            "Z values are not in the range [-200, 200]!"
        )

        # Verify that the intensity and ring values are in the expected range
        assert np.all(actual[:, 3] >= 0) and np.all(actual[:, 3] <= 1), (
            "Intensity values are not in the range [0, 1]!"
        )
        assert np.all(actual[:, 4] >= 0) and np.all(actual[:, 4] <= 127), (
            "Ring values are not in the range [0, 127]!"
        )

        # Verify that dropout and noise are applied correctly due to RandomSeed
        assert actual.shape == expected.shape, (
            f"Lidar data shape {actual.shape} does not match expected shape {expected.shape} !"
        )
        assert np.allclose(actual, expected, atol=1e-4), (
            "Lidar data does not match expected values!"
        )


@pytest.mark.parametrize("seed1, seed2", [(42, 123)])
def test_random_seed_changes_output(seed1, seed2, cfg1):
    cfg = copy.deepcopy(cfg1)
    cfg["agents"][0]["sensors"][0]["configuration"]["RandomSeed"] = seed1

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick()
        if state.__contains__("turtle0"):
            state = state["turtle0"]
        lidar1 = state["RaycastLidar"]

    cfg["agents"][0]["sensors"][0]["configuration"]["RandomSeed"] = seed2
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick()
        if state.__contains__("turtle0"):
            state = state["turtle0"]
        lidar2 = state["RaycastLidar"]
    if lidar1.shape != lidar2.shape:
        assert True  # Shapes differ, so outputs differ
    else:
        assert not np.allclose(lidar1, lidar2, atol=1e-6), (
            "Lidar outputs should differ for different seeds!"
        )


@pytest.fixture
def cfg2():
    base_config = {
        "name": "test_raycast_lidar_sensor_no_dropoff",
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
                            "DropOffGenRate": 0.0,  # General drop-off rate
                            "DropOffIntensityLimit": 0.0,  # Intensity value below which drop-off starts
                            "DropOffAtZeroIntensity": 0.0,  # Drop-off rate at zero intensity
                            "ShowDebugPoints": False,  # Show laser hit points in simulator for debugging
                            "NoiseStdDev": 0.0,  # Standard deviation of measurement noise in centimeters
                        },
                    }
                ],
                "control_scheme": 1,
                "location": [-6.5, -30, 5.0],
                "rotation": [0, 0, 0],
            }
        ],
    }
    return base_config


def test_raycast_lidar_sensor_no_dropoff(cfg2):
    """Test RaycastLidar sensor with no dropoff. Compares multiple scans to ensure consistency."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg2,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        scans = []
        for _ in range(5):  # Take multiple scans
            state = env.tick()
            if state.__contains__("turtle0"):
                state = state["turtle0"]
            lidar = state["RaycastLidar"]
            scans.append(lidar)
            print(f"Scan {_} shape: {lidar.shape}")

        # Check that all scans are identical
        for i in range(1, len(scans)):
            print(
                f"Comparing scan 0 (shape:{scans[0].shape}) to scan {i} (shape:{scans[i].shape})"
            )
            assert np.allclose(scans[0], scans[i], atol=1e-4), (
                f"Scan {i} does not match the first scan!"
            )


@pytest.fixture
def cfg3():
    base_config = {
        "name": "test_raycast_lidar_sensor_ticks_per_capture",
        "world": "TestWorld",
        "main_agent": "turtle0",
        "ticks_per_sec": 30,
        "frames_per_sec": 150,
        "agents": [
            {
                "agent_name": "turtle0",
                "agent_type": "TurtleAgent",
                "sensors": [
                    {
                        "sensor_type": "RaycastLidar",
                        "configuration": {
                            "socket": "CameraSocket",
                            "RotationFrequency": 10,  # Lidar rotation frequency in Hz
                            "ShowDebugPoints": True,
                        },
                        "Hz": 10,
                    }
                ],
                "control_scheme": 0,  # Manual control scheme
                "location": [-6.5, -30, 5.0],
                "rotation": [0, 0, 0],
            }
        ],
    }
    return base_config


@pytest.mark.parametrize("upper, lower", [(90, -90), (30, -30), (10, -10)])
def test_fov_variation(cfg2, upper, lower):
    cfg = copy.deepcopy(cfg2)
    cfg["agents"][0]["sensors"][0]["configuration"]["UpperFovLimit"] = upper
    cfg["agents"][0]["sensors"][0]["configuration"]["LowerFovLimit"] = lower

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=False,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick()
        if state.__contains__("turtle0"):
            state = state["turtle0"]
        lidar = state["RaycastLidar"]
        assert lidar.shape[0] > 0, (
            f"No lidar points collected for FOV ({upper}, {lower})"
        )


# Test the Hz ticks per capture functionality. Should return nothing 2/3 of the time, and return data 1/3 of the time.
def test_raycast_lidar_sensor_ticks_per_capture(cfg3):
    """Test RaycastLidar sensor with ticks per capture. Should return data 1/3 of the time."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg3,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        scans = []
        numScansEmpty = 0
        for _ in range(6):  # Take multiple scans
            state = env.tick()
            if state.__contains__("sv"):
                state = state["sv"]
            lidar = state["RaycastLidar"]
            if lidar.any():  # Check if lidar data is returned
                scans.append(lidar)
            else:
                numScansEmpty += 1

        assert numScansEmpty == 4, f"Expected 4 empty scans, but got {numScansEmpty}!"
        assert len(scans) == 2, f"Expected 2 scans with data, but got {len(scans)}!"
