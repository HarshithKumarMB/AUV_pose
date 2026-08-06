import holoocean
import uuid
import pytest
import numpy as np


@pytest.fixture
def cfg1():
    base_config = {
        "name": "test_raycast_semantic_lidar_sensor",
        "world": "TestWorld",
        "main_agent": "turtle0",
        "ticks_per_sec": 30,
        "agents": [
            {
                "agent_name": "turtle0",
                "agent_type": "TurtleAgent",
                "sensors": [
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
                    }
                ],
                "control_scheme": 1,
                "location": [-6.5, -30, 5.0],
                "rotation": [0, 0, 0],
            }
        ],
    }
    return base_config


@pytest.fixture
def cfg2():
    base_config = {
        "name": "test_raycast_semantic_lidar_sensor",
        "world": "TestWorld",
        "main_agent": "turtle0",
        "ticks_per_sec": 30,
        "agents": [
            {
                "agent_name": "turtle0",
                "agent_type": "TurtleAgent",
                "sensors": [
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
                    }
                ],
                "control_scheme": 1,
                "location": [47, -58, -1.045],
                "rotation": [0, 0, 0],
            }
        ],
    }
    return base_config


def test_raycast_semantic_lidar_sensor(cfg1):
    """Test basic functionality of RaycastSemanticLidar sensor. Compares one scan to the expected data."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg1,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick(1)

        actual = state["RaycastSemanticLidar"]
        expected = np.load("tests/sensors/baseline_data/expected_semantic_lidar.npy")
        expected_non_semantic = np.load(
            "tests/sensors/baseline_data/expected_lidar.npy"
        )

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

        # Dropout and noise are applied correctly due to RandomSeed
        assert actual.shape == expected.shape, (
            f"Lidar data shape {actual.shape} does not match expected shape {expected.shape}!"
        )
        # Ignore object IDS for comparison
        assert np.allclose(
            np.delete(actual, 5, axis=1), np.delete(expected, 5, axis=1), atol=1e-4
        ), "Lidar data does not match expected values!"

        # Check that the non-semantic lidar matches in x,y,z,intensity,ring
        assert np.allclose(
            np.delete(actual, [5, 6], axis=1), expected_non_semantic, atol=1e-4
        ), "Non-semantic lidar data does not match expected values!"

        # Verify that the number of unique object IDs detected matches expected
        unique_ids = np.unique(actual[:, 5])
        unique_expected_ids = np.unique(expected[:, 5])
        assert len(unique_ids) == len(unique_expected_ids), (
            f"Number of unique object IDs detected ({len(unique_ids)}) does not match expected ({len(unique_expected_ids)})!"
        )


def test_lidar_with_landscape(cfg2):
    """Makes sure that the semantic lidar is working with landscapes

    TestWorld landscape is set up with 3 material layers. The semantic lidar should
    be able to tag each material layer as something separate.

    """
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg2,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        unique_expected_ids = 1
        expected_tags = 3
        for _ in range(5):
            env.tick()

        state = env.tick()
        current_lidar = state["RaycastSemanticLidar"]
        last_two = current_lidar[:, -2:]

        # Get unique tags
        tags, _ = np.unique(last_two[:, 1], return_index=True)

        # Get unique IDs
        _, idx = np.unique(last_two[:, 0], return_index=True)

        # Get rows with unique IDs only (first occurrence)
        unique_ids = last_two[np.sort(idx)]

        # Should have 3 tags, one for each material layer
        assert len(tags) == 3, (
            f"Number of tags detected ({len(tags)}) does not match expected ({len(expected_tags)})!"
        )
        # Should have 1 unique ID for the single landscape
        assert len(unique_ids) == 1, (
            f"Number of unique object IDs detected ({len(unique_ids)}) does not match expected ({len(unique_expected_ids)})!"
        )
