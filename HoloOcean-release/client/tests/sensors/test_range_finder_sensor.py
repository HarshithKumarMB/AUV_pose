import holoocean
import uuid
import pytest
import numpy as np

from tests.utils.equality import almost_equal


@pytest.fixture(scope="module")
def cfg1(vehicle_type="SphereAgent"):
    sphere_config = {
        "name": "test_range_finder_sensor",
        "world": "TestWorld",
        "main_agent": "sphere0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "sphere0",
                "agent_type": vehicle_type,
                "sensors": [
                    {
                        "sensor_type": "RangeFinderSensor",
                        "configuration": {"LaserMaxDistance": 1, "LaserCount": 12},
                    }
                ],
                "control_scheme": 0,
                "location": [0.95, -1.75, 0.5],
            }
        ],
    }
    yield sphere_config


@pytest.mark.parametrize(
    "cfg1",
    ["SphereAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_range_finder_sensor_no_hit(cfg1):
    """Make sure the range sensor returns a -1 when nothing is within the max range of the sensor."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg1,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        state = env.tick(10)

        actual = state["RangeFinderSensor"]

        expected_count = cfg1["agents"][0]["sensors"][0]["configuration"]["LaserCount"]
        assert len(actual) == expected_count, (
            "Sensed range size did not match the expected size!"
        )

        expected_dist = -1
        assert all(np.isclose(expected_dist, actual)), (
            "Sensed range not -1 for no detected hit"
        )


@pytest.mark.parametrize(
    "cfg1",
    ["SphereAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_range_finder_sensor_range_values(cfg1):
    """Make sure the range sensor returns appropriate values across the full span of possible measurements.
    Starts before the range finder can see anything, then from 0 to the max range."""

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    max_range_setting = cfg1["agents"][0]["sensors"][0]["configuration"][
        "LaserMaxDistance"
    ]
    x_location_modifiers = np.arange(0, max_range_setting + 0.015, 0.01)
    agent_start_location = [2.78, 0.5, 0.5]

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg1,
        binary_path=binary_path,
        show_viewport=False,
        # verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        range_finder_measurements = []
        for x_location_mod in x_location_modifiers:
            new_location = np.asarray(agent_start_location)
            new_location[0] -= x_location_mod
            env.agents["sphere0"].teleport(location=new_location.tolist())
            state = env.tick()
            actual = state["RangeFinderSensor"]
            range_finder_measurements.append(float(actual[0]))
        # Check if the first measurement was a "no return" of -1
        expected_no_hit_meas = -1
        assert expected_no_hit_meas == range_finder_measurements[0], (
            "Sensed range for no detection did not match expected value!"
        )
        # Check that the rest of the measurements lie between 0 and the max range
        print(range_finder_measurements[1:])
        assert all(np.asarray(range_finder_measurements[1:]) >= 0.0), (
            "Sensed range went below 0.0 when not expected!"
        )
        assert all(np.asarray(range_finder_measurements[1:]) <= max_range_setting), (
            "Sensed range exceeded maximum value!"
        )
        # Check that the expected number of lasers matches the actual number
        expected_count = cfg1["agents"][0]["sensors"][0]["configuration"]["LaserCount"]
        assert len(actual) == expected_count, (
            "Sensed range size did not match the expected size!"
        )


@pytest.fixture(scope="module")
def cfg2(vehicle_type="UavAgent"):
    uav_config = {
        "name": "test_range_finder_sensor",
        "world": "TestWorld",
        "main_agent": "uav0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "uav0",
                "agent_type": vehicle_type,
                "sensors": [
                    {
                        "sensor_type": "RangeFinderSensor",
                        "configuration": {"LaserAngle": -90, "LaserMaxDistance": 15},
                    }
                ],
                "control_scheme": 0,
                "location": [0, 0, 10],
            }
        ],
    }
    yield uav_config


@pytest.mark.parametrize(
    "cfg2",
    ["SphereAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_range_finder_sensor_falling(cfg2):
    """Makes sure that the range sensor updates as the UAV falls, and after it comes to a rest."""

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg2,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        last_range = env.tick()["RangeFinderSensor"][0]

        for _ in range(10):
            new_range = env.tick(4)["RangeFinderSensor"][0]
            assert new_range < last_range, "UAV's range sensor did not detect falling!"
            last_range = new_range

        # Give the UAV time to bounce and settle
        env.tick(80)

        # Make sure it is stationary now
        last_range = env.tick()["RangeFinderSensor"][0]
        new_range = env.tick()["RangeFinderSensor"][0]

        assert almost_equal(last_range, new_range), "The UAV did not seem to settle!"
