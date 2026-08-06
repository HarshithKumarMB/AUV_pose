import holoocean
import uuid
from copy import deepcopy
import numpy as np
import pytest

from tests.utils.equality import almost_equal


@pytest.fixture(scope="module")
def cfg():  # vehicle_type="UavAgent"):
    uav_config = {
        "name": "test_depth_sensor",
        "world": "TestWorld",
        "main_agent": "uav0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "uav0",
                "agent_type": "UavAgent",  # vehicle_type,
                "sensors": [
                    {
                        "sensor_type": "DepthSensor",
                    },
                    {
                        "sensor_type": "DepthSensor",
                        "sensor_name": "noise",
                        "configuration": {"Sigma": 10},
                    },
                ],
                "control_scheme": 0,
                "location": [0.95, -1.75, 0.5],
            }
        ],
    }
    yield uav_config


@pytest.mark.parametrize(
    "cfg",
    ["UavAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_depth_sensor_falling(cfg):
    """Makes sure that the depth sensor updates as the UAV falls, and after it comes to a rest"""
    cfg = deepcopy(cfg)

    # Spawn the UAV 10 meters up
    cfg["agents"][0]["location"] = [0, 0, 20]

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        last_location = env.tick()["DepthSensor"]

        for _ in range(85):
            new_location = env.tick()["DepthSensor"]
            assert new_location[0] < last_location[0], (
                "UAV's location sensor did not detect falling!"
            )
            last_location = new_location

        # Give the UAV time to bounce and settle
        for _ in range(160):
            env.tick()

        # Make sure it is stationary now
        last_location = env.tick()["DepthSensor"]
        new_location = env.tick()["DepthSensor"]

        assert almost_equal(last_location, new_location), (
            "The vehicle did not seem to settle!"
        )


@pytest.mark.parametrize(
    "cfg",
    ["UavAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_depth_sensor_noise(cfg):
    """Make sure turning on noise actually turns it on"""

    config = deepcopy(cfg)

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=config,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        # let it land and then start moving forward
        for _ in range(100):
            state = env.tick()
            assert not np.allclose(state["DepthSensor"], state["noise"])
