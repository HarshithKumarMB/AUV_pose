import holoocean
import uuid
import numpy as np
from tests.utils.equality import almost_equal
import pytest


@pytest.fixture(scope="module")
def cfg(vehicle_type="TurtleAgent"):
    turtle_config = {
        "name": "test_velocity_sensor",
        "world": "TestWorld",
        "main_agent": "turtle0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "turtle0",
                "agent_type": vehicle_type,
                "sensors": [
                    {
                        "sensor_type": "PoseSensor",
                    },
                    {
                        "sensor_type": "OrientationSensor",
                    },
                    {
                        "sensor_type": "LocationSensor",
                    },
                ],
                "control_scheme": 0,
                "location": [-1.5, -1.50, 3.0],
            }
        ],
    }
    yield turtle_config


@pytest.mark.parametrize(
    "cfg",
    ["TurtleAgent", "HoveringAUV", "TorpedoAUV", "SurfaceVessel", "BlueROV2"],
    indirect=True,
)
def test_pose_sensor_straight(cfg):
    """Make sure pose sensor returns the same values as the orientation and location sensors"""

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        # let it land and then start moving forward
        for _ in range(200):
            command = np.random.random(size=2)
            state = env.step(command)
            assert almost_equal(
                state["PoseSensor"][:3, :3], state["OrientationSensor"]
            ), (
                f"Rotation in PoseSensor doesn't match that in OrientationSensor at timestep {i}"
            )
            assert almost_equal(state["PoseSensor"][:3, 3], state["LocationSensor"]), (
                f"Location in PoseSensor doesn't match that in LocationSensor at timestep {i}"
            )
