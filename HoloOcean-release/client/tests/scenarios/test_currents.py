import holoocean
import pytest
import uuid
import numpy as np


@pytest.fixture(scope="module")
def env():
    scenario = {
        "name": "test_currents",
        "package_name": "TestWorlds",
        "world": "TestWorld",
        "main_agent": "auv0",
        "current": {
            "vehicle_debugging": False,
        },
        "ticks_per_sec": 60,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False,
                        },
                    },
                ],
                "control_scheme": 0,
                "location": [18.5, -25.5, -10],
            }
        ],
    }
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=scenario,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=60,
    ) as env:
        yield env


def test_currents_underwater(env):
    """Test currents acting on underwater vehicle"""

    env.reset()
    data = env.tick()
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration, [0, 0, 0]), "Null current failed"

    env.reset()
    env.set_ocean_currents("auv0", [1, 0, 0])
    data = env.tick()
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration, [0.06320982, 0, 0]), "X current failed"

    env.reset()
    env.set_ocean_currents("auv0", [0, 1, 0])
    data = env.tick()
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration, [0, 0.06320982, 0]), "Y current failed"

    env.reset()
    env.set_ocean_currents("auv0", [0, 0, 1])
    data = env.tick()
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration, [0, 0, 0.06320982]), "Z current failed"


def test_currents_surface(env):
    """Test currents acting on surface vessel"""

    env._scenario["agents"][0]["agent_type"] = "SurfaceVessel"
    env._scenario["agents"][0]["location"] = [10, 10, 0.1]

    env.reset()
    env.tick(500)  # Let the vessel settle
    env.set_ocean_currents("auv0", [1, 0, 0])
    data = env.tick(10)
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration[:2], [0.00199408, 0], rtol=2e-2), (
        "Surface Vessel X Current Failed"
    )

    env.reset()
    env.tick(500)  # Let the vessel settle
    env.set_ocean_currents("auv0", [0, 1, 0])
    data = env.tick(10)
    dynamics_data = data["DynamicsSensor"]
    acceleration = dynamics_data[:3]
    assert np.allclose(acceleration[:2], [0, 0.00197457], rtol=2e-2), (
        "Surface Vessel Y Current Failed"
    )
