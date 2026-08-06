import holoocean
import uuid
import pytest
import numpy as np
from holoocean.fossen_dynamics import *


@pytest.fixture(scope="module")
def env():
    scenario = {
        "name": "test_torpedo",
        "world": "TestWorld",
        "frames_per_sec": False,
        "main_agent": "auv0",
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "TorpedoAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {"UseCOM": True, "UseRPY": False},
                    },
                    {
                        "sensor_type": "DynamicsSensor",
                        "sensor_name": "DynamicsSensorRPY",
                        "configuration": {"UseCOM": True, "UseRPY": True},
                    },
                ],
                "control_scheme": 1,
                "location": [0, 0, -10],
                "rotation": [0, 0, 0],
                "fossen_model": "torpedo",
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
        ticks_per_sec=30,
    ) as env:
        yield env


def test_manual_dynamics(env):
    """Test to make sure vehicle achieves the linear and angular acceleration we set"""
    des = [1, 2, 3, 0.1, 0.2, 0.3]
    env.reset()
    state = env.step(des, 20)
    lin_accel = state["DynamicsSensor"][0:3]
    ang_accel = state["DynamicsSensor"][9:12]

    assert np.allclose(des[:3], lin_accel), (
        f"Manual dynamics didn't achieve the correct linear acceleration. Should be {des[:3]}, got {lin_accel}"
    )
    assert np.allclose(des[3:], ang_accel), (
        f"Manual dynamics didn't achieve the correct angular acceleration. Should be {des[3:]}, got {ang_accel}"
    )


def test_fossen_dynamics(env):
    """Test to ensure the Fossen dynamics achieve correct accelerations"""
    env.reset()
    numSteps = 200
    accel = np.array(np.zeros(6), float)
    fossen_agents = ["auv0"]
    fossen_interface = FossenInterface(fossen_agents, env._scenario)
    fossen_interface.set_control_mode("auv0", "manualControl")
    u_control = np.array(
        [-0.087, -0.087, 0.087, 0.087, 800]
    )  # [RudderAngle, SternAngle,Thruster]
    fossen_interface.set_u_control("auv0", u_control)
    for i in range(numSteps):
        state = env.step(accel)
        accel = fossen_interface.update("auv0", state)
    pitch_heading = state["DynamicsSensorRPY"][
        16:18
    ]  # PULL OUT PITCH and heading in radians
    # Assert that pitch and heading are less than 0
    assert pitch_heading[0] < 0, (
        f"Pitch should be less than 0, but got {pitch_heading[0]}"
    )
    assert pitch_heading[1] < 0, (
        f"Heading should be less than 0, but got {pitch_heading[1]}"
    )


def test_fossen_autopilot(env):
    """Test to make sure Fossen autopilot gets vehicle to the orientation and position we command"""
    env.reset()
    numSteps = 800
    depth = 15
    heading = 50
    accel = np.array(np.zeros(6), float)
    fossen_agents = ["auv0"]
    fossen_interface = FossenInterface(fossen_agents, env._scenario)
    fossen_interface.set_control_mode("auv0", "depthHeadingAutopilot")
    fossen_interface.set_goal(
        "auv0", depth, heading, 1300
    )  # Changes depth (positive depth), heading, thruster RPM goals for controller
    for i in range(numSteps):
        state = env.step(accel)
        accel = fossen_interface.update("auv0", state)
    depth_actual = -state["DynamicsSensor"][8]  # Depth is negative Z
    heading_actual = state["DynamicsSensorRPY"][17]
    depth_error = abs(depth - depth_actual)
    heading_error = abs(heading - heading_actual)

    assert np.allclose(depth, depth_actual, atol=1.0), (
        f"Autopilot depth not achieved within 1 meter (Depth error was: {depth_error})"
    )
    assert np.allclose(heading, heading_actual, atol=10.0), (
        f"Autopilot heading not acheived within 10 degrees (Heading error was {heading_error})"
    )


def test_fin_dynamics(env):
    """Test to ensure the fins achieve correct accelerations"""
    command = np.array([30, -20, -10, 30, 50])
    env._scenario["agents"][0]["control_scheme"] = 0
    env.reset()
    state = env.step(command, 20)
    lin_accel = state["DynamicsSensor"][0:3]
    ang_accel = state["DynamicsSensor"][9:12]
    target_lin_accel = np.array([0.7525106, -0.01888119, 0.01457546])
    target_ang_accel = np.array([-0.00471005, 0.03474867, 0.03910198])
    assert np.allclose(target_lin_accel, lin_accel), (
        f"Fin dynamics didn't achieve the correct linear acceleration. Should be {target_lin_accel}, got {lin_accel}"
    )
    assert np.allclose(target_ang_accel, ang_accel), (
        f"Fin dynamics didn't achieve the correct angular acceleration. Should be {target_ang_accel}, got {ang_accel}"
    )


# TODO: Test to make sure fins saturate at max deflection
