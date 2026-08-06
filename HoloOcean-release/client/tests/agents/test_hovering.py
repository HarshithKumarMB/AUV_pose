import holoocean
import uuid
import pytest
import numpy as np


@pytest.fixture(scope="module")
def env():
    scenario = {
        "name": "test_hovering",
        "world": "TestWorld",
        "main_agent": "auv0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                    }
                ],
                "control_scheme": 0,
                "location": [0, 0, -10],
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


def test_thruster_dynamics(env):
    """Test to ensure the thrusters achieve correct accelerations"""
    command = np.array([-5, 5, -10, -10, -5, 5, -10, -10])
    env.reset()
    state = env.step(command, 20)
    lin_accel = state["DynamicsSensor"][0:3]
    ang_accel = state["DynamicsSensor"][9:12]
    target_lin_accel = np.array([-0.0476728, 0.03966415, -0.42882535])
    target_ang_accel = np.array([-0.64573175, 0.779148, -0.6961151])
    assert np.allclose(lin_accel, target_lin_accel), (
        f"Thruster dynamics didn't achieve the correct linear acceleration. Should be {target_lin_accel}, got {lin_accel}"
    )
    assert np.allclose(ang_accel, target_ang_accel), (
        f"Thruster dynamics didn't achieve the correct angular acceleration. Should be {target_ang_accel}, got {ang_accel}"
    )


def test_pid_controller(env):
    """Test to make sure PID controller gets vehicle to the orientation and position we command"""
    des = [2, 2, -12, 0, 20, 45]
    env._scenario["agents"][0]["control_scheme"] = 1
    env.reset()
    state = env.step(des, 200)
    pos = state["DynamicsSensor"][6:9]
    rpy = state["DynamicsSensor"][15:18]

    assert np.allclose(des[:3], pos, 1e-1), (
        f"PID Controller didn't make it to the correct position. Should be {des[:3]}, got {pos}"
    )
    assert np.allclose(des[3:], rpy, 5), (
        f"PID Controller didn't make it to the correct orientation. Should be {des[3:]}, got {rpy}"
    )


def test_manual_dynamics(env):
    """Test to make sure vehicle achieves the linear and angular acceleration we set"""
    des = [1, 2, 3, 0.1, 0.2, 0.3]
    env._scenario["agents"][0]["control_scheme"] = 2
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
