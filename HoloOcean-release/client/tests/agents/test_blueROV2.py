import holoocean
import uuid
import pytest
import numpy as np


@pytest.fixture(scope="module")
def env():
    scenario = {
        "name": "test_blueROV",
        "world": "TestWorld",
        "main_agent": "auv0",
        "frames_per_sec": False,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "BlueROV2",
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
    command = np.array([-2, 3, -4, -5, -2, 0, -4, -5])
    env.reset()
    state = env.step(command, 20)
    lin_accel = state["DynamicsSensor"][0:3]
    ang_accel = state["DynamicsSensor"][9:12]
    target_lin_accel = np.array([-0.07624035, 0.29059425, -0.42171726])
    target_ang_accel = np.array([-2.6503966, 3.901235, -0.6542496])
    assert np.allclose(lin_accel, target_lin_accel), (
        f"Thruster dynamics didn't achieve the correct linear acceleration. Should be {target_lin_accel}, got {lin_accel}"
    )
    assert np.allclose(ang_accel, target_ang_accel), (
        f"Thruster dynamics didn't achieve the correct angular acceleration. Should be {target_ang_accel}, got {ang_accel}"
    )


def test_pid_controller(env):
    """Test to make sure PID controller gets vehicle to the orientation and position we command"""
    des = [2, 2, -12, 0, 0, 45]
    env._scenario["agents"][0]["control_scheme"] = 1
    env.reset()
    state = env.step(des, 200)
    pos = state["DynamicsSensor"][6:9]
    rpy = state["DynamicsSensor"][15:18]
    assert np.allclose(des[:3], pos, 2e-1), (
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
