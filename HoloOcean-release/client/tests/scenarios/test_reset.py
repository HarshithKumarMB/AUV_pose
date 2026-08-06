import numpy as np
import copy
import uuid

import holoocean

from tests.utils.equality import almost_equal


def compare_agent_states(state1, state2, thresh=0.01, is_close=True, to_ignore=None):
    if to_ignore is None:
        to_ignore = []

    for sensor in state1:
        if sensor in to_ignore:
            continue
        close = almost_equal(state1[sensor], state2[sensor], thresh)
        if is_close != close:
            print("Sensor {} failed!".format(sensor))
            print(state1[sensor])
            print(state2[sensor])
            print("--------------------------------")
            assert is_close != close


def extract_poses_from_config(scenario) -> list:
    init_positions = []
    init_rotations = []
    for agent in scenario["agents"]:
        init_positions.append(agent["location"])
        init_rotations.append(agent["rotation"])
    output = np.array([init_positions, init_rotations]).flatten()
    output = output.tolist()
    return copy.deepcopy(output)


def is_full_state(state):
    return isinstance(next(iter(state.values())), dict)


def test_main_agent_after_resetting():
    """Validate that sensor data for the main agent is the same after calling .reset()
    and that random location initialization isn't being carried over after resets.
    """
    scenario_config = {
        "name": "test_location_sensor",
        "world": "TestWorld",
        "main_agent": "sphere",
        "agents": [
            {
                "agent_name": "sphere",
                "agent_type": "SphereAgent",
                "sensors": [
                    {"sensor_type": "OrientationSensor", "rotation": [0, 0, 0]}
                ],
                "control_scheme": 0,
                "location": [0, 0, 0],
                "rotation": [0, 0, 0],
                "location_randomization": [0.5, 0.5, 0],
                "rotation_randomization": [0.1, 0.1, 0],
            }
        ],
    }

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    env = holoocean.environments.HoloOceanEnvironment(
        scenario=scenario_config,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    )

    main_agent = scenario_config["main_agent"]

    test_resets = 5

    env.reset()
    init_state = env._get_full_state()[main_agent]
    agent_count = len(env.agents)
    sensor_count = sum([len(env.agents[agent].sensors) for agent in env.agents])

    init_poses = extract_poses_from_config(env._scenario)

    for _ in range(test_resets):
        env.tick()
        env.reset()
        state = env._get_full_state()[main_agent]

        compare_agent_states(
            init_state,
            state,
            0.3,
            is_close=True,
            to_ignore=["RGBCamera", "BallLocationSensor"],
        )
        assert agent_count == len(env.agents)
        assert sensor_count == sum(
            [len(env.agents[agent].sensors) for agent in env.agents]
        )

        poses = extract_poses_from_config(env._scenario)
        assert poses == init_poses
