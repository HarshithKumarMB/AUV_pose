import holoocean
import holoocean.environments
import uuid


def bio_fx(location):
    x, y, z = location
    result = 0
    if x > 0:
        result = 1
    return result


def sal_fx(location):
    x, y, z = location
    result = 0
    if y > 0:
        result = 1
    return result


def temp_fx(location):
    x, y, z = location
    result = 0
    if z > 0:
        result = 1
    return result


scenario = {
    "name": "test_currents",
    "world": "TestWorld",
    "package_name": "TestWorlds",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "BSTSensor",
                    "socket": "COM",
                    "configuration": {
                        "biomass_clusters": [
                            {"position": [10, 10, -10], "strength": 2, "falloff": 10}
                        ],
                        "salinity_clusters": [
                            {"position": [10, 10, -10], "strength": 2, "falloff": 10}
                        ],
                        "temperature_clusters": [
                            {"position": [10, 10, -10], "strength": 2, "falloff": 10}
                        ],
                    },
                },
            ],
            "control_scheme": 0,
            "location": [5, 10, -10],
        }
    ],
}


def test_clusters():
    """
    Test each of the cluster parameters to make sure the correct alterations and dropoffs occur
    """
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=scenario,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=30,
    ) as env:
        agent0 = env.agents["auv0"]
        sensor_data = env.tick()
        bst_data = sensor_data["BSTSensor"]
        assert bst_data[0] == 2.365897, "Biomass data doesn't match test case"
        assert bst_data[1] == 34.238064, "Salinity data doesn't match test case"
        assert bst_data[2] == 24.01283, "Temperature data doesn't match test case"

        agent0.teleport(location=[7, 10, -10])
        sensor_data = env.tick()
        bst_data = sensor_data["BSTSensor"]
        assert bst_data[0] == 2.6363587, "Biomass data doesn't match test case"
        assert bst_data[1] == 34.508522, "Salinity data doesn't match test case"
        assert bst_data[2] == 24.283293, "Temperature data doesn't match test case"


# test_clusters()


def custom_functions():
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=scenario,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=30,
    ) as env:
        agent0 = env.agents["auv0"]
        env.set_biomass_function("auv0", bio_fx)
        env.set_salinity_function("auv0", sal_fx)
        env.set_temperature_function("auv0", temp_fx)
        sensor_data = env.tick()
        bst_data = sensor_data["BSTSensor"]
        assert bst_data[0] == 1
        assert bst_data[1] == 1
        assert bst_data[2] == 0

        agent0.teleport(location=[-5, -10, 10])
        sensor_data = env.tick()
        bst_data = sensor_data["BSTSensor"]
        assert bst_data[0] == 0
        assert bst_data[1] == 0
        assert bst_data[2] == 1
