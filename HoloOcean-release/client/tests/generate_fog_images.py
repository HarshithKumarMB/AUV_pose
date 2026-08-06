import holoocean
import copy
import uuid
import cv2

config_water = {
    "name": "test_fog",
    "world": "TestWorld",
    "package_name": "TestWorlds",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "BlueROV2",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "Camera",
                    "socket": "CameraSocket",
                }
            ],
            "control_scheme": 0,
            "location": [6.0, -47.0, -2.5],
            "rotation": [0.0, 0.0, 220.0],
        }
    ],
}

config_air = {
    "name": "test_fog",
    "world": "TestWorld",
    "package_name": "TestWorlds",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "Camera",
                    "socket": "Platform",
                }
            ],
            "control_scheme": 0,
            "location": [0.0, -53.0, 0.5],
            "rotation": [0.0, 0.0, 220.0],
        }
    ],
}


def test_fog_water():
    global config_water

    cfg = copy.deepcopy(config_water)

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        for _ in range(20):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("standard_fog.png")
        cv2.imwrite(filepath, pixels)

        env.water_fog(5, 0.5, 1.0, 0.0, 1.0)
        for _ in range(200):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("purple_fog.png")
        cv2.imwrite(filepath, pixels)


def test_fog_air():
    global config_air

    cfg = copy.deepcopy(config_air)

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        for _ in range(20):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("no_air_fog.png")
        cv2.imwrite(filepath, pixels)

        env.air_fog(1.5, 2, 0.0, 0.5, 1.0)
        for _ in range(200):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("air_fog.png")
        cv2.imwrite(filepath, pixels)


if __name__ == "__main__":
    test_fog_water()
    test_fog_air()
