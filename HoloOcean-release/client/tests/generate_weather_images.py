import holoocean
import copy
import uuid
import cv2

config = {
    "name": "test_weather",
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
            "location": [-1.0, -55.0, 0.0],
            "rotation": [0.0, 0.0, 135.0],
        }
    ],
}

config_under = {
    "name": "test_weather",
    "world": "TestWorld",
    "package_name": "TestWorlds",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "Camera",
                    "socket": "CameraLeftSocket",
                }
            ],
            "control_scheme": 0,
            "location": [-1.0, -55.0, -1.0],
            "rotation": [0.0, 0.0, 135.0],
        }
    ],
}


def test_weather_sunny():
    global config
    cfg = copy.deepcopy(config)
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.reset()
        env.tick()
        env.change_weather(0)
        # Let simulation environment set
        for _ in range(2000):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("sunny.png")
        cv2.imwrite(filepath, pixels)


def test_weather_cloudy():
    global config
    cfg = copy.deepcopy(config)
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.reset()
        env.tick()
        env.change_weather(1)

        for _ in range(1500):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("cloudy.png")
        cv2.imwrite(filepath, pixels)


def test_weather_rain_under():
    global config_under
    cfg = copy.deepcopy(config_under)
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.reset()
        env.tick()
        env.change_weather(2)

        for _ in range(1500):
            env.tick()

        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("rainy_under.png")
        cv2.imwrite(filepath, pixels)


if __name__ == "__main__":
    test_weather_sunny()
    test_weather_cloudy()
    test_weather_rain_under()
