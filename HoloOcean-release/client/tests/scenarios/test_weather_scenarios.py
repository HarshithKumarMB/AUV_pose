import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err

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

"""Test to make sure that sunny weather gets set correctly"""


def test_weather_sunny(request):
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
        filepath = str("/baseline_images/sunny.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 5000, "Sunny weather set incorrectly"


"""Test to make sure that cloudy weather gets set correctly"""


def test_weather_cloudy(request):
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
        # Let simulation environment set
        for _ in range(1500):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/cloudy.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 2000, "Cloudy weather set incorrectly"


"""Test to make sure that rainy weather gets set correctly. Rain above surface"""


def test_weather_rain_above(request):
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
        env.change_weather(2)

        # Let simulation environment set
        for _ in range(1500):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        for _ in range(200):
            env.tick()
        pixels2 = env.tick()["Camera"][:, :, 0:3]

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("real 2", pixels2)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, pixels2)

        assert err > 25, "Rainy weather set incorrectly"


"""Test to make sure that rainy weather gets set correctly. No rain underwater"""


def test_weather_rain_under(request):
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
        # Let simulation environment set
        for _ in range(1500):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/rainy_under.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Rainy weather set incorrectly."
