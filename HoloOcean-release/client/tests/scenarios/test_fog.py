import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err

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

"""Test to make sure that water fog gets set correctly"""


def test_water_fog(request):
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
        env.reset()
        env.tick()
        # Let simulation environment set
        for _ in range(20):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/standard_fog.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Default fog set incorrectly"

        # Change to purple fog
        env.water_fog(5, 0.5, 1.0, 0.0, 1.0)
        for _ in range(200):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/purple_fog.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Purple fog set incorrectly"


"""Test to make sure that air fog gets set correctly"""


def test_air_fog(request):
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
        env.reset()
        env.tick()
        # No default fog, so no fog at beggining of simulation
        for _ in range(20):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/no_air_fog.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Air should be clear at the start"

        # Add air fog
        env.air_fog(1.5, 2, 0.0, 0.5, 1.0)
        for _ in range(200):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/air_fog.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Air fog set incorrectly"
