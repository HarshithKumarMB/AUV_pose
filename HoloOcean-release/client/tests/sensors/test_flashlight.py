import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err

config = {
    "name": "test_flashlight",
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
            "control_scheme": 2,
            "location": [-8.8, -31.8, 0.5],
            "rotation": [0.0, 0.0, 220.0],
        }
    ],
    "flashlight": [
        {
            "flashlight_name": "flashlight1",
            "intensity": 10000,
            "color_G": 0,
            "color_B": 0,  # red
        },
        {
            "flashlight_name": "flashlight2",
            "intensity": 10000,
            "color_G": 0,
            "color_R": 0,  # blue
        },
        {
            "flashlight_name": "flashlight3",
            "intensity": 10000,
            "color_R": 0,
            "color_B": 0,  # green
        },
        # flashlight 6 does not exist, should give a warning but not crash.
        {
            "flashlight_name": "flashlight6",
            "intensity": 10000,
            "color_R": 0,
            "color_B": 0,  # green
        },
    ],
}

"""Test to make sure that the flashlights work with both the config and the commands"""


def test_flashlights(request):
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

        # Let simulation environment set
        for _ in range(60):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/3_flashlights_config.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # 3 flashlights should be on with the settings from the config
        # # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)
        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Flashlights set up incorrectly"

        env.turn_off_flashlight("flashlight1")
        env.turn_off_flashlight("flashlight2")
        env.turn_off_flashlight("flashlight3")
        for _ in range(60):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/flashlights_off.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # All flashlights should be off (command)
        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Flashlights turned off incorrectly"

        env.turn_on_flashlight(
            "flashlight4", beam_width=30, location_z_offset=10, color_G=0
        )
        for _ in range(60):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/flashlight_4_on.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        assert baseline is not None, f"Failed to load baseline image from {filepath}"

        # Only flashlight 4 should be on with the settings from the command
        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Flashlight turned on incorrectly by command"

        env.turn_off_flashlight("flashlight4")
        env.turn_on_flashlight("flashlight1")
        for _ in range(60):
            env.tick()
        pixels = env.tick()["Camera"][:, :, 0:3]
        filepath = str("/baseline_images/flashlight_1_config.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))

        assert baseline is not None, f"Failed to load baseline image from {filepath}"
        # Flashlight 1 should be on by command but keep the setting from the config
        err = mean_square_err(pixels, baseline)
        assert err < 2500, "Flashlight did not keep parameters from config"
