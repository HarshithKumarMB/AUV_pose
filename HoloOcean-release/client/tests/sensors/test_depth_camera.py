import holoocean
import cv2
import copy
import uuid
import pytest
import numpy as np
from tests.utils.equality import mean_square_err


base_cfg = {
    "name": "test_semantic_camera",
    "world": "TestWorld",
    "main_agent": "sphere0",
    "frames_per_sec": False,
    "agents": [
        {
            "agent_name": "sphere0",
            "agent_type": "SphereAgent",
            "sensors": [
                {
                    "sensor_type": "DepthCamera",
                    "socket": "CameraSocket",
                    # note the different camera name. Regression test for #197
                    "sensor_name": "TestCamera",
                    "configuration": {
                        "CaptureWidth": 254,
                        "CaptureHeight": 254,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [
                0.95,
                -1.75,
                0.5,
            ],  # if you change this, you must change rotation_env too.
        }
    ],
}


base_cfg_2 = {
    "name": "test_semantic_camera",
    "world": "TestWorld",
    "main_agent": "sphere0",
    "frames_per_sec": False,
    "agents": [
        {
            "agent_name": "sphere0",
            "agent_type": "SphereAgent",
            "sensors": [
                {
                    "sensor_type": "DepthCamera",
                    "socket": "CameraSocket",
                    # note the different camera name. Regression test for #197
                    "sensor_name": "TestCamera",
                    "configuration": {
                        "CaptureWidth": 254,
                        "CaptureHeight": 254,
                    },
                },
            ],
            "control_scheme": 0,
            "location": [
                0.77,
                0.9,
                0.5,
            ],  # if you change this, you must change rotation_env too.
        }
    ],
}


@pytest.mark.skipif(
    "TestWorlds" not in holoocean.installed_packages(),
    reason="Ocean package not installed",
)
def test_depth_camera(resolution, request):
    """Makes sure that the depth camera is positioned and capturing correctly.

    Capture pixel data, and load from disk the baseline of what it should look like.
    Then, use mse() to see how different the images are.

    """

    global base_cfg

    cfg = copy.deepcopy(base_cfg)

    cfg["agents"][0]["sensors"][0]["configuration"] = {
        "CaptureWidth": resolution,
        "CaptureHeight": resolution,
    }

    # print("config: ", cfg)

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

        data = env.tick()["TestCamera"]
        depth = data["image"]  # grab the depth data from the camera output
        pixels = depth
        filepath = str("/baseline_images/baseline_depth_" + str(resolution) + ".jpg")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        err = mean_square_err(pixels[:, :, 0], baseline[:, :, 0])

        # Uncomment to visually compare images
        # cv2.imshow("Baselines", baseline)
        # cv2.imshow("Pixels", pixels)
        # cv2.waitKey(0)
        assert err < 2000


def test_depth_output():
    """Agent is placed 2 meters away from a wall. Makes sure depth camera output returns 2
    meters at the center pixel.

    Capture pixel data, then use mse() to see how close to 2 meters.

    """

    global base_cfg_2

    cfg = copy.deepcopy(base_cfg_2)

    cfg["agents"][0]["sensors"][0]["configuration"] = {
        "CaptureWidth": 256,
        "CaptureHeight": 256,
    }

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

        data = env.tick()["TestCamera"]
        depth = data["depth"]  # grab the depth data from the camera output
        pixels = depth
        # print(pixels[128,128])
        # cv2.imshow("Pixels", pixels)
        # cv2.waitKey(0)

        real_wall_distance = 2.0

        assert np.isclose(pixels[128, 128], real_wall_distance, 1e-2), (
            "Depth camera return is not correct!"
        )


shared_ticks_per_capture_env = None
