import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err

config = {
    "name": "test_time",
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
            "location": [-1.0, -50.0, 1.0],
            "rotation": [0.0, 0.0, 90.0],
        }
    ],
}

"""Test to make sure that the sun is rotating correctly correctly"""


def test_time_cycle(request):
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

        env.change_time_of_day(0)
        for _ in range(200):
            env.tick()
        pixels1 = env.tick()["Camera"][:, :, 0:3]
        filepath1 = str("/baseline_images/time_0.png")
        baseline1 = cv2.imread(str(request.fspath.dirname + filepath1))

        # to show pictures visually
        # cv2.imshow("real", pixels1)
        # cv2.imshow("meant to be", baseline1)
        # cv2.waitKey(0)

        env.change_time_of_day(6)
        for _ in range(200):
            env.tick()
        pixels2 = env.tick()["Camera"][:, :, 0:3]
        filepath2 = str("/baseline_images/time_6.png")
        baseline2 = cv2.imread(str(request.fspath.dirname + filepath2))

        # to show pictures visually
        # cv2.imshow("real", pixels2)
        # cv2.imshow("meant to be", baseline2)
        # cv2.waitKey(0)

        env.change_time_of_day(12)
        for _ in range(200):
            env.tick()
        pixels3 = env.tick()["Camera"][:, :, 0:3]
        filepath3 = str("/baseline_images/time_12.png")
        baseline3 = cv2.imread(str(request.fspath.dirname + filepath3))

        # to show pictures visually
        # cv2.imshow("real", pixels3)
        # cv2.imshow("meant to be", baseline3)
        # cv2.waitKey(0)

        env.change_time_of_day(18)
        for _ in range(200):
            env.tick()
        pixels4 = env.tick()["Camera"][:, :, 0:3]
        filepath4 = str("/baseline_images/time_18.png")
        baseline4 = cv2.imread(str(request.fspath.dirname + filepath4))

        # to show pictures visually
        # cv2.imshow("real", pixels4)
        # cv2.imshow("meant to be", baseline4)
        # cv2.waitKey(0)

        err1 = mean_square_err(pixels1, baseline1)
        err2 = mean_square_err(pixels2, baseline2)
        err3 = mean_square_err(pixels3, baseline3)
        err4 = mean_square_err(pixels4, baseline4)

        assert err1 < 3000, (
            f"Sun rotation mismatch at cycle 1: MSE={err1:.2f}, expected < 3000"
        )
        assert err2 < 3600, (
            f"Sun rotation mismatch at cycle 2: MSE={err2:.2f}, expected < 3600"
        )
        assert err3 < 3000, (
            f"Sun rotation mismatch at cycle 3: MSE={err3:.2f}, expected < 3000"
        )
        assert err4 < 3000, (
            f"Sun rotation mismatch at cycle 4: MSE={err4:.2f}, expected < 3000"
        )
