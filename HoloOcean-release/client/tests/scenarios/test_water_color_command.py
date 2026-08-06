import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err


base_cfg = {
    "name": "test",
    "world": "TestWorld",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "main_agent": "sv0",
    "agents": [
        {
            "agent_name": "sv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "RGBCamera",
                    "socket": "CameraRightSocket",
                }
            ],
            "control_scheme": 3,
            "location": [0, -52, -1],
            "rotation": [0, 0, 0],
        }
    ],
}


def test_change_watercolor(request):
    global base_cfg
    cfg = copy.deepcopy(base_cfg)
    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.reset()
        for _ in range(20):
            env.tick()
        env.water_color(1, 0, 0)
        for _ in range(20):
            env.tick()
        pixels = env.tick()["RGBCamera"][:, :, 0:3]
        filepath = str("/baseline_images/red_water.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))
        err = mean_square_err(pixels, baseline)
        assert err < 2500, "The water color is wrong"
