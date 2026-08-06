import holoocean
import uuid
import cv2
import copy

from tests.utils.equality import mean_square_err


base_cfg = {
    "name": "test_tides",
    "world": "TestWorld",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "tide_cycle": {"active": True, "amplitude": 05.0, "frequency": 100},
    "main_agent": "a0",
    "agents": [
        {
            "agent_name": "a0",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {"sensor_type": "RGBCamera", "socket": "Platform"},
                {"sensor_type": "DynamicsSensor", "socket": "Payload"},
            ],
            "control_scheme": 3,
            "location": [0, -50, 0],
            "rotation": [0, 0, 90],
        }
    ],
}

"""Test to make sure that floatables, our water surface, and our agent all float together and correctly"""


def test_auto_tide(request):
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
        pixels = env.tick()["RGBCamera"][:, :, 0:3]
        filepath = str("/baseline_images/baseline_tide.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))

        # to show pictures visually
        # cv2.imshow("real", pixels)
        # cv2.imshow("meant to be", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 3000, "before tide moved this was already wrong"

        for _ in range(25):
            env.tick()
        pixels = env.tick()["RGBCamera"][:, :, 0:3]
        filepath = str("/baseline_images/baseline_tide_high.png")
        baseline = cv2.imread(str(request.fspath.dirname + filepath))

        # to show pictures visually
        # cv2.imshow("real2", pixels)
        # cv2.imshow("meant to be2", baseline)
        # cv2.waitKey(0)

        err = mean_square_err(pixels, baseline)
        assert err < 2500, "this is wrong after the tide moved"
        # this is to account for weird lighting differences


"""Test to make sure that our agent collides and rises properly with obsticles while the tide is going"""


def test_tide_collisions():
    global base_cfg
    cfg = copy.deepcopy(base_cfg)

    cfg["tide_cycle"]["active"] = False

    binary_path = holoocean.packagemanager.get_binary_path_for_package("TestWorlds")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=cfg,
        binary_path=binary_path,
        show_viewport=False,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.tide(-500, 1)
        for _ in range(200):
            env.tick()
        locationZ = env.tick()["DynamicsSensor"][8]
        assert abs(locationZ + 2.6) < 0.25, (
            f"our agent did not collide location = {locationZ}"
        )
        env.tide(0, 1)
        for _ in range(100):
            env.tick()
        locationZ = env.tick()["DynamicsSensor"][8]
        assert abs(locationZ) < 0.6, (
            f"our agent did not rise correctly location = {locationZ}"
        )
