import holoocean
import cv2
import copy
import uuid

# from tests.utils.equality import mean_square_err

base_cfg = {
    "name": "test_viewport_capture",
    "world": "TestWorld",
    "package_name": "TestWorlds",
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
            "location": [0.95, -1.75, 0.5],
        }
    ],
}


def test_depth_camera(resolution):
    """Makes sure that the RGB camera is positioned and capturing correctly.

    Capture pixel data, and load from disk the baseline of what it should look like.
    Then, use mse() to see how different the images are.

    """

    global base_cfg

    cfg = copy.deepcopy(base_cfg)

    cfg["agents"][0]["sensors"][0]["sensor_type"] = "DepthCamera"
    cfg["agents"][0]["sensors"][0]["socket"] = "CameraSocket"
    cfg["agents"][0]["sensors"][0]["configuration"] = {
        "CaptureWidth": resolution,
        "CaptureHeight": resolution,
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

        val = env.tick()
        data = val["TestCamera"]
        depth = data["image"]  # grab the depth data from the camera output
        pixels = depth
        filepath = str("baseline_depth_" + str(resolution) + ".jpg")
        cv2.imwrite(filepath, pixels)
        cv2.imshow("image", pixels)
        cv2.waitKey(0)


if __name__ == "__main__":
    print("Hello")
    test_depth_camera(256)
    test_depth_camera(500)
    test_depth_camera(1000)
