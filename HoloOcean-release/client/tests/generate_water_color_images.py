import holoocean
import cv2

base_cfg = {
    "name": "test",
    "world": "TestWorld",
    "package_name": "TestWorlds",
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

with holoocean.make(scenario_cfg=base_cfg) as env:
    env.reset()
    for _ in range(20):
        env.tick()
    env.water_color(1, 0, 0)
    for _ in range(20):
        env.tick()

    pixels = env.tick()["RGBCamera"][:, :, 0:3]
    filepath = str("red_water.png")
    cv2.imwrite(filepath, pixels)
