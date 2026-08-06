import holoocean
import cv2

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

with holoocean.make(scenario_cfg=config) as env:
    env.change_time_of_day(0)
    for _ in range(200):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("time_0.png")
    cv2.imwrite(filepath, pixels)

    env.change_time_of_day(6)
    for _ in range(200):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("time_6.png")
    cv2.imwrite(filepath, pixels)

    env.change_time_of_day(12)
    for _ in range(200):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("time_12.png")
    cv2.imwrite(filepath, pixels)

    env.change_time_of_day(18)
    for _ in range(200):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("time_18.png")
    cv2.imwrite(filepath, pixels)

    for _ in range(200):
        env.tick()
