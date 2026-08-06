import holoocean
import cv2

base_cfg = {
    "name": "test",
    "world": "TestWorld",
    "package_name": "TestWorlds",
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
            ],
            "control_scheme": 3,
            "location": [0, -50, 0],
            "rotation": [0, 0, 90],
        }
    ],
}

with holoocean.make(scenario_cfg=base_cfg) as env:
    for _ in range(20):
        env.tick()

    pixels = env.tick()["RGBCamera"][:, :, 0:3]
    filepath = str("baseline_tide.png")
    cv2.imwrite(filepath, pixels)

    for _ in range(25):
        env.tick()

    pixels = env.tick()["RGBCamera"][:, :, 0:3]
    filepath = str("baseline_tide_high.png")
    cv2.imwrite(filepath, pixels)

    env.tick()
