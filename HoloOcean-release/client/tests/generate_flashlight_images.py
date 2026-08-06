import holoocean
import cv2

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

with holoocean.make(scenario_cfg=config, show_viewport=False) as env:
    for _ in range(60):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("3_flashlights_config.png")
    cv2.imwrite(filepath, pixels)

    env.turn_off_flashlight("flashlight1")
    env.turn_off_flashlight("flashlight2")
    env.turn_off_flashlight("flashlight3")
    for _ in range(60):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("flashlights_off.png")
    cv2.imwrite(filepath, pixels)

    env.turn_on_flashlight(
        "flashlight4", beam_width=30, location_z_offset=10, color_G=0
    )
    for _ in range(60):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("flashlight_4_on.png")
    cv2.imwrite(filepath, pixels)

    env.turn_off_flashlight("flashlight4")
    env.turn_on_flashlight("flashlight1")
    for _ in range(60):
        env.tick()
    pixels = env.tick()["Camera"][:, :, 0:3]
    filepath = str("flashlight_1_config.png")
    cv2.imwrite(filepath, pixels)
