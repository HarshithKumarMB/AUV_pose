"""This file contains multiple examples of how you might use HoloOcean."""

import numpy as np

import holoocean
from holoocean.environments import *
from holoocean.fossen_dynamics import *


def hovering_example():
    """A basic example of how to use the HoveringAUV agent."""
    env = holoocean.make("SimpleUnderwater-Hovering")

    # This command tells the AUV go forward with a power of "10"
    # The last four elements correspond to the horizontal thrusters (see docs for more info)
    command = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    for _ in range(1000):
        state = env.step(command)
        # To access specific sensor data:
        if "PoseSensor" in state:
            pose = state["PoseSensor"]
        # Some sensors don't tick every timestep, so we check if it's received.
        if "DVLSensor" in state:
            dvl = state["DVLSensor"]

    # This command tells the AUV to go down with a power of "10"
    # The first four elements correspond to the vertical thrusters
    command = np.array([-10, -10, -10, -10, 0, 0, 0, 0])
    for _ in range(1000):
        # We alternatively use the act function
        env.act("auv0", command)
        state = env.tick()

    # You can control the AgentFollower camera (what you see) by pressing v to toggle spectator
    # mode. This detaches the camera and allows you to move freely about the world.
    # Press h to view the agents x-y-z location
    # You can also press c to snap to the location of the camera to see the world from the perspective of the
    # agent. See the Controls section of the ReadMe for more details.


def torpedo_example():
    """A basic example of how to use the TorpedoAUV agent."""
    env = holoocean.make("SimpleUnderwater-Torpedo")

    # This command tells the AUV go forward with a power of "50"
    # The last four elements correspond to
    command = np.array([0, 0, 0, 0, 50])
    for _ in range(1000):
        state = env.step(command)

    # Now turn the top and bottom fins to turn left
    command = np.array([0, -45, 0, 45, 50])
    for _ in range(1000):
        state = env.step(command)


def editor_example():
    """This editor example shows how to interact with HoloOcean worlds while they are being built
    in the Unreal Engine Editor. Most people that use HoloOcean will not need this.

    This example uses a custom scenario, see
    https://holoocean.readthedocs.io/en/latest/usage/examples/custom-scenarios.html

    Note: When launching HoloOcean from the editor, press the down arrow next to "Play" and select
    "Standalone Game", otherwise the editor will lock up when the client stops ticking it.
    """

    scenario = {
        "name": "test",
        "world": "ExampleLevel",
        "main_agent": "auv0",
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [
                    {
                        "sensor_type": "LocationSensor",
                    },
                    {"sensor_type": "VelocitySensor"},
                    {"sensor_type": "RGBCamera"},
                ],
                "control_scheme": 1,
                "location": [0, 0, 1],
            }
        ],
    }

    env = HoloOceanEnvironment(scenario=scenario, start_world=False)
    command = [0, 0, 10, 50]

    for i in range(10):
        env.reset()
        for _ in range(1000):
            state = env.step(command)


def editor_multi_agent_example():
    """This editor example shows how to interact with HoloOcean worlds that have multiple agents.
    This is specifically for when working with UE4 directly and not a prebuilt binary.

    Note: When launching HoloOcean from the editor, press the down arrow next to "Play" and select
    "Standalone Game", otherwise the editor will lock up when the client stops ticking it.
    """
    scenario = {
        "name": "test_handagent",
        "world": "ExampleLevel",
        "main_agent": "auv0",
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [],
                "control_scheme": 1,
                "location": [0, 0, 1],
            },
            {
                "agent_name": "auv1",
                "agent_type": "TorpedoAUV",
                "sensors": [],
                "control_scheme": 1,
                "location": [0, 0, 5],
            },
        ],
    }

    env = HoloOceanEnvironment(scenario=scenario, start_world=False)

    cmd0 = np.array([0, 0, -2, 10])
    cmd1 = np.array([0, 0, 5, 10])

    for i in range(10):
        env.reset()
        env.act("uav0", cmd0)
        env.act("uav1", cmd1)
        for _ in range(1000):
            states = env.tick()


def fossen_dynamics():
    """
    This exmaple steps through using fossen dynamics with multiple
    agents in HoloOcean. This example is very similar to the single
    agent example but with two agents.

    Demonstrates the capability to use fossen dynamics with a topedo
    agent and surface vessel agent.
    """

    ticks_per_sec = 50
    numSteps = 1500
    print("Total Simulation Time:", (numSteps / ticks_per_sec))
    agent_name = "auv0"

    initial_location = [0, 20, -280]  # Translation in NWU coordinate system
    initial_rotation = [
        0,
        0,
        0,
    ]  # Roll, pitch, Yaw in Euler angle order ZYX and in degrees NWU coordinate system

    scenario = {
        "name": "torpedo_dynamics",
        "package_name": "Ocean",
        "world": "OpenWater",
        "main_agent": agent_name,
        "ticks_per_sec": ticks_per_sec,
        "agents": [
            {
                "agent_name": agent_name,
                "agent_type": "CougUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False,  # Use quaternion for dynamics
                        },
                    },
                ],
                "control_scheme": 1,  # Control scheme 1 is how custom dynamics are applied to CougUV agents
                "location": initial_location,
                "rotation": initial_rotation,
                "fossen_model": "torpedo",
                "control_mode": "depthHeadingAutopilot",  # Initial control mode
                "dynamics": {
                    # Environment parameters:
                    "rho": 1026,  # Density of water in kg/m^3
                    # Vehicle physical parameters:
                    "mass": 31.03,  # Mass of vehicle in kg
                    "length": 1.6,  # Length of vehicle in m
                    "diam": 0.19,  # Diameter of vehicle in m
                    "r_bg": [
                        0,
                        0,
                        0.02,
                    ],  # Center of gravity of the vehicle (x, y, z) in body frame x forward, y right, z down
                    "r_bb": [
                        0,
                        0,
                        0,
                    ],  # Center of boyancy of the vehicle (x, y, z) in body frame x forward, y right, z down
                    "area_fraction": 0.7,  # relates vehicle effective area to length and width. pi/4 for a spheroid
                    # Low-speed linear damping matrix parameters:
                    "T_surge": 20,  # Surge time constant (s)
                    "T_sway": 20,  # Sway time constant (s)
                    "zeta_roll": 0.3,  # Roll damping ratio
                    "zeta_pitch": 0.8,  # Pitch damping ratio
                    "T_yaw": 1,  # Yaw time constant (s)
                    "K_nomoto": 0.25,  # Nomoto gain
                    # Other damping parameters:
                    "r44": 0.3,  # Added moment of inertia in roll: A44 = r44 * Ix
                    "Cd": 0.42,  # Coefficient of drag
                    "e": 0.7,  # Oswald efficiency factor for vehicle drag
                },
                "actuator": {
                    # Fins:
                    "fin_count": 4,  # Number of equally spaced fins on the vehicle
                    "fin_offset_deg": 0,  # Angle offset of first fin around x axis (deg) starting from positive y with z down
                    # 0 deg: fin on port side
                    # 90 deg: fin on bottom
                    "fin_center": 0.1,  # radius (m) from COP on the fin to the COM in the YZ plane
                    "fin_area": 0.00697,  # Surface area of one side of a fin
                    "x_fin": -0.8,  # Body frame x distance (x forward) from center of mass to fin COP
                    "CL_delta": 0.5,  # Coefficient of lift for fin
                    "deltaMax_fin_deg": 15,  # Max deflection of the fin (degrees)
                    "T_delta": 0.1,  # Time constant for fin actuation. (s)
                    # Propellor:
                    "nMax": 1525,  # Max rpm of the thruster
                    "T_n": 1.0,  # Time constant for thruster actuation. (s)
                    "D_prop": 0.14,  # Propeller diameter
                    "t_prop": 0.1,  # Propeller pitch
                    "KT_0": 0.4566,  # Thrust coefficient at zero rpm
                    "KQ_0": 0.0700,  # Torque coefficient at zero rpm
                    "KT_max": 0.1798,  # Max thrust coefficient
                    "KQ_max": 0.0312,  # Max torque coefficient
                    "w": 0.056,  # wake fraction number
                    "Ja_max": 0.6632,  # Max advance ratio
                },
                "autopilot": {
                    "depth": {
                        "wn_d_z": 0.12,  # Damped natural frequency for low pass filter for depth commands
                        "Kp_z": 0.153,  # Proportional gain for depth controller
                        "T_z": 100,  # Time constant for depth controller
                        "Kp_theta": 39.78,  # Proportional gain for pitch angle for depth controller
                        "Kd_theta": 17.1,  # Derivative gain for pitch angle for depth controller
                        "Ki_theta": 0.5,  # Integral gain for pitch angle for depth controller
                        "wn_d_theta": 0.25,  # Damped natural frequency for low pass filter for depth commands
                        "K_w": 0.0,  # Optional heave velocity feedback gain
                        "theta_max_deg": 15,  # Max output of pitch controller inner loop
                        "outer_loop_threshold": 2.91,  # Threshold for outer loop to switch to surge control
                        "surge_threshold": 0.6,  # Surge threshold for running depth controller.
                    },
                    "heading": {
                        "wn_d": 0.4,  # Damped natural frequency of input commands for low pass filter
                        "zeta_d": 1.0,  # Damping coefficient
                        "r_max": 0.87,
                        "lam": 0.1,
                        "phi_b": 0.1,
                        "K_d": 0.5,
                        "K_sigma": 0.05,
                    },
                },
            }
        ],
    }

    # Initialize the environment
    env = holoocean.make(scenario_cfg=scenario)
    env.set_render_quality(
        0
    )  # OPTIONAL: Set render quality to low for faster performance

    # Initialize Fossen dynamics model for the agent
    fossen_agents = [agent_name]
    fossen_interface = FossenInterface(fossen_agents, scenario)
    accel = np.zeros(6, float)  # Initial acceleration input to HoloOcean

    ############## Manual Control Example ##############

    fossen_interface.set_control_mode(agent_name, "manualControl")

    fins_degrees = np.array([10, 10, -10, -10])  # Fin deflections in degrees
    fin_radians = np.radians(fins_degrees)
    thruster_rpm = 800
    u_control = np.append(fin_radians, thruster_rpm)

    for i in range(numSteps):
        state = env.step(accel)
        fossen_interface.set_u_control(agent_name, u_control)
        accel = fossen_interface.update(agent_name, state)

    ############## Depth-Heading Autopilot Example ##############

    depth = 279
    heading = -10
    fossen_interface.set_control_mode(agent_name, "depthHeadingAutopilot")
    fossen_interface.set_goal(agent_name, depth, heading, 1525)

    for i in range(numSteps):
        state = env.step(accel)
        accel = fossen_interface.update(agent_name, state)

        pos = state["DynamicsSensor"][6:9]  # [x, y, z]
        x_end = pos[0] + 3 * np.cos(np.deg2rad(heading))
        y_end = pos[1] + 3 * np.sin(np.deg2rad(heading))

        color = [0, 255, 0] if abs(depth + pos[2]) <= 2.0 else [255, 0, 0]
        env.draw_arrow(
            pos.tolist(),
            end=[x_end, y_end, -depth],
            color=color,
            thickness=5,
            lifetime=0.03,
        )


def multi_agent_fossen():
    numSteps = 1000

    scenario = {
        "name": "multi_agent_fossen",
        "world": "OpenWater",
        "package_name": "Ocean",
        "main_agent": "auv0",
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "TorpedoAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False,  # Use quaternion for dynamics
                        },
                    },
                ],
                "control_scheme": 1,  # Control scheme 1 is how custom dynamics are applied to TAUV
                "location": [10, 0, 0],
                "rotation": [0, 0, 0],
                "fossen_model": "torpedo",
                "control_mode": "manualControl",
            },
            {
                "agent_name": "sv1",
                "agent_type": "SurfaceVessel",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False,  # Use quaternion for dynamics
                        },
                    },
                ],
                "control_scheme": 2,  # Control scheme 2 is how custom dynamics are applied to SV
                "location": [10, -10, 0],
                "rotation": [0, 0, 0],
                "fossen_model": "otter",
                "control_mode": "manualControl",
            },
        ],
    }

    env = holoocean.make(scenario_cfg=scenario)

    main_agent = "auv0"
    sv_agent = "sv1"
    fossen_agents = [main_agent, sv_agent]
    fossen_interface = FossenInterface(fossen_agents, scenario)

    accel = np.array(np.zeros(6), float)  # HoloOcean parameter input

    ############# MANUAL CONTROL EXAMPLE: ###########
    # Set control surfaces command
    fins_degrees = np.array(
        [10, -10, -10, 10]
    )  # Rudder and Stern Fin Deflection (degrees)
    fin_radians = np.radians(fins_degrees)
    thruster_rpm = 1000
    u_control_torpedo = np.append(
        fin_radians, thruster_rpm
    )  # [RudderAngle, SternAngle,Thruster] IN RADIANS
    fossen_interface.set_control_mode(main_agent, "manualControl")

    u_control_otter = np.array([105, 80])
    fossen_interface.set_control_mode(sv_agent, "manualControl")

    states = env.tick()  # Get the inital states of the agent for the dynamics

    for i in range(numSteps):
        fossen_interface.set_u_control(
            main_agent, u_control_torpedo
        )  # If desired you can change control command here
        fossen_interface.set_u_control(sv_agent, u_control_otter)

        for agent in fossen_agents:
            accel = fossen_interface.update(
                agent, states
            )  # Calculate accelerations to be applied to HoloOcean agent
            env.act(agent, accel)

        states = env.tick()
