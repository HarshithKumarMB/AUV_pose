import numpy as np
from holoocean.fossen_dynamics.helper_functions import (
    convert_NWU_to_NED,
    convert_NED_to_NWU,
)
from holoocean.fossen_dynamics.torpedo import Torpedo
from holoocean.fossen_dynamics.otter import Otter

fossen_model_keys = {
    "torpedo": Torpedo,
    "otter": Otter,
}


class FossenInterface:
    """
    Class for handling the connection between a holocean agent and
    the fossen dynamic model to control the motion of the vehicle

    :param list (str) vehicle_names: list of name of vehicles to initialize that matches agent in scenario dictionary
    :param dict scenario: scenario dictionary for holoocean
    :param bool multi_agent: If there are multiple agents within the scenario
    """

    def __init__(
        self,
        vehicle_names,
        scenario,
        multi_agent=False,  # If multiple agents
    ):
        self.vehicles = {}

        self.sampleTime = get_sample_period(scenario)

        for name in vehicle_names:
            check_scenario_configuration(scenario, name)

            vehicle_model, control_mode = get_vehicle_model(scenario, name)
            vehicle = vehicle_model(control_mode)

            if vehicle.config_fnc is not None:
                vehicle.config_fnc(scenario, name, vehicle)

            # IF VEHICLE HAS A DYNAMICS MANAGER THEN DO THIS
            # self.configure_from_scenario(scenario, name, vehicle)

            self.vehicles[name] = vehicle
            vehicle.u_control = np.zeros(vehicle.dimU, float)
            vehicle.u_actual = np.zeros(vehicle.dimU, float)

            # TODO: Check to make sure has no initial velocity set on the fossens side?

        # For Parsing the state in a multi agent scenario
        if len(self.vehicles) > 1 or multi_agent:
            self.multi_agent_scenario = True
        else:
            self.multi_agent_scenario = False

    def update(self, vehicle_name, state):
        """
        Given the state of the vehicle and control inputs
        return the accelerations of the vehicle in holoocean global frame
        """
        vehicle = self.vehicles[vehicle_name]
        if self.multi_agent_scenario:
            x = state[vehicle_name]["DynamicsSensor"]
        else:
            x = state["DynamicsSensor"]

        # From world frame state change to body frame in correct coordinate sys
        eta, nu = convert_NWU_to_NED(x)

        t = state["t"]

        # Vehicle specific control systems
        if vehicle.controlMode == "depthAutopilot":
            vehicle.u_control = vehicle.depthAutopilot(eta, nu, self.sampleTime)
        elif vehicle.controlMode == "headingAutopilot":
            vehicle.u_control = vehicle.headingAutopilot(eta, nu, self.sampleTime)
        elif vehicle.controlMode == "depthHeadingAutopilot":
            vehicle.u_control = vehicle.depthHeadingAutopilot(eta, nu, self.sampleTime)
        elif vehicle.controlMode == "DPcontrol":
            vehicle.u_control = vehicle.DPcontrol(eta, nu, self.sampleTime)
        elif vehicle.controlMode == "stepInput":
            vehicle.u_control = vehicle.stepInput(t)
        elif vehicle.controlMode == "manualControl":
            pass
        else:
            raise ValueError(f"Unknown control mode: {vehicle.controlMode}")

        # Compute dynamics in body frame, NED coordinate system
        nu_dot, vehicle.u_actual = vehicle.dynamics(
            eta, nu, vehicle.u_actual, vehicle.u_control, self.sampleTime
        )

        # return the acceleration in world frame for holoocean
        return convert_NED_to_NWU(x, nu_dot)

    def set_u_control(self, vehicle_name, u_control):
        """Set the control surface commands for the vehicle (angles in radians)"""
        vehicle = self.vehicles[vehicle_name]

        if len(u_control) == vehicle.dimU:
            vehicle.u_control = u_control
        else:
            raise ValueError(
                f"Vehicle Control command does not match expected length, got {len(u_control)}, expected: {vehicle.dimU}"
            )

    def get_u_control(self, vehicle_name):
        """Get the control surface positions for the vehicle (angles in radians)"""
        vehicle = self.vehicles[vehicle_name]

        return vehicle.u_control

    def get_control_mode(self, vehicle_name):
        """
        Getys the control mode for the vehicle.

        :param str vehicle name: The vehicle control system is asked for

        :returns: str vehicle control mode
        """
        vehicle = self.vehicles[vehicle_name]
        return vehicle.controlMode

    def set_control_mode(self, vehicle_name, controlSystem):
        """
        Sets the control mode for the vehicle.

        :param str controlSystem: The control system to use. Possible values are:

        - ``"depthHeadingAutopilot"``: Depth and heading autopilots.
        - ``"manualControl"``: Manual input control with set_u_control().
        - ``"stepInput"``: Step inputs

        - Any other value: controlSystem is set to "stepInput".

        :returns: None
        """
        # TODO update documenatation
        vehicle = self.vehicles[vehicle_name]

        if controlSystem == "depthHeadingAutopilot":
            vehicle.controlDescription = "Depth and heading autopilots"
        elif controlSystem == "stepInput":
            vehicle.controlDescription = "Step inputs"
        elif vehicle.controlMode == "DPcontrol":
            vehicle.controlDescription = "DPcontrol"
        elif vehicle.controlMode == "headingAutopilot":
            vehicle.controlDescription = "Heading autopilot"
        elif vehicle.controlMode == "depthAutopilot":
            vehicle.controlDescription = "Depth autopilot"
        elif controlSystem == "manualControl":
            vehicle.controlDescription = "Manual input control with set_u_control()"
        else:
            vehicle.controlDescription = "Step inputs"
            controlSystem = "stepInput"
        vehicle.controlMode = controlSystem
        print(vehicle.controlDescription)

    # NOTE: Could be useful to have a flag that initializes the LP filter to current state
    def set_goal(self, vehicle_name, depth=None, heading=None, rpm=None):
        """
        Set the goals for the autopilot.

        :param float depth: Desired depth (m), positive downwards.
        :param float heading: Desired yaw angle (deg). (-180 to 180)
        :param float rpm: Desired propeller revolution (rpm).

        :returns: None
        """
        if rpm is not None:
            self.set_rpm_goal(vehicle_name, rpm)
        if heading is not None:
            self.set_heading_goal(vehicle_name, heading)
        if depth is not None:
            self.set_depth_goal(vehicle_name, depth)

    def set_heading_goal(self, vehicle_name, heading):
        """
        Set the heading goals for the autopilot.

        :param float depth: Desired heading (deg), -180 to 180 in NWU frame

        :returns: None
        """
        vehicle = self.vehicles[vehicle_name]

        # Convert to NED frame
        heading = -heading

        vehicle.ref_psi = heading
        if abs(heading) > 180.0:
            raise ValueError(
                "The heading command value should be on the interval -180 to 180"
            )

    def set_depth_goal(self, vehicle_name, depth):
        """
        Set the depth goals for the autopilot.

        :param float depth: Desired depth (m), positive downward in world frame.

        :returns: None
        """
        vehicle = self.vehicles[vehicle_name]

        vehicle.ref_z = depth

        if depth < 0.0:
            raise ValueError("The depth command value should be greater than 0")

    def set_rpm_goal(self, vehicle_name, rpm):
        """
        Set the rpm goals for the torpedo autopilot.

        :param float depth: Desired rpm for thruster

        :returns: None
        """
        vehicle = self.vehicles[vehicle_name]

        vehicle.ref_n = rpm
        vehicle.surge_control = False

        if rpm < 0.0 or rpm > vehicle.controls[-1].nMax:
            raise ValueError(
                f"The RPM value should be in the interval 0-{vehicle.controls[-1].nMax}"
            )


def check_scenario_configuration(scenario, vehicle_name):
    if "agents" not in scenario:
        raise ValueError("The scenario must contain an 'agents' key.")

    agent = next(
        (agent for agent in scenario["agents"] if agent["agent_name"] == vehicle_name),
        None,
    )
    if agent is None:
        raise ValueError(
            f"No agent with the name '{vehicle_name}' found in the scenario."
        )

    # TODO check the scenario control scheme for the vehicle. Use Holoocean builtin functions?
    # if agent.get("control_scheme") != 1:
    #     raise ValueError(f"Agent {agent['agent_name']} should have a control scheme of 1.")

    sensors = agent.get("sensors", [])
    dynamics_sensor_found = False

    for sensor in sensors:
        if sensor.get("sensor_type") == "DynamicsSensor":
            if (
                "sensor_name" not in sensor
                or sensor.get("sensor_name") == "DynamicsSensor"
            ):
                config = sensor.get("configuration", {})

                if not config.get("UseCOM"):
                    raise ValueError(
                        f"Agent {agent['agent_name']} DynamicsSensor must have 'UseCOM' set to True."
                    )

                if config.get("UseRPY"):
                    raise ValueError(
                        f"Agent {agent['agent_name']} DynamicsSensor must have 'UseRPY' set to False."
                    )
                dynamics_sensor_found = True

    if not dynamics_sensor_found:
        raise ValueError(
            f"Agent {agent['agent_name']} must have a 'DynamicsSensor' in its sensors that is named DynamicsSensor"
        )


def get_vehicle_model(scenario, vehicle_name):
    agent = next(
        (agent for agent in scenario["agents"] if agent["agent_name"] == vehicle_name),
        None,
    )

    if agent is None:
        raise KeyError(f"No agent found with name '{vehicle_name}'")

    model_name = agent.get("fossen_model", None)
    if model_name is None:
        raise KeyError(
            f"Agent {agent['agent_name']} is missing the 'fossen_model' key. Cannot be initialized with Fossen Dynamics."
        )

    if model_name not in fossen_model_keys:
        raise KeyError(
            f"Model name '{model_name}' is not in the list of available Fossen model types: {list(fossen_model_keys.keys())}"
        )

    control_mode = agent.get("control_mode", None)
    if control_mode is None:
        print(
            f"Agent {agent['agent_name']} is missing the 'control_mode' key. Defaults to Manual Control."
        )
        control_mode = "manualControl"

    # Return a vehicle object based on the fossen model defined in the scenario
    return fossen_model_keys[model_name], control_mode


def get_sample_period(scenario):
    ticks_per_sec = scenario.get("ticks_per_sec", None)

    if ticks_per_sec is None:
        ticks_per_sec = 30
        print(f"Warning: Using default ticks per second: {ticks_per_sec}")

    return 1.0 / ticks_per_sec
