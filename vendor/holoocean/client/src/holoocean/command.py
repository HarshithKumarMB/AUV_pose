"""This module contains the classes used for formatting and sending commands to the HoloOcean
backend. Most of these commands are just used internally by HoloOcean, regular users do not need to
worry about these.

"""

import numpy as np
from holoocean.exceptions import HoloOceanException


class CommandsGroup:
    """Represents a list of commands

    Can convert list of commands to json.

    """

    def __init__(self):
        self._commands = []

    def add_command(self, command):
        """Adds a command to the list

        Args:
            command (:class:`Command`): A command to add."""
        self._commands.append(command)

    def to_json(self):
        """
        Returns:
             :obj:`str`: Json for commands array object and all of the commands inside the array.

        """
        commands = ",".join(map(lambda x: x.to_json(), self._commands))
        return '{"commands": [' + commands + "]}"

    def clear(self):
        """Clear the list of commands."""
        self._commands.clear()

    @property
    def size(self):
        """
        Returns:
            int: Size of commands group"""
        return len(self._commands)


class Command:
    """Base class for Command objects.

    Commands are used for IPC between the holoocean python bindings and holoocean
    binaries.

    Derived classes must set the ``_command_type``.

    The order in which :meth:`add_number_parameters` and :meth:`add_string_parameters` are called
    is significant, they are added to an ordered list. Ensure that you are adding parameters in
    the order the client expects them.

    """

    def __init__(self):
        self._parameters = []
        self._command_type = ""

    def set_command_type(self, command_type):
        """Set the type of the command.

        Args:
            command_type (:obj:`str`): This is the name of the command that it will be set to.

        """
        self._command_type = command_type

    def add_number_parameters(self, number):
        """Add given number parameters to the internal list.

        Args:
            number (:obj:`list` of :obj:`int`/:obj:`float`, or singular :obj:`int`/:obj:`float`):
                A number or list of numbers to add to the parameters.

        """
        if isinstance(number, list) or isinstance(number, tuple):
            for x in number:
                self.add_number_parameters(x)
            return
        self._parameters.append('{ "value": ' + str(number) + " }")

    def add_string_parameters(self, string):
        """Add given string parameters to the internal list.

        Args:
            string (:obj:`list` of :obj:`str` or :obj:`str`):
                A string or list of strings to add to the parameters.

        """
        if isinstance(string, list) or isinstance(string, tuple):
            for x in string:
                self.add_string_parameters(x)
            return
        self._parameters.append('{ "value": "' + string + '" }')

    def to_json(self):
        """Converts to json.

        Returns:
            :obj:`str`: This object as a json string.

        """
        to_return = (
            '{ "type": "'
            + self._command_type
            + '", "params": ['
            + ",".join(self._parameters)
            + "]}"
        )
        return to_return


class CommandCenter:
    """Manages pending commands to send to the client (the engine).

    Args:
        client (:class:`~holoocean.holooceanclient.HoloOceanClient`): Client to send commands to

    """

    def __init__(self, client):
        self._client = client

        # Set up command buffer
        self._command_bool_ptr = self._client.malloc("command_bool", [1], np.bool_)
        # This is the size of the command buffer that HoloOcean expects/will read.
        self.max_buffer = 1048576
        self._command_buffer_ptr = self._client.malloc(
            "command_buffer", [self.max_buffer], np.byte
        )
        self._commands = CommandsGroup()
        self._should_write_to_command_buffer = False

    def clear(self):
        """Clears pending commands"""
        self._commands.clear()

    def handle_buffer(self):
        """Writes the list of commands into the command buffer, if needed.

        Checks if we should write to the command buffer, writes all of the queued commands to the
        buffer, and then clears the contents of the self._commands list

        """
        if self._should_write_to_command_buffer:
            self._write_to_command_buffer(self._commands.to_json())
            self._should_write_to_command_buffer = False
            self._commands.clear()

    def enqueue_command(self, command_to_send):
        """Adds command to outgoing queue.

        Args:
            command_to_send (:class:`Command`): Command to add to queue

        """
        self._should_write_to_command_buffer = True
        self._commands.add_command(command_to_send)

    def _write_to_command_buffer(self, to_write):
        """Write input to the command buffer.

        Reformat input string to the correct format.

        Args:
            to_write (:class:`str`): The string to write to the command buffer.

        """
        np.copyto(self._command_bool_ptr, True)
        to_write += "0"  # The gason JSON parser in holoocean expects a 0 at the end of the file.
        input_bytes = str.encode(to_write)
        if len(input_bytes) > self.max_buffer:
            raise HoloOceanException("Error: Command length exceeds buffer size")
        for index, val in enumerate(input_bytes):
            self._command_buffer_ptr[index] = val

    @property
    def queue_size(self):
        """
        Returns:
            int: Size of commands queue"""
        return self._commands.size


class SpawnAgentCommand(Command):
    """Spawn an agent in the world.

    Args:
        location (:obj:`list` of :obj:`float`): ``[x, y, z]`` location to spawn agent (see :ref:`coordinate-system`)
        name (:obj:`str`): The name of the agent.
        agent_type (:obj:`str` or type): The type of agent to spawn (UAVAgent, NavAgent, ...)

    """

    def __init__(self, location, rotation, name, agent_type, is_main_agent=False):
        super(SpawnAgentCommand, self).__init__()
        self._command_type = "SpawnAgent"
        self.set_location(location)
        self.set_rotation(rotation)
        self.set_type(agent_type)
        self.set_name(name)
        self.add_number_parameters(int(is_main_agent))

    def set_location(self, location):
        """Set where agent will be spawned.

        Args:
            location (:obj:`list` of :obj:`float`): ``[x, y, z]`` location to spawn agent (see :ref:`coordinate-system`)

        """
        if len(location) != 3:
            raise HoloOceanException("Invalid location given to spawn agent command")
        self.add_number_parameters(location)

    def set_rotation(self, rotation):
        """Set where agent will be spawned.

        Args:
            rotation (:obj:`list` of :obj:`float`): ``[roll, pitch, yaw]`` rotation for agent.
                (see :ref:`rotations`)

        """
        if len(rotation) != 3:
            raise HoloOceanException("Invalid rotation given to spawn agent command")
        self.add_number_parameters(rotation)

    def set_name(self, name):
        """Set agents name

        Args:
            name (:obj:`str`): The name to set the agent to.

        """
        self.add_string_parameters(name)

    def set_type(self, agent_type):
        """Set the type of agent.

        Args:
            agent_type (:obj:`str` or :obj:`type`): The type of agent to spawn.

        """
        if not isinstance(agent_type, str):
            agent_type = agent_type.agent_type  # Get str from type
        self.add_string_parameters(agent_type)


class DebugDrawCommand(Command):
    """Draw debug geometry in the world.

    Args:
        draw_type (:obj:`int`) : The type of object to draw

            - ``0``: line
            - ``1``: arrow
            - ``2``: box
            - ``3``: point

        start (:obj:`list` of :obj:`float`): The start  ``[x, y, z]`` location in meters of the object.
            (see :ref:`coordinate-system`)
        end (:obj:`list` of :obj:`float`): The end ``[x, y, z]`` location in meters of the object
            (not used for point, and extent for box)
        color (:obj:`list` of :obj:`float`): ``[r, g, b]`` color value (from 0 to 255).
        thickness (:obj:`float`): thickness of the line/object
        lifetime (:obj:`float`): Number of simulation seconds the object should persist. If 0, makes persistent

    """

    def __init__(self, draw_type, start, end, color, thickness, lifetime):
        super(DebugDrawCommand, self).__init__()
        self._command_type = "DebugDraw"

        # Check type of variable is not numpy array

        variables = [start, end, color]

        for variable in variables:
            if not isinstance(variable, list):
                raise TypeError(f"Expected a list, but got {type(variable).__name__}")

        self.add_number_parameters(draw_type)
        self.add_number_parameters(start)
        self.add_number_parameters(end)
        self.add_number_parameters(color)
        self.add_number_parameters(thickness)
        self.add_number_parameters(lifetime)


class TeleportCameraCommand(Command):
    """Move the viewport camera (agent follower)

    Args:
        location (:obj:`list` of :obj:`float`): The ``[x, y, z]`` location to give the camera
            (see :ref:`coordinate-system`)
        rotation (:obj:`list` of :obj:`float`): The ``[roll, pitch, yaw]`` rotation to give the camera
            (see :ref:`rotations`)

    """

    def __init__(self, location, rotation):
        Command.__init__(self)
        self._command_type = "TeleportCamera"
        self.add_number_parameters(location)
        self.add_number_parameters(rotation)


class AddSensorCommand(Command):
    """Add a sensor to an agent

    Args:
        sensor_definition (~holoocean.sensors.SensorDefinition): Sensor to add
    """

    def __init__(self, sensor_definition):
        Command.__init__(self)
        self._command_type = "AddSensor"
        self.add_string_parameters(sensor_definition.agent_name)
        self.add_string_parameters(sensor_definition.sensor_name)
        self.add_string_parameters(sensor_definition.type.sensor_type)
        self.add_string_parameters(sensor_definition.get_config_json_string())
        self.add_string_parameters(sensor_definition.socket)

        self.add_number_parameters(sensor_definition.location[0])
        self.add_number_parameters(sensor_definition.location[1])
        self.add_number_parameters(sensor_definition.location[2])

        self.add_number_parameters(sensor_definition.rotation[0])
        self.add_number_parameters(sensor_definition.rotation[1])
        self.add_number_parameters(sensor_definition.rotation[2])


class OceanCurrentsCommand(Command):
    """Share individual current velocities for each agent

    Args:
        ocean_currents: current velocity per agent
    """

    def __init__(self, name, x, y, z, vehicle_debugging):
        Command.__init__(self)
        self._command_type = "OceanCurrents"
        self.add_string_parameters(name)
        self.add_number_parameters(x)
        self.add_number_parameters(y)
        self.add_number_parameters(z)
        self.add_number_parameters(vehicle_debugging)


class RemoveSensorCommand(Command):
    """Remove a sensor from an agent

    Args:
        agent (:obj:`str`): Name of agent to modify
        sensor (:obj:`str`): Name of the sensor to remove

    """

    def __init__(self, agent, sensor):
        Command.__init__(self)
        self._command_type = "RemoveSensor"
        self.add_string_parameters(agent)
        self.add_string_parameters(sensor)


class RotateSensorCommand(Command):
    """Rotate a sensor on the agent. Multiple rotations do not accumulate, but rather sets to the last rotation.

    Args:
        agent (:obj:`str`): Name of agent
        sensor (:obj:`str`): Name of the sensor to rotate
        rotation (:obj:`list` of :obj:`float`): ``[roll, pitch, yaw]`` rotation for sensor.

    """

    def __init__(self, agent, sensor, rotation):
        Command.__init__(self)
        self._command_type = "RotateSensor"
        self.add_string_parameters(agent)
        self.add_string_parameters(sensor)
        self.add_number_parameters(rotation)


class RenderViewportCommand(Command):
    """Enable or disable the viewport. Note that this does not prevent the viewport from being shown,
    it just prevents it from being updated.

    Args:
        render_viewport (:obj:`bool`): If viewport should be rendered

    """

    def __init__(self, render_viewport):
        Command.__init__(self)
        self.set_command_type("RenderViewport")
        self.add_number_parameters(int(bool(render_viewport)))


class RGBCameraRateCommand(Command):
    """Set the number of ticks between captures of the RGB camera.

    Args:
        agent_name (:obj:`str`): name of the agent to modify
        sensor_name (:obj:`str`): name of the sensor to modify
        ticks_per_capture (:obj:`int`): number of ticks between captures

    """

    def __init__(self, agent_name, sensor_name, ticks_per_capture):
        Command.__init__(self)
        self._command_type = "RGBCameraRate"
        self.add_string_parameters(agent_name)
        self.add_string_parameters(sensor_name)
        self.add_number_parameters(ticks_per_capture)


class RaycastLidarRateCommand(Command):
    """Set the number of ticks between captures of the Raycast Lidar.

    Args:
        agent_name (:obj:`str`): name of the agent to modify
        sensor_name (:obj:`str`): name of the sensor to modify
        ticks_per_capture (:obj:`int`): number of ticks between captures

    """

    def __init__(self, agent_name, sensor_name, ticks_per_capture):
        Command.__init__(self)
        self._command_type = "RaycastLidarRate"
        self.add_string_parameters(agent_name)
        self.add_string_parameters(sensor_name)
        self.add_number_parameters(ticks_per_capture)


class RenderQualityCommand(Command):
    """Adjust the rendering quality of HoloOcean

    Args:
        render_quality (:obj:`int`): 0 = low, 1 = medium, 3 = high, 3 = epic

    """

    def __init__(self, render_quality):
        Command.__init__(self)
        self.set_command_type("AdjustRenderQuality")
        self.add_number_parameters(int(render_quality))


class WaterColorCommand(Command):
    """Changes the water color in the current world.

    Args:
        red (:obj:`float`): The red intensity value of the new water, between 0 and 1.
        green (:obj:`float`): The green intensity value of the new water, between 0 and 1.
        blue (:obj:`float`): The blue intensity value of the new water, between 0 and 1.
    """

    def __init__(self, red, green, blue):
        Command.__init__(self)
        self.set_command_type("WaterColor")
        self.add_number_parameters(float(red))
        self.add_number_parameters(float(green))
        self.add_number_parameters(float(blue))


class TideCommand(Command):
    """Changes the water level in the current world.

    Args:
        adjustment (:obj:`float`): The amount in meters you want to adjust the water level by or the level you want to set the water at.
        absolute (:obj:`bool`): True if you want to set a new surface level and False if you want to adjust the tides by an offset.
    """

    def __init__(self, adjustment, absolute):
        Command.__init__(self)
        self.set_command_type("Tide")
        self.add_number_parameters(float(adjustment))
        self.add_number_parameters(int(bool(absolute)))


class ChangeWeatherCommand(Command):
    """Changes the weather in the world.

    Args:
        weather (:obj:`int`): The weather wanted (0 - sunny, 1 - cloudy, 2 - rainy).
    """

    def __init__(self, weather):
        Command.__init__(self)
        self.set_command_type("ChangeWeather")
        self.add_number_parameters(int(weather))


class SetRainParametersCommand(Command):
    """Changes the rain's velocity and spawn rate.
    You must first activate weather and set it to ``2 - rainy`` using the Change Weather Command
    before this will have any visible effect.

    Args:
        vel_x (:obj:`float`): Rain velocity on the x axis.
        vel_y (:obj:`float`): VRain velocity on the y axis.
        vel_z (:obj:`float`): Rain velocity on the z axis. Should be a negative value.
        spawnRate(:obj:`float`): Rain's spawn rate (number of particles).
    """

    def __init__(self, vel_x, vel_y, vel_z, spawnRate):
        Command.__init__(self)
        self.set_command_type("SetRainParameters")
        self.add_number_parameters(float(vel_x))
        self.add_number_parameters(float(vel_y))
        self.add_number_parameters(float(vel_z))
        self.add_number_parameters(float(spawnRate))


class ChangeTimeOfDayCommand(Command):
    """Changes the world's time of day.

    Args:
        TimeOfDay (:obj:`float`): time of day desired, a number between 0 and 23 inclusive.
    """

    def __init__(self, TimeOfDay):
        Command.__init__(self)
        self.set_command_type("ChangeTimeOfDay")
        self.add_number_parameters(TimeOfDay)


class SetFPSCommand(Command):
    """Set the frames per second of the simulation.

    Args:
        fps (:obj:`int`): The number of frames per second to set the simulation to.

    """

    def __init__(self, fps):
        Command.__init__(self)
        self.set_command_type("AdjustFPS")
        self.add_number_parameters(int(fps))


class SetTPSCommand(Command):
    """Set the ticks per second of the simulation.

    Args:
        tps (:obj:`int`): The number of ticks per second to set the simulation to.

    """

    def __init__(self, tps):
        Command.__init__(self)
        self.set_command_type("AdjustTPS")
        self.add_number_parameters(int(tps))


class TurnOnFlashlightCommand(Command):
    """Turns on the vehicle's flashlight and sets its visual parameters.

    Args:
        flashlight_name(:obj:`str`): The name of the flashlight to turn on. (e.g., flashlight1)
        intensity (:obj:`float`): The brightness of the flashlight. Recommended range: 0 to 100000. (Default = 5000)
        beam_width (:obj:`float`): The beam's spread angle in degrees. Recommended range: 1 to 80. (Default = 45)
        location_x_offset (:obj:`float`): x component of the flashlight location offset. (Default = 0)
        location_y_offset (:obj:`float`): y component of the flashlight location offset. (Default = 0)
        location_z_offset (:obj:`float`): z component of the flashlight location offset. (Default = 0)
        angle_pitch (:obj:`float`): The pitch angle (in degrees) for flashlight direction. Range: -70 to 70. (Default = -30)
        angle_yaw (:obj:`float`): The yaw angle (in degrees) for flashlight direction. Range: -70 to 70. (Default = 0)
        color_R (:obj:`float`): Red component of the flashlight color. Range: 0.0 to 1.0. (Default = 1)
        color_G (:obj:`float`): Green component of the flashlight color. Range: 0.0 to 1.0. (Default = 1)
        color_B (:obj:`float`): Blue component of the flashlight color. Range: 0.0 to 1.0. (Default = 1)
    """

    def __init__(
        self,
        flashlight_name,
        intensity=5000,
        beam_width=45,
        location_x_offset=0,
        location_y_offset=0,
        location_z_offset=0,
        angle_pitch=-30,
        angle_yaw=0,
        color_R=1,
        color_G=1,
        color_B=1,
    ):
        Command.__init__(self)
        self.set_command_type("TurnOnFlashlight")
        self.add_string_parameters(flashlight_name)
        self.add_number_parameters(float(intensity))
        self.add_number_parameters(float(beam_width))
        self.add_number_parameters(float(location_x_offset))
        self.add_number_parameters(float(location_y_offset))
        self.add_number_parameters(float(location_z_offset))
        self.add_number_parameters(float(angle_pitch))
        self.add_number_parameters(float(angle_yaw))
        self.add_number_parameters(float(color_R))
        self.add_number_parameters(float(color_G))
        self.add_number_parameters(float(color_B))


class TurnOffFlashlightCommand(Command):
    """Turns off the vehicle's flashlight.

    Args:
        flashlight_name(:obj:`str`): The name of the flashlight to turn off. Vehicles have 4 flashlights (flashlight1-flashlight4)
    """

    def __init__(self, flashlight_name):
        Command.__init__(self)
        self.set_command_type("TurnOffFlashlight")
        self.add_string_parameters(flashlight_name)


class AirFogCommand(Command):
    """Changes the air fog density, depth, and color.

    Args:
        fogDensity (:obj:`float`): The density value for the fog. Range: 0.0 to 10.0.
        fogDepth (:obj:`float`): The distance at which the fog begins. Range: 0.0 to 10.0. (Default = 3).
        color_R (:obj:`float`): The red component of the fog's color. Range: 0.0 to 1.0. (Default = 0.45).
        color_G (:obj:`float`): The green component of the fog's color. Range: 0.0 to 1.0. (Default = 0.5).
        color_B (:obj:`float`): The blue component of the fog's color. Range: 0.0 to 1.0. (Default = 0.6).
    """

    def __init__(
        self, fogDensity, fogDepth=3.0, color_R=0.45, color_G=0.5, color_B=0.6
    ):
        Command.__init__(self)
        self.set_command_type("AirFog")
        self.add_number_parameters(float(fogDensity))
        self.add_number_parameters(float(fogDepth))
        self.add_number_parameters(float(color_R))
        self.add_number_parameters(float(color_G))
        self.add_number_parameters(float(color_B))


class WaterFogCommand(Command):
    """Changes the water fog density, depth, and color.

    Args:
        fogDensity (:obj:`float`): The density value for the fog. Range: 0.0 to 10.0.
        fogDepth (:obj:`float`): The distance at which the fog begins. Range: 0.0 to 10.0. (Default = 3).
        color_R (:obj:`float`): The red component of the fog's color. Range: 0.0 to 1.0. (Default = 0.4).
        color_G (:obj:`float`): The green component of the fog's color. Range: 0.0 to 1.0. (Default = 0.6).
        color_B (:obj:`float`): The blue component of the fog's color. Range: 0.0 to 1.0. (Default = 1.0).
    """

    def __init__(self, fogDensity, fogDepth=3.0, color_R=0.4, color_G=0.6, color_B=1.0):
        Command.__init__(self)
        self.set_command_type("WaterFog")
        self.add_number_parameters(float(fogDensity))
        self.add_number_parameters(float(fogDepth))
        self.add_number_parameters(float(color_R))
        self.add_number_parameters(float(color_G))
        self.add_number_parameters(float(color_B))


class CustomCommand(Command):
    """Send a custom command to the currently loaded world.

    Args:
        name (:obj:`str`): The name of the command, ex "OpenDoor"
        num_params (obj:`list` of :obj:`int`): List of arbitrary number parameters
        string_params (obj:`list` of :obj:`int`): List of arbitrary string parameters

    """

    def __init__(self, name, num_params=None, string_params=None):
        if num_params is None:
            num_params = []

        if string_params is None:
            string_params = []

        Command.__init__(self)
        self.set_command_type("CustomCommand")
        self.add_string_parameters(name)
        self.add_number_parameters(num_params)
        self.add_string_parameters(string_params)


######################## HOLOOCEAN CUSTOM COMMANDS ###########################


class SendAcousticMessageCommand(Command):
    """Set the number of ticks between captures of the RGB camera.

    Args:
        agent_name (:obj:`str`): name of the agent to modify
        sensor_name (:obj:`str`): name of the sensor to modify
        num (:obj:`int`): number of ticks between captures

    """

    def __init__(
        self, from_agent_name, from_sensor_name, to_agent_name, to_sensor_name
    ):
        Command.__init__(self)
        self._command_type = "SendAcousticMessage"
        self.add_string_parameters(from_agent_name)
        self.add_string_parameters(from_sensor_name)
        self.add_string_parameters(to_agent_name)
        self.add_string_parameters(to_sensor_name)


class SendOpticalMessageCommand(Command):
    """Send information through OpticalModem."""

    def __init__(
        self, from_agent_name, from_sensor_name, to_agent_name, to_sensor_name
    ):
        Command.__init__(self)
        self._command_type = "SendOpticalMessage"
        self.add_string_parameters(from_agent_name)
        self.add_string_parameters(from_sensor_name)
        self.add_string_parameters(to_agent_name)
        self.add_string_parameters(to_sensor_name)
