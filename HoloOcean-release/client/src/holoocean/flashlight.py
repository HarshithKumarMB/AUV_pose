from holoocean.command import TurnOnFlashlightCommand


class FlashlightController:
    """
    A controller to set up a flashlight in a HoloOcean scenario.

    Flashlight parameters:
        - Light intensity
        - Beam width
        - Light position (x, y, z)
        - Light angle (Yaw, Pitch)
        - Light color (R,G,B)
    """

    def __init__(
        self,
        client,
        flashlight_name,
        intensity,
        beam_width,
        location_x_offset,
        location_y_offset,
        location_z_offset,
        angle_pitch,
        angle_yaw,
        color_R,
        color_G,
        color_B,
    ):
        self._client = client
        self.flashlight_name = flashlight_name
        self.intensity = intensity
        self.beam_width = beam_width
        self.location_x_offset = location_x_offset
        self.location_y_offset = location_y_offset
        self.location_z_offset = location_z_offset
        self.angle_pitch = angle_pitch
        self.angle_yaw = angle_yaw
        self.color_R = color_R
        self.color_G = color_G
        self.color_B = color_B

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_beam_width(self, beam_width):
        self.beam_width = beam_width

    def set_location_x_offset(self, location_x_offset):
        self.location_x_offset = location_x_offset

    def set_location_y_offset(self, location_y_offset):
        self.location_y_offset = location_y_offset

    def set_location_z_offset(self, location_z_offset):
        self.location_z_offset = location_z_offset

    def set_angle_pitch(self, angle_pitch):
        self.angle_pitch = angle_pitch

    def set_angle_yaw(self, angle_yaw):
        self.angle_yaw = angle_yaw

    def set_color_R(self, color_R):
        self.color_R = color_R

    def set_color_G(self, color_G):
        self.color_G = color_G

    def set_color_B(self, color_B):
        self.color_B = color_B

    def set_flashlight(self):
        self._client.command_center.enqueue_command(
            TurnOnFlashlightCommand(
                self.flashlight_name,
                self.intensity,
                self.beam_width,
                self.location_x_offset,
                self.location_y_offset,
                self.location_z_offset,
                self.angle_pitch,
                self.angle_yaw,
                self.color_R,
                self.color_G,
                self.color_B,
            )
        )
