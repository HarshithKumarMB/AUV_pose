from holoocean.command import TideCommand
import numpy
import random


class TideController:
    def __init__(
        self, client, amplitude=100.0, frequency=2000, tide_type="TideNormal", active=0
    ):
        self.amplitude = amplitude  # in cm
        self.frequency = frequency  # in ticks
        self.tide_type = tide_type  # might use later
        self.active = active
        self._client = client
        self._start_time = 0
        self._tick = None

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_active(self, active):
        self.active = active

    def set_tide_type(self, tide_type):
        self.tide_type = tide_type

    # lambda function that gets the tick data from env
    def get_ticks(self, tick_source):
        self._tick = tick_source

    def get_height(self):
        sim_age = self._tick() - self._start_time
        if self.tide_type == "TideNormal":
            return self.amplitude * numpy.sin(
                ((2 * numpy.pi) / self.frequency) * sim_age
            )
        if self.tide_type == "TideFreaky":
            return random.randint(-self.amplitude * 100, self.amplitude * 100)
        return 0

    # if self.a
    def update_tide(self):
        if self.active:
            height = self.get_height()
            command_to_send = TideCommand(height, 1)
            self._client.command_center.enqueue_command(command_to_send)
