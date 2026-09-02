from holoocean.command import ChangeTimeOfDayCommand


class TimeController:
    def __init__(self, client, frequency=0, hour=12, active=False):
        self.frequency = frequency
        self.hour = hour
        self.active = active
        self._client = client
        self._start_time = None
        self._tick = None

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_hour(self, hour):
        self.hour = hour

    def set_active(self, active):
        self.active = active
        if active and self._tick is not None:
            self._start_time = self._tick()

    def get_ticks(self, tick_source):
        self._tick = tick_source
        if self.active and self._start_time is None:
            self._start_time = self._tick()

    def get_time_of_day(self):
        if self._tick is None or self._start_time is None:
            return 0
        sim_age = self._tick() - self._start_time
        if self.frequency != 0:
            cycle_position = sim_age % self.frequency
            day_fraction = cycle_position / self.frequency
            return float(day_fraction * 24)
        return 0

    def update_time(self):
        if self.active:
            if self.frequency != 0:
                hour = self.get_time_of_day()
            else:
                hour = self.hour

            command_to_send = ChangeTimeOfDayCommand(hour)
            self._client.command_center.enqueue_command(command_to_send)
