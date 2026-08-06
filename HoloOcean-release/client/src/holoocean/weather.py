from holoocean.command import ChangeWeatherCommand
from holoocean.exceptions import HoloOceanException


class WeatherController:
    """
    A controller to manage weather in a HoloOcean scenario.

    Weather types:
        0 - Sunny
        1 - Cloudy
        2 - Rainy
    """

    def __init__(self, client):
        self._client = client
        self.cur_weather = None

    def get_weather(self):
        """Returns the current weather setting (0, 1, or 2)."""
        return self.cur_weather

    def set_weather(self, weather_type: int):
        """
        Sets the weather in the HoloOcean scenario.

        Args:
            weather_type (int): The type of weather (0 = sunny, 1 = cloudy, 2 = rainy).
        """

        if weather_type not in [0, 1, 2]:
            raise HoloOceanException("Invalid weather type")

        if self.cur_weather == weather_type:
            # Avoid sending redundant command
            return

        self.cur_weather = weather_type
        self._client.command_center.enqueue_command(
            ChangeWeatherCommand(self.cur_weather)
        )
