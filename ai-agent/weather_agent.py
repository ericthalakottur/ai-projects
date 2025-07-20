import os
import json
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

weather_agent = Agent('google-gla:gemini-2.0-flash')


@dataclass
class TomorrowWeatherAPI:
    TOMORROW_API_KEY: Optional[str] = os.getenv('TOMORROW_API_KEY')

    def get_city_temperature(self, city: str) -> str:
        response = requests.get("https://api.tomorrow.io/v4/weather/realtime", {
            "location": city,
            "apikey": self.TOMORROW_API_KEY,
        }, headers={
            'Accept': 'application/json',
            'Accept-Encoding': 'deflate, gzip, br'
        })
        response_json = response.json()
        weather_info = {
            "city": response_json["location"]["name"],
            "temperature": response_json["data"]["values"]["temperature"]
        }
        return json.dumps(weather_info)


class WeatherInfo(BaseModel):
    city: str = Field(description='City name')
    temperature: float = Field(description='Current temperature in city')


@weather_agent.tool
async def get_weather_info(ctx: RunContext[TomorrowWeatherAPI], city: str) -> WeatherInfo:
    """Returns the current temperature in a given city

        Args:
                ctx: context
                city: City name to get weather data

        Returns:
                str: {WeatherInfo}
    """
    return WeatherInfo.model_validate_json(ctx.deps.get_city_temperature(city))


def main():
    while True:
        message = input("Message > ")
        if message == "q":
            break

        result = weather_agent.run_sync(message, deps=TomorrowWeatherAPI())

        print(result.output)


if __name__ == '__main__':
    main()
