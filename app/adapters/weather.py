"""Weather adapter for meteorological data services.

This module provides adapters for various weather services including OpenWeatherMap,
WeatherAPI, and other weather data providers.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

import aiohttp
from app.core.logging import LoggerMixin
from app.core.config import settings


class BaseWeatherAdapter(ABC, LoggerMixin):
    """Abstract base class for weather adapters."""
    
    def __init__(self):
        super().__init__()
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    @abstractmethod
    async def get_current_weather(self, lat: float, lng: float, units: str = "metric") -> Dict[str, Any]:
        """Get current weather conditions."""
        pass
    
    @abstractmethod
    async def get_forecast(
        self,
        lat: float,
        lng: float,
        days: int = 5,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get weather forecast."""
        pass
    
    @abstractmethod
    async def get_hourly_forecast(
        self,
        lat: float,
        lng: float,
        hours: int = 24,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get hourly weather forecast."""
        pass


class OpenWeatherMapAdapter(BaseWeatherAdapter):
    """OpenWeatherMap API adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.OPENWEATHERMAP_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            self.logger.warning("OpenWeatherMap API key not configured")
    
    async def get_current_weather(self, lat: float, lng: float, units: str = "metric") -> Dict[str, Any]:
        """Get current weather from OpenWeatherMap."""
        
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")
        
        try:
            params = {
                "lat": lat,
                "lon": lng,
                "units": units,
                "appid": self.api_key
            }
            
            url = f"{self.base_url}/weather"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"OpenWeatherMap API error: {response.status}")
                
                data = await response.json()
                
                return self._format_current_weather(data, units)
                
        except Exception as e:
            self.logger.error(f"OpenWeatherMap current weather failed: {e}")
            raise
    
    def _format_current_weather(self, data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Format OpenWeatherMap current weather response."""
        
        # Temperature unit
        temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        
        # Wind speed unit
        wind_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"
        
        return {
            "success": True,
            "provider": "openweathermap",
            "location": {
                "name": data.get("name", ""),
                "country": data.get("sys", {}).get("country", ""),
                "latitude": data.get("coord", {}).get("lat", lat),
                "longitude": data.get("coord", {}).get("lon", lng)
            },
            "current": {
                "temperature": data["main"]["temp"],
                "temperature_unit": temp_unit,
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                "uv_index": None,  # Not available in current weather
                "weather": {
                    "main": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                    "icon": data["weather"][0]["icon"]
                },
                "wind": {
                    "speed": data.get("wind", {}).get("speed", 0),
                    "speed_unit": wind_unit,
                    "direction": data.get("wind", {}).get("deg", 0),
                    "gust": data.get("wind", {}).get("gust", 0)
                },
                "clouds": data.get("clouds", {}).get("all", 0),
                "precipitation": {
                    "rain_1h": data.get("rain", {}).get("1h", 0),
                    "snow_1h": data.get("snow", {}).get("1h", 0)
                }
            },
            "timestamp": data["dt"],
            "timezone": data["timezone"]
        }
    
    async def get_forecast(
        self,
        lat: float,
        lng: float,
        days: int = 5,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get weather forecast from OpenWeatherMap."""
        
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")
        
        try:
            params = {
                "lat": lat,
                "lon": lng,
                "units": units,
                "appid": self.api_key,
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            url = f"{self.base_url}/forecast"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"OpenWeatherMap API error: {response.status}")
                
                data = await response.json()
                
                return self._format_forecast(data, units, days)
                
        except Exception as e:
            self.logger.error(f"OpenWeatherMap forecast failed: {e}")
            raise
    
    def _format_forecast(self, data: Dict[str, Any], units: str, days: int) -> Dict[str, Any]:
        """Format OpenWeatherMap forecast response."""
        
        temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        wind_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"
        
        # Group forecasts by day
        daily_forecasts = {}
        hourly_forecasts = []
        
        for item in data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            date_key = dt.strftime("%Y-%m-%d")
            
            forecast_item = {
                "datetime": item["dt"],
                "temperature": item["main"]["temp"],
                "temperature_unit": temp_unit,
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "weather": {
                    "main": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"],
                    "icon": item["weather"][0]["icon"]
                },
                "wind": {
                    "speed": item.get("wind", {}).get("speed", 0),
                    "speed_unit": wind_unit,
                    "direction": item.get("wind", {}).get("deg", 0)
                },
                "clouds": item.get("clouds", {}).get("all", 0),
                "precipitation": {
                    "rain_3h": item.get("rain", {}).get("3h", 0),
                    "snow_3h": item.get("snow", {}).get("3h", 0)
                },
                "visibility": item.get("visibility", 10000) / 1000
            }
            
            # Add to hourly forecasts
            hourly_forecasts.append(forecast_item)
            
            # Group by day for daily summary
            if date_key not in daily_forecasts:
                daily_forecasts[date_key] = {
                    "date": date_key,
                    "temperatures": [],
                    "conditions": [],
                    "forecasts": []
                }
            
            daily_forecasts[date_key]["temperatures"].append(item["main"]["temp"])
            daily_forecasts[date_key]["conditions"].append(item["weather"][0]["main"])
            daily_forecasts[date_key]["forecasts"].append(forecast_item)
        
        # Create daily summaries
        daily_summaries = []
        for date_key, day_data in list(daily_forecasts.items())[:days]:
            temps = day_data["temperatures"]
            most_common_condition = max(set(day_data["conditions"]), key=day_data["conditions"].count)
            
            daily_summaries.append({
                "date": date_key,
                "temperature_min": min(temps),
                "temperature_max": max(temps),
                "temperature_unit": temp_unit,
                "condition": most_common_condition,
                "forecasts": day_data["forecasts"]
            })
        
        return {
            "success": True,
            "provider": "openweathermap",
            "location": {
                "name": data["city"]["name"],
                "country": data["city"]["country"],
                "latitude": data["city"]["coord"]["lat"],
                "longitude": data["city"]["coord"]["lon"]
            },
            "daily_forecasts": daily_summaries,
            "hourly_forecasts": hourly_forecasts[:days * 8]
        }
    
    async def get_hourly_forecast(
        self,
        lat: float,
        lng: float,
        hours: int = 24,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get hourly weather forecast."""
        
        # Use the regular forecast API and filter
        forecast_result = await self.get_forecast(lat, lng, days=5, units=units)
        
        if forecast_result["success"]:
            # Take only the requested number of hours
            hourly_forecasts = forecast_result["hourly_forecasts"][:hours]
            
            return {
                "success": True,
                "provider": "openweathermap",
                "location": forecast_result["location"],
                "hourly_forecasts": hourly_forecasts
            }
        
        return forecast_result


class WeatherAPIAdapter(BaseWeatherAdapter):
    """WeatherAPI.com adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.WEATHERAPI_KEY
        self.base_url = "https://api.weatherapi.com/v1"
        
        if not self.api_key:
            self.logger.warning("WeatherAPI key not configured")
    
    async def get_current_weather(self, lat: float, lng: float, units: str = "metric") -> Dict[str, Any]:
        """Get current weather from WeatherAPI."""
        
        if not self.api_key:
            raise ValueError("WeatherAPI key not configured")
        
        try:
            params = {
                "key": self.api_key,
                "q": f"{lat},{lng}",
                "aqi": "yes"  # Include air quality data
            }
            
            url = f"{self.base_url}/current.json"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"WeatherAPI error: {response.status}")
                
                data = await response.json()
                
                return self._format_weatherapi_current(data, units)
                
        except Exception as e:
            self.logger.error(f"WeatherAPI current weather failed: {e}")
            raise
    
    def _format_weatherapi_current(self, data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Format WeatherAPI current weather response."""
        
        current = data["current"]
        location = data["location"]
        
        # Convert temperature if needed
        if units == "imperial":
            temp = current["temp_f"]
            feels_like = current["feelslike_f"]
            temp_unit = "°F"
            wind_speed = current["wind_mph"]
            wind_unit = "mph"
        else:
            temp = current["temp_c"]
            feels_like = current["feelslike_c"]
            temp_unit = "°C"
            wind_speed = current["wind_kph"] / 3.6  # Convert to m/s
            wind_unit = "m/s"
        
        return {
            "success": True,
            "provider": "weatherapi",
            "location": {
                "name": location["name"],
                "country": location["country"],
                "latitude": location["lat"],
                "longitude": location["lon"]
            },
            "current": {
                "temperature": temp,
                "temperature_unit": temp_unit,
                "feels_like": feels_like,
                "humidity": current["humidity"],
                "pressure": current["pressure_mb"],
                "visibility": current["vis_km"],
                "uv_index": current["uv"],
                "weather": {
                    "main": current["condition"]["text"],
                    "description": current["condition"]["text"],
                    "icon": current["condition"]["icon"]
                },
                "wind": {
                    "speed": wind_speed,
                    "speed_unit": wind_unit,
                    "direction": current["wind_degree"],
                    "gust": current.get("gust_kph", 0) / 3.6 if units == "metric" else current.get("gust_mph", 0)
                },
                "clouds": current["cloud"],
                "precipitation": {
                    "rain_1h": current.get("precip_mm", 0),
                    "snow_1h": 0  # WeatherAPI doesn't separate rain/snow
                }
            },
            "air_quality": data.get("current", {}).get("air_quality", {}),
            "timestamp": current["last_updated_epoch"],
            "timezone": location["tz_id"]
        }
    
    async def get_forecast(
        self,
        lat: float,
        lng: float,
        days: int = 5,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get weather forecast from WeatherAPI."""
        
        if not self.api_key:
            raise ValueError("WeatherAPI key not configured")
        
        try:
            params = {
                "key": self.api_key,
                "q": f"{lat},{lng}",
                "days": min(days, 10),  # WeatherAPI max is 10 days
                "aqi": "yes",
                "alerts": "yes"
            }
            
            url = f"{self.base_url}/forecast.json"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"WeatherAPI error: {response.status}")
                
                data = await response.json()
                
                return self._format_weatherapi_forecast(data, units)
                
        except Exception as e:
            self.logger.error(f"WeatherAPI forecast failed: {e}")
            raise
    
    def _format_weatherapi_forecast(self, data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Format WeatherAPI forecast response."""
        
        location = data["location"]
        
        daily_forecasts = []
        hourly_forecasts = []
        
        for day in data["forecast"]["forecastday"]:
            day_data = day["day"]
            
            # Temperature and wind units
            if units == "imperial":
                temp_min = day_data["mintemp_f"]
                temp_max = day_data["maxtemp_f"]
                temp_unit = "°F"
                wind_speed = day_data["maxwind_mph"]
                wind_unit = "mph"
            else:
                temp_min = day_data["mintemp_c"]
                temp_max = day_data["maxtemp_c"]
                temp_unit = "°C"
                wind_speed = day_data["maxwind_kph"] / 3.6
                wind_unit = "m/s"
            
            # Daily summary
            daily_forecasts.append({
                "date": day["date"],
                "temperature_min": temp_min,
                "temperature_max": temp_max,
                "temperature_unit": temp_unit,
                "condition": day_data["condition"]["text"],
                "icon": day_data["condition"]["icon"],
                "chance_of_rain": day_data["daily_chance_of_rain"],
                "chance_of_snow": day_data["daily_chance_of_snow"],
                "total_precipitation": day_data["totalprecip_mm"],
                "max_wind": {
                    "speed": wind_speed,
                    "speed_unit": wind_unit
                },
                "avg_humidity": day_data["avghumidity"],
                "uv_index": day_data["uv"]
            })
            
            # Hourly data
            for hour in day["hour"]:
                if units == "imperial":
                    temp = hour["temp_f"]
                    feels_like = hour["feelslike_f"]
                    temp_unit = "°F"
                    wind_speed = hour["wind_mph"]
                    wind_unit = "mph"
                else:
                    temp = hour["temp_c"]
                    feels_like = hour["feelslike_c"]
                    temp_unit = "°C"
                    wind_speed = hour["wind_kph"] / 3.6
                    wind_unit = "m/s"
                
                hourly_forecasts.append({
                    "datetime": hour["time_epoch"],
                    "temperature": temp,
                    "temperature_unit": temp_unit,
                    "feels_like": feels_like,
                    "humidity": hour["humidity"],
                    "pressure": hour["pressure_mb"],
                    "weather": {
                        "main": hour["condition"]["text"],
                        "description": hour["condition"]["text"],
                        "icon": hour["condition"]["icon"]
                    },
                    "wind": {
                        "speed": wind_speed,
                        "speed_unit": wind_unit,
                        "direction": hour["wind_degree"],
                        "gust": hour.get("gust_kph", 0) / 3.6 if units == "metric" else hour.get("gust_mph", 0)
                    },
                    "clouds": hour["cloud"],
                    "precipitation": {
                        "rain_1h": hour.get("precip_mm", 0),
                        "snow_1h": 0
                    },
                    "visibility": hour["vis_km"],
                    "uv_index": hour["uv"],
                    "chance_of_rain": hour["chance_of_rain"],
                    "chance_of_snow": hour["chance_of_snow"]
                })
        
        return {
            "success": True,
            "provider": "weatherapi",
            "location": {
                "name": location["name"],
                "country": location["country"],
                "latitude": location["lat"],
                "longitude": location["lon"]
            },
            "daily_forecasts": daily_forecasts,
            "hourly_forecasts": hourly_forecasts,
            "alerts": data.get("alerts", {}).get("alert", [])
        }
    
    async def get_hourly_forecast(
        self,
        lat: float,
        lng: float,
        hours: int = 24,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get hourly weather forecast."""
        
        # WeatherAPI provides hourly data in the forecast endpoint
        days_needed = (hours // 24) + 1
        forecast_result = await self.get_forecast(lat, lng, days=days_needed, units=units)
        
        if forecast_result["success"]:
            # Take only the requested number of hours
            hourly_forecasts = forecast_result["hourly_forecasts"][:hours]
            
            return {
                "success": True,
                "provider": "weatherapi",
                "location": forecast_result["location"],
                "hourly_forecasts": hourly_forecasts,
                "alerts": forecast_result.get("alerts", [])
            }
        
        return forecast_result
