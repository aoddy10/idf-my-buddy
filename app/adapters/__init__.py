"""Adapters package for external API integrations."""

from .maps import GoogleMapsAdapter, OpenStreetMapAdapter
from .places import FoursquareAdapter, GooglePlacesAdapter
from .weather import OpenWeatherMapAdapter, WeatherAPIAdapter

__all__ = [
    "GoogleMapsAdapter",
    "OpenStreetMapAdapter",
    "OpenWeatherMapAdapter",
    "WeatherAPIAdapter",
    "GooglePlacesAdapter",
    "FoursquareAdapter"
]
