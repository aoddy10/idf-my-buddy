"""Adapters package for external API integrations."""

from .maps import GoogleMapsAdapter, OpenStreetMapAdapter
from .weather import OpenWeatherMapAdapter, WeatherAPIAdapter
from .places import GooglePlacesAdapter, FoursquareAdapter

__all__ = [
    "GoogleMapsAdapter",
    "OpenStreetMapAdapter", 
    "OpenWeatherMapAdapter",
    "WeatherAPIAdapter",
    "GooglePlacesAdapter",
    "FoursquareAdapter"
]
