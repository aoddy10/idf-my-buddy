"""Places adapter for POI and location search services.

This module provides adapters for various places and points of interest services
including Google Places, Foursquare, and other location-based APIs.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

import aiohttp
from app.core.logging import LoggerMixin
from app.core.config import settings


class BasePlacesAdapter(ABC, LoggerMixin):
    """Abstract base class for places adapters."""
    
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
    async def search_places(
        self,
        query: str,
        lat: float,
        lng: float,
        radius: int = 5000,
        place_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for places around a location."""
        pass
    
    @abstractmethod
    async def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific place."""
        pass
    
    @abstractmethod
    async def search_nearby(
        self,
        lat: float,
        lng: float,
        place_type: str,
        radius: int = 5000
    ) -> Dict[str, Any]:
        """Search for nearby places of a specific type."""
        pass


class GooglePlacesAdapter(BasePlacesAdapter):
    """Google Places API adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.GOOGLE_PLACES_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
        if not self.api_key:
            self.logger.warning("Google Places API key not configured")
    
    async def search_places(
        self,
        query: str,
        lat: float,
        lng: float,
        radius: int = 5000,
        place_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for places using Google Places Text Search."""
        
        if not self.api_key:
            raise ValueError("Google Places API key not configured")
        
        try:
            params = {
                "query": query,
                "location": f"{lat},{lng}",
                "radius": radius,
                "key": self.api_key,
                "language": "en"
            }
            
            if place_type:
                params["type"] = self._map_place_type_google(place_type)
            
            url = f"{self.base_url}/textsearch/json"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google Places API error: {response.status}")
                
                data = await response.json()
                
                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    raise Exception(f"Google Places API error: {data.get('status')}")
                
                return self._format_google_places_results(data, "text_search")
                
        except Exception as e:
            self.logger.error(f"Google Places search failed: {e}")
            raise
    
    async def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """Get place details using Google Places Details API."""
        
        if not self.api_key:
            raise ValueError("Google Places API key not configured")
        
        try:
            params = {
                "place_id": place_id,
                "fields": "place_id,name,formatted_address,geometry,rating,user_ratings_total," +
                         "price_level,opening_hours,phone_number,website,photos,reviews,types," +
                         "formatted_phone_number,international_phone_number,url",
                "key": self.api_key,
                "language": "en"
            }
            
            url = f"{self.base_url}/details/json"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google Places API error: {response.status}")
                
                data = await response.json()
                
                if data.get("status") != "OK":
                    raise Exception(f"Google Places API error: {data.get('status')}")
                
                return self._format_google_place_details(data["result"])
                
        except Exception as e:
            self.logger.error(f"Google Places details failed: {e}")
            raise
    
    async def search_nearby(
        self,
        lat: float,
        lng: float,
        place_type: str,
        radius: int = 5000
    ) -> Dict[str, Any]:
        """Search for nearby places using Google Places Nearby Search."""
        
        if not self.api_key:
            raise ValueError("Google Places API key not configured")
        
        try:
            params = {
                "location": f"{lat},{lng}",
                "radius": radius,
                "type": self._map_place_type_google(place_type),
                "key": self.api_key,
                "language": "en"
            }
            
            url = f"{self.base_url}/nearbysearch/json"
            
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google Places API error: {response.status}")
                
                data = await response.json()
                
                if data.get("status") not in ["OK", "ZERO_RESULTS"]:
                    raise Exception(f"Google Places API error: {data.get('status')}")
                
                return self._format_google_places_results(data, "nearby_search")
                
        except Exception as e:
            self.logger.error(f"Google Places nearby search failed: {e}")
            raise
    
    def _map_place_type_google(self, place_type: str) -> str:
        """Map generic place type to Google Places type."""
        
        type_mapping = {
            "restaurant": "restaurant",
            "food": "food",
            "hotel": "lodging",
            "accommodation": "lodging",
            "gas_station": "gas_station",
            "hospital": "hospital",
            "pharmacy": "pharmacy",
            "bank": "bank",
            "atm": "atm",
            "tourist_attraction": "tourist_attraction",
            "shopping_mall": "shopping_mall",
            "store": "store",
            "supermarket": "supermarket",
            "convenience_store": "convenience_store",
            "cafe": "cafe",
            "bar": "bar",
            "night_club": "night_club",
            "museum": "museum",
            "park": "park",
            "gym": "gym",
            "beauty_salon": "beauty_salon",
            "car_repair": "car_repair",
            "airport": "airport",
            "subway_station": "subway_station",
            "bus_station": "bus_station",
            "taxi_stand": "taxi_stand"
        }
        
        return type_mapping.get(place_type.lower(), place_type)
    
    def _format_google_places_results(self, data: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Format Google Places search results."""
        
        places = []
        for result in data.get("results", []):
            place = self._format_google_place(result)
            places.append(place)
        
        return {
            "success": True,
            "provider": "google_places",
            "search_type": search_type,
            "places": places,
            "next_page_token": data.get("next_page_token")
        }
    
    def _format_google_place(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single Google Places result."""
        
        location = result.get("geometry", {}).get("location", {})
        
        return {
            "id": result["place_id"],
            "name": result.get("name", ""),
            "address": result.get("formatted_address", result.get("vicinity", "")),
            "location": {
                "latitude": location.get("lat", 0),
                "longitude": location.get("lng", 0)
            },
            "rating": result.get("rating", 0),
            "user_ratings_total": result.get("user_ratings_total", 0),
            "price_level": result.get("price_level"),
            "types": result.get("types", []),
            "business_status": result.get("business_status"),
            "opening_hours": {
                "open_now": result.get("opening_hours", {}).get("open_now"),
                "weekday_text": result.get("opening_hours", {}).get("weekday_text", [])
            },
            "photos": [
                {
                    "photo_reference": photo["photo_reference"],
                    "width": photo["width"],
                    "height": photo["height"]
                }
                for photo in result.get("photos", [])
            ],
            "plus_code": result.get("plus_code", {}),
            "permanently_closed": result.get("permanently_closed", False)
        }
    
    def _format_google_place_details(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format Google Places place details."""
        
        location = result.get("geometry", {}).get("location", {})
        
        return {
            "success": True,
            "provider": "google_places",
            "place": {
                "id": result["place_id"],
                "name": result.get("name", ""),
                "address": result.get("formatted_address", ""),
                "location": {
                    "latitude": location.get("lat", 0),
                    "longitude": location.get("lng", 0)
                },
                "rating": result.get("rating", 0),
                "user_ratings_total": result.get("user_ratings_total", 0),
                "price_level": result.get("price_level"),
                "types": result.get("types", []),
                "phone_number": result.get("formatted_phone_number", ""),
                "international_phone_number": result.get("international_phone_number", ""),
                "website": result.get("website", ""),
                "url": result.get("url", ""),
                "opening_hours": {
                    "open_now": result.get("opening_hours", {}).get("open_now"),
                    "periods": result.get("opening_hours", {}).get("periods", []),
                    "weekday_text": result.get("opening_hours", {}).get("weekday_text", [])
                },
                "photos": [
                    {
                        "photo_reference": photo["photo_reference"],
                        "width": photo["width"],
                        "height": photo["height"]
                    }
                    for photo in result.get("photos", [])
                ],
                "reviews": [
                    {
                        "author_name": review["author_name"],
                        "rating": review["rating"],
                        "text": review["text"],
                        "time": review["time"],
                        "relative_time_description": review["relative_time_description"]
                    }
                    for review in result.get("reviews", [])
                ]
            }
        }


class FoursquareAdapter(BasePlacesAdapter):
    """Foursquare Places API adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.FOURSQUARE_API_KEY
        self.base_url = "https://api.foursquare.com/v3/places"
        
        if not self.api_key:
            self.logger.warning("Foursquare API key not configured")
    
    async def search_places(
        self,
        query: str,
        lat: float,
        lng: float,
        radius: int = 5000,
        place_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for places using Foursquare Places Search."""
        
        if not self.api_key:
            raise ValueError("Foursquare API key not configured")
        
        try:
            params = {
                "query": query,
                "ll": f"{lat},{lng}",
                "radius": min(radius, 100000),  # Foursquare max is 100km
                "limit": 50
            }
            
            if place_type:
                params["categories"] = self._map_place_type_foursquare(place_type)
            
            headers = {
                "Authorization": self.api_key,
                "Accept": "application/json"
            }
            
            url = f"{self.base_url}/search"
            
            async with self._session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Foursquare API error: {response.status}")
                
                data = await response.json()
                
                return self._format_foursquare_results(data, "search")
                
        except Exception as e:
            self.logger.error(f"Foursquare search failed: {e}")
            raise
    
    async def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """Get place details using Foursquare Places Details."""
        
        if not self.api_key:
            raise ValueError("Foursquare API key not configured")
        
        try:
            headers = {
                "Authorization": self.api_key,
                "Accept": "application/json"
            }
            
            url = f"{self.base_url}/{place_id}"
            
            async with self._session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Foursquare API error: {response.status}")
                
                data = await response.json()
                
                return self._format_foursquare_place_details(data)
                
        except Exception as e:
            self.logger.error(f"Foursquare place details failed: {e}")
            raise
    
    async def search_nearby(
        self,
        lat: float,
        lng: float,
        place_type: str,
        radius: int = 5000
    ) -> Dict[str, Any]:
        """Search for nearby places using Foursquare."""
        
        if not self.api_key:
            raise ValueError("Foursquare API key not configured")
        
        try:
            params = {
                "ll": f"{lat},{lng}",
                "radius": min(radius, 100000),
                "categories": self._map_place_type_foursquare(place_type),
                "limit": 50
            }
            
            headers = {
                "Authorization": self.api_key,
                "Accept": "application/json"
            }
            
            url = f"{self.base_url}/nearby"
            
            async with self._session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Foursquare API error: {response.status}")
                
                data = await response.json()
                
                return self._format_foursquare_results(data, "nearby")
                
        except Exception as e:
            self.logger.error(f"Foursquare nearby search failed: {e}")
            raise
    
    def _map_place_type_foursquare(self, place_type: str) -> str:
        """Map generic place type to Foursquare category ID."""
        
        # Foursquare uses category IDs - these are some common ones
        type_mapping = {
            "restaurant": "13065",  # Restaurant
            "food": "13000",  # Food and Dining
            "hotel": "19014",  # Hotel
            "accommodation": "19000",  # Travel and Tourism
            "gas_station": "17069",  # Gas Station
            "hospital": "15014",  # Hospital
            "pharmacy": "17102",  # Pharmacy
            "bank": "10017",  # Bank
            "atm": "10018",  # ATM
            "tourist_attraction": "16032",  # Landmark
            "shopping_mall": "17012",  # Mall
            "store": "17000",  # Retail
            "supermarket": "17069",  # Grocery Store
            "convenience_store": "17067",  # Convenience Store
            "cafe": "13034",  # Café
            "bar": "13003",  # Bar
            "night_club": "10032",  # Nightclub
            "museum": "12026",  # Museum
            "park": "16000",  # Outdoors and Recreation
            "gym": "18021",  # Gym
            "beauty_salon": "18016",  # Salon / Barbershop
            "car_repair": "17016",  # Automotive
            "airport": "19040",  # Airport
            "subway_station": "19046",  # Metro Station
            "bus_station": "19041",  # Bus Station
            "taxi_stand": "19057"  # Taxi
        }
        
        return type_mapping.get(place_type.lower(), "19000")  # Default to Travel
    
    def _format_foursquare_results(self, data: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Format Foursquare search results."""
        
        places = []
        for result in data.get("results", []):
            place = self._format_foursquare_place(result)
            places.append(place)
        
        return {
            "success": True,
            "provider": "foursquare",
            "search_type": search_type,
            "places": places
        }
    
    def _format_foursquare_place(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single Foursquare place result."""
        
        geocodes = result.get("geocodes", {}).get("main", {})
        location = result.get("location", {})
        
        return {
            "id": result["fsq_id"],
            "name": result.get("name", ""),
            "address": location.get("formatted_address", ""),
            "location": {
                "latitude": geocodes.get("latitude", 0),
                "longitude": geocodes.get("longitude", 0)
            },
            "rating": result.get("rating", 0) / 2,  # Convert 0-10 to 0-5 scale
            "categories": [
                {
                    "id": cat["id"],
                    "name": cat["name"],
                    "icon": cat.get("icon", {}).get("prefix", "") + "64" + cat.get("icon", {}).get("suffix", "")
                }
                for cat in result.get("categories", [])
            ],
            "distance": result.get("distance", 0),
            "timezone": result.get("timezone", ""),
            "chains": result.get("chains", [])
        }
    
    def _format_foursquare_place_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Foursquare place details."""
        
        result = data
        geocodes = result.get("geocodes", {}).get("main", {})
        location = result.get("location", {})
        
        return {
            "success": True,
            "provider": "foursquare",
            "place": {
                "id": result["fsq_id"],
                "name": result.get("name", ""),
                "address": location.get("formatted_address", ""),
                "location": {
                    "latitude": geocodes.get("latitude", 0),
                    "longitude": geocodes.get("longitude", 0)
                },
                "rating": result.get("rating", 0) / 2,
                "categories": [
                    {
                        "id": cat["id"],
                        "name": cat["name"],
                        "icon": cat.get("icon", {}).get("prefix", "") + "64" + cat.get("icon", {}).get("suffix", "")
                    }
                    for cat in result.get("categories", [])
                ],
                "hours": result.get("hours", {}),
                "hours_popular": result.get("hours_popular", []),
                "tel": result.get("tel", ""),
                "website": result.get("website", ""),
                "social_media": result.get("social_media", {}),
                "verified": result.get("verified", False),
                "stats": result.get("stats", {}),
                "popularity": result.get("popularity", 0),
                "price": result.get("price", 0),
                "tips": result.get("tips", []),
                "tastes": result.get("tastes", []),
                "features": result.get("features", {}),
                "store_id": result.get("store_id", ""),
                "date_closed": result.get("date_closed", ""),
                "photos": [
                    {
                        "id": photo["id"],
                        "url": photo["prefix"] + "original" + photo["suffix"]
                    }
                    for photo in result.get("photos", [])
                ]
            }
        }
