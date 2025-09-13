"""Maps adapter for routing and navigation services.

This module provides adapters for various mapping services including Google Maps,
OpenStreetMap, and other navigation providers.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import json

import aiohttp
from app.core.logging import LoggerMixin
from app.core.config import settings


class BaseMapsAdapter(ABC, LoggerMixin):
    """Abstract base class for maps adapters."""
    
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
    async def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Get route between points."""
        pass
    
    @abstractmethod
    async def geocode(self, address: str) -> Dict[str, Any]:
        """Convert address to coordinates."""
        pass
    
    @abstractmethod
    async def reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        """Convert coordinates to address."""
        pass
    
    @abstractmethod
    async def get_distance_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Get distance matrix between multiple points."""
        pass


class GoogleMapsAdapter(BaseMapsAdapter):
    """Google Maps API adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or settings.GOOGLE_MAPS_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api"
        
        if not self.api_key:
            self.logger.warning("Google Maps API key not configured")
    
    async def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Get route using Google Directions API."""
        
        if not self.api_key:
            raise ValueError("Google Maps API key not configured")
        
        try:
            # Format coordinates
            origin_str = f"{origin[0]},{origin[1]}"
            destination_str = f"{destination[0]},{destination[1]}"
            
            # Build request parameters
            params = {
                "origin": origin_str,
                "destination": destination_str,
                "mode": travel_mode.lower(),
                "key": self.api_key,
                "alternatives": "true",
                "language": "en"
            }
            
            # Add waypoints if provided
            if waypoints:
                waypoints_str = "|".join([f"{wp[0]},{wp[1]}" for wp in waypoints])
                params["waypoints"] = waypoints_str
            
            # Make API request
            url = f"{self.base_url}/directions/json"
            
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") != "OK":
                    raise Exception(f"Google Maps API error: {data.get('status')}")
                
                # Process and format response
                routes = []
                for route in data.get("routes", []):
                    processed_route = self._process_google_route(route)
                    routes.append(processed_route)
                
                return {
                    "success": True,
                    "routes": routes,
                    "provider": "google_maps"
                }
                
        except Exception as e:
            self.logger.error(f"Google Maps routing failed: {e}")
            raise
    
    def _process_google_route(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Process Google Maps route data."""
        
        # Extract basic route info
        leg = route["legs"][0]  # Assuming single leg for now
        
        # Decode polyline
        polyline_points = self._decode_polyline(route["overview_polyline"]["points"])
        
        return {
            "duration": leg["duration"]["value"],  # seconds
            "duration_text": leg["duration"]["text"],
            "distance": leg["distance"]["value"],  # meters
            "distance_text": leg["distance"]["text"],
            "start_address": leg["start_address"],
            "end_address": leg["end_address"],
            "steps": [
                {
                    "instruction": step["html_instructions"],
                    "distance": step["distance"]["value"],
                    "duration": step["duration"]["value"],
                    "start_location": step["start_location"],
                    "end_location": step["end_location"]
                }
                for step in leg["steps"]
            ],
            "polyline": polyline_points,
            "bounds": route["bounds"]
        }
    
    def _decode_polyline(self, polyline_str: str) -> List[Tuple[float, float]]:
        """Decode Google Maps polyline string."""
        # Simplified polyline decoding - in production use proper library
        # This is a basic implementation
        points = []
        index = lat = lng = 0
        
        while index < len(polyline_str):
            b = shift = result = 0
            
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            
            dlat = ~(result >> 1) if result & 1 else (result >> 1)
            lat += dlat
            
            shift = result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            
            dlng = ~(result >> 1) if result & 1 else (result >> 1)
            lng += dlng
            
            points.append((lat / 1e5, lng / 1e5))
        
        return points
    
    async def geocode(self, address: str) -> Dict[str, Any]:
        """Geocode address using Google Geocoding API."""
        
        if not self.api_key:
            raise ValueError("Google Maps API key not configured")
        
        try:
            params = {
                "address": address,
                "key": self.api_key,
                "language": "en"
            }
            
            url = f"{self.base_url}/geocode/json"
            
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") != "OK":
                    raise Exception(f"Geocoding error: {data.get('status')}")
                
                results = []
                for result in data.get("results", []):
                    location = result["geometry"]["location"]
                    results.append({
                        "formatted_address": result["formatted_address"],
                        "latitude": location["lat"],
                        "longitude": location["lng"],
                        "place_id": result["place_id"],
                        "types": result["types"]
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "provider": "google_maps"
                }
                
        except Exception as e:
            self.logger.error(f"Google geocoding failed: {e}")
            raise
    
    async def reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        """Reverse geocode coordinates using Google Geocoding API."""
        
        if not self.api_key:
            raise ValueError("Google Maps API key not configured")
        
        try:
            params = {
                "latlng": f"{lat},{lng}",
                "key": self.api_key,
                "language": "en"
            }
            
            url = f"{self.base_url}/geocode/json"
            
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") != "OK":
                    raise Exception(f"Reverse geocoding error: {data.get('status')}")
                
                results = []
                for result in data.get("results", []):
                    results.append({
                        "formatted_address": result["formatted_address"],
                        "place_id": result["place_id"],
                        "types": result["types"],
                        "address_components": result["address_components"]
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "provider": "google_maps"
                }
                
        except Exception as e:
            self.logger.error(f"Google reverse geocoding failed: {e}")
            raise
    
    async def get_distance_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Get distance matrix using Google Distance Matrix API."""
        
        if not self.api_key:
            raise ValueError("Google Maps API key not configured")
        
        try:
            # Format coordinates
            origins_str = "|".join([f"{o[0]},{o[1]}" for o in origins])
            destinations_str = "|".join([f"{d[0]},{d[1]}" for d in destinations])
            
            params = {
                "origins": origins_str,
                "destinations": destinations_str,
                "mode": travel_mode.lower(),
                "key": self.api_key,
                "language": "en"
            }
            
            url = f"{self.base_url}/distancematrix/json"
            
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") != "OK":
                    raise Exception(f"Distance matrix error: {data.get('status')}")
                
                # Process matrix results
                matrix = []
                for i, row in enumerate(data["rows"]):
                    matrix_row = []
                    for j, element in enumerate(row["elements"]):
                        if element["status"] == "OK":
                            matrix_row.append({
                                "distance": element["distance"]["value"],
                                "distance_text": element["distance"]["text"],
                                "duration": element["duration"]["value"],
                                "duration_text": element["duration"]["text"],
                                "status": "OK"
                            })
                        else:
                            matrix_row.append({
                                "status": element["status"],
                                "distance": None,
                                "duration": None
                            })
                    matrix.append(matrix_row)
                
                return {
                    "success": True,
                    "matrix": matrix,
                    "origins": data["origin_addresses"],
                    "destinations": data["destination_addresses"],
                    "provider": "google_maps"
                }
                
        except Exception as e:
            self.logger.error(f"Google distance matrix failed: {e}")
            raise


class OpenStreetMapAdapter(BaseMapsAdapter):
    """OpenStreetMap adapter using OSRM and Nominatim."""
    
    def __init__(self):
        super().__init__()
        self.osrm_base_url = "https://router.project-osrm.org"
        self.nominatim_base_url = "https://nominatim.openstreetmap.org"
    
    async def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Get route using OSRM."""
        
        try:
            # Build coordinates list
            coordinates = [origin]
            if waypoints:
                coordinates.extend(waypoints)
            coordinates.append(destination)
            
            # Format coordinates for OSRM (lng,lat format)
            coords_str = ";".join([f"{coord[1]},{coord[0]}" for coord in coordinates])
            
            # OSRM profile mapping
            profile_map = {
                "driving": "driving",
                "walking": "foot",
                "cycling": "cycling",
                "transit": "driving"  # Fallback
            }
            
            profile = profile_map.get(travel_mode.lower(), "driving")
            
            # Make OSRM request
            url = f"{self.osrm_base_url}/route/v1/{profile}/{coords_str}"
            params = {
                "overview": "full",
                "geometries": "geojson",
                "steps": "true"
            }
            
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("code") != "Ok":
                    raise Exception(f"OSRM error: {data.get('message', 'Unknown error')}")
                
                # Process routes
                routes = []
                for route in data.get("routes", []):
                    processed_route = self._process_osrm_route(route)
                    routes.append(processed_route)
                
                return {
                    "success": True,
                    "routes": routes,
                    "provider": "openstreetmap"
                }
                
        except Exception as e:
            self.logger.error(f"OpenStreetMap routing failed: {e}")
            raise
    
    def _process_osrm_route(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Process OSRM route data."""
        
        # Extract polyline coordinates
        polyline_points = []
        if "geometry" in route:
            polyline_points = route["geometry"]["coordinates"]
            # Convert from [lng, lat] to [lat, lng]
            polyline_points = [(coord[1], coord[0]) for coord in polyline_points]
        
        # Process steps
        steps = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                steps.append({
                    "instruction": step.get("maneuver", {}).get("instruction", "Continue"),
                    "distance": step.get("distance", 0),
                    "duration": step.get("duration", 0),
                    "start_location": {
                        "lat": step.get("maneuver", {}).get("location", [0, 0])[1],
                        "lng": step.get("maneuver", {}).get("location", [0, 0])[0]
                    }
                })
        
        return {
            "duration": route.get("duration", 0),  # seconds
            "duration_text": f"{int(route.get('duration', 0) / 60)} min",
            "distance": route.get("distance", 0),  # meters
            "distance_text": f"{route.get('distance', 0) / 1000:.1f} km",
            "steps": steps,
            "polyline": polyline_points
        }
    
    async def geocode(self, address: str) -> Dict[str, Any]:
        """Geocode using Nominatim."""
        
        try:
            params = {
                "q": address,
                "format": "json",
                "limit": 5,
                "addressdetails": 1
            }
            
            headers = {
                "User-Agent": "IDF-My-Buddy/1.0"  # Required by Nominatim
            }
            
            url = f"{self.nominatim_base_url}/search"
            
            async with self._session.get(url, params=params, headers=headers) as response:
                data = await response.json()
                
                results = []
                for result in data:
                    results.append({
                        "formatted_address": result.get("display_name", ""),
                        "latitude": float(result.get("lat", 0)),
                        "longitude": float(result.get("lon", 0)),
                        "place_id": result.get("place_id", ""),
                        "importance": result.get("importance", 0),
                        "address": result.get("address", {})
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "provider": "openstreetmap"
                }
                
        except Exception as e:
            self.logger.error(f"OpenStreetMap geocoding failed: {e}")
            raise
    
    async def reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        """Reverse geocode using Nominatim."""
        
        try:
            params = {
                "lat": lat,
                "lon": lng,
                "format": "json",
                "addressdetails": 1
            }
            
            headers = {
                "User-Agent": "IDF-My-Buddy/1.0"
            }
            
            url = f"{self.nominatim_base_url}/reverse"
            
            async with self._session.get(url, params=params, headers=headers) as response:
                data = await response.json()
                
                results = [{
                    "formatted_address": data.get("display_name", ""),
                    "address_components": data.get("address", {}),
                    "place_id": data.get("place_id", ""),
                    "importance": data.get("importance", 0)
                }]
                
                return {
                    "success": True,
                    "results": results,
                    "provider": "openstreetmap"
                }
                
        except Exception as e:
            self.logger.error(f"OpenStreetMap reverse geocoding failed: {e}")
            raise
    
    async def get_distance_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """Calculate distance matrix using multiple route requests."""
        
        # OSRM doesn't have a built-in distance matrix API
        # We'll make multiple route requests
        
        try:
            matrix = []
            
            for origin in origins:
                matrix_row = []
                for destination in destinations:
                    try:
                        route_result = await self.get_route(origin, destination, travel_mode=travel_mode)
                        
                        if route_result["success"] and route_result["routes"]:
                            route = route_result["routes"][0]
                            matrix_row.append({
                                "distance": route["distance"],
                                "distance_text": route["distance_text"],
                                "duration": route["duration"],
                                "duration_text": route["duration_text"],
                                "status": "OK"
                            })
                        else:
                            matrix_row.append({
                                "status": "NO_ROUTE",
                                "distance": None,
                                "duration": None
                            })
                    except Exception:
                        matrix_row.append({
                            "status": "ERROR",
                            "distance": None,
                            "duration": None
                        })
                
                matrix.append(matrix_row)
            
            return {
                "success": True,
                "matrix": matrix,
                "provider": "openstreetmap"
            }
            
        except Exception as e:
            self.logger.error(f"OpenStreetMap distance matrix failed: {e}")
            raise
