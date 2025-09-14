"""Navigation and Maps service for My Buddy application.

This module provides comprehensive location and navigation services using Google Maps API 
and OpenStreetMap with intelligent fallback mechanisms, optimized for edge computing
and privacy-first location processing.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import httpx
from geopy import distance
from geopy.geocoders import Nominatim
from shapely.geometry import Point

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.schemas.navigation import (
    TransportMode, POICategory, NavigationRequest, NavigationStep, 
    RouteInfo, PointOfInterest, Coordinates
)

try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    googlemaps = None  # type: ignore
    GOOGLEMAPS_AVAILABLE = False

try:
    import overpass
    OVERPASS_AVAILABLE = True
except ImportError:
    overpass = None  # type: ignore
    OVERPASS_AVAILABLE = False


class NavigationService(LoggerMixin):
    """Navigation and location services with Google Maps and OpenStreetMap support.
    
    Provides GPS location services, route calculation, POI discovery, and geocoding
    with intelligent fallback between Google Maps API and OpenStreetMap data.
    Optimized for privacy-first processing and edge computing capabilities.
    """

    def __init__(self):
        super().__init__()
        self._google_client = None
        self._nominatim_client = None
        self._overpass_api = None
        
        # Performance metrics
        self._route_cache = {}
        self._poi_cache = {}
        self._geocode_cache = {}
        
        # Initialize based on available services
        self._setup_service()

    def _setup_service(self):
        """Setup navigation service based on available resources and configuration."""
        try:
            # Initialize Google Maps client if available and configured
            if (GOOGLEMAPS_AVAILABLE and 
                settings.navigation.use_google_maps and 
                settings.navigation.google_maps_api_key):
                
                self._google_client = googlemaps.Client(key=settings.navigation.google_maps_api_key)  # type: ignore
                self.logger.info("Google Maps client initialized successfully")
            
            # Initialize OpenStreetMap services if available
            if settings.navigation.use_openstreetmap:
                if OVERPASS_AVAILABLE:
                    self._overpass_api = overpass.API()  # type: ignore
                    self.logger.info("Overpass API initialized for OpenStreetMap data")
                
                # Initialize Nominatim for geocoding
                self._nominatim_client = Nominatim(user_agent="my-buddy-travel-assistant")
                self.logger.info("Nominatim geocoding service initialized")
            
            if not self._google_client and not self._nominatim_client:
                self.logger.warning("No mapping services available - ensure API keys are configured")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize navigation service: {str(e)}")
            raise

    async def get_current_location_info(
        self,
        coordinates: Coordinates,
        include_address: bool = True,
        include_pois: bool = False,
        poi_categories: Optional[List[POICategory]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive information about a location.
        
        Args:
            coordinates: GPS coordinates
            include_address: Whether to include reverse geocoded address
            include_pois: Whether to include nearby POIs
            poi_categories: Specific POI categories to search for
            
        Returns:
            Dict containing location information, address, and optional POIs
        """
        try:
            start_time = time.time()
            result = {
                "coordinates": coordinates,
                "timestamp": datetime.now(timezone.utc),
            }
            
            # Reverse geocode to get address if requested
            if include_address:
                address_info = await self._reverse_geocode(coordinates)
                result.update(address_info)
            
            # Get nearby POIs if requested
            if include_pois:
                pois = await self.search_nearby_pois(
                    coordinates=coordinates,
                    categories=poi_categories or [POICategory.RESTAURANT, POICategory.GAS_STATION],
                    radius_meters=500,
                    max_results=10
                )
                result["nearby_pois"] = pois
            
            processing_time = time.time() - start_time
            self.logger.info(f"Location info retrieved in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get location info: {str(e)}")
            raise

    async def calculate_route(
        self,
        request: NavigationRequest
    ) -> List[RouteInfo]:
        """Calculate route between origin and destination with waypoints.
        
        Args:
            request: Navigation request with origin, destination, and preferences
            
        Returns:
            List of calculated route options with turn-by-turn instructions
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_route_cache_key(request)
            if cache_key in self._route_cache:
                cached_route = self._route_cache[cache_key]
                if self._is_cache_valid(cached_route["timestamp"]):
                    self.logger.debug("Returning cached route")
                    return cached_route["routes"]
            
            routes = []
            
            # Try Google Maps API first
            if self._google_client:
                try:
                    google_routes = await self._calculate_google_route(request)
                    routes.extend(google_routes)
                except Exception as e:
                    self.logger.warning(f"Google Maps route calculation failed: {str(e)}")
            
            # Fallback to OpenStreetMap if no routes from Google Maps
            if not routes and self._nominatim_client:
                try:
                    osm_routes = await self._calculate_osm_route(request)
                    routes.extend(osm_routes)
                except Exception as e:
                    self.logger.warning(f"OpenStreetMap route calculation failed: {str(e)}")
            
            if not routes:
                raise Exception("Unable to calculate route with any available service")
            
            # Cache the results
            if settings.navigation.enable_route_caching:
                self._route_cache[cache_key] = {
                    "routes": routes,
                    "timestamp": datetime.now(timezone.utc)
                }
            
            processing_time = time.time() - start_time
            self.logger.info(f"Route calculated in {processing_time:.2f}s")
            
            return routes
            
        except Exception as e:
            self.logger.error(f"Route calculation failed: {str(e)}")
            raise

    async def search_nearby_pois(
        self,
        coordinates: Coordinates,
        categories: Optional[List[POICategory]] = None,
        radius_meters: int = 1000,
        keyword: Optional[str] = None,
        max_results: int = 20
    ) -> List[PointOfInterest]:
        """Search for Points of Interest near a location.
        
        Args:
            coordinates: Center point for search
            categories: POI categories to search for
            radius_meters: Search radius in meters
            keyword: Optional search keyword
            max_results: Maximum number of results
            
        Returns:
            List of found POIs with details
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_poi_cache_key(coordinates, categories, radius_meters, keyword)
            if cache_key in self._poi_cache:
                cached_pois = self._poi_cache[cache_key]
                if self._is_cache_valid(cached_pois["timestamp"]):
                    self.logger.debug("Returning cached POI results")
                    return cached_pois["pois"]
            
            pois = []
            
            # Try Google Places API first
            if self._google_client:
                try:
                    google_pois = await self._search_google_places(
                        coordinates, categories, radius_meters, keyword, max_results
                    )
                    pois.extend(google_pois)
                except Exception as e:
                    self.logger.warning(f"Google Places search failed: {str(e)}")
            
            # Fallback to OpenStreetMap if needed
            if len(pois) < max_results and self._overpass_api:
                try:
                    osm_pois = await self._search_osm_places(
                        coordinates, categories, radius_meters, keyword, max_results - len(pois)
                    )
                    pois.extend(osm_pois)
                except Exception as e:
                    self.logger.warning(f"OpenStreetMap POI search failed: {str(e)}")
            
            # Cache the results
            if settings.navigation.enable_poi_caching:
                self._poi_cache[cache_key] = {
                    "pois": pois,
                    "timestamp": datetime.now(timezone.utc)
                }
            
            processing_time = time.time() - start_time
            self.logger.info(f"POI search completed in {processing_time:.2f}s, found {len(pois)} results")
            
            return pois[:max_results]
            
        except Exception as e:
            self.logger.error(f"POI search failed: {str(e)}")
            raise

    async def geocode_address(self, address: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Convert address to coordinates.
        
        Args:
            address: Address string to geocode
            country_code: Optional country code for biasing results
            
        Returns:
            Dict containing coordinates and geocoding information
        """
        try:
            cache_key = f"geocode:{address}:{country_code or 'any'}"
            if cache_key in self._geocode_cache:
                cached = self._geocode_cache[cache_key]
                if self._is_cache_valid(cached["timestamp"]):
                    return cached["result"]
            
            result = None
            
            # Try Google Geocoding API first
            if self._google_client:
                try:
                    result = await self._google_geocode(address, country_code)
                except Exception as e:
                    self.logger.warning(f"Google Geocoding failed: {str(e)}")
            
            # Fallback to Nominatim
            if not result and self._nominatim_client:
                try:
                    result = await self._nominatim_geocode(address, country_code)
                except Exception as e:
                    self.logger.warning(f"Nominatim geocoding failed: {str(e)}")
            
            if not result:
                raise Exception(f"Unable to geocode address: {address}")
            
            # Cache the result
            self._geocode_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now(timezone.utc)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Geocoding failed: {str(e)}")
            raise

    async def reverse_geocode(self, coordinates: Coordinates) -> Dict[str, Any]:
        """Convert coordinates to address.
        
        Args:
            coordinates: GPS coordinates to reverse geocode
            
        Returns:
            Dict containing address information
        """
        try:
            return await self._reverse_geocode(coordinates)
        except Exception as e:
            self.logger.error(f"Reverse geocoding failed: {str(e)}")
            raise

    # Private helper methods

    async def _calculate_google_route(self, request: NavigationRequest) -> List[RouteInfo]:
        """Calculate route using Google Directions API."""
        if not self._google_client:
            raise Exception("Google Maps client not available")
        
        # Convert coordinates to Google Maps format
        origin = f"{request.origin.latitude},{request.origin.longitude}"
        destination = f"{request.destination.latitude},{request.destination.longitude}"
        waypoints = [f"{wp.latitude},{wp.longitude}" for wp in request.waypoints] if request.waypoints else None
        
        # Map transport modes
        mode_mapping = {
            TransportMode.WALKING: "walking",
            TransportMode.DRIVING: "driving", 
            TransportMode.BICYCLING: "bicycling",
            TransportMode.TRANSIT: "transit"
        }
        
        # Calculate route
        directions = await asyncio.to_thread(
            self._google_client.directions,  # type: ignore
            origin=origin,
            destination=destination,
            waypoints=waypoints,
            mode=mode_mapping.get(request.transport_mode, "walking"),
            avoid=self._get_google_avoid_options(request),
            departure_time=request.departure_time,
            alternatives=True
        )
        
        routes = []
        for route in directions:
            route_info = self._parse_google_route(route)
            routes.append(route_info)
        
        return routes

    async def _calculate_osm_route(self, request: NavigationRequest) -> List[RouteInfo]:
        """Calculate route using OpenStreetMap data."""
        # This is a simplified implementation
        # In a real system, you'd use OSRM or GraphHopper for routing
        
        # For now, return a basic direct route
        origin_point = Point(request.origin.longitude, request.origin.latitude)
        dest_point = Point(request.destination.longitude, request.destination.latitude)
        
        # Calculate direct distance
        direct_distance = distance.distance(
            (request.origin.latitude, request.origin.longitude),
            (request.destination.latitude, request.destination.longitude)
        ).meters
        
        # Estimate duration based on transport mode
        speed_mapping = {
            TransportMode.WALKING: 5.0,  # km/h
            TransportMode.BICYCLING: 15.0,
            TransportMode.DRIVING: 50.0,
            TransportMode.TRANSIT: 30.0
        }
        
        speed_kmh = speed_mapping.get(request.transport_mode, 5.0)
        estimated_duration = int((direct_distance / 1000) / speed_kmh * 3600)
        
        # Create a basic route
        route_info = RouteInfo(
            distance_meters=int(direct_distance),
            duration_seconds=estimated_duration,
            duration_in_traffic_seconds=None,
            overview_polyline=None,
            steps=[
                NavigationStep(
                    instruction=f"Head to destination",
                    distance_meters=int(direct_distance),
                    duration_seconds=estimated_duration,
                    maneuver="straight",
                    start_location=request.origin,
                    end_location=request.destination,
                    travel_mode=request.transport_mode,
                    polyline=None
                )
            ],
            bounds={
                "northeast": Coordinates(
                    latitude=max(request.origin.latitude, request.destination.latitude),
                    longitude=max(request.origin.longitude, request.destination.longitude),
                    accuracy=None
                ),
                "southwest": Coordinates(
                    latitude=min(request.origin.latitude, request.destination.latitude), 
                    longitude=min(request.origin.longitude, request.destination.longitude),
                    accuracy=None
                )
            },
            warnings=["Route calculated using direct path - limited navigation available"],
            copyrights="© OpenStreetMap contributors"
        )
        
        return [route_info]

    async def _search_google_places(
        self, 
        coordinates: Coordinates, 
        categories: Optional[List[POICategory]], 
        radius_meters: int,
        keyword: Optional[str],
        max_results: int
    ) -> List[PointOfInterest]:
        """Search POIs using Google Places API."""
        if not self._google_client:
            return []
        
        pois = []
        
        # Map POI categories to Google Places types
        type_mapping = {
            POICategory.RESTAURANT: "restaurant",
            POICategory.HOTEL: "lodging",
            POICategory.GAS_STATION: "gas_station",
            POICategory.ATM: "atm",
            POICategory.HOSPITAL: "hospital",
            POICategory.PHARMACY: "pharmacy",
            POICategory.TOURIST_ATTRACTION: "tourist_attraction",
            POICategory.SHOPPING: "shopping_mall",
        }
        
        search_types = [type_mapping.get(cat, "establishment") for cat in categories] if categories else ["establishment"]
        
        for place_type in search_types:
            try:
                places_result = await asyncio.to_thread(
                    self._google_client.places_nearby,  # type: ignore
                    location=(coordinates.latitude, coordinates.longitude),
                    radius=radius_meters,
                    type=place_type,
                    keyword=keyword
                )
                
                for place in places_result.get("results", [])[:max_results]:
                    poi = self._parse_google_place(place, coordinates)
                    if poi:
                        pois.append(poi)
                
                if len(pois) >= max_results:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Google Places search for {place_type} failed: {str(e)}")
                continue
        
        return pois[:max_results]

    async def _search_osm_places(
        self,
        coordinates: Coordinates,
        categories: Optional[List[POICategory]], 
        radius_meters: int,
        keyword: Optional[str],
        max_results: int
    ) -> List[PointOfInterest]:
        """Search POIs using OpenStreetMap Overpass API."""
        if not self._overpass_api:
            return []
        
        # This is a simplified implementation
        # In production, you'd build proper Overpass QL queries
        return []

    def _parse_google_route(self, route: Dict[str, Any]) -> RouteInfo:
        """Parse Google Directions API route response."""
        leg = route["legs"][0]  # Assuming single leg for simplicity
        
        steps = []
        for step in leg["steps"]:
            nav_step = NavigationStep(
                instruction=step["html_instructions"],
                distance_meters=step["distance"]["value"],
                duration_seconds=step["duration"]["value"], 
                maneuver=step.get("maneuver", "straight"),
                start_location=Coordinates(
                    latitude=step["start_location"]["lat"],
                    longitude=step["start_location"]["lng"],
                    accuracy=None
                ),
                end_location=Coordinates(
                    latitude=step["end_location"]["lat"],
                    longitude=step["end_location"]["lng"],
                    accuracy=None
                ),
                travel_mode=TransportMode.WALKING,  # Would need mapping
                polyline=step.get("polyline", {}).get("points")
            )
            steps.append(nav_step)
        
        return RouteInfo(
            distance_meters=leg["distance"]["value"],
            duration_seconds=leg["duration"]["value"],
            duration_in_traffic_seconds=leg.get("duration_in_traffic", {}).get("value"),
            steps=steps,
            overview_polyline=route["overview_polyline"]["points"],
            bounds={
                "northeast": Coordinates(
                    latitude=route["bounds"]["northeast"]["lat"],
                    longitude=route["bounds"]["northeast"]["lng"],
                    accuracy=None
                ),
                "southwest": Coordinates(
                    latitude=route["bounds"]["southwest"]["lat"], 
                    longitude=route["bounds"]["southwest"]["lng"],
                    accuracy=None
                )
            },
            warnings=route.get("warnings", []),
            copyrights=route.get("copyrights", "")
        )

    def _parse_google_place(self, place: Dict[str, Any], search_center: Coordinates) -> Optional[PointOfInterest]:
        """Parse Google Places API place response."""
        try:
            place_location = Coordinates(
                latitude=place["geometry"]["location"]["lat"],
                longitude=place["geometry"]["location"]["lng"],
                accuracy=None
            )
            
            # Calculate distance from search center
            dist = distance.distance(
                (search_center.latitude, search_center.longitude),
                (place_location.latitude, place_location.longitude)
            ).meters
            
            return PointOfInterest(
                id=place.get("place_id", str(uuid4())),
                name=place.get("name", "Unknown"),
                category=POICategory.OTHER,  # Would need proper mapping
                subcategory=None,
                coordinates=place_location,
                distance_meters=int(dist),
                phone_number=None,
                website_url=None,
                business_hours=None,
                rating=place.get("rating"),
                review_count=place.get("user_ratings_total"),
                price_level=place.get("price_level"),
                formatted_address=place.get("vicinity"),
                currently_open=place.get("opening_hours", {}).get("open_now")
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse Google place: {str(e)}")
            return None

    async def _reverse_geocode(self, coordinates: Coordinates) -> Dict[str, Any]:
        """Reverse geocode coordinates to address."""
        # Try Google first, then Nominatim
        if self._google_client:
            try:
                result = await asyncio.to_thread(
                    self._google_client.reverse_geocode,  # type: ignore
                    (coordinates.latitude, coordinates.longitude)
                )
                if result:
                    return self._parse_google_reverse_geocode(result[0])
            except Exception as e:
                self.logger.warning(f"Google reverse geocoding failed: {str(e)}")
        
        # Fallback to Nominatim
        if self._nominatim_client:
            try:
                location = await asyncio.to_thread(
                    self._nominatim_client.reverse,
                    f"{coordinates.latitude},{coordinates.longitude}"
                )
                if location:
                    return self._parse_nominatim_result(location)
            except Exception as e:
                self.logger.warning(f"Nominatim reverse geocoding failed: {str(e)}")
        
        return {"formatted_address": f"{coordinates.latitude}, {coordinates.longitude}"}

    def _parse_google_reverse_geocode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google reverse geocoding result."""
        return {
            "formatted_address": result.get("formatted_address"),
            "place_id": result.get("place_id"),
            "address_components": result.get("address_components", [])
        }

    def _parse_nominatim_result(self, location) -> Dict[str, Any]:
        """Parse Nominatim geocoding result."""
        return {
            "formatted_address": location.address,
            "place_id": None,
            "address_components": []
        }

    async def _google_geocode(self, address: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Geocode address using Google Geocoding API."""
        result = await asyncio.to_thread(
            self._google_client.geocode,  # type: ignore
            address,
            region=country_code
        )
        
        if result:
            location = result[0]["geometry"]["location"]
            return {
                "coordinates": Coordinates(
                    latitude=location["lat"],
                    longitude=location["lng"],
                    accuracy=None
                ),
                "formatted_address": result[0]["formatted_address"],
                "place_id": result[0].get("place_id"),
                "accuracy": result[0]["geometry"]["location_type"]
            }
        
        raise Exception("No results found")

    async def _nominatim_geocode(self, address: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Geocode address using Nominatim."""
        location = await asyncio.to_thread(
            self._nominatim_client.geocode,  # type: ignore
            address,
            country_codes=country_code
        )
        
        if location:
            return {
                "coordinates": Coordinates(
                    latitude=location.latitude,  # type: ignore
                    longitude=location.longitude,  # type: ignore
                    accuracy=None
                ),
                "formatted_address": location.address,  # type: ignore
                "place_id": None,
                "accuracy": "APPROXIMATE"
            }
        
        raise Exception("No results found")

    def _get_google_avoid_options(self, request: NavigationRequest) -> List[str]:
        """Get Google Maps avoid options based on request preferences."""
        avoid = []
        if request.avoid_tolls:
            avoid.append("tolls")
        if request.avoid_highways:
            avoid.append("highways")
        if request.avoid_ferries:
            avoid.append("ferries")
        return avoid

    def _get_route_cache_key(self, request: NavigationRequest) -> str:
        """Generate cache key for route request."""
        waypoints_str = ",".join([f"{wp.latitude},{wp.longitude}" for wp in request.waypoints])
        return (f"route:{request.origin.latitude},{request.origin.longitude}:"
                f"{request.destination.latitude},{request.destination.longitude}:"
                f"{waypoints_str}:{request.transport_mode.value}")

    def _get_poi_cache_key(
        self, 
        coordinates: Coordinates, 
        categories: Optional[List[POICategory]], 
        radius: int, 
        keyword: Optional[str]
    ) -> str:
        """Generate cache key for POI search."""
        cats_str = ",".join([cat.value for cat in categories]) if categories else "all"
        keyword_str = keyword or "none"
        return f"poi:{coordinates.latitude},{coordinates.longitude}:{cats_str}:{radius}:{keyword_str}"

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid."""
        if not timestamp:
            return False
        
        age_minutes = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
        return age_minutes < settings.navigation.route_cache_ttl_minutes
