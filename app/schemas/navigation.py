"""Navigation schemas for My Buddy API.

This module contains Pydantic schemas for navigation-related endpoints,
including route planning, directions, and location services.
"""

from datetime import datetime, time
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator

from app.schemas.common import (
    BaseResponse, Coordinates, Location, Address,
    LanguageCode, TranslationResponse
)


class TransportMode(str, Enum):
    """Transportation modes for navigation."""
    WALKING = "walking"
    DRIVING = "driving"
    PUBLIC_TRANSIT = "public_transit"
    CYCLING = "cycling"
    TAXI = "taxi"
    RIDESHARE = "rideshare"


class RoutePreference(str, Enum):
    """Route optimization preferences."""
    FASTEST = "fastest"
    SHORTEST = "shortest"
    SCENIC = "scenic"
    AVOID_TOLLS = "avoid_tolls"
    AVOID_HIGHWAYS = "avoid_highways"
    ACCESSIBLE = "accessible"


class NavigationRequest(BaseModel):
    """Navigation route request schema."""
    
    origin: Location = Field(description="Starting location")
    destination: Location = Field(description="Destination location")
    waypoints: List[Location] = Field(default_factory=list, description="Intermediate waypoints")
    transport_mode: TransportMode = Field(default=TransportMode.WALKING, description="Transportation mode")
    route_preference: RoutePreference = Field(default=RoutePreference.FASTEST, description="Route optimization")
    avoid_areas: List[Coordinates] = Field(default_factory=list, description="Areas to avoid")
    departure_time: Optional[datetime] = Field(None, description="Preferred departure time")
    arrival_time: Optional[datetime] = Field(None, description="Preferred arrival time")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Language for instructions")
    accessibility_required: bool = Field(default=False, description="Require accessible routes")
    
    @validator('waypoints')
    def validate_waypoints(cls, v):
        """Validate waypoint count."""
        if len(v) > 10:
            raise ValueError("Maximum 10 waypoints allowed")
        return v


class RouteStep(BaseModel):
    """Individual route step/instruction."""
    
    step_number: int = Field(ge=1, description="Step sequence number")
    instruction: str = Field(description="Turn-by-turn instruction")
    translated_instruction: Optional[str] = Field(None, description="Translated instruction")
    distance_meters: float = Field(ge=0, description="Step distance in meters")
    duration_seconds: int = Field(ge=0, description="Estimated step duration")
    start_location: Coordinates = Field(description="Step start coordinates")
    end_location: Coordinates = Field(description="Step end coordinates")
    polyline: Optional[str] = Field(None, description="Encoded polyline for step geometry")
    maneuver_type: Optional[str] = Field(None, description="Type of maneuver")
    road_name: Optional[str] = Field(None, description="Road or street name")
    transport_mode: TransportMode = Field(description="Transport mode for this step")


class Route(BaseModel):
    """Complete route information."""
    
    route_id: UUID = Field(description="Unique route identifier")
    summary: str = Field(description="Route summary description")
    total_distance_meters: float = Field(ge=0, description="Total route distance")
    total_duration_seconds: int = Field(ge=0, description="Estimated total duration")
    steps: List[RouteStep] = Field(description="Turn-by-turn route steps")
    polyline: str = Field(description="Encoded polyline for entire route")
    bounds: Dict[str, Coordinates] = Field(description="Route bounding box (northeast, southwest)")
    warnings: List[str] = Field(default_factory=list, description="Route warnings or alerts")
    toll_info: Optional[Dict[str, Any]] = Field(None, description="Toll road information")
    traffic_info: Optional[Dict[str, Any]] = Field(None, description="Current traffic conditions")


class AlternativeRoute(BaseModel):
    """Alternative route option."""
    
    route: Route = Field(description="Alternative route details")
    time_difference_seconds: int = Field(description="Time difference vs primary route")
    distance_difference_meters: float = Field(description="Distance difference vs primary route")
    advantages: List[str] = Field(default_factory=list, description="Route advantages")
    disadvantages: List[str] = Field(default_factory=list, description="Route disadvantages")


class NavigationResponse(BaseResponse):
    """Navigation response with routes."""
    
    primary_route: Route = Field(description="Primary recommended route")
    alternative_routes: List[AlternativeRoute] = Field(default_factory=list, description="Alternative route options")
    request_id: UUID = Field(description="Request tracking identifier")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Response generation time")


# Live Navigation Schemas
class NavigationUpdate(BaseModel):
    """Live navigation update."""
    
    current_location: Coordinates = Field(description="Current user location")
    next_instruction: str = Field(description="Next navigation instruction")
    distance_to_next_meters: float = Field(ge=0, description="Distance to next turn")
    time_to_next_seconds: int = Field(ge=0, description="Time to next instruction")
    remaining_distance_meters: float = Field(ge=0, description="Remaining total distance")
    remaining_time_seconds: int = Field(ge=0, description="Remaining total time")
    current_road: Optional[str] = Field(None, description="Current road name")
    speed_limit: Optional[int] = Field(None, description="Current speed limit")
    traffic_delay_seconds: Optional[int] = Field(None, description="Traffic-related delays")


class NavigationAlert(BaseModel):
    """Navigation alert or warning."""
    
    alert_type: str = Field(description="Type of alert (traffic, road_closure, etc.)")
    severity: str = Field(description="Alert severity (low, medium, high)")
    message: str = Field(description="Alert message")
    translated_message: Optional[str] = Field(None, description="Translated alert message")
    location: Optional[Coordinates] = Field(None, description="Alert location")
    affects_route: bool = Field(description="Whether alert affects current route")
    suggested_action: Optional[str] = Field(None, description="Suggested user action")


# Points of Interest (POI) Schemas
class POICategory(str, Enum):
    """Point of Interest categories."""
    RESTAURANT = "restaurant"
    GAS_STATION = "gas_station"
    ATM = "atm"
    HOSPITAL = "hospital"
    PHARMACY = "pharmacy"
    TOURIST_ATTRACTION = "tourist_attraction"
    SHOPPING = "shopping"
    ACCOMMODATION = "accommodation"
    TRANSPORT_HUB = "transport_hub"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    RELIGIOUS = "religious"
    GOVERNMENT = "government"


class PointOfInterest(BaseModel):
    """Point of Interest information."""
    
    poi_id: str = Field(description="Unique POI identifier")
    name: str = Field(description="POI name")
    translated_name: Optional[str] = Field(None, description="Translated POI name")
    category: POICategory = Field(description="POI category")
    location: Location = Field(description="POI location")
    distance_meters: float = Field(ge=0, description="Distance from reference point")
    rating: Optional[float] = Field(None, ge=0, le=5, description="User rating (0-5)")
    price_level: Optional[int] = Field(None, ge=1, le=4, description="Price level (1-4)")
    opening_hours: Optional[Dict[str, str]] = Field(None, description="Opening hours by day")
    phone: Optional[str] = Field(None, description="Contact phone number")
    website: Optional[str] = Field(None, description="Website URL")
    description: Optional[str] = Field(None, description="POI description")
    amenities: List[str] = Field(default_factory=list, description="Available amenities")
    accessibility_features: List[str] = Field(default_factory=list, description="Accessibility features")


class POISearchRequest(BaseModel):
    """POI search request schema."""
    
    location: Coordinates = Field(description="Search center coordinates")
    radius_meters: int = Field(default=1000, ge=100, le=50000, description="Search radius in meters")
    category: Optional[POICategory] = Field(None, description="POI category filter")
    query: Optional[str] = Field(None, max_length=100, description="Search query text")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating filter")
    max_price_level: Optional[int] = Field(None, ge=1, le=4, description="Maximum price level")
    open_now: bool = Field(default=False, description="Filter for currently open POIs")
    accessible_only: bool = Field(default=False, description="Filter for accessible POIs only")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Language for results")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum number of results")


class POISearchResponse(BaseResponse):
    """POI search response."""
    
    pois: List[PointOfInterest] = Field(description="Found points of interest")
    search_center: Coordinates = Field(description="Search center location")
    search_radius_meters: int = Field(description="Search radius used")
    total_found: int = Field(description="Total POIs found")
    query_id: UUID = Field(description="Search query identifier")


# Geocoding Schemas
class GeocodingRequest(BaseModel):
    """Geocoding (address to coordinates) request."""
    
    address: str = Field(min_length=1, max_length=500, description="Address to geocode")
    region_bias: Optional[str] = Field(None, description="Region bias for results")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    restrict_to_country: Optional[str] = Field(None, description="Country code restriction")


class GeocodingResult(BaseModel):
    """Geocoding result."""
    
    location: Location = Field(description="Geocoded location")
    confidence: float = Field(ge=0, le=1, description="Geocoding confidence")
    result_type: str = Field(description="Type of result (address, establishment, etc.)")
    partial_match: bool = Field(description="Whether result is a partial match")


class GeocodingResponse(BaseResponse):
    """Geocoding response."""
    
    results: List[GeocodingResult] = Field(description="Geocoding results")
    query: str = Field(description="Original query")


# Reverse Geocoding Schemas
class ReverseGeocodingRequest(BaseModel):
    """Reverse geocoding (coordinates to address) request."""
    
    coordinates: Coordinates = Field(description="Coordinates to reverse geocode")
    result_types: List[str] = Field(default_factory=list, description="Types of results to return")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")


class ReverseGeocodingResponse(BaseResponse):
    """Reverse geocoding response."""
    
    results: List[Address] = Field(description="Address results")
    coordinates: Coordinates = Field(description="Input coordinates")


# Real-time Location Tracking
class LocationUpdate(BaseModel):
    """Location update for tracking."""
    
    coordinates: Coordinates = Field(description="Current coordinates")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Location timestamp")
    speed_mps: Optional[float] = Field(None, ge=0, description="Speed in meters per second")
    heading_degrees: Optional[float] = Field(None, ge=0, lt=360, description="Heading in degrees")
    altitude_meters: Optional[float] = Field(None, description="Altitude in meters")


class LocationHistory(BaseModel):
    """Location history for a user."""
    
    user_id: UUID = Field(description="User identifier")
    updates: List[LocationUpdate] = Field(description="Location updates")
    start_time: datetime = Field(description="History start time")
    end_time: datetime = Field(description="History end time")
    total_distance_meters: float = Field(ge=0, description="Total distance traveled")
    average_speed_mps: float = Field(ge=0, description="Average speed")
