"""Navigation and location services schemas for My Buddy API.

This module contains Pydantic schemas for navigation-related endpoints,
including GPS location services, route planning, directions, POI discovery,
and voice-guided navigation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Dict
from uuid import UUID

from pydantic import BaseModel, Field, validator

from app.schemas.common import BaseResponse, LanguageCode, Coordinates, Location, Address


class TransportMode(str, Enum):
    """Transportation modes for navigation."""
    WALKING = "walking"
    DRIVING = "driving"
    BICYCLING = "bicycling"
    TRANSIT = "transit"


class RoutePreference(str, Enum):
    """Route optimization preferences."""
    FASTEST = "fastest"
    SHORTEST = "shortest"
    AVOID_TOLLS = "avoid_tolls"
    AVOID_HIGHWAYS = "avoid_highways"
    AVOID_FERRIES = "avoid_ferries"


class POICategory(str, Enum):
    """Point of Interest categories."""
    RESTAURANT = "restaurant"
    HOTEL = "hotel"
    GAS_STATION = "gas_station"
    ATM = "atm"
    HOSPITAL = "hospital"
    PHARMACY = "pharmacy"
    TOURIST_ATTRACTION = "tourist_attraction"
    SHOPPING = "shopping"
    TRANSPORT = "transport"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


class NavigationRequest(BaseModel):
    """Navigation route request schema."""

    origin: Coordinates = Field(description="Starting location coordinates")
    destination: Coordinates = Field(description="Destination coordinates")
    waypoints: List[Coordinates] = Field(default_factory=list, description="Intermediate waypoints")
    transport_mode: TransportMode = Field(default=TransportMode.WALKING, description="Transportation mode")
    route_preference: RoutePreference = Field(default=RoutePreference.FASTEST, description="Route optimization")
    avoid_tolls: bool = Field(default=False, description="Avoid toll roads")
    avoid_highways: bool = Field(default=False, description="Avoid highways")
    avoid_ferries: bool = Field(default=False, description="Avoid ferries")
    departure_time: Optional[datetime] = Field(None, description="Planned departure time")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Language for instructions")

    @validator("waypoints")
    def validate_waypoints(cls, v):
        """Validate waypoints list."""
        if len(v) > 8:
            raise ValueError("Maximum 8 waypoints allowed")
        return v


class NavigationStep(BaseModel):
    """Individual step in navigation instructions."""

    instruction: str = Field(description="Human-readable instruction")
    distance_meters: int = Field(description="Distance for this step in meters")
    duration_seconds: int = Field(description="Estimated time for this step in seconds")
    maneuver: str = Field(description="Maneuver type (turn-left, turn-right, straight, etc.)")
    start_location: Coordinates = Field(description="Step start coordinates")
    end_location: Coordinates = Field(description="Step end coordinates")
    travel_mode: TransportMode = Field(description="Transportation mode for this step")
    polyline: Optional[str] = Field(None, description="Encoded polyline for step visualization")


class RouteInfo(BaseModel):
    """Detailed route information."""

    distance_meters: int = Field(description="Total route distance in meters")
    duration_seconds: int = Field(description="Estimated travel time in seconds")
    duration_in_traffic_seconds: Optional[int] = Field(None, description="Time considering current traffic")
    steps: List[NavigationStep] = Field(description="Turn-by-turn navigation instructions")
    overview_polyline: Optional[str] = Field(None, description="Route overview encoded polyline")
    bounds: Dict[str, Coordinates] = Field(description="Route bounding box (northeast, southwest)")
    warnings: List[str] = Field(default_factory=list, description="Route warnings and alerts")
    copyrights: Optional[str] = Field(None, description="Route data attribution and copyrights")


class NavigationResponse(BaseResponse):
    """Navigation route calculation response."""

    routes: List[RouteInfo] = Field(description="Available route options")
    status: str = Field(description="Route calculation status")
    geocoded_waypoints: List[Dict[str, Any]] = Field(default_factory=list, description="Geocoded waypoint information")


class LocationUpdateRequest(BaseModel):
    """Real-time location update during active navigation."""

    session_id: UUID = Field(description="Navigation session identifier")
    current_location: Coordinates = Field(description="Current user location")
    accuracy_meters: Optional[float] = Field(None, ge=0.0, description="Location accuracy in meters")
    bearing_degrees: Optional[float] = Field(None, ge=0.0, lt=360.0, description="Compass bearing (0-359)")
    speed_mps: Optional[float] = Field(None, ge=0.0, description="Current speed in meters per second")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Location timestamp")


class TrafficAlert(BaseModel):
    """Traffic or route alert information."""

    alert_type: str = Field(description="Alert type (traffic, construction, incident, road_closure)")
    severity: str = Field(description="Alert severity (low, medium, high, critical)")
    description: str = Field(description="Human-readable alert description")
    location: Optional[Coordinates] = Field(None, description="Alert location coordinates")
    distance_to_alert_meters: Optional[int] = Field(None, ge=0, description="Distance to alert from current location")
    estimated_delay_seconds: Optional[int] = Field(None, ge=0, description="Estimated delay caused by alert")


class NavigationUpdate(BaseResponse):
    """Real-time navigation status update."""

    session_id: UUID = Field(description="Navigation session identifier")
    current_instruction: Optional[NavigationStep] = Field(None, description="Current navigation instruction")
    next_instruction: Optional[NavigationStep] = Field(None, description="Next navigation instruction")
    distance_remaining_meters: int = Field(ge=0, description="Remaining distance to destination")
    time_remaining_seconds: int = Field(ge=0, description="Estimated remaining time")
    progress_percentage: float = Field(ge=0.0, le=100.0, description="Navigation progress percentage")
    off_route: bool = Field(default=False, description="Whether user has deviated from planned route")
    should_reroute: bool = Field(default=False, description="Whether route recalculation is recommended")
    traffic_alerts: List[TrafficAlert] = Field(default_factory=list, description="Current traffic alerts ahead")


class PointOfInterest(BaseModel):
    """Point of Interest information with business details."""

    id: str = Field(description="POI unique identifier")
    name: str = Field(description="POI name or business name")
    category: POICategory = Field(description="POI category type")
    subcategory: Optional[str] = Field(None, description="Specific subcategory (e.g., 'Italian Restaurant')")
    coordinates: Coordinates = Field(description="POI location coordinates")
    distance_meters: Optional[int] = Field(None, ge=0, description="Distance from search point")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Average rating (0-5 stars)")
    review_count: Optional[int] = Field(None, ge=0, description="Total number of reviews")
    price_level: Optional[int] = Field(None, ge=0, le=4, description="Price level (0=free, 1=$, 2=$$, 3=$$$, 4=$$$$)")
    phone_number: Optional[str] = Field(None, description="Contact phone number")
    website_url: Optional[str] = Field(None, description="Official website URL")
    formatted_address: Optional[str] = Field(None, description="Human-readable address")
    business_hours: Optional[Dict[str, Any]] = Field(None, description="Operating hours by day of week")
    photos: List[str] = Field(default_factory=list, description="Photo URLs")
    currently_open: Optional[bool] = Field(None, description="Whether establishment is currently open")


class POISearchRequest(BaseModel):
    """Point of Interest search request."""

    location: Coordinates = Field(description="Search center coordinates")
    radius_meters: int = Field(default=1000, ge=100, le=50000, description="Search radius in meters")
    categories: List[POICategory] = Field(default_factory=list, description="POI categories to search for")
    keyword: Optional[str] = Field(None, max_length=100, description="Search keyword or business name")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of results to return")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language for results")
    open_now: bool = Field(default=False, description="Filter for establishments currently open")
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Minimum rating filter")


class POISearchResponse(BaseResponse):
    """Point of Interest search results."""

    pois: List[PointOfInterest] = Field(description="Found points of interest")
    search_center: Coordinates = Field(description="Search center coordinates used")
    search_radius_meters: int = Field(description="Actual search radius used")
    total_results: int = Field(ge=0, description="Total POIs found (may exceed returned count)")
    next_page_token: Optional[str] = Field(None, description="Token for retrieving additional results")


class GeocodeRequest(BaseModel):
    """Address to coordinates geocoding request."""

    address: str = Field(max_length=200, description="Address string to geocode")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language for results")
    country_code: Optional[str] = Field(None, max_length=2, description="Country code hint (ISO 3166-1 alpha-2)")
    bounds: Optional[Dict[str, Coordinates]] = Field(None, description="Bounding box for search bias")


class GeocodeResponse(BaseResponse):
    """Geocoding response with location results."""

    results: List[Dict[str, Any]] = Field(description="Geocoding candidate results")
    coordinates: Coordinates = Field(description="Best match coordinates")
    formatted_address: str = Field(description="Formatted address string")
    place_id: Optional[str] = Field(None, description="Google Places or OSM place identifier")
    accuracy: str = Field(description="Geocoding accuracy level (ROOFTOP, RANGE_INTERPOLATED, etc.)")
    address_components: List[Dict[str, Any]] = Field(description="Structured address components")


class ReverseGeocodeRequest(BaseModel):
    """Coordinates to address reverse geocoding request."""

    coordinates: Coordinates = Field(description="Coordinates to reverse geocode")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language for address results")
    result_types: List[str] = Field(default_factory=list, description="Filter result types (street_address, route, etc.)")


class ReverseGeocodeResponse(BaseResponse):
    """Reverse geocoding response with address results."""

    results: List[Address] = Field(description="Address results from most specific to least specific")
    coordinates: Coordinates = Field(description="Input coordinates")
    formatted_addresses: List[str] = Field(description="Human-readable formatted address strings")
    place_id: Optional[str] = Field(None, description="Google Places or OSM place identifier")


class CurrentLocationRequest(BaseModel):
    """Current location information request."""

    coordinates: Coordinates = Field(description="Current coordinates")
    accuracy_meters: Optional[float] = Field(None, ge=0.0, description="Location accuracy in meters")
    include_address: bool = Field(default=True, description="Include reverse geocoded address")
    include_nearby_pois: bool = Field(default=False, description="Include nearby points of interest")
    poi_categories: List[POICategory] = Field(default_factory=list, description="POI categories to include")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Language for address and POI names")


class CurrentLocationResponse(BaseResponse):
    """Current location response with contextual information."""

    coordinates: Coordinates = Field(description="Current location coordinates")
    accuracy_meters: Optional[float] = Field(None, description="Location accuracy in meters")
    address: Optional[Address] = Field(None, description="Reverse geocoded address information")
    formatted_address: Optional[str] = Field(None, description="Human-readable address")
    nearby_pois: List[PointOfInterest] = Field(default_factory=list, description="Nearby points of interest")
    timezone: Optional[str] = Field(None, description="Local timezone identifier")
    country_code: Optional[str] = Field(None, description="ISO country code")
    administrative_area: Optional[str] = Field(None, description="State, province, or administrative area")
    locality: Optional[str] = Field(None, description="City or locality name")


class VoiceNavigationRequest(BaseModel):
    """Voice-guided navigation configuration request."""

    session_id: UUID = Field(description="Navigation session identifier")
    enable_voice: bool = Field(default=True, description="Enable voice instruction announcements")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Voice instruction language")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice playback speed multiplier")
    distance_units: str = Field(default="metric", description="Distance units (metric/imperial)")
    announce_traffic: bool = Field(default=True, description="Announce traffic alerts and incidents")
    announce_maneuvers_distance: int = Field(default=100, ge=50, le=500, description="Distance before maneuver to announce (meters)")


class VoiceNavigationResponse(BaseResponse):
    """Voice navigation configuration confirmation."""

    session_id: UUID = Field(description="Navigation session identifier")
    voice_settings: Dict[str, Any] = Field(description="Applied voice configuration settings")
    supported_languages: List[str] = Field(description="Available voice instruction languages")
    current_instruction: Optional[str] = Field(None, description="Current voice instruction if navigation is active")


class NavigationSessionRequest(BaseModel):
    """Navigation session initialization request."""

    route_id: Optional[UUID] = Field(None, description="Pre-calculated route identifier")
    navigation_request: Optional[NavigationRequest] = Field(None, description="Route calculation parameters")
    enable_voice_guidance: bool = Field(default=True, description="Enable voice turn-by-turn instructions")
    voice_language: LanguageCode = Field(default=LanguageCode.EN, description="Voice instruction language")
    user_id: Optional[UUID] = Field(None, description="User identifier (optional for anonymous sessions)")

    @validator("navigation_request")
    def validate_route_source(cls, v, values):
        """Ensure either route_id or navigation_request is provided."""
        route_id = values.get("route_id")
        if not route_id and not v:
            raise ValueError("Either route_id or navigation_request must be provided")
        if route_id and v:
            raise ValueError("Provide either route_id or navigation_request, not both")
        return v


class NavigationSessionResponse(BaseResponse):
    """Navigation session creation confirmation."""

    session_id: UUID = Field(description="Created navigation session identifier")
    route: RouteInfo = Field(description="Route information for navigation")
    initial_instruction: Optional[NavigationStep] = Field(None, description="First navigation instruction")
    estimated_arrival_time: datetime = Field(description="Estimated arrival time at destination")
    voice_enabled: bool = Field(description="Voice guidance activation status")
    session_expires_at: datetime = Field(description="Session expiration time")
