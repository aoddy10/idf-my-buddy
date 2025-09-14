"""Navigation API endpoints for My Buddy application.

Provides comprehensive location-based services including GPS location, route calculation,
turn-by-turn navigation, POI discovery, geocoding, and voice-guided navigation.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.deps import get_current_user_optional
from app.services.maps import NavigationService
from app.services.nllb import NLLBTranslationService
from app.services.tts import TTSService
from app.services.voice_navigation import VoiceNavigationService
from app.schemas.navigation import (
    NavigationRequest, NavigationResponse, NavigationStep, RouteInfo,
    LocationUpdateRequest, NavigationUpdate, TrafficAlert,
    POISearchRequest, POISearchResponse, PointOfInterest,
    GeocodeRequest, GeocodeResponse, ReverseGeocodeRequest, ReverseGeocodeResponse,
    CurrentLocationRequest, CurrentLocationResponse,
    VoiceNavigationRequest, VoiceNavigationResponse,
    NavigationSessionRequest, NavigationSessionResponse,
    TransportMode, POICategory, Coordinates
)
from app.schemas.common import BaseResponse, LanguageCode

router = APIRouter()

# Service instances - these would typically be dependency injected
navigation_service = NavigationService()
translation_service = NLLBTranslationService()
tts_service = TTSService()
voice_navigation_service = VoiceNavigationService()

# In-memory session storage (in production, use Redis or database)
active_sessions = {}


@router.get("/")
async def navigation_status():
    """Get navigation service status and health check."""
    try:
        # Quick health check
        test_coords = Coordinates(latitude=40.7128, longitude=-74.0060, accuracy=10.0)  # NYC
        await navigation_service.get_current_location_info(test_coords, include_address=False)
        
        from app.schemas.common import SuccessResponse
        return SuccessResponse(
            success=True,
            message="Navigation service is operational",
            data={
                "google_maps_available": navigation_service._google_client is not None,
                "openstreetmap_available": navigation_service._nominatim_client is not None,
                "supported_transport_modes": [mode.value for mode in TransportMode],
                "supported_poi_categories": [cat.value for cat in POICategory],
                "version": "1.0.0"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Navigation service unavailable: {str(e)}")


@router.post("/routes/calculate", response_model=NavigationResponse)
async def calculate_route(
    request: NavigationRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> NavigationResponse:
    """Calculate route between origin and destination with optional waypoints.
    
    Provides turn-by-turn navigation instructions optimized for the specified
    transport mode with support for avoiding tolls, highways, or ferries.
    """
    try:
        routes = await navigation_service.calculate_route(request)
        
        # Translate instructions if requested language is not English
        if request.language != LanguageCode.EN:
            for route in routes:
                for step in route.steps:
                    try:
                        translated = await translation_service.translate_text(
                            step.instruction,
                            target_language=request.language.value
                        )
                        step.instruction = translated.get("translated_text", step.instruction)
                    except Exception as e:
                        # Log translation error but don't fail the entire request
                        pass
        
        return NavigationResponse(
            success=True,
            message=f"Found {len(routes)} route option(s)",
            routes=routes,
            status="OK" if routes else "ZERO_RESULTS",
            geocoded_waypoints=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Route calculation failed: {str(e)}")


@router.post("/location/current", response_model=CurrentLocationResponse)
async def get_current_location(
    request: CurrentLocationRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> CurrentLocationResponse:
    """Get comprehensive information about a specific location.
    
    Includes reverse geocoded address, nearby POIs, timezone, and
    administrative area information based on provided coordinates.
    """
    try:
        location_info = await navigation_service.get_current_location_info(
            coordinates=request.coordinates,
            include_address=request.include_address,
            include_pois=request.include_nearby_pois,
            poi_categories=request.poi_categories
        )
        
        return CurrentLocationResponse(
            success=True,
            message="Location information retrieved successfully",
            coordinates=request.coordinates,
            accuracy_meters=request.accuracy_meters,
            address=location_info.get("address"),
            formatted_address=location_info.get("formatted_address"),
            nearby_pois=location_info.get("nearby_pois", []),
            timezone=location_info.get("timezone"),
            country_code=location_info.get("country_code"),
            administrative_area=location_info.get("administrative_area"),
            locality=location_info.get("locality")
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Location lookup failed: {str(e)}")


@router.post("/poi/search", response_model=POISearchResponse)
async def search_points_of_interest(
    request: POISearchRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> POISearchResponse:
    """Search for Points of Interest near a location.
    
    Find restaurants, hotels, gas stations, hospitals, and other POIs
    within a specified radius with filtering by category, rating, and keywords.
    """
    try:
        pois = await navigation_service.search_nearby_pois(
            coordinates=request.location,
            categories=request.categories,
            radius_meters=request.radius_meters,
            keyword=request.keyword or "",
            max_results=request.max_results
        )
        
        # Translate POI names and descriptions if requested
        if request.language != LanguageCode.EN:
            for poi in pois:
                try:
                    if poi.name:
                        translated = await translation_service.translate_text(
                            poi.name,
                            target_language=request.language.value
                        )
                        # Store original name in subcategory if translation available
                        if translated.get("translated_text") != poi.name:
                            poi.subcategory = f"{poi.subcategory or ''} ({poi.name})".strip()
                            poi.name = translated["translated_text"]
                except Exception:
                    pass  # Continue without translation on error
        
        return POISearchResponse(
            success=True,
            message=f"Found {len(pois)} POI(s)",
            pois=pois,
            search_center=request.location,
            search_radius_meters=request.radius_meters,
            total_results=len(pois),
            next_page_token=None  # Implement pagination if needed
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"POI search failed: {str(e)}")


@router.post("/geocode", response_model=GeocodeResponse)
async def geocode_address(
    request: GeocodeRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> GeocodeResponse:
    """Convert address to coordinates (geocoding).
    
    Transforms human-readable addresses into precise GPS coordinates
    with accuracy information and structured address components.
    """
    try:
        result = await navigation_service.geocode_address(
            address=request.address,
            country_code=request.country_code or ""
        )
        
        return GeocodeResponse(
            success=True,
            message="Address geocoded successfully",
            results=[result],
            coordinates=result["coordinates"],
            formatted_address=result["formatted_address"],
            place_id=result.get("place_id"),
            accuracy=result["accuracy"],
            address_components=result.get("address_components", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geocoding failed: {str(e)}")


@router.post("/geocode/reverse", response_model=ReverseGeocodeResponse)
async def reverse_geocode_coordinates(
    request: ReverseGeocodeRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> ReverseGeocodeResponse:
    """Convert coordinates to address (reverse geocoding).
    
    Transforms GPS coordinates into human-readable addresses
    with varying levels of detail from specific addresses to general areas.
    """
    try:
        result = await navigation_service.reverse_geocode(request.coordinates)
        
        # Create Address object from result
        from app.schemas.common import Address
        address = Address(
            street_number="",
            street_name="",
            city="",
            state="",
            country="",
            country_code="",
            postal_code="",
            formatted_address=result.get("formatted_address", "")
        )
        
        return ReverseGeocodeResponse(
            success=True,
            message="Coordinates reverse geocoded successfully", 
            results=[address],
            coordinates=request.coordinates,
            formatted_addresses=[result.get("formatted_address", "")],
            place_id=result.get("place_id")
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reverse geocoding failed: {str(e)}")


@router.post("/sessions/start", response_model=NavigationSessionResponse)
async def start_navigation_session(
    request: NavigationSessionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> NavigationSessionResponse:
    """Start a new navigation session with voice guidance.
    
    Initiates turn-by-turn navigation with optional voice instructions
    in multiple languages. Supports both pre-calculated routes and
    real-time route calculation.
    """
    try:
        session_id = uuid4()
        
        # Calculate route if not provided
        if request.navigation_request:
            routes = await navigation_service.calculate_route(request.navigation_request)
            if not routes:
                raise HTTPException(status_code=400, detail="Unable to calculate route")
            route = routes[0]  # Use primary route
        else:
            # In a real implementation, fetch route by ID from database
            raise HTTPException(status_code=400, detail="Route ID lookup not implemented")
        
        # Create navigation session
        session = {
            "id": session_id,
            "route": route,
            "user_id": current_user.get("id") if current_user else None,
            "started_at": datetime.now(timezone.utc),
            "voice_enabled": request.enable_voice_guidance,
            "voice_language": request.voice_language,
            "current_step": 0,
            "status": "active"
        }
        
        active_sessions[session_id] = session
        
        # Get first instruction
        initial_instruction = route.steps[0] if route.steps else None
        
        # Schedule cleanup task
        background_tasks.add_task(cleanup_expired_session, session_id)
        
        return NavigationSessionResponse(
            success=True,
            message="Navigation session started successfully",
            session_id=session_id,
            route=route,
            initial_instruction=initial_instruction,
            estimated_arrival_time=datetime.now(timezone.utc),  # Would calculate properly
            voice_enabled=request.enable_voice_guidance,
            session_expires_at=datetime.now(timezone.utc)  # Would set proper expiration
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to start navigation session: {str(e)}")


@router.post("/sessions/{session_id}/update", response_model=NavigationUpdate)
async def update_navigation_location(
    session_id: UUID,
    request: LocationUpdateRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> NavigationUpdate:
    """Update current location during active navigation.
    
    Provides real-time navigation updates based on user's current position,
    including next instructions, remaining distance/time, and rerouting suggestions.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Navigation session not found")
        
        session = active_sessions[session_id]
        route = session["route"]
        
        # In a real implementation, this would:
        # 1. Check if user is on route
        # 2. Calculate remaining distance/time
        # 3. Determine if rerouting is needed
        # 4. Update current instruction based on position
        
        # For demo, return basic update
        current_step = min(session["current_step"], len(route.steps) - 1)
        next_step = current_step + 1 if current_step < len(route.steps) - 1 else None
        
        return NavigationUpdate(
            success=True,
            message="Navigation updated successfully",
            session_id=session_id,
            current_instruction=route.steps[current_step] if current_step < len(route.steps) else None,
            next_instruction=route.steps[next_step] if next_step and next_step < len(route.steps) else None,
            distance_remaining_meters=route.distance_meters,  # Would calculate actual remaining
            time_remaining_seconds=route.duration_seconds,    # Would calculate actual remaining
            progress_percentage=0.0,  # Would calculate based on position
            off_route=False,          # Would check actual position
            should_reroute=False,     # Would determine based on conditions
            traffic_alerts=[]         # Would include real traffic data
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Navigation update failed: {str(e)}")


@router.post("/sessions/{session_id}/voice", response_model=VoiceNavigationResponse)
async def configure_voice_navigation(
    session_id: UUID,
    request: VoiceNavigationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> VoiceNavigationResponse:
    """Configure voice-guided navigation settings.
    
    Enable/disable voice instructions, set language and speed,
    configure announcement preferences for traffic alerts and maneuvers.
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Navigation session not found")
        
        session = active_sessions[session_id]
        
        # Update voice settings
        session["voice_enabled"] = request.enable_voice
        session["voice_language"] = request.language
        session["voice_speed"] = request.voice_speed
        session["distance_units"] = request.distance_units
        session["announce_traffic"] = request.announce_traffic
        
        # Configure voice navigation service for this session
        voice_navigation_service.set_voice_settings(
            session_id=str(session_id),
            language=request.language,
            use_imperial=(request.distance_units == "imperial"),
            voice_speed=request.voice_speed,
            announce_distance_threshold=500  # meters
        )
        
        # If voice is enabled and there's a current instruction, speak it
        current_instruction = None
        if (request.enable_voice and 
            session["current_step"] < len(session["route"].steps)):
            
            step = session["route"].steps[session["current_step"]]
            current_instruction = step.instruction
            
            # Schedule background task to speak the instruction
            if request.enable_voice and session["current_step"] < len(session["route"].steps):
                current_step_data = session["route"].steps[session["current_step"]]
                background_tasks.add_task(
                    speak_navigation_instruction,
                    session_id=str(session_id),
                    maneuver="straight",  # Would get from step data
                    distance_meters=100,  # Would get from step data
                    street_name=getattr(current_step_data, 'street_name', ""),
                    language=request.language
                )
        
        return VoiceNavigationResponse(
            success=True,
            message="Voice navigation settings updated",
            session_id=session_id,
            voice_settings={
                "enabled": request.enable_voice,
                "language": request.language.value,
                "speed": request.voice_speed,
                "distance_units": request.distance_units,
                "announce_traffic": request.announce_traffic
            },
            supported_languages=[lang.value for lang in LanguageCode],
            current_instruction=current_instruction
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Voice configuration failed: {str(e)}")


@router.delete("/sessions/{session_id}")
async def stop_navigation_session(
    session_id: UUID,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> BaseResponse:
    """Stop and cleanup a navigation session."""
    try:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            session["status"] = "stopped"
            session["ended_at"] = datetime.now(timezone.utc)
            del active_sessions[session_id]
            
            # Cleanup voice navigation service
            voice_navigation_service.cleanup_session(str(session_id))
        
        return BaseResponse(
            success=True,
            message="Navigation session stopped successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to stop session: {str(e)}")


# Background task functions

async def cleanup_expired_session(session_id: UUID):
    """Background task to cleanup expired sessions."""
    await asyncio.sleep(3600)  # 1 hour timeout
    if session_id in active_sessions:
        del active_sessions[session_id]
        # Cleanup voice navigation service
        voice_navigation_service.cleanup_session(str(session_id))


async def speak_navigation_instruction(
    session_id: str, 
    maneuver: str, 
    distance_meters: float,
    street_name: str = "",
    language: LanguageCode = LanguageCode.EN
):
    """Background task to speak navigation instruction using voice navigation templates."""
    try:
        # Generate contextual voice instruction
        instruction = voice_navigation_service.get_voice_instruction_for_step(
            session_id=session_id,
            maneuver=maneuver,
            distance_meters=distance_meters,
            street_name=street_name
        )
        
        if instruction:
            # Speak the instruction using synthesize_text method
            result = await tts_service.synthesize_text(
                text=instruction,
                language=language.value,
                voice="default",
                speed=1.0,
                output_format="wav"
            )
            # In a real implementation, would play the audio file
            # For now, just log that TTS was called
        
    except Exception as e:
        # Log error but don't fail - voice is optional
        pass
