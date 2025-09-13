"""Safety schemas for My Buddy API.

This module contains Pydantic schemas for safety-related endpoints,
including emergency services, safety alerts, and travel safety features.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator

from app.schemas.common import (
    BaseResponse, Coordinates, Location, Address,
    LanguageCode, FileUpload
)


class EmergencyType(str, Enum):
    """Types of emergency situations."""
    MEDICAL = "medical"
    FIRE = "fire"
    POLICE = "police"
    ACCIDENT = "accident"
    NATURAL_DISASTER = "natural_disaster"
    THEFT = "theft"
    ASSAULT = "assault"
    LOST = "lost"
    STRANDED = "stranded"
    HARASSMENT = "harassment"
    FRAUD = "fraud"
    GENERAL = "general"


class SeverityLevel(str, Enum):
    """Severity levels for safety alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of safety alerts."""
    CRIME = "crime"
    WEATHER = "weather"
    HEALTH = "health"
    TRAFFIC = "traffic"
    POLITICAL = "political"
    NATURAL_DISASTER = "natural_disaster"
    TERRORISM = "terrorism"
    CIVIL_UNREST = "civil_unrest"
    DISEASE_OUTBREAK = "disease_outbreak"
    TRAVEL_ADVISORY = "travel_advisory"


class EmergencyContact(BaseModel):
    """Emergency contact information."""
    
    contact_id: str = Field(description="Unique contact identifier")
    name: str = Field(description="Contact name/organization")
    translated_name: Optional[str] = Field(None, description="Translated contact name")
    emergency_type: EmergencyType = Field(description="Type of emergency handled")
    
    # Contact methods
    phone_primary: str = Field(description="Primary phone number")
    phone_secondary: Optional[str] = Field(None, description="Secondary phone number")
    sms_number: Optional[str] = Field(None, description="SMS number")
    email: Optional[str] = Field(None, description="Email address")
    
    # Location and availability
    location: Optional[Location] = Field(None, description="Contact location")
    coverage_area: Optional[str] = Field(None, description="Service coverage area")
    operates_24_7: bool = Field(default=True, description="24/7 availability")
    languages: List[LanguageCode] = Field(default_factory=list, description="Supported languages")
    
    # Additional info
    services: List[str] = Field(default_factory=list, description="Services provided")
    instructions: Optional[str] = Field(None, description="How to contact/what to say")
    translated_instructions: Optional[str] = Field(None, description="Translated instructions")
    response_time_minutes: Optional[int] = Field(None, description="Typical response time")
    
    # Metadata
    verified: bool = Field(default=False, description="Verified contact information")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class EmergencyContactsRequest(BaseModel):
    """Request for emergency contacts."""
    
    location: Coordinates = Field(description="Current location")
    emergency_types: List[EmergencyType] = Field(
        default_factory=list,
        description="Types of emergency contacts needed"
    )
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language")
    radius_meters: int = Field(default=25000, ge=1000, le=100000, description="Search radius")


class EmergencyContactsResponse(BaseResponse):
    """Emergency contacts response."""
    
    contacts: List[EmergencyContact] = Field(description="Emergency contact information")
    location: Coordinates = Field(description="Request location")
    country: Optional[str] = Field(None, description="Current country")
    local_emergency_number: Optional[str] = Field(None, description="Local universal emergency number")
    important_notes: List[str] = Field(default_factory=list, description="Important safety notes")


class SafetyAlert(BaseModel):
    """Safety alert information."""
    
    alert_id: str = Field(description="Unique alert identifier")
    alert_type: AlertType = Field(description="Type of alert")
    severity: SeverityLevel = Field(description="Alert severity level")
    title: str = Field(description="Alert title")
    translated_title: Optional[str] = Field(None, description="Translated alert title")
    
    # Alert content
    description: str = Field(description="Alert description")
    translated_description: Optional[str] = Field(None, description="Translated description")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    translated_recommendations: List[str] = Field(
        default_factory=list,
        description="Translated recommendations"
    )
    
    # Location and timing
    affected_areas: List[str] = Field(default_factory=list, description="Affected areas/regions")
    coordinates: Optional[List[Coordinates]] = Field(None, description="Affected coordinates")
    start_time: Optional[datetime] = Field(None, description="Alert start time")
    end_time: Optional[datetime] = Field(None, description="Alert end time")
    
    # Source and reliability
    source: str = Field(description="Alert source")
    reliability_score: float = Field(ge=0, le=1, description="Source reliability (0-1)")
    verified: bool = Field(default=False, description="Verified alert")
    
    # Metadata
    issued_at: datetime = Field(default_factory=datetime.utcnow, description="Alert issue time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    expires_at: Optional[datetime] = Field(None, description="Alert expiration time")
    
    # User interaction
    user_affected: bool = Field(default=False, description="User is in affected area")
    distance_meters: Optional[float] = Field(None, description="Distance from user location")


class SafetyAlertsRequest(BaseModel):
    """Request for safety alerts."""
    
    location: Coordinates = Field(description="Current location")
    radius_meters: int = Field(default=50000, ge=1000, le=500000, description="Alert search radius")
    alert_types: List[AlertType] = Field(default_factory=list, description="Types of alerts to include")
    min_severity: SeverityLevel = Field(default=SeverityLevel.LOW, description="Minimum severity level")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language")
    include_expired: bool = Field(default=False, description="Include expired alerts")
    max_age_hours: int = Field(default=24, ge=1, le=168, description="Maximum alert age in hours")


class SafetyAlertsResponse(BaseResponse):
    """Safety alerts response."""
    
    alerts: List[SafetyAlert] = Field(description="Safety alerts")
    location: Coordinates = Field(description="Request location")
    active_alerts_count: int = Field(description="Number of active alerts")
    high_priority_count: int = Field(description="Number of high/critical alerts")
    summary: Optional[str] = Field(None, description="Overall safety summary")


class EmergencyReportRequest(BaseModel):
    """Emergency report/SOS request."""
    
    emergency_type: EmergencyType = Field(description="Type of emergency")
    location: Coordinates = Field(description="Emergency location")
    description: str = Field(min_length=10, max_length=1000, description="Emergency description")
    severity: SeverityLevel = Field(description="Severity assessment")
    
    # Reporter information
    reporter_name: Optional[str] = Field(None, description="Reporter name")
    reporter_phone: Optional[str] = Field(None, description="Reporter phone")
    reporter_language: LanguageCode = Field(default=LanguageCode.EN, description="Reporter language")
    
    # Additional context
    people_involved: int = Field(default=1, ge=1, description="Number of people involved")
    injuries: bool = Field(default=False, description="Are there injuries?")
    immediate_danger: bool = Field(default=False, description="Is there immediate danger?")
    assistance_needed: List[str] = Field(default_factory=list, description="Types of assistance needed")
    
    # Media evidence
    photos: List[FileUpload] = Field(default_factory=list, description="Evidence photos")
    audio_recordings: List[FileUpload] = Field(default_factory=list, description="Audio evidence")


class EmergencyReportResponse(BaseResponse):
    """Emergency report response."""
    
    report_id: UUID = Field(description="Emergency report identifier")
    status: str = Field(description="Report status")
    recommended_actions: List[str] = Field(description="Recommended immediate actions")
    emergency_contacts: List[EmergencyContact] = Field(description="Relevant emergency contacts")
    estimated_response_time: Optional[str] = Field(None, description="Estimated response time")
    reference_number: Optional[str] = Field(None, description="Official reference number")
    follow_up_required: bool = Field(default=False, description="Follow-up required")


class SafeZone(BaseModel):
    """Safe zone/area information."""
    
    zone_id: str = Field(description="Safe zone identifier")
    name: str = Field(description="Zone name")
    translated_name: Optional[str] = Field(None, description="Translated zone name")
    zone_type: str = Field(description="Type of safe zone")
    
    # Location
    location: Location = Field(description="Zone location")
    area_coordinates: List[Coordinates] = Field(description="Zone boundary coordinates")
    
    # Safety features
    security_level: int = Field(ge=1, le=5, description="Security level (1-5)")
    safety_features: List[str] = Field(description="Available safety features")
    surveillance: bool = Field(default=False, description="CCTV surveillance available")
    security_personnel: bool = Field(default=False, description="Security personnel present")
    lighting: str = Field(description="Lighting quality")
    
    # Access and timing
    access_hours: Dict[str, str] = Field(description="Access hours by day")
    accessible_24_7: bool = Field(default=False, description="24/7 accessibility")
    entry_requirements: List[str] = Field(default_factory=list, description="Entry requirements")
    
    # Amenities
    facilities: List[str] = Field(default_factory=list, description="Available facilities")
    emergency_services_nearby: bool = Field(default=False, description="Emergency services nearby")
    
    # Ratings and feedback
    safety_rating: Optional[float] = Field(None, ge=0, le=5, description="Safety rating")
    user_reports: int = Field(default=0, description="Number of user safety reports")
    
    # Distance (when from search)
    distance_meters: Optional[float] = Field(None, ge=0, description="Distance from search location")


class SafeZonesRequest(BaseModel):
    """Request for safe zones."""
    
    location: Coordinates = Field(description="Current location")
    radius_meters: int = Field(default=5000, ge=500, le=50000, description="Search radius")
    zone_types: List[str] = Field(default_factory=list, description="Types of safe zones")
    min_security_level: int = Field(default=1, ge=1, le=5, description="Minimum security level")
    accessible_now: bool = Field(default=False, description="Currently accessible zones only")
    emergency_services: bool = Field(default=False, description="Near emergency services")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language")


class SafeZonesResponse(BaseResponse):
    """Safe zones response."""
    
    safe_zones: List[SafeZone] = Field(description="Safe zones found")
    location: Coordinates = Field(description="Search location")
    nearest_zone_distance: Optional[float] = Field(None, description="Distance to nearest zone")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")


class PersonalSafetyRequest(BaseModel):
    """Personal safety assessment request."""
    
    location: Coordinates = Field(description="Current location")
    time_of_day: Optional[datetime] = Field(None, description="Time for assessment")
    activity: Optional[str] = Field(None, description="Planned activity")
    duration_hours: Optional[int] = Field(None, ge=1, description="Expected duration")
    traveling_alone: bool = Field(default=True, description="Traveling alone")
    local_knowledge: int = Field(default=1, ge=1, le=5, description="Local knowledge level (1-5)")
    language_barriers: bool = Field(default=False, description="Language barriers expected")
    special_concerns: List[str] = Field(default_factory=list, description="Special safety concerns")


class PersonalSafetyAssessment(BaseModel):
    """Personal safety assessment result."""
    
    overall_risk_level: SeverityLevel = Field(description="Overall risk assessment")
    risk_factors: List[str] = Field(description="Identified risk factors")
    safety_score: float = Field(ge=0, le=10, description="Safety score (0-10)")
    
    # Recommendations
    precautions: List[str] = Field(description="Recommended precautions")
    alternative_suggestions: List[str] = Field(default_factory=list, description="Alternative suggestions")
    emergency_preparations: List[str] = Field(description="Emergency preparations")
    
    # Contextual information
    local_conditions: Dict[str, Any] = Field(description="Local safety conditions")
    cultural_considerations: List[str] = Field(default_factory=list, description="Cultural considerations")
    legal_considerations: List[str] = Field(default_factory=list, description="Legal considerations")


class PersonalSafetyResponse(BaseResponse):
    """Personal safety assessment response."""
    
    assessment: PersonalSafetyAssessment = Field(description="Safety assessment")
    location: Coordinates = Field(description="Assessment location")
    assessed_at: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    valid_until: datetime = Field(description="Assessment validity period")


class TravelDocumentInfo(BaseModel):
    """Travel document information."""
    
    document_type: str = Field(description="Type of document (passport, visa, etc.)")
    required: bool = Field(description="Document is required")
    description: str = Field(description="Document description")
    translated_description: Optional[str] = Field(None, description="Translated description")
    
    # Validity requirements
    minimum_validity_months: Optional[int] = Field(None, description="Minimum validity period")
    blank_pages_required: Optional[int] = Field(None, description="Required blank pages")
    
    # Process information
    how_to_obtain: List[str] = Field(default_factory=list, description="How to obtain document")
    processing_time: Optional[str] = Field(None, description="Typical processing time")
    cost_range: Optional[str] = Field(None, description="Cost information")
    
    # Additional notes
    special_requirements: List[str] = Field(default_factory=list, description="Special requirements")
    exemptions: List[str] = Field(default_factory=list, description="Exemption conditions")


class TravelDocumentsRequest(BaseModel):
    """Travel documents requirements request."""
    
    origin_country: str = Field(description="Country of origin")
    destination_country: str = Field(description="Destination country")
    nationality: str = Field(description="Traveler nationality")
    trip_purpose: str = Field(default="tourism", description="Purpose of trip")
    duration_days: Optional[int] = Field(None, description="Trip duration in days")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Response language")


class TravelDocumentsResponse(BaseResponse):
    """Travel documents requirements response."""
    
    required_documents: List[TravelDocumentInfo] = Field(description="Required documents")
    recommended_documents: List[TravelDocumentInfo] = Field(description="Recommended documents")
    country_requirements: Dict[str, Any] = Field(description="Country-specific requirements")
    important_notes: List[str] = Field(description="Important travel notes")
    embassy_contacts: List[EmergencyContact] = Field(description="Embassy/consulate contacts")
