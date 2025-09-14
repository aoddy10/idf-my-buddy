"""Restaurant and dining schemas for My Buddy API.

This module contains Pydantic schemas for restaurant search, recommendations,
menu parsing, and dining-related features.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator

from app.schemas.common import (
    BaseResponse,
    Coordinates,
    DietaryRestriction,
    FileUpload,
    LanguageCode,
    Location,
)


class CuisineType(str, Enum):
    """Types of cuisine."""
    THAI = "thai"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    INDIAN = "indian"
    ITALIAN = "italian"
    FRENCH = "french"
    AMERICAN = "american"
    MEXICAN = "mexican"
    MEDITERRANEAN = "mediterranean"
    MIDDLE_EASTERN = "middle_eastern"
    VIETNAMESE = "vietnamese"
    INDONESIAN = "indonesian"
    GERMAN = "german"
    SPANISH = "spanish"
    GREEK = "greek"
    TURKISH = "turkish"
    BRAZILIAN = "brazilian"
    AFRICAN = "african"
    FUSION = "fusion"
    INTERNATIONAL = "international"


class MealType(str, Enum):
    """Types of meals."""
    BREAKFAST = "breakfast"
    BRUNCH = "brunch"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    DESSERT = "dessert"
    DRINKS = "drinks"


class ServiceType(str, Enum):
    """Restaurant service types."""
    DINE_IN = "dine_in"
    TAKEAWAY = "takeaway"
    DELIVERY = "delivery"
    BUFFET = "buffet"
    FAST_FOOD = "fast_food"
    FINE_DINING = "fine_dining"
    CASUAL_DINING = "casual_dining"
    FOOD_TRUCK = "food_truck"
    STREET_FOOD = "street_food"


class PriceRange(str, Enum):
    """Restaurant price ranges."""
    BUDGET = "budget"        # $ (under $10)
    MODERATE = "moderate"    # $$ ($10-30)
    EXPENSIVE = "expensive"  # $$$ ($30-60)
    LUXURY = "luxury"        # $$$$ ($60+)


class Restaurant(BaseModel):
    """Restaurant information schema."""

    restaurant_id: str = Field(description="Unique restaurant identifier")
    name: str = Field(description="Restaurant name")
    translated_name: str | None = Field(None, description="Translated restaurant name")
    cuisine_types: list[CuisineType] = Field(description="Types of cuisine served")
    service_types: list[ServiceType] = Field(description="Available service types")
    location: Location = Field(description="Restaurant location")

    # Contact and basic info
    phone: str | None = Field(None, description="Restaurant phone number")
    website: str | None = Field(None, description="Restaurant website")
    email: str | None = Field(None, description="Restaurant email")

    # Ratings and reviews
    rating: float | None = Field(None, ge=0, le=5, description="Average rating (0-5)")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    price_range: PriceRange | None = Field(None, description="Price range")

    # Operating hours
    opening_hours: dict[str, str] = Field(default_factory=dict, description="Opening hours by day")
    is_open_now: bool | None = Field(None, description="Currently open status")

    # Features and amenities
    features: list[str] = Field(default_factory=list, description="Restaurant features")
    amenities: list[str] = Field(default_factory=list, description="Available amenities")
    accessibility_features: list[str] = Field(default_factory=list, description="Accessibility features")

    # Dietary accommodation
    dietary_accommodations: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Supported dietary restrictions"
    )

    # Photos and media
    photos: list[str] = Field(default_factory=list, description="Restaurant photo URLs")
    menu_photos: list[str] = Field(default_factory=list, description="Menu photo URLs")

    # Distance (when from search)
    distance_meters: float | None = Field(None, ge=0, description="Distance from search location")

    # Additional metadata
    description: str | None = Field(None, description="Restaurant description")
    popular_dishes: list[str] = Field(default_factory=list, description="Popular dish names")
    atmosphere: str | None = Field(None, description="Restaurant atmosphere/ambiance")
    dress_code: str | None = Field(None, description="Dress code requirements")
    reservation_required: bool | None = Field(None, description="Reservation requirement")
    accepts_credit_cards: bool | None = Field(None, description="Credit card acceptance")

    @validator('cuisine_types')
    def validate_cuisine_types(cls, v):
        """Validate cuisine types list."""
        if not v:
            raise ValueError("At least one cuisine type is required")
        return v


class RestaurantSearchRequest(BaseModel):
    """Restaurant search request schema."""

    location: Coordinates = Field(description="Search location")
    radius_meters: int = Field(default=2000, ge=100, le=25000, description="Search radius")

    # Filters
    cuisine_types: list[CuisineType] = Field(default_factory=list, description="Cuisine type filters")
    meal_type: MealType | None = Field(None, description="Meal type filter")
    price_range: PriceRange | None = Field(None, description="Price range filter")
    dietary_restrictions: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Required dietary accommodations"
    )

    # Preferences
    min_rating: float | None = Field(None, ge=0, le=5, description="Minimum rating")
    open_now: bool = Field(default=False, description="Only show currently open restaurants")
    accessible_only: bool = Field(default=False, description="Only accessible restaurants")
    has_delivery: bool = Field(default=False, description="Delivery available")
    has_takeaway: bool = Field(default=False, description="Takeaway available")
    accepts_reservations: bool = Field(default=False, description="Accepts reservations")

    # Search parameters
    query: str | None = Field(None, max_length=100, description="Text search query")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class RestaurantSearchResponse(BaseResponse):
    """Restaurant search response."""

    restaurants: list[Restaurant] = Field(description="Found restaurants")
    search_center: Coordinates = Field(description="Search center location")
    search_radius_meters: int = Field(description="Search radius used")
    total_found: int = Field(description="Total restaurants found")
    filters_applied: dict[str, Any] = Field(description="Applied search filters")
    query_id: UUID = Field(description="Search query identifier")


class MenuItemCategory(str, Enum):
    """Menu item categories."""
    APPETIZER = "appetizer"
    SOUP = "soup"
    SALAD = "salad"
    MAIN_COURSE = "main_course"
    SIDE_DISH = "side_dish"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    ALCOHOL = "alcohol"
    SPECIAL = "special"


class MenuItemIngredient(BaseModel):
    """Menu item ingredient."""

    name: str = Field(description="Ingredient name")
    translated_name: str | None = Field(None, description="Translated ingredient name")
    is_allergen: bool = Field(default=False, description="Is a common allergen")
    dietary_flags: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Dietary restrictions this ingredient violates"
    )


class MenuItem(BaseModel):
    """Menu item information."""

    item_id: str = Field(description="Unique menu item identifier")
    name: str = Field(description="Item name")
    translated_name: str | None = Field(None, description="Translated item name")
    description: str | None = Field(None, description="Item description")
    translated_description: str | None = Field(None, description="Translated description")

    # Categorization
    category: MenuItemCategory = Field(description="Item category")
    cuisine_type: CuisineType | None = Field(None, description="Cuisine type")
    meal_types: list[MealType] = Field(default_factory=list, description="Suitable meal types")

    # Pricing
    price: Decimal | None = Field(None, description="Item price")
    currency: str | None = Field(None, description="Price currency")
    price_range: str | None = Field(None, description="Price range indicator")

    # Dietary information
    ingredients: list[MenuItemIngredient] = Field(default_factory=list, description="Ingredients")
    allergens: list[str] = Field(default_factory=list, description="Known allergens")
    dietary_compatible: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Compatible dietary restrictions"
    )
    dietary_violations: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Violated dietary restrictions"
    )

    # Nutritional info (if available)
    calories: int | None = Field(None, ge=0, description="Calories per serving")
    spice_level: int | None = Field(None, ge=0, le=5, description="Spice level (0-5)")

    # Metadata
    is_popular: bool = Field(default=False, description="Popular item")
    is_signature: bool = Field(default=False, description="Signature dish")
    is_seasonal: bool = Field(default=False, description="Seasonal availability")
    availability: str | None = Field(None, description="Availability notes")

    # Media
    photo_url: str | None = Field(None, description="Item photo URL")


class Menu(BaseModel):
    """Restaurant menu."""

    menu_id: str = Field(description="Unique menu identifier")
    restaurant_id: str = Field(description="Restaurant identifier")
    menu_name: str | None = Field(None, description="Menu name (breakfast, lunch, etc.)")
    items: list[MenuItem] = Field(description="Menu items")
    categories: list[MenuItemCategory] = Field(description="Available categories")
    last_updated: datetime = Field(description="Last menu update")
    language: LanguageCode = Field(description="Menu language")
    currency: str | None = Field(None, description="Menu currency")


class MenuParsingRequest(BaseModel):
    """Menu photo parsing request."""

    restaurant_id: str | None = Field(None, description="Restaurant identifier")
    menu_photos: list[FileUpload] = Field(description="Menu photos to parse")
    language_hint: LanguageCode | None = Field(None, description="Expected menu language")
    currency_hint: str | None = Field(None, description="Expected currency")
    parse_prices: bool = Field(default=True, description="Extract price information")
    parse_ingredients: bool = Field(default=True, description="Extract ingredients")
    detect_allergens: bool = Field(default=True, description="Detect allergens")


class MenuParsingResponse(BaseResponse):
    """Menu parsing response."""

    parsed_menu: Menu = Field(description="Parsed menu information")
    confidence_score: float = Field(ge=0, le=1, description="Overall parsing confidence")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    detected_language: LanguageCode | None = Field(None, description="Detected menu language")
    detected_currency: str | None = Field(None, description="Detected currency")
    parsing_errors: list[str] = Field(default_factory=list, description="Parsing errors")
    low_confidence_items: list[str] = Field(
        default_factory=list,
        description="Items with low confidence scores"
    )


class DietaryAnalysisRequest(BaseModel):
    """Dietary analysis request for menu items."""

    menu_items: list[str] = Field(description="Menu item IDs to analyze")
    dietary_restrictions: list[DietaryRestriction] = Field(
        description="User's dietary restrictions"
    )
    allergen_concerns: list[str] = Field(default_factory=list, description="Specific allergen concerns")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Response language")


class DietaryAnalysisResult(BaseModel):
    """Dietary analysis result for a menu item."""

    item_id: str = Field(description="Menu item identifier")
    is_compatible: bool = Field(description="Compatible with dietary restrictions")
    violations: list[str] = Field(default_factory=list, description="Dietary violations found")
    allergen_warnings: list[str] = Field(default_factory=list, description="Allergen warnings")
    safe_alternatives: list[str] = Field(default_factory=list, description="Safe alternative items")
    modification_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested modifications"
    )
    confidence_score: float = Field(ge=0, le=1, description="Analysis confidence")


class DietaryAnalysisResponse(BaseResponse):
    """Dietary analysis response."""

    results: list[DietaryAnalysisResult] = Field(description="Analysis results")
    summary: dict[str, int] = Field(description="Summary statistics")
    recommendations: list[str] = Field(default_factory=list, description="General recommendations")


class ReservationRequest(BaseModel):
    """Restaurant reservation request."""

    restaurant_id: str = Field(description="Restaurant identifier")
    party_size: int = Field(ge=1, le=20, description="Number of diners")
    preferred_date: datetime = Field(description="Preferred reservation date/time")
    alternative_times: list[datetime] = Field(
        default_factory=list,
        description="Alternative time preferences"
    )
    special_requests: str | None = Field(None, description="Special requests or notes")
    contact_phone: str = Field(description="Contact phone number")
    contact_email: str | None = Field(None, description="Contact email")

    @validator('alternative_times')
    def validate_alternative_times(cls, v):
        """Limit alternative times."""
        if len(v) > 5:
            raise ValueError("Maximum 5 alternative times allowed")
        return v


class ReservationResponse(BaseResponse):
    """Reservation response."""

    reservation_id: str | None = Field(None, description="Reservation confirmation ID")
    status: str = Field(description="Reservation status")
    confirmed_datetime: datetime | None = Field(None, description="Confirmed date/time")
    restaurant_contact: str | None = Field(None, description="Restaurant contact info")
    special_instructions: str | None = Field(None, description="Special instructions")
    cancellation_policy: str | None = Field(None, description="Cancellation policy")


# ============================================================================
# Restaurant Intelligence Schemas (Menu OCR, Allergen Detection, Voice)
# ============================================================================

class DishCategory(str, Enum):
    """Menu item categories."""
    APPETIZER = "appetizer"
    SOUP = "soup"
    SALAD = "salad"
    MAIN_COURSE = "main_course"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SNACK = "snack"
    SPECIAL = "special"
    UNKNOWN = "unknown"


class AllergenType(str, Enum):
    """Allergen types for detection."""
    MILK_DAIRY = "milk_dairy"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT_GLUTEN = "wheat_gluten"
    SOYBEANS = "soybeans"
    SESAME = "sesame"
    SULFITES = "sulfites"
    MOLLUSCS = "molluscs"
    CELERY = "celery"
    MUSTARD = "mustard"
    LUPIN = "lupin"
    OTHER = "other"


class AllergenRiskLevel(str, Enum):
    """Risk levels for allergen warnings."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MenuItemAnalysis(BaseModel):
    """Analysis result for a single menu item."""
    item_name: str = Field(description="Name of the menu item")
    original_text: str = Field(description="Original text from menu")
    translated_name: str | None = Field(None, description="Translated item name")
    category: DishCategory = Field(description="Classified dish category")
    estimated_price: str | None = Field(None, description="Extracted price")
    price_currency: str | None = Field(None, description="Currency symbol")
    description: str | None = Field(None, description="Item description")
    ingredients: list[str] = Field(default_factory=list, description="Identified ingredients")
    allergen_warnings: list[str] = Field(default_factory=list, description="Allergen warnings")
    allergen_risk_level: str = Field(default="unknown", description="Risk assessment")
    dietary_tags: list[str] = Field(default_factory=list, description="Dietary tags")
    spice_level: str | None = Field(None, description="Spice level indicator")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Analysis confidence")


class MenuAnalysisRequest(BaseModel):
    """Request for menu image analysis."""
    user_language: str = Field(default="en", description="User's preferred language")
    include_allergen_warnings: bool = Field(default=True, description="Include allergen detection")
    target_currency: str | None = Field(None, description="Target currency for conversion")
    user_allergens: list[str] | None = Field(None, description="User's known allergies")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="OCR confidence threshold")


class MenuAnalysisResponse(BaseResponse):
    """Response from menu image analysis."""
    restaurant_name: str | None = Field(None, description="Detected restaurant name")
    cuisine_type: str = Field(description="Detected cuisine type")
    detected_language: str = Field(description="Menu language detected")
    currency: str | None = Field(None, description="Currency found in menu")
    menu_items: list[MenuItemAnalysis] = Field(description="Analyzed menu items")
    categories_found: list[str] = Field(description="Menu categories discovered")
    price_range: dict[str, float | None] = Field(description="Price range analysis")
    allergen_summary: dict[str, int] = Field(description="Allergen occurrence counts")
    processing_time: float = Field(description="Analysis processing time")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    recommendations: list[str] = Field(description="Personalized recommendations")


class DishExplanationRequest(BaseModel):
    """Request for dish explanation."""
    dish_name: str = Field(min_length=1, max_length=200, description="Name of dish to explain")
    cuisine_type: str | None = Field(None, description="Type of cuisine for context")
    user_language: str = Field(default="en", description="User's preferred language")
    include_audio: bool = Field(default=False, description="Generate audio explanation")
    cultural_context: str | None = Field(None, description="User's cultural background")


class DishExplanationResponse(BaseResponse):
    """Response for dish explanation request."""
    explanation: str = Field(..., description="Detailed dish explanation")
    cuisine_type: CuisineType = Field(..., description="Detected cuisine type")
    confidence: float = Field(..., description="Explanation confidence (0-1)")
    ingredients: List[str] = Field(default_factory=list, description="Key ingredients")
    allergen_warnings: List[str] = Field(default_factory=list, description="Allergen warnings")
    cultural_notes: Optional[str] = Field(None, description="Cultural significance")
    preparation_tips: List[str] = Field(default_factory=list, description="Preparation advice")


class VoiceExplanationRequest(BaseModel):
    """Request for voice dish explanation."""
    dish_name: str = Field(..., description="Name of dish to explain")
    language: LanguageCode = Field(default="en", description="Response language")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice synthesis speed")
    include_cultural_context: bool = Field(default=True, description="Include cultural information")


class VoiceExplanationResponse(BaseResponse):
    """Response for voice dish explanation."""
    dish_name: str = Field(..., description="Name of explained dish")
    text_explanation: str = Field(..., description="Text version of explanation")
    audio_available: bool = Field(..., description="Whether audio was generated")
    audio_duration_seconds: Optional[float] = Field(None, description="Audio duration")
    language: LanguageCode = Field(..., description="Response language")
    authenticity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dish authenticity score")


class VoiceCommandRequest(BaseModel):
    """Request for voice command processing."""
    command_language: LanguageCode = Field(default="en", description="Command language")
    context: Optional[str] = Field(None, description="Additional context")


class VoiceCommandResponse(BaseResponse):
    """Response for voice command processing."""
    transcribed_text: str = Field(..., description="Transcribed voice command")
    command_type: str = Field(..., description="Detected command type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")
    response_text: str = Field(..., description="Response to command")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Command recognition confidence")


class AllergenMatch(BaseModel):
    """Details of an allergen match."""
    allergen_type: AllergenType = Field(description="Type of allergen detected")
    matched_text: str = Field(description="Text that matched allergen pattern")
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence score")
    position: int = Field(ge=0, description="Position in text where found")
    severity_notes: str | None = Field(None, description="Additional severity information")


class AllergenCheckRequest(BaseModel):
    """Request for allergen checking."""
    text: str = Field(min_length=1, description="Text to analyze for allergens")
    language: str = Field(default="en", description="Text language")
    user_allergens: list[str] | None = Field(None, description="User's specific allergens")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Detection threshold")
    include_cross_contamination: bool = Field(default=True, description="Check cross-contamination risks")


class AllergenCheckResponse(BaseResponse):
    """Response from allergen analysis."""
    text_analyzed: str = Field(description="Text that was analyzed")
    language: str = Field(description="Language of analyzed text")
    detected_allergens: list[AllergenMatch] = Field(description="Found allergen matches")
    risk_level: AllergenRiskLevel = Field(description="Overall risk assessment")
    safety_warnings: list[str] = Field(description="Safety warnings and precautions")
    user_specific_warnings: list[str] = Field(description="Warnings specific to user")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    processing_time: float = Field(description="Detection processing time")
    recommendations: list[str] = Field(description="Safety recommendations")
    cross_contamination_risks: list[str] = Field(default_factory=list, description="Cross-contamination warnings")


class VoiceMenuRequest(BaseModel):
    """Request for voice menu assistance."""
    query: str = Field(min_length=1, description="Voice query or question")
    language: str = Field(default="en", description="User's language")
    menu_context: str | None = Field(None, description="Current menu context")
    user_preferences: dict[str, Any] = Field(default_factory=dict, description="User dining preferences")
    conversation_history: list[str] = Field(default_factory=list, description="Previous conversation")


class VoiceMenuResponse(BaseResponse):
    """Response from voice menu assistance."""
    query: str = Field(description="Original user query")
    response_text: str = Field(description="Text response to query")
    language: str = Field(description="Response language")
    audio_available: bool = Field(default=False, description="Whether audio response available")
    audio_format: str | None = Field(None, description="Audio format")
    audio_duration: float | None = Field(None, description="Audio duration")
    suggested_items: list[str] = Field(default_factory=list, description="Suggested menu items")
    follow_up_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")
    processing_time: float = Field(description="Voice processing time")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Response confidence")


class RestaurantIntelligenceStatus(BaseModel):
    """Status of restaurant intelligence services."""
    ocr_available: bool = Field(description="OCR service availability")
    translation_available: bool = Field(description="Translation service availability")
    allergen_detection_available: bool = Field(description="Allergen detection availability")
    tts_available: bool = Field(description="Text-to-speech availability")
    supported_languages: list[str] = Field(description="Supported languages")
    supported_cuisines: list[str] = Field(description="Supported cuisine types")
    max_image_size: int = Field(description="Maximum image size in bytes")
    processing_statistics: dict[str, Any] = Field(description="Usage statistics")


class MenuScanSession(BaseModel):
    """Menu scanning session information."""
    session_id: str = Field(description="Unique session identifier")
    user_id: str = Field(description="User identifier")
    restaurant_id: str | None = Field(None, description="Restaurant identifier if known")
    image_url: str = Field(description="Stored menu image URL")
    status: str = Field(description="Processing status")
    created_at: datetime = Field(description="Session creation time")
    results: MenuAnalysisResponse | None = Field(None, description="Analysis results")
    error_message: str | None = Field(None, description="Error message if failed")


class UserAllergenProfile(BaseModel):
    """User's allergen profile."""
    user_id: str = Field(description="User identifier")
    allergens: list[AllergenType] = Field(description="User's allergens")
    severity_levels: dict[str, AllergenRiskLevel] = Field(description="Severity per allergen")
    notes: dict[str, str] = Field(default_factory=dict, description="Additional notes per allergen")
    emergency_contact: str | None = Field(None, description="Emergency contact information")
    medical_alert: bool = Field(default=False, description="Medical alert requirement")
    last_updated: datetime = Field(description="Profile last update time")


class DiningRecommendation(BaseModel):
    """Personalized dining recommendation."""
    item_name: str = Field(description="Recommended menu item")
    reason: str = Field(description="Why this item is recommended")
    safety_score: float = Field(ge=0.0, le=1.0, description="Safety score for user")
    match_score: float = Field(ge=0.0, le=1.0, description="Preference match score")
    warnings: list[str] = Field(default_factory=list, description="Any warnings or cautions")
    alternatives: list[str] = Field(default_factory=list, description="Alternative options")
    cultural_notes: str | None = Field(None, description="Cultural context for item")
