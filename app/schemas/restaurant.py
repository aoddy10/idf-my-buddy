"""Restaurant and dining schemas for My Buddy API.

This module contains Pydantic schemas for restaurant search, recommendations,
menu parsing, and dining-related features.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
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
