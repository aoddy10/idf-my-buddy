"""Restaurant and dining schemas for My Buddy API.

This module contains Pydantic schemas for restaurant search, recommendations,
menu parsing, and dining-related features.
"""

from datetime import datetime, time
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, validator

from app.schemas.common import (
    BaseResponse, Coordinates, Location, Address,
    LanguageCode, DietaryRestriction, FileUpload, ProcessedMedia
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
    translated_name: Optional[str] = Field(None, description="Translated restaurant name")
    cuisine_types: List[CuisineType] = Field(description="Types of cuisine served")
    service_types: List[ServiceType] = Field(description="Available service types")
    location: Location = Field(description="Restaurant location")
    
    # Contact and basic info
    phone: Optional[str] = Field(None, description="Restaurant phone number")
    website: Optional[str] = Field(None, description="Restaurant website")
    email: Optional[str] = Field(None, description="Restaurant email")
    
    # Ratings and reviews
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating (0-5)")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    price_range: Optional[PriceRange] = Field(None, description="Price range")
    
    # Operating hours
    opening_hours: Dict[str, str] = Field(default_factory=dict, description="Opening hours by day")
    is_open_now: Optional[bool] = Field(None, description="Currently open status")
    
    # Features and amenities
    features: List[str] = Field(default_factory=list, description="Restaurant features")
    amenities: List[str] = Field(default_factory=list, description="Available amenities")
    accessibility_features: List[str] = Field(default_factory=list, description="Accessibility features")
    
    # Dietary accommodation
    dietary_accommodations: List[DietaryRestriction] = Field(
        default_factory=list,
        description="Supported dietary restrictions"
    )
    
    # Photos and media
    photos: List[str] = Field(default_factory=list, description="Restaurant photo URLs")
    menu_photos: List[str] = Field(default_factory=list, description="Menu photo URLs")
    
    # Distance (when from search)
    distance_meters: Optional[float] = Field(None, ge=0, description="Distance from search location")
    
    # Additional metadata
    description: Optional[str] = Field(None, description="Restaurant description")
    popular_dishes: List[str] = Field(default_factory=list, description="Popular dish names")
    atmosphere: Optional[str] = Field(None, description="Restaurant atmosphere/ambiance")
    dress_code: Optional[str] = Field(None, description="Dress code requirements")
    reservation_required: Optional[bool] = Field(None, description="Reservation requirement")
    accepts_credit_cards: Optional[bool] = Field(None, description="Credit card acceptance")
    
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
    cuisine_types: List[CuisineType] = Field(default_factory=list, description="Cuisine type filters")
    meal_type: Optional[MealType] = Field(None, description="Meal type filter")
    price_range: Optional[PriceRange] = Field(None, description="Price range filter")
    dietary_restrictions: List[DietaryRestriction] = Field(
        default_factory=list,
        description="Required dietary accommodations"
    )
    
    # Preferences
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    open_now: bool = Field(default=False, description="Only show currently open restaurants")
    accessible_only: bool = Field(default=False, description="Only accessible restaurants")
    has_delivery: bool = Field(default=False, description="Delivery available")
    has_takeaway: bool = Field(default=False, description="Takeaway available")
    accepts_reservations: bool = Field(default=False, description="Accepts reservations")
    
    # Search parameters
    query: Optional[str] = Field(None, max_length=100, description="Text search query")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class RestaurantSearchResponse(BaseResponse):
    """Restaurant search response."""
    
    restaurants: List[Restaurant] = Field(description="Found restaurants")
    search_center: Coordinates = Field(description="Search center location")
    search_radius_meters: int = Field(description="Search radius used")
    total_found: int = Field(description="Total restaurants found")
    filters_applied: Dict[str, Any] = Field(description="Applied search filters")
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
    translated_name: Optional[str] = Field(None, description="Translated ingredient name")
    is_allergen: bool = Field(default=False, description="Is a common allergen")
    dietary_flags: List[DietaryRestriction] = Field(
        default_factory=list,
        description="Dietary restrictions this ingredient violates"
    )


class MenuItem(BaseModel):
    """Menu item information."""
    
    item_id: str = Field(description="Unique menu item identifier")
    name: str = Field(description="Item name")
    translated_name: Optional[str] = Field(None, description="Translated item name")
    description: Optional[str] = Field(None, description="Item description")
    translated_description: Optional[str] = Field(None, description="Translated description")
    
    # Categorization
    category: MenuItemCategory = Field(description="Item category")
    cuisine_type: Optional[CuisineType] = Field(None, description="Cuisine type")
    meal_types: List[MealType] = Field(default_factory=list, description="Suitable meal types")
    
    # Pricing
    price: Optional[Decimal] = Field(None, description="Item price")
    currency: Optional[str] = Field(None, description="Price currency")
    price_range: Optional[str] = Field(None, description="Price range indicator")
    
    # Dietary information
    ingredients: List[MenuItemIngredient] = Field(default_factory=list, description="Ingredients")
    allergens: List[str] = Field(default_factory=list, description="Known allergens")
    dietary_compatible: List[DietaryRestriction] = Field(
        default_factory=list,
        description="Compatible dietary restrictions"
    )
    dietary_violations: List[DietaryRestriction] = Field(
        default_factory=list,
        description="Violated dietary restrictions"
    )
    
    # Nutritional info (if available)
    calories: Optional[int] = Field(None, ge=0, description="Calories per serving")
    spice_level: Optional[int] = Field(None, ge=0, le=5, description="Spice level (0-5)")
    
    # Metadata
    is_popular: bool = Field(default=False, description="Popular item")
    is_signature: bool = Field(default=False, description="Signature dish")
    is_seasonal: bool = Field(default=False, description="Seasonal availability")
    availability: Optional[str] = Field(None, description="Availability notes")
    
    # Media
    photo_url: Optional[str] = Field(None, description="Item photo URL")


class Menu(BaseModel):
    """Restaurant menu."""
    
    menu_id: str = Field(description="Unique menu identifier")
    restaurant_id: str = Field(description="Restaurant identifier")
    menu_name: Optional[str] = Field(None, description="Menu name (breakfast, lunch, etc.)")
    items: List[MenuItem] = Field(description="Menu items")
    categories: List[MenuItemCategory] = Field(description="Available categories")
    last_updated: datetime = Field(description="Last menu update")
    language: LanguageCode = Field(description="Menu language")
    currency: Optional[str] = Field(None, description="Menu currency")


class MenuParsingRequest(BaseModel):
    """Menu photo parsing request."""
    
    restaurant_id: Optional[str] = Field(None, description="Restaurant identifier")
    menu_photos: List[FileUpload] = Field(description="Menu photos to parse")
    language_hint: Optional[LanguageCode] = Field(None, description="Expected menu language")
    currency_hint: Optional[str] = Field(None, description="Expected currency")
    parse_prices: bool = Field(default=True, description="Extract price information")
    parse_ingredients: bool = Field(default=True, description="Extract ingredients")
    detect_allergens: bool = Field(default=True, description="Detect allergens")


class MenuParsingResponse(BaseResponse):
    """Menu parsing response."""
    
    parsed_menu: Menu = Field(description="Parsed menu information")
    confidence_score: float = Field(ge=0, le=1, description="Overall parsing confidence")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    detected_language: Optional[LanguageCode] = Field(None, description="Detected menu language")
    detected_currency: Optional[str] = Field(None, description="Detected currency")
    parsing_errors: List[str] = Field(default_factory=list, description="Parsing errors")
    low_confidence_items: List[str] = Field(
        default_factory=list,
        description="Items with low confidence scores"
    )


class DietaryAnalysisRequest(BaseModel):
    """Dietary analysis request for menu items."""
    
    menu_items: List[str] = Field(description="Menu item IDs to analyze")
    dietary_restrictions: List[DietaryRestriction] = Field(
        description="User's dietary restrictions"
    )
    allergen_concerns: List[str] = Field(default_factory=list, description="Specific allergen concerns")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Response language")


class DietaryAnalysisResult(BaseModel):
    """Dietary analysis result for a menu item."""
    
    item_id: str = Field(description="Menu item identifier")
    is_compatible: bool = Field(description="Compatible with dietary restrictions")
    violations: List[str] = Field(default_factory=list, description="Dietary violations found")
    allergen_warnings: List[str] = Field(default_factory=list, description="Allergen warnings")
    safe_alternatives: List[str] = Field(default_factory=list, description="Safe alternative items")
    modification_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested modifications"
    )
    confidence_score: float = Field(ge=0, le=1, description="Analysis confidence")


class DietaryAnalysisResponse(BaseResponse):
    """Dietary analysis response."""
    
    results: List[DietaryAnalysisResult] = Field(description="Analysis results")
    summary: Dict[str, int] = Field(description="Summary statistics")
    recommendations: List[str] = Field(default_factory=list, description="General recommendations")


class ReservationRequest(BaseModel):
    """Restaurant reservation request."""
    
    restaurant_id: str = Field(description="Restaurant identifier")
    party_size: int = Field(ge=1, le=20, description="Number of diners")
    preferred_date: datetime = Field(description="Preferred reservation date/time")
    alternative_times: List[datetime] = Field(
        default_factory=list,
        description="Alternative time preferences"
    )
    special_requests: Optional[str] = Field(None, description="Special requests or notes")
    contact_phone: str = Field(description="Contact phone number")
    contact_email: Optional[str] = Field(None, description="Contact email")
    
    @validator('alternative_times')
    def validate_alternative_times(cls, v):
        """Limit alternative times."""
        if len(v) > 5:
            raise ValueError("Maximum 5 alternative times allowed")
        return v


class ReservationResponse(BaseResponse):
    """Reservation response."""
    
    reservation_id: Optional[str] = Field(None, description="Reservation confirmation ID")
    status: str = Field(description="Reservation status")
    confirmed_datetime: Optional[datetime] = Field(None, description="Confirmed date/time")
    restaurant_contact: Optional[str] = Field(None, description="Restaurant contact info")
    special_instructions: Optional[str] = Field(None, description="Special instructions")
    cancellation_policy: Optional[str] = Field(None, description="Cancellation policy")
