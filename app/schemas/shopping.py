"""Shopping schemas for My Buddy API.

This module contains Pydantic schemas for shopping-related endpoints,
including product search, price comparison, and shopping assistance.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import (
    BaseResponse,
    Coordinates,
    FileUpload,
    LanguageCode,
    Location,
)


class ProductCategory(str, Enum):
    """Product categories for shopping."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    SHOES = "shoes"
    ACCESSORIES = "accessories"
    BEAUTY = "beauty"
    HEALTH = "health"
    FOOD = "food"
    BEVERAGES = "beverages"
    HOME = "home"
    FURNITURE = "furniture"
    BOOKS = "books"
    SPORTS = "sports"
    TOYS = "toys"
    AUTOMOTIVE = "automotive"
    JEWELRY = "jewelry"
    SOUVENIRS = "souvenirs"
    GIFTS = "gifts"
    TRAVEL_GEAR = "travel_gear"
    PHARMACY = "pharmacy"
    TECHNOLOGY = "technology"


class ShopType(str, Enum):
    """Types of shopping venues."""
    DEPARTMENT_STORE = "department_store"
    SHOPPING_MALL = "shopping_mall"
    BOUTIQUE = "boutique"
    MARKET = "market"
    STREET_MARKET = "street_market"
    FLOATING_MARKET = "floating_market"
    NIGHT_MARKET = "night_market"
    SUPERMARKET = "supermarket"
    CONVENIENCE_STORE = "convenience_store"
    SPECIALTY_STORE = "specialty_store"
    OUTLET = "outlet"
    DUTY_FREE = "duty_free"
    PHARMACY = "pharmacy"
    ELECTRONICS_STORE = "electronics_store"
    BOOKSTORE = "bookstore"
    SOUVENIR_SHOP = "souvenir_shop"


class PriceRange(str, Enum):
    """Price ranges for products/shops."""
    BUDGET = "budget"        # $
    MODERATE = "moderate"    # $$
    EXPENSIVE = "expensive"  # $$$
    LUXURY = "luxury"        # $$$$


class PaymentMethod(str, Enum):
    """Accepted payment methods."""
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOBILE_PAYMENT = "mobile_payment"
    DIGITAL_WALLET = "digital_wallet"
    TRAVELERS_CHECK = "travelers_check"
    FOREIGN_CURRENCY = "foreign_currency"


class Shop(BaseModel):
    """Shopping venue information."""

    shop_id: str = Field(description="Unique shop identifier")
    name: str = Field(description="Shop name")
    translated_name: str | None = Field(None, description="Translated shop name")
    shop_type: ShopType = Field(description="Type of shop")
    categories: list[ProductCategory] = Field(description="Product categories available")
    location: Location = Field(description="Shop location")

    # Contact information
    phone: str | None = Field(None, description="Shop phone number")
    website: str | None = Field(None, description="Shop website")

    # Operating details
    opening_hours: dict[str, str] = Field(default_factory=dict, description="Opening hours by day")
    is_open_now: bool | None = Field(None, description="Currently open status")

    # Ratings and pricing
    rating: float | None = Field(None, ge=0, le=5, description="Average rating (0-5)")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    price_range: PriceRange | None = Field(None, description="General price range")

    # Payment and services
    payment_methods: list[PaymentMethod] = Field(default_factory=list, description="Accepted payment methods")
    services: list[str] = Field(default_factory=list, description="Additional services offered")
    features: list[str] = Field(default_factory=list, description="Shop features")

    # Tax and tourist services
    tax_free_shopping: bool = Field(default=False, description="Offers tax-free shopping")
    tourist_discounts: bool = Field(default=False, description="Tourist discounts available")
    multilingual_staff: bool = Field(default=False, description="Multilingual staff available")

    # Distance and photos
    distance_meters: float | None = Field(None, ge=0, description="Distance from search location")
    photos: list[str] = Field(default_factory=list, description="Shop photo URLs")

    # Additional info
    description: str | None = Field(None, description="Shop description")
    popular_products: list[str] = Field(default_factory=list, description="Popular product types")
    accessibility_features: list[str] = Field(default_factory=list, description="Accessibility features")


class Product(BaseModel):
    """Product information."""

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    translated_name: str | None = Field(None, description="Translated product name")
    brand: str | None = Field(None, description="Product brand")
    category: ProductCategory = Field(description="Product category")

    # Pricing
    price: Decimal | None = Field(None, description="Product price")
    original_price: Decimal | None = Field(None, description="Original price (before discount)")
    currency: str = Field(description="Price currency")
    discount_percentage: float | None = Field(None, ge=0, le=100, description="Discount percentage")

    # Product details
    description: str | None = Field(None, description="Product description")
    translated_description: str | None = Field(None, description="Translated description")
    specifications: dict[str, str] = Field(default_factory=dict, description="Product specifications")

    # Availability
    in_stock: bool = Field(default=True, description="Product in stock")
    stock_level: str | None = Field(None, description="Stock level indicator")
    sizes_available: list[str] = Field(default_factory=list, description="Available sizes")
    colors_available: list[str] = Field(default_factory=list, description="Available colors")

    # Reviews and ratings
    rating: float | None = Field(None, ge=0, le=5, description="Product rating")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")

    # Images and media
    images: list[str] = Field(default_factory=list, description="Product image URLs")

    # Shop information
    shop_id: str = Field(description="Shop selling this product")
    shop_name: str | None = Field(None, description="Shop name")

    # Metadata
    is_popular: bool = Field(default=False, description="Popular product")
    is_local_specialty: bool = Field(default=False, description="Local specialty item")
    is_tourist_favorite: bool = Field(default=False, description="Tourist favorite")
    tags: list[str] = Field(default_factory=list, description="Product tags")


class ShopSearchRequest(BaseModel):
    """Shop search request schema."""

    location: Coordinates = Field(description="Search location")
    radius_meters: int = Field(default=2000, ge=100, le=25000, description="Search radius")

    # Filters
    shop_types: list[ShopType] = Field(default_factory=list, description="Shop type filters")
    categories: list[ProductCategory] = Field(default_factory=list, description="Product category filters")
    price_range: PriceRange | None = Field(None, description="Price range filter")

    # Preferences
    min_rating: float | None = Field(None, ge=0, le=5, description="Minimum rating")
    open_now: bool = Field(default=False, description="Only show currently open shops")
    tax_free_only: bool = Field(default=False, description="Only tax-free shops")
    tourist_discounts: bool = Field(default=False, description="Shops with tourist discounts")
    accepts_cards: bool = Field(default=False, description="Accepts credit/debit cards")
    multilingual_staff: bool = Field(default=False, description="Has multilingual staff")

    # Search parameters
    query: str | None = Field(None, max_length=100, description="Text search query")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class ShopSearchResponse(BaseResponse):
    """Shop search response."""

    shops: list[Shop] = Field(description="Found shops")
    search_center: Coordinates = Field(description="Search center location")
    search_radius_meters: int = Field(description="Search radius used")
    total_found: int = Field(description="Total shops found")
    filters_applied: dict[str, Any] = Field(description="Applied search filters")
    query_id: UUID = Field(description="Search query identifier")


class ProductSearchRequest(BaseModel):
    """Product search request schema."""

    query: str = Field(min_length=1, max_length=200, description="Product search query")
    location: Coordinates | None = Field(None, description="Search location for local results")
    radius_meters: int | None = Field(None, ge=100, le=25000, description="Search radius")

    # Filters
    category: ProductCategory | None = Field(None, description="Product category filter")
    min_price: Decimal | None = Field(None, ge=0, description="Minimum price filter")
    max_price: Decimal | None = Field(None, ge=0, description="Maximum price filter")
    currency: str | None = Field(None, description="Currency for price filters")
    brand: str | None = Field(None, description="Brand filter")

    # Availability filters
    in_stock_only: bool = Field(default=True, description="Only show in-stock products")
    size: str | None = Field(None, description="Size filter")
    color: str | None = Field(None, description="Color filter")

    # Quality filters
    min_rating: float | None = Field(None, ge=0, le=5, description="Minimum rating")
    popular_only: bool = Field(default=False, description="Only popular products")
    local_specialties: bool = Field(default=False, description="Include local specialties")

    # Search parameters
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=30, ge=1, le=100, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class ProductSearchResponse(BaseResponse):
    """Product search response."""

    products: list[Product] = Field(description="Found products")
    search_query: str = Field(description="Original search query")
    total_found: int = Field(description="Total products found")
    filters_applied: dict[str, Any] = Field(description="Applied search filters")
    query_id: UUID = Field(description="Search query identifier")


class PriceComparisonRequest(BaseModel):
    """Price comparison request schema."""

    product_name: str = Field(description="Product name to compare")
    location: Coordinates = Field(description="Location for comparison")
    radius_meters: int = Field(default=5000, ge=500, le=25000, description="Comparison radius")
    category: ProductCategory | None = Field(None, description="Product category")
    specifications: dict[str, str] = Field(default_factory=dict, description="Product specifications")
    max_shops: int = Field(default=10, ge=1, le=20, description="Maximum shops to compare")


class PriceComparison(BaseModel):
    """Price comparison for a product across shops."""

    product_name: str = Field(description="Product name")
    shops: list[dict[str, Any]] = Field(description="Shops with pricing information")
    price_range: dict[str, Decimal] = Field(description="Price range (min, max, average)")
    best_deal: dict[str, Any] | None = Field(None, description="Best deal found")
    savings_opportunity: Decimal | None = Field(None, description="Maximum potential savings")
    currency: str = Field(description="Currency for prices")


class PriceComparisonResponse(BaseResponse):
    """Price comparison response."""

    comparison: PriceComparison = Field(description="Price comparison results")
    search_location: Coordinates = Field(description="Search center location")
    comparison_date: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")
    recommendations: list[str] = Field(default_factory=list, description="Shopping recommendations")


class ProductRecognitionRequest(BaseModel):
    """Product recognition from image request."""

    image: FileUpload = Field(description="Product image to analyze")
    location: Coordinates | None = Field(None, description="User location for local results")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    find_similar: bool = Field(default=True, description="Find similar products")
    price_comparison: bool = Field(default=True, description="Include price comparison")


class ProductRecognitionResponse(BaseResponse):
    """Product recognition response."""

    recognized_products: list[Product] = Field(description="Recognized products")
    similar_products: list[Product] = Field(default_factory=list, description="Similar products")
    confidence_score: float = Field(ge=0, le=1, description="Recognition confidence")
    processing_time_ms: int = Field(description="Processing time")
    price_comparison: PriceComparison | None = Field(None, description="Price comparison if requested")


class ShoppingListItem(BaseModel):
    """Shopping list item."""

    item_id: UUID = Field(description="Unique item identifier")
    name: str = Field(description="Item name")
    category: ProductCategory | None = Field(None, description="Item category")
    quantity: int = Field(default=1, ge=1, description="Quantity needed")
    notes: str | None = Field(None, description="Additional notes")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    estimated_price: Decimal | None = Field(None, description="Estimated price")
    found: bool = Field(default=False, description="Item found/purchased")
    shop_found: str | None = Field(None, description="Shop where item was found")
    actual_price: Decimal | None = Field(None, description="Actual price paid")


class ShoppingList(BaseModel):
    """User shopping list."""

    list_id: UUID = Field(description="Unique list identifier")
    name: str = Field(description="List name")
    user_id: UUID = Field(description="User identifier")
    items: list[ShoppingListItem] = Field(description="Shopping list items")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="List creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    budget: Decimal | None = Field(None, description="Shopping budget")
    currency: str | None = Field(None, description="Budget currency")


class ShoppingRecommendationRequest(BaseModel):
    """Shopping recommendation request."""

    location: Coordinates = Field(description="User location")
    budget: Decimal | None = Field(None, description="Shopping budget")
    currency: str | None = Field(None, description="Budget currency")
    interests: list[ProductCategory] = Field(default_factory=list, description="Shopping interests")
    time_available: int | None = Field(None, description="Available shopping time (minutes)")
    trip_purpose: str | None = Field(None, description="Trip purpose (leisure, business, etc.)")
    shopping_style: str | None = Field(None, description="Shopping preference")
    looking_for_souvenirs: bool = Field(default=False, description="Looking for souvenirs")
    local_specialties: bool = Field(default=False, description="Interested in local specialties")


class ShoppingRecommendation(BaseModel):
    """Shopping recommendation."""

    recommended_shops: list[Shop] = Field(description="Recommended shops")
    suggested_products: list[Product] = Field(description="Suggested products")
    shopping_route: dict[str, Any] | None = Field(None, description="Optimized shopping route")
    budget_breakdown: dict[str, Decimal] | None = Field(None, description="Suggested budget allocation")
    tips: list[str] = Field(default_factory=list, description="Shopping tips and advice")


class ShoppingRecommendationResponse(BaseResponse):
    """Shopping recommendation response."""

    recommendations: ShoppingRecommendation = Field(description="Shopping recommendations")
    location: Coordinates = Field(description="User location")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Recommendation timestamp")
