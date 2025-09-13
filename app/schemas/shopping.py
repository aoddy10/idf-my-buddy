"""Shopping schemas for My Buddy API.

This module contains Pydantic schemas for shopping-related endpoints,
including product search, price comparison, and shopping assistance.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, validator

from app.schemas.common import (
    BaseResponse, Coordinates, Location, Address,
    LanguageCode, FileUpload, ProcessedMedia
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
    translated_name: Optional[str] = Field(None, description="Translated shop name")
    shop_type: ShopType = Field(description="Type of shop")
    categories: List[ProductCategory] = Field(description="Product categories available")
    location: Location = Field(description="Shop location")
    
    # Contact information
    phone: Optional[str] = Field(None, description="Shop phone number")
    website: Optional[str] = Field(None, description="Shop website")
    
    # Operating details
    opening_hours: Dict[str, str] = Field(default_factory=dict, description="Opening hours by day")
    is_open_now: Optional[bool] = Field(None, description="Currently open status")
    
    # Ratings and pricing
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating (0-5)")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    price_range: Optional[PriceRange] = Field(None, description="General price range")
    
    # Payment and services
    payment_methods: List[PaymentMethod] = Field(default_factory=list, description="Accepted payment methods")
    services: List[str] = Field(default_factory=list, description="Additional services offered")
    features: List[str] = Field(default_factory=list, description="Shop features")
    
    # Tax and tourist services
    tax_free_shopping: bool = Field(default=False, description="Offers tax-free shopping")
    tourist_discounts: bool = Field(default=False, description="Tourist discounts available")
    multilingual_staff: bool = Field(default=False, description="Multilingual staff available")
    
    # Distance and photos
    distance_meters: Optional[float] = Field(None, ge=0, description="Distance from search location")
    photos: List[str] = Field(default_factory=list, description="Shop photo URLs")
    
    # Additional info
    description: Optional[str] = Field(None, description="Shop description")
    popular_products: List[str] = Field(default_factory=list, description="Popular product types")
    accessibility_features: List[str] = Field(default_factory=list, description="Accessibility features")


class Product(BaseModel):
    """Product information."""
    
    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    translated_name: Optional[str] = Field(None, description="Translated product name")
    brand: Optional[str] = Field(None, description="Product brand")
    category: ProductCategory = Field(description="Product category")
    
    # Pricing
    price: Optional[Decimal] = Field(None, description="Product price")
    original_price: Optional[Decimal] = Field(None, description="Original price (before discount)")
    currency: str = Field(description="Price currency")
    discount_percentage: Optional[float] = Field(None, ge=0, le=100, description="Discount percentage")
    
    # Product details
    description: Optional[str] = Field(None, description="Product description")
    translated_description: Optional[str] = Field(None, description="Translated description")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Product specifications")
    
    # Availability
    in_stock: bool = Field(default=True, description="Product in stock")
    stock_level: Optional[str] = Field(None, description="Stock level indicator")
    sizes_available: List[str] = Field(default_factory=list, description="Available sizes")
    colors_available: List[str] = Field(default_factory=list, description="Available colors")
    
    # Reviews and ratings
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    
    # Images and media
    images: List[str] = Field(default_factory=list, description="Product image URLs")
    
    # Shop information
    shop_id: str = Field(description="Shop selling this product")
    shop_name: Optional[str] = Field(None, description="Shop name")
    
    # Metadata
    is_popular: bool = Field(default=False, description="Popular product")
    is_local_specialty: bool = Field(default=False, description="Local specialty item")
    is_tourist_favorite: bool = Field(default=False, description="Tourist favorite")
    tags: List[str] = Field(default_factory=list, description="Product tags")


class ShopSearchRequest(BaseModel):
    """Shop search request schema."""
    
    location: Coordinates = Field(description="Search location")
    radius_meters: int = Field(default=2000, ge=100, le=25000, description="Search radius")
    
    # Filters
    shop_types: List[ShopType] = Field(default_factory=list, description="Shop type filters")
    categories: List[ProductCategory] = Field(default_factory=list, description="Product category filters")
    price_range: Optional[PriceRange] = Field(None, description="Price range filter")
    
    # Preferences
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    open_now: bool = Field(default=False, description="Only show currently open shops")
    tax_free_only: bool = Field(default=False, description="Only tax-free shops")
    tourist_discounts: bool = Field(default=False, description="Shops with tourist discounts")
    accepts_cards: bool = Field(default=False, description="Accepts credit/debit cards")
    multilingual_staff: bool = Field(default=False, description="Has multilingual staff")
    
    # Search parameters
    query: Optional[str] = Field(None, max_length=100, description="Text search query")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class ShopSearchResponse(BaseResponse):
    """Shop search response."""
    
    shops: List[Shop] = Field(description="Found shops")
    search_center: Coordinates = Field(description="Search center location")
    search_radius_meters: int = Field(description="Search radius used")
    total_found: int = Field(description="Total shops found")
    filters_applied: Dict[str, Any] = Field(description="Applied search filters")
    query_id: UUID = Field(description="Search query identifier")


class ProductSearchRequest(BaseModel):
    """Product search request schema."""
    
    query: str = Field(min_length=1, max_length=200, description="Product search query")
    location: Optional[Coordinates] = Field(None, description="Search location for local results")
    radius_meters: Optional[int] = Field(None, ge=100, le=25000, description="Search radius")
    
    # Filters
    category: Optional[ProductCategory] = Field(None, description="Product category filter")
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price filter")
    currency: Optional[str] = Field(None, description="Currency for price filters")
    brand: Optional[str] = Field(None, description="Brand filter")
    
    # Availability filters
    in_stock_only: bool = Field(default=True, description="Only show in-stock products")
    size: Optional[str] = Field(None, description="Size filter")
    color: Optional[str] = Field(None, description="Color filter")
    
    # Quality filters
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    popular_only: bool = Field(default=False, description="Only popular products")
    local_specialties: bool = Field(default=False, description="Include local specialties")
    
    # Search parameters
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    max_results: int = Field(default=30, ge=1, le=100, description="Maximum results")
    sort_by: str = Field(default="relevance", description="Sort criteria")


class ProductSearchResponse(BaseResponse):
    """Product search response."""
    
    products: List[Product] = Field(description="Found products")
    search_query: str = Field(description="Original search query")
    total_found: int = Field(description="Total products found")
    filters_applied: Dict[str, Any] = Field(description="Applied search filters")
    query_id: UUID = Field(description="Search query identifier")


class PriceComparisonRequest(BaseModel):
    """Price comparison request schema."""
    
    product_name: str = Field(description="Product name to compare")
    location: Coordinates = Field(description="Location for comparison")
    radius_meters: int = Field(default=5000, ge=500, le=25000, description="Comparison radius")
    category: Optional[ProductCategory] = Field(None, description="Product category")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Product specifications")
    max_shops: int = Field(default=10, ge=1, le=20, description="Maximum shops to compare")


class PriceComparison(BaseModel):
    """Price comparison for a product across shops."""
    
    product_name: str = Field(description="Product name")
    shops: List[Dict[str, Any]] = Field(description="Shops with pricing information")
    price_range: Dict[str, Decimal] = Field(description="Price range (min, max, average)")
    best_deal: Optional[Dict[str, Any]] = Field(None, description="Best deal found")
    savings_opportunity: Optional[Decimal] = Field(None, description="Maximum potential savings")
    currency: str = Field(description="Currency for prices")


class PriceComparisonResponse(BaseResponse):
    """Price comparison response."""
    
    comparison: PriceComparison = Field(description="Price comparison results")
    search_location: Coordinates = Field(description="Search center location")
    comparison_date: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")
    recommendations: List[str] = Field(default_factory=list, description="Shopping recommendations")


class ProductRecognitionRequest(BaseModel):
    """Product recognition from image request."""
    
    image: FileUpload = Field(description="Product image to analyze")
    location: Optional[Coordinates] = Field(None, description="User location for local results")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Result language")
    find_similar: bool = Field(default=True, description="Find similar products")
    price_comparison: bool = Field(default=True, description="Include price comparison")


class ProductRecognitionResponse(BaseResponse):
    """Product recognition response."""
    
    recognized_products: List[Product] = Field(description="Recognized products")
    similar_products: List[Product] = Field(default_factory=list, description="Similar products")
    confidence_score: float = Field(ge=0, le=1, description="Recognition confidence")
    processing_time_ms: int = Field(description="Processing time")
    price_comparison: Optional[PriceComparison] = Field(None, description="Price comparison if requested")


class ShoppingListItem(BaseModel):
    """Shopping list item."""
    
    item_id: UUID = Field(description="Unique item identifier")
    name: str = Field(description="Item name")
    category: Optional[ProductCategory] = Field(None, description="Item category")
    quantity: int = Field(default=1, ge=1, description="Quantity needed")
    notes: Optional[str] = Field(None, description="Additional notes")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    estimated_price: Optional[Decimal] = Field(None, description="Estimated price")
    found: bool = Field(default=False, description="Item found/purchased")
    shop_found: Optional[str] = Field(None, description="Shop where item was found")
    actual_price: Optional[Decimal] = Field(None, description="Actual price paid")


class ShoppingList(BaseModel):
    """User shopping list."""
    
    list_id: UUID = Field(description="Unique list identifier")
    name: str = Field(description="List name")
    user_id: UUID = Field(description="User identifier")
    items: List[ShoppingListItem] = Field(description="Shopping list items")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="List creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    budget: Optional[Decimal] = Field(None, description="Shopping budget")
    currency: Optional[str] = Field(None, description="Budget currency")


class ShoppingRecommendationRequest(BaseModel):
    """Shopping recommendation request."""
    
    location: Coordinates = Field(description="User location")
    budget: Optional[Decimal] = Field(None, description="Shopping budget")
    currency: Optional[str] = Field(None, description="Budget currency")
    interests: List[ProductCategory] = Field(default_factory=list, description="Shopping interests")
    time_available: Optional[int] = Field(None, description="Available shopping time (minutes)")
    trip_purpose: Optional[str] = Field(None, description="Trip purpose (leisure, business, etc.)")
    shopping_style: Optional[str] = Field(None, description="Shopping preference")
    looking_for_souvenirs: bool = Field(default=False, description="Looking for souvenirs")
    local_specialties: bool = Field(default=False, description="Interested in local specialties")


class ShoppingRecommendation(BaseModel):
    """Shopping recommendation."""
    
    recommended_shops: List[Shop] = Field(description="Recommended shops")
    suggested_products: List[Product] = Field(description="Suggested products")
    shopping_route: Optional[Dict[str, Any]] = Field(None, description="Optimized shopping route")
    budget_breakdown: Optional[Dict[str, Decimal]] = Field(None, description="Suggested budget allocation")
    tips: List[str] = Field(default_factory=list, description="Shopping tips and advice")


class ShoppingRecommendationResponse(BaseResponse):
    """Shopping recommendation response."""
    
    recommendations: ShoppingRecommendation = Field(description="Shopping recommendations")
    location: Coordinates = Field(description="User location")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Recommendation timestamp")
