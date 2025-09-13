"""Package marker for schemas module."""

from .common import *
from .auth import *
from .navigation import *
from .restaurant import *
from .shopping import *
from .safety import *
from .voice import *

__all__ = [
    # Common schemas
    "BaseResponse",
    "ErrorResponse", 
    "SuccessResponse",
    "PaginatedResponse",
    "Coordinates",
    "Location",
    "Address",
    "LanguageCode",
    "TranslationRequest",
    "TranslationResponse",
    
    # Auth schemas
    "UserRegistrationRequest",
    "UserLoginRequest",
    "LoginResponse",
    "UserProfileResponse",
    
    # Navigation schemas  
    "NavigationRequest",
    "NavigationResponse",
    "Route",
    "POISearchRequest",
    
    # Restaurant schemas
    "RestaurantSearchRequest",
    "RestaurantSearchResponse", 
    "Restaurant",
    "MenuParsingRequest",
    
    # Shopping schemas
    "ShopSearchRequest",
    "ProductSearchRequest",
    "PriceComparisonRequest",
    
    # Safety schemas
    "EmergencyContactsRequest",
    "SafetyAlertsRequest",
    "EmergencyReportRequest",
    
    # Voice schemas
    "SpeechRecognitionRequest",
    "TextToSpeechRequest",
    "VoiceConversationRequest",
    "AudioTranslationRequest"
]
