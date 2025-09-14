"""
Restaurant Intelligence API Endpoints

Provides REST API endpoints for restaurant-related AI services including:
- Menu OCR and processing
- Dish explanations and recommendations
- Allergen detection and safety warnings
- Voice-enabled menu assistance

Author: AI Assistant
Date: 2024
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional
import logging
from datetime import datetime
from io import BytesIO

# Schemas will be imported when needed
from app.core.logging import get_logger

# Initialize router
router = APIRouter(prefix="/restaurant", tags=["restaurant"])
logger = get_logger(__name__)

# Service instances (lazy-loaded to avoid import issues)
restaurant_service = None
allergen_service = None


async def get_restaurant_service():
    """Dependency to get restaurant intelligence service."""
    global restaurant_service
    if restaurant_service is None:
        from app.services.restaurant_intelligence import RestaurantIntelligenceService
        restaurant_service = RestaurantIntelligenceService()
        await restaurant_service.initialize_services()
    return restaurant_service


async def get_allergen_service():
    """Dependency to get allergen detection service."""
    global allergen_service
    if allergen_service is None:
        from app.services.allergen_detection import AllergenDetectionService
        allergen_service = AllergenDetectionService()
    return allergen_service


@router.post("/analyze-menu")
async def analyze_menu_image(
    image: UploadFile = File(..., description="Menu image file"),
    user_language: str = Form("en", description="User's preferred language"),
    include_allergen_warnings: bool = Form(True, description="Include allergen detection"),
    target_currency: Optional[str] = Form(None, description="Target currency for conversion"),
    user_allergens: Optional[str] = Form(None, description="Comma-separated user allergies"),
):
    """
    Analyze menu image for comprehensive restaurant intelligence.
    
    This endpoint processes a menu photo and returns:
    - Extracted menu text with OCR
    - Detected cuisine type and language
    - Individual dish analysis and categorization
    - Price analysis and currency detection
    - Allergen warnings and safety information
    - Personalized recommendations
    """
    try:
        logger.info("Processing menu analysis request")
        
        # Validate image file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read image data
        image_data = await image.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Get service instance
        service = await get_restaurant_service()
        
        # Parse user allergens
        user_allergen_list = None
        if user_allergens:
            user_allergen_list = [allergen.strip() for allergen in user_allergens.split(",")]
        
        # Analyze menu
        analysis_result = await service.analyze_menu_image(
            image_data=image_data,
            user_language=user_language,
            include_allergen_warnings=include_allergen_warnings,
            target_currency=target_currency,
            user_allergen_preferences=user_allergen_list
        )
        
        # Convert to response format
        return {
            "success": True,
            "restaurant_name": analysis_result.restaurant_name,
            "cuisine_type": analysis_result.cuisine_type.value,
            "detected_language": analysis_result.detected_language,
            "currency": analysis_result.currency,
            "menu_items": [
                {
                    "item_name": item.item_name,
                    "original_text": item.original_text,
                    "translated_name": item.translated_name,
                    "category": item.category.value,
                    "estimated_price": item.estimated_price,
                    "price_currency": item.price_currency,
                    "description": item.description,
                    "ingredients": item.ingredients,
                    "allergen_warnings": item.allergen_warnings,
                    "allergen_risk_level": item.allergen_risk_level,
                    "dietary_tags": item.dietary_tags,
                    "spice_level": item.spice_level,
                    "confidence_score": item.confidence_score
                }
                for item in analysis_result.items
            ],
            "categories_found": [cat.value for cat in analysis_result.categories_found],
            "price_range": analysis_result.price_range,
            "allergen_summary": analysis_result.allergen_summary,
            "processing_time": analysis_result.processing_time,
            "confidence_score": analysis_result.confidence_score,
            "recommendations": analysis_result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Menu analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Menu analysis failed: {str(e)}")


@router.post("/explain-dish")
async def explain_dish(
    dish_name: str,
    cuisine_type: Optional[str] = None,
    user_language: str = "en",
    include_audio: bool = False
):
    """
    Get detailed explanation of a specific dish including cultural context.
    
    Provides comprehensive information about:
    - Dish ingredients and preparation methods
    - Cultural background and significance
    - Common allergens and dietary considerations
    - Serving suggestions and accompaniments
    """
    try:
        logger.info(f"Explaining dish '{dish_name}'")
        
        service = await get_restaurant_service()
        
        explanation_result = await service.explain_dish(
            dish_name=dish_name,
            cuisine_type=cuisine_type or "unknown",
            user_language=user_language,
            include_audio=include_audio
        )
        
        response_data = {
            "success": explanation_result["success"],
            "dish_name": explanation_result["dish_name"],
            "explanation": explanation_result.get("explanation", ""),
            "cuisine_context": explanation_result.get("cuisine_context", ""),
            "language": explanation_result.get("language", user_language)
        }
        
        # Add audio data if generated
        if include_audio and "audio_data" in explanation_result:
            response_data.update({
                "audio_available": True,
                "audio_format": explanation_result.get("audio_format", "wav"),
                "audio_duration": explanation_result.get("audio_duration")
            })
        else:
            response_data["audio_available"] = False
        
        if not explanation_result["success"]:
            response_data["error"] = explanation_result.get("error", "Unknown error")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Dish explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dish explanation failed: {str(e)}")


@router.post("/check-allergens")
async def check_allergens(
    text: str,
    language: str = "en",
    user_allergens: Optional[List[str]] = None,
    confidence_threshold: float = 0.3
):
    """
    Safety-critical allergen detection for menu items.
    
    Provides comprehensive allergen analysis with:
    - Multi-language allergen pattern detection
    - Risk level assessment (none, low, medium, high, critical)
    - User-specific warnings based on known allergies
    - Safety recommendations and precautions
    - Cross-contamination warnings
    """
    try:
        logger.info(f"Checking allergens for text: '{text[:50]}...'")
        
        service = await get_allergen_service()
        
        detection_result = await service.analyze_text(
            text=text,
            language=language,
            user_allergens=user_allergens,
            confidence_threshold=confidence_threshold
        )
        
        return {
            "success": True,
            "text_analyzed": detection_result.text_analyzed,
            "language": detection_result.language,
            "detected_allergens": [
                {
                    "allergen_type": match.allergen_type.value,
                    "matched_text": match.matched_text,
                    "confidence": match.confidence,
                    "position": match.position,
                    "severity_notes": match.severity_notes
                }
                for match in detection_result.detected_allergens
            ],
            "risk_level": detection_result.risk_level.value,
            "safety_warnings": detection_result.safety_warnings,
            "user_specific_warnings": detection_result.user_specific_warnings,
            "confidence_score": detection_result.confidence_score,
            "processing_time": detection_result.processing_time,
            "recommendations": detection_result.recommendations
        }
        
    except Exception as e:
        logger.error(f"Allergen check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Allergen check failed: {str(e)}")


@router.get("/health")
async def restaurant_health_check():
    """Health check endpoint for restaurant intelligence services."""
    try:
        # Check service availability
        service_status = {
            "restaurant_intelligence": restaurant_service is not None,
            "allergen_detection": allergen_service is not None,
            "timestamp": "2024-12-28T10:00:00Z"
        }
        
        # Get allergen service statistics if available
        if allergen_service:
            stats = allergen_service.get_detection_statistics()
            service_status["allergen_stats"] = {
                "total_analyses": stats["total_analyses"],
                "detection_rate": stats["detection_rate"],
                "critical_rate": stats["critical_rate"]
            }
        
        return {
            "status": "healthy",
            "services": service_status,
            "capabilities": [
                "menu_ocr",
                "dish_explanation", 
                "allergen_detection",
                "voice_assistance",
                "multi_language_support"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported languages for restaurant intelligence."""
    return {
        "languages": {
            "en": "English",
            "th": "Thai", 
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "de": "German",
            "pt": "Portuguese"
        },
        "default": "en",
        "ocr_supported": ["en", "th", "zh", "ja", "ko", "fr", "es", "it", "de"],
        "allergen_detection": ["en", "th"],
        "tts_supported": ["en", "th", "zh", "ja", "ko", "fr", "es", "it", "de", "pt"]
    }


@router.get("/cuisine-types")
async def get_supported_cuisine_types():
    """Get list of supported cuisine types for classification."""
    return {
        "cuisine_types": [
            "thai", "chinese", "japanese", "korean", "italian", "french", 
            "indian", "mexican", "american", "mediterranean", "fusion", "local", "unknown"
        ],
        "auto_detection": True,
        "confidence_threshold": 0.6
    }


@router.get("/allergen-types")
async def get_allergen_types():
    """Get list of allergen types that can be detected."""
    return {
        "allergen_types": [
            "milk_dairy", "eggs", "fish", "shellfish", "tree_nuts", "peanuts",
            "wheat_gluten", "soybeans", "sesame", "sulfites", "molluscs", 
            "celery", "mustard", "lupin", "other"
        ],
        "risk_levels": ["none", "low", "medium", "high", "critical"],
        "critical_allergens": ["peanuts", "tree_nuts", "shellfish", "fish"],
        "detection_languages": ["en", "th"]
    }


@router.post("/explain-dish/voice")
async def explain_dish_voice(
    dish_name: str = Form(..., description="Name of the dish to explain"),
    language: str = Form("en", description="Response language (en/th)"),
    voice_speed: float = Form(1.0, description="Voice synthesis speed")
):
    """Generate voice explanation for a dish using TTS."""
    try:
        from app.services.restaurant_intelligence import RestaurantIntelligenceService
        
        service = RestaurantIntelligenceService()
        result = await service.explain_dish_with_voice(
            dish_name=dish_name,
            language=language,
            voice_speed=voice_speed
        )
        
        if result.get("success") and result.get("audio_data"):
            # Return audio as response
            from fastapi.responses import Response
            return Response(
                content=result["audio_data"],
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=dish_explanation_{dish_name}.wav",
                    "X-Text-Explanation": result["text_explanation"],
                    "X-Dish-Name": dish_name,
                    "X-Language": language
                }
            )
        else:
            # Return JSON response if audio generation failed
            return {
                "success": result.get("success", False),
                "text_explanation": result.get("text_explanation"),
                "audio_available": False,
                "error": result.get("error"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Voice explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice explanation failed: {str(e)}"
        )


@router.post("/voice-command")
async def process_voice_command(
    audio_file: UploadFile = File(..., description="Audio file with voice command"),
    command_language: str = Form("en", description="Command language (en/th)")
):
    """Process voice command for restaurant intelligence."""
    try:
        # Validate audio file
        if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Process voice command
        from app.services.restaurant_intelligence import RestaurantIntelligenceService
        service = RestaurantIntelligenceService()
        result = await service.process_voice_command(
            audio_data=audio_data,
            command_language=command_language
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Voice command processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice command processing failed: {str(e)}"
        )
