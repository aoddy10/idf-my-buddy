"""
Restaurant Intelligence Service

Provides comprehensive restaurant-related AI services including:
- Menu OCR and processing
- Dish explanation and recommendations
- Price analysis and currency conversion
- Allergen detection and safety warnings
- Menu item categorization and filtering

This service integrates OCR, translation, and safety services to provide
intelligent restaurant assistance for travelers.

Author: AI Assistant
Date: 2024
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from app.services.ocr import OCRService
from app.core.logging import get_logger


class DishCategory(str, Enum):
    """Standard dish categories for menu classification."""
    APPETIZER = "appetizer"
    SOUP = "soup"
    SALAD = "salad"
    MAIN_COURSE = "main_course"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SNACK = "snack"
    SPECIAL = "special"
    UNKNOWN = "unknown"


class CuisineType(str, Enum):
    """Common cuisine types for context and recommendations."""
    THAI = "thai"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    ITALIAN = "italian"
    FRENCH = "french"
    INDIAN = "indian"
    MEXICAN = "mexican"
    AMERICAN = "american"
    MEDITERRANEAN = "mediterranean"
    FUSION = "fusion"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class MenuItemAnalysis:
    """Analysis results for a single menu item."""
    item_name: str
    original_text: str
    translated_name: Optional[str] = None
    category: DishCategory = DishCategory.UNKNOWN
    estimated_price: Optional[str] = None
    price_currency: Optional[str] = None
    description: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    allergen_warnings: List[str] = field(default_factory=list)
    allergen_risk_level: str = "unknown"  # low, medium, high, critical
    dietary_tags: List[str] = field(default_factory=list)  # vegetarian, vegan, gluten-free, etc.
    spice_level: Optional[str] = None  # mild, medium, hot, very_hot
    confidence_score: float = 0.0


@dataclass
@dataclass
class MenuAnalysis:
    """Complete menu analysis results."""
    restaurant_name: Optional[str] = None
    cuisine_type: CuisineType = CuisineType.UNKNOWN
    detected_language: str = "unknown"
    currency: Optional[str] = None
    items: List[MenuItemAnalysis] = field(default_factory=list)
    categories_found: List[DishCategory] = field(default_factory=list)
    price_range: Dict[str, Optional[float]] = field(default_factory=lambda: {"min": None, "max": None, "average": None})
    allergen_summary: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    authenticity_scores: Dict[str, float] = field(default_factory=dict)
    cultural_recommendations: List[str] = field(default_factory=list)


class RestaurantIntelligenceService:
    """Main service for restaurant intelligence and menu processing."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.ocr_service = None
        self.translation_service = None
        self.tts_service = None
        self.whisper_service = None
        
        # Common allergens to detect (expandable based on region)
        self.common_allergens = {
            "en": [
                "milk", "dairy", "cheese", "butter", "cream", "yogurt",
                "eggs", "egg", "mayonnaise",
                "peanuts", "peanut", "groundnut",
                "tree nuts", "almonds", "walnuts", "cashews", "pistachios", 
                "hazelnuts", "pecans", "brazil nuts", "macadamia",
                "fish", "salmon", "tuna", "cod", "sardines", "anchovies",
                "shellfish", "shrimp", "crab", "lobster", "oysters", "mussels", "clams",
                "wheat", "gluten", "flour", "bread", "pasta", "noodles",
                "soy", "soya", "tofu", "soybean", "miso",
                "sesame", "tahini", "sesame oil",
                "sulfites", "wine", "dried fruit"
            ],
            "th": [
                "นม", "เนย", "ครีม", "เนยแข็ง", "โยเกิร์ต",
                "ไข่", "ไข่ไก่", "มายองเนส",
                "ถั่วลิสง", "ถั่วเหลือง", "เต้าหู้",
                "ปลา", "ปลาทู", "ปลาแซลมอน", "ปลากะพง",
                "กุ้ง", "ปู", "หอย", "ปลาหมึก",
                "แป้งสาลี", "แป้ง", "ข้าวสาลี", "บะหมี่",
                "งา", "น้ำมันงา"
            ]
        }
        
        # Cuisine indicators for automatic detection
        self.cuisine_indicators = {
            CuisineType.THAI: ["pad thai", "tom yum", "green curry", "massaman", "som tam", "mango sticky rice"],
            CuisineType.CHINESE: ["kung pao", "sweet and sour", "dim sum", "chow mein", "fried rice", "peking duck"],
            CuisineType.JAPANESE: ["sushi", "sashimi", "tempura", "ramen", "udon", "yakitori", "miso soup"],
            CuisineType.KOREAN: ["kimchi", "bulgogi", "bibimbap", "galbi", "japchae", "kimchi jjigae"],
            CuisineType.ITALIAN: ["pasta", "pizza", "risotto", "carbonara", "bolognese", "tiramisu", "gelato"],
            CuisineType.FRENCH: ["escargot", "coq au vin", "bouillabaisse", "ratatouille", "crème brûlée"],
            CuisineType.INDIAN: ["curry", "tandoori", "biryani", "naan", "samosa", "masala", "dal"],
            CuisineType.MEXICAN: ["tacos", "burritos", "quesadillas", "guacamole", "salsa", "enchiladas"],
        }
    
    async def initialize_services(self):
        """Initialize required services if not already done."""
        if not self.ocr_service:
            self.ocr_service = OCRService()
            
        if not self.tts_service:
            try:
                from app.services.tts import TTSService
                self.tts_service = TTSService()
            except ImportError:
                self.logger.warning("TTS service not available")
                
        if not self.whisper_service:
            try:
                from app.services.whisper import WhisperService
                self.whisper_service = WhisperService()
            except ImportError:
                self.logger.warning("Whisper service not available")
                
        # Add translation service when available
        if not self.translation_service:
            try:
                from app.services.nllb import NLLBTranslationService
                self.translation_service = NLLBTranslationService()
            except ImportError:
                self.logger.warning("Translation service not available")
    
    async def analyze_menu_image(
        self,
        image_data: bytes,
        user_language: str = "en",
        include_allergen_warnings: bool = True,
        target_currency: Optional[str] = None,
        user_allergen_preferences: Optional[List[str]] = None
    ) -> MenuAnalysis:
        """
        Comprehensive menu image analysis.
        
        Args:
            image_data: Raw image bytes of the menu
            user_language: User's preferred language for responses
            include_allergen_warnings: Whether to perform allergen detection
            target_currency: Currency to convert prices to (if possible)
            user_allergen_preferences: User's specific allergen concerns
            
        Returns:
            Complete menu analysis with safety warnings and recommendations
        """
        await self.initialize_services()
        
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Starting comprehensive menu analysis")
            
            # Check if OCR service is available
            if not self.ocr_service:
                return MenuAnalysis(
                    processing_time=time.time() - start_time,
                    recommendations=["OCR service not available"]
                )
            
            # Step 1: OCR Processing
            ocr_result = await self.ocr_service.process_menu_image(
                image_data,
                language="en",  # Start with English, will detect language
                include_bounding_boxes=False,  # Not needed for text analysis
                confidence_threshold=0.5
            )
            
            if not ocr_result["success"]:
                return MenuAnalysis(
                    processing_time=time.time() - start_time,
                    recommendations=[f"OCR failed: {ocr_result.get('error', 'Unknown error')}"]
                )
            
            # Step 2: Extract menu structure and items
            menu_structure = ocr_result.get("menu_structure", {})
            detected_language = ocr_result.get("detected_language", "unknown")
            
            # Step 3: Analyze individual menu items
            menu_items = []
            for item_info in menu_structure.get("items", []):
                item_analysis = await self._analyze_menu_item(
                    item_info,
                    detected_language,
                    user_language,
                    include_allergen_warnings,
                    user_allergen_preferences
                )
                menu_items.append(item_analysis)
            
            # Step 4: Detect cuisine type
            cuisine_type = self._detect_cuisine_type(ocr_result["extracted_text"])
            
            # Step 5: Currency and price analysis
            price_analysis = self._analyze_prices(menu_structure.get("prices", []))
            
            # Step 6: Generate allergen summary
            allergen_summary = self._generate_allergen_summary(menu_items)
            
            # Step 7: Calculate authenticity scores
            authenticity_scores = {}
            cultural_recommendations = []
            
            for item in menu_items:
                if item.item_name:
                    dish_name = item.item_name
                    authenticity_score = await self._calculate_authenticity_score(
                        dish_name, cuisine_type, ocr_result["extracted_text"]
                    )
                    authenticity_scores[dish_name] = authenticity_score
                    
                    # Generate cultural recommendations for highly authentic dishes
                    if authenticity_score > 0.7:
                        cultural_rec = await self._get_cultural_recommendation(
                            dish_name, user_language
                        )
                        if cultural_rec:
                            cultural_recommendations.append(cultural_rec)
            
            # Step 8: Generate recommendations
            recommendations = self._generate_menu_recommendations(
                menu_items, cuisine_type, allergen_summary, user_language
            )
            
            # Add cultural recommendations to overall recommendations
            recommendations.extend(cultural_recommendations[:3])  # Top 3 cultural recommendations
            
            # Step 9: Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                ocr_result["confidence_score"], menu_items
            )
            
            return MenuAnalysis(
                cuisine_type=cuisine_type,
                detected_language=detected_language,
                currency=price_analysis.get("currency"),
                items=menu_items,
                categories_found=list(set(item.category for item in menu_items)),
                price_range=price_analysis.get("range", {}),
                allergen_summary=allergen_summary,
                processing_time=time.time() - start_time,
                confidence_score=confidence_score,
                recommendations=recommendations,
                authenticity_scores=authenticity_scores,
                cultural_recommendations=cultural_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Menu analysis failed: {e}")
            return MenuAnalysis(
                processing_time=time.time() - start_time,
                recommendations=[f"Analysis failed: {str(e)}"]
            )
    
    async def _analyze_menu_item(
        self,
        item_info: Dict[str, Any],
        detected_language: str,
        user_language: str,
        include_allergen_warnings: bool,
        user_allergen_preferences: Optional[List[str]]
    ) -> MenuItemAnalysis:
        """Analyze a single menu item for categorization, translation, and allergens."""
        
        item_text = item_info.get("text", "")
        item_prices = item_info.get("prices", [])
        
        # Basic item analysis
        analysis = MenuItemAnalysis(
            item_name=' '.join(item_text.split()[0:3]),  # First few words as name
            original_text=item_text
        )
        
        try:
            # Price extraction
            if item_prices:
                analysis.estimated_price = item_prices[0]
                analysis.price_currency = self._extract_currency(item_prices[0])
            
            # Category classification
            analysis.category = self._classify_dish_category(item_text)
            
            # Translation if needed
            if detected_language != user_language and self.translation_service:
                translation_result = await self.translation_service.translate_text(
                    text=item_text,
                    source_language=detected_language,
                    target_language=user_language
                )
                if translation_result["success"]:
                    analysis.translated_name = translation_result["translation"]
            
            # Allergen detection
            if include_allergen_warnings:
                allergen_info = self._detect_allergens(
                    item_text, detected_language, user_allergen_preferences
                )
                analysis.allergen_warnings = allergen_info["warnings"]
                analysis.allergen_risk_level = allergen_info["risk_level"]
            
            # Dietary tags
            analysis.dietary_tags = self._detect_dietary_tags(item_text)
            
            # Spice level (primarily for Asian cuisines)
            analysis.spice_level = self._detect_spice_level(item_text, detected_language)
            
            # Basic confidence based on text length and structure
            analysis.confidence_score = min(0.9, len(item_text.split()) / 10.0)
            
        except Exception as e:
            self.logger.warning(f"Item analysis failed for '{item_text}': {e}")
            analysis.confidence_score = 0.1
        
        return analysis
    
    def _classify_dish_category(self, text: str) -> DishCategory:
        """Classify dish into standard categories based on keywords."""
        text_lower = text.lower()
        
        # Category keywords (expandable)
        category_keywords = {
            DishCategory.APPETIZER: ["appetizer", "starter", "apps", "finger food", "small plate"],
            DishCategory.SOUP: ["soup", "broth", "bisque", "chowder", "consommé"],
            DishCategory.SALAD: ["salad", "greens", "caesar", "garden"],
            DishCategory.MAIN_COURSE: ["main", "entree", "grilled", "roasted", "fried", "steak", "chicken", "fish"],
            DishCategory.DESSERT: ["dessert", "cake", "ice cream", "pudding", "pie", "sweet", "chocolate"],
            DishCategory.BEVERAGE: ["drink", "juice", "coffee", "tea", "soda", "beer", "wine", "cocktail"],
            DishCategory.SPECIAL: ["special", "chef", "signature", "house", "today"]
        }
        
        # Check for category matches
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return DishCategory.UNKNOWN
    
    def _detect_cuisine_type(self, menu_text: str) -> CuisineType:
        """Detect cuisine type based on dish names and keywords."""
        text_lower = menu_text.lower()
        
        # Score each cuisine type
        cuisine_scores = {}
        for cuisine, indicators in self.cuisine_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in text_lower)
            if score > 0:
                cuisine_scores[cuisine] = score
        
        if cuisine_scores:
            return max(cuisine_scores.keys(), key=lambda x: cuisine_scores[x])
        
        return CuisineType.UNKNOWN
    
    def _detect_allergens(
        self,
        text: str,
        language: str,
        user_preferences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect potential allergens in menu item text."""
        
        text_lower = text.lower()
        allergen_list = self.common_allergens.get(language, self.common_allergens["en"])
        
        detected_allergens = []
        for allergen in allergen_list:
            if allergen.lower() in text_lower:
                detected_allergens.append(allergen)
        
        # Risk level assessment
        risk_level = "low"
        critical_allergens = ["peanut", "shellfish", "fish", "nuts"] if language == "en" else ["ถั่วลิสง", "กุ้ง", "ปลา"]
        
        if any(allergen in detected_allergens for allergen in critical_allergens):
            risk_level = "critical"
        elif len(detected_allergens) >= 3:
            risk_level = "high" 
        elif len(detected_allergens) >= 1:
            risk_level = "medium"
        
        # User-specific warnings
        user_specific_warnings = []
        if user_preferences:
            for pref in user_preferences:
                if any(pref.lower() in allergen.lower() for allergen in detected_allergens):
                    user_specific_warnings.append(f"CAUTION: Contains {pref} (user allergen)")
        
        return {
            "warnings": detected_allergens + user_specific_warnings,
            "risk_level": risk_level
        }
    
    def _detect_dietary_tags(self, text: str) -> List[str]:
        """Detect dietary preference tags (vegetarian, vegan, etc.)."""
        text_lower = text.lower()
        tags = []
        
        dietary_indicators = {
            "vegetarian": ["vegetarian", "veggie", "no meat"],
            "vegan": ["vegan", "plant-based", "dairy-free"],
            "gluten-free": ["gluten-free", "gluten free", "gf"],
            "halal": ["halal"],
            "kosher": ["kosher"],
            "organic": ["organic"],
            "low-sodium": ["low sodium", "low salt"],
            "sugar-free": ["sugar-free", "no sugar", "sugarless"]
        }
        
        for tag, indicators in dietary_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                tags.append(tag)
        
        return tags
    
    def _detect_spice_level(self, text: str, language: str) -> Optional[str]:
        """Detect spice level indicators."""
        text_lower = text.lower()
        
        spice_indicators = {
            "mild": ["mild", "light", "gentle"],
            "medium": ["medium", "moderate"],
            "hot": ["hot", "spicy", "chili", "pepper"],
            "very_hot": ["very hot", "extra spicy", "fiery", "volcano"]
        }
        
        # Thai-specific spice indicators
        if language == "th":
            thai_spice = {
                "mild": ["ไม่เผ็ด", "อ่อน"],
                "medium": ["เผ็ดกลาง", "เผ็ดปานกลาง"],
                "hot": ["เผ็ด", "พริก"],
                "very_hot": ["เผ็ดมาก", "เผ็ดร้อน", "เผ็ดเลิศ"]
            }
            spice_indicators.update(thai_spice)
        
        for level, indicators in spice_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return level
        
        return None
    
    def _analyze_prices(self, prices: List[str]) -> Dict[str, Any]:
        """Analyze price information and currency."""
        if not prices:
            return {"currency": None, "range": {}}
        
        # Extract numeric values and determine currency
        import re
        numeric_prices = []
        currency = None
        
        for price_str in prices:
            # Extract currency symbol
            currency_match = re.search(r'[\$£€¥₹₽]|USD|EUR|GBP|THB|CNY', price_str)
            if currency_match and not currency:
                currency = currency_match.group()
            
            # Extract numeric value
            numeric_match = re.search(r'\d+(?:[.,]\d{1,2})?', price_str)
            if numeric_match:
                try:
                    price_val = float(numeric_match.group().replace(',', '.'))
                    numeric_prices.append(price_val)
                except ValueError:
                    continue
        
        if numeric_prices:
            return {
                "currency": currency,
                "range": {
                    "min": min(numeric_prices),
                    "max": max(numeric_prices),
                    "average": sum(numeric_prices) / len(numeric_prices)
                }
            }
        
        return {"currency": currency, "range": {}}
    
    def _extract_currency(self, price_str: str) -> Optional[str]:
        """Extract currency from price string."""
        import re
        match = re.search(r'[\$£€¥₹₽]|USD|EUR|GBP|THB|CNY', price_str)
        return match.group() if match else None
    
    def _generate_allergen_summary(self, items: List[MenuItemAnalysis]) -> Dict[str, int]:
        """Generate summary of allergen warnings across menu."""
        allergen_counts = {}
        
        for item in items:
            for allergen in item.allergen_warnings:
                allergen_counts[allergen] = allergen_counts.get(allergen, 0) + 1
        
        return allergen_counts
    
    def _generate_menu_recommendations(
        self,
        items: List[MenuItemAnalysis],
        cuisine_type: CuisineType,
        allergen_summary: Dict[str, int],
        user_language: str
    ) -> List[str]:
        """Generate helpful recommendations for the user."""
        recommendations = []
        
        # Cuisine-specific recommendations
        if cuisine_type != CuisineType.UNKNOWN:
            recommendations.append(f"This appears to be {cuisine_type.value.title()} cuisine")
        
        # Safety recommendations based on allergen detection
        if allergen_summary:
            most_common_allergen = max(allergen_summary.keys(), key=lambda x: allergen_summary[x])
            recommendations.append(
                f"Most common allergen detected: {most_common_allergen} "
                f"({allergen_summary[most_common_allergen]} items)"
            )
        
        # Dietary preference recommendations
        vegetarian_items = [item for item in items if "vegetarian" in item.dietary_tags]
        if vegetarian_items:
            recommendations.append(f"{len(vegetarian_items)} vegetarian options available")
        
        # Price range recommendation
        prices = [item for item in items if item.estimated_price]
        if len(prices) > 3:
            recommendations.append("Good variety of price points available")
        
        return recommendations
    
    def _calculate_overall_confidence(
        self,
        ocr_confidence: float,
        items: List[MenuItemAnalysis]
    ) -> float:
        """Calculate overall analysis confidence score."""
        if not items:
            return ocr_confidence * 0.5  # Low confidence if no items processed
        
        item_confidences = [item.confidence_score for item in items]
        avg_item_confidence = sum(item_confidences) / len(item_confidences)
        
        # Weight OCR confidence and item analysis confidence
        return (ocr_confidence * 0.6) + (avg_item_confidence * 0.4)
    
    async def explain_dish(
        self,
        dish_name: str,
        cuisine_type: str = "unknown",
        user_language: str = "en",
        include_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Provide detailed explanation of a dish including ingredients and cultural context.
        
        Args:
            dish_name: Name of the dish to explain
            cuisine_type: Type of cuisine for context
            user_language: User's preferred language
            include_audio: Whether to generate audio explanation
            
        Returns:
            Detailed dish explanation with cultural context
        """
        await self.initialize_services()
        
        try:
            # This is a simplified implementation - in production, this would
            # integrate with a comprehensive dish database or LLM
            explanation = await self._generate_dish_explanation(
                dish_name, cuisine_type, user_language
            )
            
            result = {
                "dish_name": dish_name,
                "explanation": explanation,
                "cuisine_context": cuisine_type,
                "language": user_language,
                "success": True
            }
            
            # Generate audio if requested
            if include_audio and self.tts_service:
                try:
                    audio_result = await self.tts_service.synthesize_text(
                        explanation, user_language
                    )
                    if audio_result.get("success"):
                        result["audio_data"] = audio_result["audio_data"]
                        result["audio_format"] = audio_result.get("format", "wav")
                except Exception as e:
                    self.logger.warning(f"TTS generation failed: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dish explanation failed: {e}")
            return {
                "dish_name": dish_name,
                "success": False,
                "error": str(e)
            }
    
    async def _generate_dish_explanation(
        self,
        dish_name: str,
        cuisine_type: str,
        language: str
    ) -> str:
        """Generate comprehensive explanation for a dish using cultural database."""
        
        # Import cultural database
        try:
            from app.services.cultural_dining import cultural_dining_db
            
            # Get detailed dish information from cultural database
            dish_info = cultural_dining_db.get_dish_info(dish_name)
            
            if dish_info:
                # Generate comprehensive explanation
                explanation_parts = []
                
                # Basic description
                if language == "th":
                    explanation_parts.append(f"{dish_info.name} เป็นอาหารดั้งเดิมจาก{dish_info.origin_country}")
                else:
                    explanation_parts.append(f"{dish_info.name} is a traditional dish from {dish_info.origin_country}")
                
                # Cultural significance
                if dish_info.cultural_significance:
                    if language == "th":
                        explanation_parts.append(f"ความสำคัญทางวัฒนธรรม: {dish_info.cultural_significance}")
                    else:
                        explanation_parts.append(f"Cultural significance: {dish_info.cultural_significance}")
                
                # Traditional ingredients
                if dish_info.traditional_ingredients:
                    ingredients_text = ", ".join(dish_info.traditional_ingredients[:5])  # Top 5 ingredients
                    if language == "th":
                        explanation_parts.append(f"ส่วนผสมหลัก: {ingredients_text}")
                    else:
                        explanation_parts.append(f"Key ingredients: {ingredients_text}")
                
                # Preparation method
                if dish_info.preparation_method:
                    if language == "th":
                        explanation_parts.append(f"วิธีทำ: {dish_info.preparation_method}")
                    else:
                        explanation_parts.append(f"Preparation: {dish_info.preparation_method}")
                
                # Serving style
                if dish_info.serving_style:
                    if language == "th":
                        explanation_parts.append(f"วิธีเสิร์ฟ: {dish_info.serving_style}")
                    else:
                        explanation_parts.append(f"Serving: {dish_info.serving_style}")
                
                # Common variations
                if dish_info.common_variations:
                    variations_text = ", ".join(dish_info.common_variations[:3])  # Top 3 variations
                    if language == "th":
                        explanation_parts.append(f"ชนิดที่พบบ่อย: {variations_text}")
                    else:
                        explanation_parts.append(f"Common variations: {variations_text}")
                
                return ". ".join(explanation_parts) + "."
            
        except ImportError:
            self.logger.warning("Cultural dining database not available")
        
        # Fallback to basic explanations
        basic_explanations = {
            "pad thai": {
                "en": "Pad Thai is a popular Thai stir-fried noodle dish made with rice noodles, "
                      "eggs, tofu or shrimp, bean sprouts, and a sweet-tangy sauce made from "
                      "tamarind, fish sauce, and palm sugar. Garnished with lime and peanuts.",
                "th": "ผัดไทยเป็นอาหารไทยที่นิยมทำจากเส้นใหญ่ ไข่ เต้าหู้หรือกุ้ง ถั่วงอก "
                      "และซอสรสหวานเปรี้ยวจากมะขามเปียก น้ำปลา และน้ำตาลปี๊บ เสิร์ฟพร้อมมะนาวและถั่วลิสง"
            },
            "tom yum": {
                "en": "Tom Yum is a famous Thai hot and sour soup, typically made with shrimp, "
                      "mushrooms, tomatoes, lemongrass, galangal, and lime leaves. Known for its "
                      "bold flavors combining spicy, sour, and aromatic elements.",
                "th": "ต้มยำเป็นอาหารไทยแบบแกงเผ็ดร้อน โดยทำจากกุ้ง เห็ด มะเขือเทศ ตะไคร้ "
                      "ข่า และใบมะกรูด มีรสเผ็ดเปรี้ยวและหอมที่โดดเด่น"
            },
            "sushi": {
                "en": "Sushi is a traditional Japanese dish featuring vinegared rice combined with "
                      "various ingredients such as raw fish, vegetables, and nori seaweed. "
                      "It represents centuries of Japanese culinary artistry.",
                "th": "ซูชิเป็นอาหารญี่ปุ่นดั้งเดิมที่ทำจากข้าวปรุงรสด้วยน้ำส้มสายชูผสมกับ "
                      "ส่วนผสมต่างๆ เช่น ปลาดิบ ผัก และสาหร่ายนอริ"
            }
        }
        
        # Look up dish in basic database
        dish_key = dish_name.lower().strip().replace(" ", "").replace("-", "")
        for key, info in basic_explanations.items():
            if dish_key in key or key in dish_key:
                return info.get(language, info.get("en", ""))
        
        # Default explanation for unknown dishes
        if language == "th":
            return f"{dish_name} เป็นอาหารจากครัว{cuisine_type} กรุณาสอบถามพนักงานเพื่อข้อมูลเพิ่มเติม"
        else:
            return f"{dish_name} is a {cuisine_type} dish. Please ask your server for more details about ingredients and preparation."
    
    async def _calculate_authenticity_score(
        self,
        dish_name: str,
        cuisine_type: str,
        menu_text: str
    ) -> float:
        """Calculate authenticity score for a dish using cultural database."""
        
        try:
            from app.services.cultural_dining import cultural_dining_db
            
            # Extract ingredients from menu text (basic approach)
            ingredients = []
            common_ingredients = [
                "rice", "noodles", "chicken", "beef", "pork", "shrimp", "fish", "tofu",
                "vegetables", "onion", "garlic", "ginger", "chili", "coconut", "lime"
            ]
            
            menu_lower = menu_text.lower()
            for ingredient in common_ingredients:
                if ingredient in menu_lower:
                    ingredients.append(ingredient)
            
            # Get authenticity score from cultural database
            authenticity_result = cultural_dining_db.calculate_authenticity_score(
                dish_name=dish_name,
                ingredients=ingredients,
                preparation_notes=menu_text[:200],  # First 200 chars as preparation notes
                presentation_style="traditional",  # Assume traditional for menu analysis
                restaurant_context="casual"  # Default context
            )
            
            authenticity_score = authenticity_result.get("overall_score", 0.5)
            
            return authenticity_score
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not calculate authenticity score: {e}")
            return 0.5  # Neutral score when unable to calculate
    
    async def _get_cultural_recommendation(
        self,
        dish_name: str,
        language: str
    ) -> Optional[str]:
        """Get cultural recommendation for a dish."""
        
        try:
            from app.services.cultural_dining import cultural_dining_db
            
            # Get dish information
            dish_info = cultural_dining_db.get_dish_info(dish_name)
            
            if not dish_info:
                return None
            
            # Generate cultural recommendation based on dish characteristics
            if language == "th":
                recommendation = f"แนะนำ: {dish_info.name} - อาหารต้นตำรับจาก{dish_info.origin_country}"
                
                if dish_info.cultural_significance:
                    recommendation += f" มีความหมายทางวัฒนธรรม: {dish_info.cultural_significance[:100]}..."
                
                if dish_info.serving_style:
                    recommendation += f" เสิร์ฟแบบ: {dish_info.serving_style}"
                    
            else:
                recommendation = f"Recommended: {dish_info.name} - Traditional dish from {dish_info.origin_country}"
                
                if dish_info.cultural_significance:
                    recommendation += f". Cultural significance: {dish_info.cultural_significance[:100]}..."
                
                if dish_info.serving_style:
                    recommendation += f" Served: {dish_info.serving_style}"
            
            return recommendation
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not generate cultural recommendation: {e}")
            return None
    
    async def explain_dish_with_voice(
        self,
        dish_name: str,
        language: str = "en",
        voice_speed: float = 1.0
    ) -> Dict[str, Any]:
        """Generate voice explanation for a dish using TTS."""
        
        await self.initialize_services()
        
        try:
            # Get text explanation
            text_explanation = await self._generate_dish_explanation(
                dish_name, "unknown", language
            )
            
            # Generate voice if TTS service is available
            audio_data = None
            if self.tts_service:
                try:
                    audio_result = await self.tts_service.synthesize_text(
                        text=text_explanation,
                        language=language
                    )
                    
                    if audio_result.get("success"):
                        audio_data = audio_result.get("audio_data")
                        
                except Exception as e:
                    self.logger.warning(f"TTS generation failed: {e}")
            
            return {
                "success": True,
                "dish_name": dish_name,
                "text_explanation": text_explanation,
                "audio_available": audio_data is not None,
                "audio_data": audio_data,
                "language": language,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Voice explanation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def process_voice_command(
        self,
        audio_data: bytes,
        command_language: str = "en"
    ) -> Dict[str, Any]:
        """Process voice command for restaurant intelligence."""
        
        await self.initialize_services()
        
        try:
            # Transcribe audio to text
            if not self.whisper_service:
                return {
                    "success": False,
                    "error": "Voice recognition not available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            transcription_result = await self.whisper_service.transcribe_audio(
                audio_data=audio_data,
                language=command_language
            )
            
            if not transcription_result.get("success"):
                return {
                    "success": False,
                    "error": f"Transcription failed: {transcription_result.get('error')}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            transcribed_text = transcription_result.get("text", "").lower().strip()
            
            # Parse voice commands
            command_result = await self._parse_voice_command(transcribed_text, command_language)
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "command_type": command_result.get("command_type"),
                "parameters": command_result.get("parameters"),
                "response": command_result.get("response"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Voice command processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _parse_voice_command(
        self,
        transcribed_text: str,
        language: str
    ) -> Dict[str, Any]:
        """Parse transcribed voice command and execute appropriate action."""
        
        # Define command patterns
        command_patterns = {
            "explain_dish": {
                "en": ["explain", "tell me about", "what is", "describe"],
                "th": ["อธิบาย", "บอกเกี่ยวกับ", "คืออะไร", "อธิบายให้ฟัง"]
            },
            "check_allergens": {
                "en": ["allergens", "allergic", "safe to eat", "contains"],
                "th": ["แพ้", "สารก่อภูมิแพ้", "กินได้ไหม", "มีส่วนผสม"]
            },
            "recommend_dish": {
                "en": ["recommend", "suggest", "what should i order", "best dish"],
                "th": ["แนะนำ", "เสนอ", "ควรสั่งอะไร", "จานเด็ด"]
            }
        }
        
        # Detect command type
        command_type = "unknown"
        for cmd_type, patterns in command_patterns.items():
            lang_patterns = patterns.get(language, patterns.get("en", []))
            for pattern in lang_patterns:
                if pattern in transcribed_text:
                    command_type = cmd_type
                    break
            if command_type != "unknown":
                break
        
        # Extract dish name or parameters
        parameters = {}
        response = ""
        
        if command_type == "explain_dish":
            # Extract dish name from command
            dish_name = self._extract_dish_name_from_text(transcribed_text, language)
            parameters["dish_name"] = dish_name
            
            if dish_name:
                explanation = await self._generate_dish_explanation(dish_name, "unknown", language)
                response = explanation
            else:
                response = "Sorry, I couldn't identify the dish name. Could you repeat it?" if language == "en" else "ขอโทษครับ ไม่สามารถระบุชื่ออาหารได้ ช่วยพูดอีกครั้งได้ไหมครับ"
        
        elif command_type == "check_allergens":
            # Extract dish name for allergen checking
            dish_name = self._extract_dish_name_from_text(transcribed_text, language)
            parameters["dish_name"] = dish_name
            
            if dish_name:
                response = f"Checking allergen information for {dish_name}..." if language == "en" else f"กำลังตรวจสอบข้อมูลสารก่อภูมิแพ้สำหรับ {dish_name}..."
            else:
                response = "Please specify which dish you'd like to check for allergens." if language == "en" else "กรุณาระบุชื่ออาหารที่ต้องการตรวจสอบสารก่อภูมิแพ้"
        
        elif command_type == "recommend_dish":
            response = "Based on this menu, I recommend trying the authentic traditional dishes with high authenticity scores." if language == "en" else "จากเมนูนี้ แนะนำให้ลองอาหารต้นตำรับที่มีคะแนนความถูกต้องสูง"
        
        else:
            response = "I didn't understand that command. Try asking about dishes, allergens, or recommendations." if language == "en" else "ไม่เข้าใจคำสั่งนั้น ลองถามเกี่ยวกับอาหาร สารก่อภูมิแพ้ หรือการแนะนำ"
        
        return {
            "command_type": command_type,
            "parameters": parameters,
            "response": response
        }
    
    def _extract_dish_name_from_text(self, text: str, language: str) -> Optional[str]:
        """Extract dish name from voice command text."""
        
        # Remove common command words
        remove_words = {
            "en": ["explain", "tell", "me", "about", "what", "is", "describe", "the", "a", "an"],
            "th": ["อธิบาย", "บอก", "เกี่ยวกับ", "คือ", "อะไร", "ให้", "ฟัง", "หน่อย", "ครับ", "ค่ะ"]
        }
        
        words = text.split()
        lang_remove_words = remove_words.get(language, remove_words.get("en", []))
        
        # Filter out command words
        dish_words = [word for word in words if word.lower() not in lang_remove_words]
        
        if dish_words:
            return " ".join(dish_words[:3])  # Take first 3 words as dish name
        
        return None
