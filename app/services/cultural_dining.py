"""
Cultural Dining Database

Comprehensive database of cultural dining customs, dish characteristics, 
etiquette rules, and regional variations to provide authentic cultural 
context for restaurant intelligence services.

This database supports:
- Cultural dining etiquette by country/region
- Traditional dish characteristics and preparation methods
- Regional variations and authenticity scoring
- Cultural dining customs and social norms
- Dietary restrictions based on cultural/religious practices

Author: AI Assistant
Date: 2024
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class CulturalRegion(str, Enum):
    """Major cultural regions for dining customs."""
    EAST_ASIA = "east_asia"
    SOUTHEAST_ASIA = "southeast_asia"
    SOUTH_ASIA = "south_asia"
    MIDDLE_EAST = "middle_east"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    LATIN_AMERICA = "latin_america"
    AFRICA = "africa"
    OCEANIA = "oceania"


class DiningStyle(str, Enum):
    """Different dining styles and service methods."""
    FAMILY_STYLE = "family_style"
    INDIVIDUAL_PORTIONS = "individual_portions"
    BUFFET = "buffet"
    TAPAS_SMALL_PLATES = "tapas_small_plates"
    STREET_FOOD = "street_food"
    FINE_DINING = "fine_dining"
    CASUAL_DINING = "casual_dining"
    FAST_FOOD = "fast_food"


@dataclass
class DishCharacteristics:
    """Characteristics of a traditional dish."""
    name: str
    origin_country: str
    region: CulturalRegion
    traditional_ingredients: List[str]
    preparation_method: str
    serving_style: str
    cultural_significance: str
    authenticity_markers: List[str]
    common_variations: List[str]
    seasonal_availability: Optional[str] = None
    festival_association: Optional[str] = None
    religious_considerations: Optional[str] = None


@dataclass
class DiningEtiquette:
    """Dining etiquette rules for a culture."""
    country: str
    region: CulturalRegion
    greeting_customs: List[str]
    table_manners: List[str]
    utensil_usage: Dict[str, str]
    eating_customs: List[str]
    payment_etiquette: List[str]
    tipping_culture: str
    dress_code: str
    conversation_norms: List[str]
    gestures_to_avoid: List[str]
    religious_considerations: List[str]


@dataclass
class CulturalDietaryInfo:
    """Cultural and religious dietary information."""
    culture: str
    common_restrictions: List[str]
    religious_dietary_laws: List[str]
    fasting_periods: List[str]
    prohibited_foods: List[str]
    preferred_foods: List[str]
    meal_timing_customs: Dict[str, str]
    special_occasion_foods: Dict[str, List[str]]


class CulturalDiningDatabase:
    """Comprehensive cultural dining database."""
    
    def __init__(self):
        self.dish_database = self._initialize_dish_database()
        self.etiquette_database = self._initialize_etiquette_database()
        self.dietary_database = self._initialize_dietary_database()
        self.authenticity_rules = self._initialize_authenticity_rules()
    
    def _initialize_dish_database(self) -> Dict[str, DishCharacteristics]:
        """Initialize the dish characteristics database."""
        return {
            # Thai Cuisine
            "pad_thai": DishCharacteristics(
                name="Pad Thai",
                origin_country="Thailand",
                region=CulturalRegion.SOUTHEAST_ASIA,
                traditional_ingredients=[
                    "rice noodles", "tamarind paste", "fish sauce", "palm sugar",
                    "eggs", "tofu or shrimp", "bean sprouts", "garlic chives",
                    "peanuts", "lime", "banana flower"
                ],
                preparation_method="Stir-fried in wok with high heat, ingredients added in specific order",
                serving_style="Individual portions with lime wedges and condiments on side",
                cultural_significance="National dish of Thailand, symbol of Thai culinary identity",
                authenticity_markers=[
                    "tamarind-based sauce", "properly soaked rice noodles",
                    "wok hei (breath of wok)", "lime and peanut garnish"
                ],
                common_variations=[
                    "Pad Thai Gai (chicken)", "Pad Thai Goong (shrimp)",
                    "Pad Thai Jay (vegetarian)", "Pad Thai Woon Sen (glass noodles)"
                ],
                seasonal_availability="Year-round"
            ),
            
            "tom_yum_goong": DishCharacteristics(
                name="Tom Yum Goong",
                origin_country="Thailand",
                region=CulturalRegion.SOUTHEAST_ASIA,
                traditional_ingredients=[
                    "shrimp", "lemongrass", "galangal", "lime leaves",
                    "bird's eye chilies", "fish sauce", "lime juice",
                    "mushrooms", "tomatoes", "Thai chilies"
                ],
                preparation_method="Clear soup base with aromatic herbs, balanced sour-spicy-salty flavors",
                serving_style="Served hot in bowl with fresh herbs on side",
                cultural_significance="Iconic Thai soup representing balance of flavors",
                authenticity_markers=[
                    "clear broth (not coconut milk)", "proper herb balance",
                    "fresh lime juice added at end", "visible herbs and chilies"
                ],
                common_variations=[
                    "Tom Yum Gai (chicken)", "Tom Yum Pla (fish)",
                    "Tom Yum Nam Khon (with coconut milk)"
                ]
            ),
            
            # Japanese Cuisine
            "sushi": DishCharacteristics(
                name="Sushi",
                origin_country="Japan",
                region=CulturalRegion.EAST_ASIA,
                traditional_ingredients=[
                    "sushi rice", "nori seaweed", "fresh fish", "wasabi",
                    "soy sauce", "pickled ginger", "rice vinegar"
                ],
                preparation_method="Properly seasoned sushi rice with precise fish cutting techniques",
                serving_style="Individual pieces eaten in one bite, specific ordering",
                cultural_significance="Traditional Japanese culinary art form dating back centuries",
                authenticity_markers=[
                    "properly seasoned sushi rice", "knife skills for fish cutting",
                    "seasonal fish selection", "minimal ingredients"
                ],
                common_variations=[
                    "Nigiri", "Maki", "Sashimi", "Chirashi", "Temaki"
                ],
                seasonal_availability="Varies by fish type"
            ),
            
            # Italian Cuisine
            "carbonara": DishCharacteristics(
                name="Carbonara",
                origin_country="Italy",
                region=CulturalRegion.EUROPE,
                traditional_ingredients=[
                    "pasta (spaghetti or tonnarelli)", "guanciale", "pecorino romano",
                    "eggs", "black pepper"
                ],
                preparation_method="Emulsification technique creating creamy sauce without cream",
                serving_style="Served immediately while hot, individual portions",
                cultural_significance="Roman pasta dish with strict traditional preparation rules",
                authenticity_markers=[
                    "no cream used", "guanciale not pancetta", "pecorino romano cheese",
                    "proper emulsification technique"
                ],
                common_variations=[
                    "Traditional Roman style", "Regional variations with pancetta"
                ]
            ),
            
            # Chinese Cuisine
            "peking_duck": DishCharacteristics(
                name="Peking Duck",
                origin_country="China",
                region=CulturalRegion.EAST_ASIA,
                traditional_ingredients=[
                    "whole duck", "thin pancakes", "hoisin sauce",
                    "scallions", "cucumber", "plum sauce"
                ],
                preparation_method="Multiple-day preparation with air-drying and roasting",
                serving_style="Carved tableside, assembled by diners in pancakes",
                cultural_significance="Imperial dish from Beijing, symbol of Chinese culinary excellence",
                authenticity_markers=[
                    "crispy skin", "proper carving technique", "thin pancakes",
                    "traditional three-course service"
                ],
                common_variations=[
                    "Beijing style", "Cantonese style", "Modern interpretations"
                ]
            ),
            
            # Indian Cuisine
            "biryani": DishCharacteristics(
                name="Biryani",
                origin_country="India",
                region=CulturalRegion.SOUTH_ASIA,
                traditional_ingredients=[
                    "basmati rice", "meat or vegetables", "yogurt", "onions",
                    "spices (cardamom, cinnamon, bay leaves)", "saffron",
                    "mint", "cilantro", "ghee"
                ],
                preparation_method="Layered cooking method with partially cooked rice and meat",
                serving_style="Served with raita and shorba, communal or individual portions",
                cultural_significance="Mughal-era dish symbolizing Indian culinary complexity",
                authenticity_markers=[
                    "layered cooking (dum method)", "aromatic long-grain rice",
                    "balanced spice blend", "saffron coloring"
                ],
                common_variations=[
                    "Hyderabadi biryani", "Lucknowi biryani", "Kolkata biryani",
                    "Malabar biryani", "Sindhi biryani"
                ]
            )
        }
    
    def _initialize_etiquette_database(self) -> Dict[str, DiningEtiquette]:
        """Initialize the dining etiquette database."""
        return {
            "thailand": DiningEtiquette(
                country="Thailand",
                region=CulturalRegion.SOUTHEAST_ASIA,
                greeting_customs=[
                    "Wai greeting with palms together",
                    "Wait for host to seat you",
                    "Remove shoes if dining at floor level"
                ],
                table_manners=[
                    "Keep feet flat on floor, don't point at food",
                    "Don't stick chopsticks upright in rice",
                    "Try small portions of everything offered"
                ],
                utensil_usage={
                    "fork_and_spoon": "Primary utensils - fork pushes food onto spoon",
                    "chopsticks": "Only for noodle dishes",
                    "hands": "Acceptable for certain foods like sticky rice"
                },
                eating_customs=[
                    "Meals are communal - share dishes",
                    "Rice is served with every meal",
                    "Eat slowly and enjoy conversation",
                    "Leave small amount on plate to show you're satisfied"
                ],
                payment_etiquette=[
                    "Host typically pays",
                    "Offer to pay but don't insist if declined",
                    "Split bills are becoming more common among friends"
                ],
                tipping_culture="Not traditional but 10% appreciated in restaurants",
                dress_code="Neat casual, avoid revealing clothing",
                conversation_norms=[
                    "Avoid controversial political topics",
                    "Compliment the food",
                    "Family and food are safe conversation topics"
                ],
                gestures_to_avoid=[
                    "Pointing with feet or index finger",
                    "Touching someone's head",
                    "Public displays of anger"
                ],
                religious_considerations=[
                    "Many Buddhists avoid beef",
                    "Some may be vegetarian on Buddhist holy days",
                    "Respect for food - don't waste"
                ]
            ),
            
            "japan": DiningEtiquette(
                country="Japan",
                region=CulturalRegion.EAST_ASIA,
                greeting_customs=[
                    "Bow when entering and leaving",
                    "Say 'Itadakimasu' before eating",
                    "Say 'Gochisousama' after eating"
                ],
                table_manners=[
                    "Don't stick chopsticks upright in rice",
                    "Don't pass food chopstick to chopstick",
                    "Bring bowl to mouth when eating",
                    "Slurping noodles is acceptable and encouraged"
                ],
                utensil_usage={
                    "chopsticks": "Primary utensil, specific etiquette rules apply",
                    "spoon": "Only for certain dishes like curry",
                    "hands": "Acceptable for sushi and some foods"
                },
                eating_customs=[
                    "Finish everything on your plate",
                    "Don't pour your own drink - serve others",
                    "Wait for 'kanpai' before drinking alcohol",
                    "Eat quietly and respectfully"
                ],
                payment_etiquette=[
                    "Pay at counter, not at table",
                    "No splitting bills - one person pays",
                    "Don't tip - it's considered rude"
                ],
                tipping_culture="Not practiced - can be offensive",
                dress_code="Clean, conservative dress",
                conversation_norms=[
                    "Keep voice low in restaurants",
                    "Avoid business talk during meals",
                    "Compliment the chef if eating sushi"
                ],
                gestures_to_avoid=[
                    "Pointing with chopsticks",
                    "Loud talking or laughing",
                    "Waving chopsticks around"
                ],
                religious_considerations=[
                    "Buddhist dietary restrictions may apply",
                    "Halal options available but not widespread"
                ]
            ),
            
            "italy": DiningEtiquette(
                country="Italy",
                region=CulturalRegion.EUROPE,
                greeting_customs=[
                    "Friendly greeting, handshakes common",
                    "Wait to be seated",
                    "Dress well - appearance matters"
                ],
                table_manners=[
                    "Keep hands visible on table",
                    "Don't cut pasta with knife",
                    "Eat courses in proper order",
                    "Don't ask for cheese with seafood pasta"
                ],
                utensil_usage={
                    "fork_and_knife": "Standard for most dishes",
                    "fork_only": "For pasta - twirl against spoon or plate",
                    "hands": "Acceptable for pizza in casual settings"
                },
                eating_customs=[
                    "Meals are long, social affairs",
                    "Coffee after meals, never with meals",
                    "Aperitivo before dinner",
                    "Don't rush through courses"
                ],
                payment_etiquette=[
                    "Split bills (fare alla romana) common",
                    "Whoever invites typically pays",
                    "Round up bill or leave small tip"
                ],
                tipping_culture="Small tip (5-10%) or round up bill",
                dress_code="Smart casual to formal, avoid sportswear",
                conversation_norms=[
                    "Animated discussion normal",
                    "Food and family are favorite topics",
                    "Passionate conversation about food welcomed"
                ],
                gestures_to_avoid=[
                    "Don't put feet on chairs",
                    "Avoid loud American-style enthusiasm",
                    "Don't ask for modifications to traditional dishes"
                ],
                religious_considerations=[
                    "Catholic traditions may influence meal timing",
                    "Friday fish traditions in some regions"
                ]
            )
        }
    
    def _initialize_dietary_database(self) -> Dict[str, CulturalDietaryInfo]:
        """Initialize cultural dietary information database."""
        return {
            "hindu": CulturalDietaryInfo(
                culture="Hindu",
                common_restrictions=["beef", "pork (some sects)", "alcohol (some sects)"],
                religious_dietary_laws=[
                    "Ahimsa (non-violence) principles",
                    "Many are vegetarian (lacto-vegetarian)",
                    "Beef strictly forbidden (cow is sacred)"
                ],
                fasting_periods=[
                    "Ekadashi (twice monthly)", "Navratri", "Karva Chauth",
                    "Maha Shivratri", "Various personal vows"
                ],
                prohibited_foods=["beef", "sometimes pork", "sometimes eggs", "sometimes alcohol"],
                preferred_foods=["vegetables", "lentils", "dairy products", "grains", "fruits"],
                meal_timing_customs={
                    "breakfast": "Light, often after prayers",
                    "lunch": "Main meal, variety of dishes",
                    "dinner": "Earlier than Western customs"
                },
                special_occasion_foods={
                    "diwali": ["sweets", "mithai", "dry fruits"],
                    "holi": ["gujiya", "bhang lassi", "festive sweets"],
                    "weddings": ["elaborate vegetarian feasts", "multiple courses"]
                }
            ),
            
            "muslim": CulturalDietaryInfo(
                culture="Muslim (Islamic)",
                common_restrictions=["pork", "alcohol", "non-halal meat"],
                religious_dietary_laws=[
                    "Halal dietary laws",
                    "No pork or pork products",
                    "No alcohol",
                    "Meat must be halal-slaughtered"
                ],
                fasting_periods=[
                    "Ramadan (month-long sunrise to sunset)",
                    "Monday and Thursday voluntary fasting",
                    "Various voluntary fasts"
                ],
                prohibited_foods=["pork", "alcohol", "non-halal meat", "gelatin (if not halal)"],
                preferred_foods=["halal meat", "vegetables", "grains", "dairy", "dates"],
                meal_timing_customs={
                    "iftar": "Break fast at sunset during Ramadan",
                    "suhur": "Pre-dawn meal during Ramadan",
                    "regular_meals": "No specific restrictions outside fasting periods"
                },
                special_occasion_foods={
                    "eid": ["dates", "sweet dishes", "elaborate meat dishes"],
                    "ramadan": ["dates to break fast", "iftar specialties"],
                    "weddings": ["biryani", "kebabs", "traditional sweets"]
                }
            ),
            
            "buddhist": CulturalDietaryInfo(
                culture="Buddhist",
                common_restrictions=["meat (strict practitioners)", "alcohol", "onions/garlic (some sects)"],
                religious_dietary_laws=[
                    "First precept: no killing",
                    "Many practice vegetarianism",
                    "Some avoid root vegetables",
                    "Mindful eating practices"
                ],
                fasting_periods=[
                    "Uposatha days (Buddhist lunar calendar)",
                    "Vesak Day",
                    "Personal meditation retreats"
                ],
                prohibited_foods=["meat (for strict practitioners)", "alcohol", "onions/garlic (some traditions)"],
                preferred_foods=["vegetables", "grains", "legumes", "fruits", "tofu"],
                meal_timing_customs={
                    "no_evening_meals": "Some monks don't eat after noon",
                    "mindful_eating": "Slow, contemplative eating",
                    "moderation": "Eating in moderation"
                },
                special_occasion_foods={
                    "vesak": ["vegetarian feast", "sweet offerings"],
                    "temple_offerings": ["fruits", "vegetarian dishes", "sweets"]
                }
            )
        }
    
    def _initialize_authenticity_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize authenticity scoring rules."""
        return {
            "ingredient_authenticity": {
                "weight": 0.4,
                "rules": {
                    "traditional_ingredients": 10,
                    "acceptable_substitutions": 7,
                    "modern_adaptations": 5,
                    "fusion_elements": 3,
                    "inauthentic_additions": 1
                }
            },
            "preparation_method": {
                "weight": 0.3,
                "rules": {
                    "traditional_technique": 10,
                    "slight_modifications": 7,
                    "modern_equipment_traditional_method": 6,
                    "simplified_preparation": 4,
                    "completely_different_method": 1
                }
            },
            "presentation_style": {
                "weight": 0.2,
                "rules": {
                    "traditional_presentation": 10,
                    "restaurant_style": 7,
                    "modern_plating": 5,
                    "fusion_presentation": 3,
                    "completely_different": 1
                }
            },
            "cultural_context": {
                "weight": 0.1,
                "rules": {
                    "proper_cultural_setting": 10,
                    "respectful_adaptation": 7,
                    "fusion_context": 5,
                    "cultural_appropriation": 1
                }
            }
        }
    
    def get_dish_info(self, dish_name: str) -> Optional[DishCharacteristics]:
        """Get detailed information about a dish."""
        dish_key = dish_name.lower().replace(" ", "_").replace("-", "_")
        return self.dish_database.get(dish_key)
    
    def get_etiquette_info(self, country: str) -> Optional[DiningEtiquette]:
        """Get dining etiquette for a specific country."""
        return self.etiquette_database.get(country.lower())
    
    def get_dietary_info(self, culture: str) -> Optional[CulturalDietaryInfo]:
        """Get dietary information for a specific culture."""
        return self.dietary_database.get(culture.lower())
    
    def calculate_authenticity_score(
        self, 
        dish_name: str,
        ingredients: List[str],
        preparation_notes: str,
        presentation_style: str,
        restaurant_context: str
    ) -> Dict[str, Any]:
        """Calculate authenticity score for a dish."""
        
        dish_info = self.get_dish_info(dish_name)
        if not dish_info:
            return {
                "overall_score": 0,
                "message": "Dish not found in database",
                "recommendations": ["Verify dish name and origin"]
            }
        
        scores = {}
        
        # Ingredient authenticity
        ingredient_score = self._score_ingredients(ingredients, dish_info.traditional_ingredients)
        scores["ingredients"] = ingredient_score
        
        # Preparation method
        preparation_score = self._score_preparation(preparation_notes, dish_info.preparation_method)
        scores["preparation"] = preparation_score
        
        # Presentation style
        presentation_score = self._score_presentation(presentation_style, dish_info.serving_style)
        scores["presentation"] = presentation_score
        
        # Cultural context
        context_score = self._score_cultural_context(restaurant_context, dish_info.origin_country)
        scores["cultural_context"] = context_score
        
        # Calculate weighted overall score
        rules = self.authenticity_rules
        overall_score = (
            scores["ingredients"] * rules["ingredient_authenticity"]["weight"] +
            scores["preparation"] * rules["preparation_method"]["weight"] +
            scores["presentation"] * rules["presentation_style"]["weight"] +
            scores["cultural_context"] * rules["cultural_context"]["weight"]
        )
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": scores,
            "authenticity_level": self._get_authenticity_level(overall_score),
            "recommendations": self._generate_authenticity_recommendations(scores, dish_info),
            "cultural_notes": dish_info.cultural_significance
        }
    
    def _score_ingredients(self, actual_ingredients: List[str], traditional_ingredients: List[str]) -> float:
        """Score ingredient authenticity."""
        if not actual_ingredients:
            return 0
        
        actual_lower = [ing.lower() for ing in actual_ingredients]
        traditional_lower = [ing.lower() for ing in traditional_ingredients]
        
        # Calculate overlap
        matches = sum(1 for ing in traditional_lower if any(ing in actual for actual in actual_lower))
        total_traditional = len(traditional_lower)
        
        if total_traditional == 0:
            return 5  # Neutral score if no traditional ingredients defined
        
        match_ratio = matches / total_traditional
        return min(10, match_ratio * 10)
    
    def _score_preparation(self, preparation_notes: str, traditional_method: str) -> float:
        """Score preparation method authenticity."""
        if not preparation_notes:
            return 5  # Neutral score if no information
        
        # Simple keyword matching (in production, this would be more sophisticated)
        prep_lower = preparation_notes.lower()
        trad_lower = traditional_method.lower()
        
        # Look for key method indicators
        key_methods = ["stir-fry", "wok", "steam", "roast", "grill", "boil", "simmer", "ferment"]
        matches = sum(1 for method in key_methods if method in prep_lower and method in trad_lower)
        
        return min(10, matches * 2 + 5)  # Base score of 5, bonus for matches
    
    def _score_presentation(self, presentation_style: str, traditional_serving: str) -> float:
        """Score presentation authenticity."""
        if not presentation_style:
            return 5
        
        # Simple scoring based on style description
        if "traditional" in presentation_style.lower():
            return 9
        elif "authentic" in presentation_style.lower():
            return 8
        elif "modern" in presentation_style.lower():
            return 6
        else:
            return 5
    
    def _score_cultural_context(self, restaurant_context: str, dish_origin: str) -> float:
        """Score cultural context appropriateness."""
        if not restaurant_context:
            return 5
        
        context_lower = restaurant_context.lower()
        origin_lower = dish_origin.lower()
        
        if origin_lower in context_lower:
            return 9
        elif "authentic" in context_lower:
            return 7
        elif "fusion" in context_lower:
            return 5
        else:
            return 4
    
    def _get_authenticity_level(self, score: float) -> str:
        """Convert numeric score to authenticity level."""
        if score >= 8.5:
            return "Highly Authentic"
        elif score >= 7.0:
            return "Mostly Authentic"
        elif score >= 5.5:
            return "Moderately Authentic"
        elif score >= 3.5:
            return "Fusion/Adapted"
        else:
            return "Heavily Modified"
    
    def _generate_authenticity_recommendations(
        self, 
        scores: Dict[str, float], 
        dish_info: DishCharacteristics
    ) -> List[str]:
        """Generate recommendations for improving authenticity."""
        recommendations = []
        
        if scores["ingredients"] < 7:
            recommendations.append(
                f"Consider using traditional ingredients: {', '.join(dish_info.traditional_ingredients[:3])}"
            )
        
        if scores["preparation"] < 7:
            recommendations.append(
                f"Traditional preparation: {dish_info.preparation_method}"
            )
        
        if scores["presentation"] < 7:
            recommendations.append(
                f"Traditional serving: {dish_info.serving_style}"
            )
        
        if not recommendations:
            recommendations.append("Excellent authenticity! This appears to be a traditional preparation.")
        
        return recommendations
    
    def get_cultural_dining_tips(self, country: str) -> Dict[str, Any]:
        """Get comprehensive cultural dining tips for a country."""
        etiquette = self.get_etiquette_info(country)
        
        if not etiquette:
            return {
                "available": False,
                "message": f"Dining etiquette information for {country} not available"
            }
        
        return {
            "available": True,
            "country": etiquette.country,
            "region": etiquette.region.value,
            "greeting_customs": etiquette.greeting_customs,
            "table_manners": etiquette.table_manners,
            "utensil_guidance": etiquette.utensil_usage,
            "eating_customs": etiquette.eating_customs,
            "payment_etiquette": etiquette.payment_etiquette,
            "tipping_culture": etiquette.tipping_culture,
            "dress_code": etiquette.dress_code,
            "conversation_tips": etiquette.conversation_norms,
            "things_to_avoid": etiquette.gestures_to_avoid,
            "religious_considerations": etiquette.religious_considerations
        }
    
    def search_dishes_by_cuisine(self, cuisine_type: str) -> List[DishCharacteristics]:
        """Search for dishes by cuisine type."""
        cuisine_lower = cuisine_type.lower()
        
        # Map cuisine types to countries
        cuisine_country_map = {
            "thai": "thailand",
            "japanese": "japan",
            "italian": "italy",
            "chinese": "china",
            "indian": "india"
        }
        
        target_country = cuisine_country_map.get(cuisine_lower)
        if not target_country:
            return []
        
        return [
            dish for dish in self.dish_database.values()
            if dish.origin_country.lower() == target_country
        ]
    
    def get_dietary_recommendations(self, culture: str, dietary_restrictions: List[str]) -> Dict[str, Any]:
        """Get dietary recommendations based on culture and restrictions."""
        cultural_info = self.get_dietary_info(culture)
        
        recommendations = {
            "cultural_guidelines": [],
            "safe_foods": [],
            "foods_to_avoid": [],
            "special_considerations": []
        }
        
        if cultural_info:
            recommendations["cultural_guidelines"] = cultural_info.religious_dietary_laws
            recommendations["safe_foods"] = cultural_info.preferred_foods
            recommendations["foods_to_avoid"] = cultural_info.prohibited_foods
            
            # Check for conflicts between personal restrictions and cultural foods
            conflicts = []
            for restriction in dietary_restrictions:
                if restriction.lower() in [food.lower() for food in cultural_info.preferred_foods]:
                    conflicts.append(f"Personal restriction '{restriction}' conflicts with cultural preference")
            
            if conflicts:
                recommendations["special_considerations"] = conflicts
        
        return recommendations


# Global instance for easy access
cultural_dining_db = CulturalDiningDatabase()
