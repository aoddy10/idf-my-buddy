"""
Tests for Cultural Dining Database functionality.
Validates authenticity scoring, cultural context, and dining recommendations.
"""

import pytest
from unittest.mock import patch, Mock
from app.services.cultural_dining import CulturalDiningDatabase, DishCharacteristics
from app.models.entities.restaurant import MenuItem

class TestCulturalDiningDatabase:
    """Test Cultural Dining Database core functionality."""
    
    @pytest.fixture
    def cultural_db(self):
        """Create cultural database instance for testing."""
        return CulturalDiningDatabase()
    
    @pytest.fixture
    def sample_thai_dish(self):
        """Sample authentic Thai dish for testing."""
        return MenuItem(
            name="Pad Thai",
            description="Traditional Thai stir-fried noodles",
            ingredients=["rice noodles", "tamarind paste", "fish sauce", "palm sugar", "peanuts", "shrimp"],
            allergens=["peanuts", "shellfish", "fish"],
            spice_level=2,
            preparation_method="wok_fried"
        )
    
    @pytest.fixture
    def sample_italian_dish(self):
        """Sample authentic Italian dish for testing."""
        return MenuItem(
            name="Carbonara",
            description="Traditional Roman pasta with eggs and guanciale",
            ingredients=["spaghetti", "guanciale", "pecorino romano", "eggs", "black pepper"],
            allergens=["gluten", "eggs", "milk"],
            preparation_method="traditional"
        )

class TestDishCharacteristics:
    """Test dish characteristics and authenticity scoring."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_thai_cuisine_characteristics(self, cultural_db):
        """Test Thai cuisine dish characteristics."""
        characteristics = cultural_db._get_thai_characteristics("Pad Thai", [
            "rice noodles", "tamarind", "fish sauce", "palm sugar", "peanuts"
        ])
        
        assert isinstance(characteristics, DishCharacteristics)
        assert characteristics.authenticity_score >= 0.8
        assert "tamarind" in characteristics.key_ingredients
        assert characteristics.cultural_significance is not None
        assert characteristics.preparation_notes is not None
    
    def test_italian_cuisine_characteristics(self, cultural_db):
        """Test Italian cuisine dish characteristics."""
        characteristics = cultural_db._get_italian_characteristics("Carbonara", [
            "spaghetti", "guanciale", "pecorino romano", "eggs", "black pepper"
        ])
        
        assert characteristics.authenticity_score >= 0.9  # Very authentic ingredients
        assert "guanciale" in characteristics.key_ingredients
        assert "Roman" in characteristics.cultural_significance
        assert "no cream" in characteristics.preparation_notes.lower()
    
    def test_japanese_cuisine_characteristics(self, cultural_db):
        """Test Japanese cuisine dish characteristics."""
        characteristics = cultural_db._get_japanese_characteristics("Ramen", [
            "ramen noodles", "miso", "chashu", "nori", "scallions"
        ])
        
        assert characteristics.authenticity_score > 0.7
        assert characteristics.umami_profile is not None
        assert "traditional" in characteristics.preparation_notes.lower()
    
    def test_indian_cuisine_characteristics(self, cultural_db):
        """Test Indian cuisine dish characteristics."""
        characteristics = cultural_db._get_indian_characteristics("Butter Chicken", [
            "chicken", "tomato", "cream", "garam masala", "fenugreek"
        ])
        
        assert characteristics.spice_complexity >= 0.7
        assert characteristics.regional_origin is not None
        assert len(characteristics.key_ingredients) >= 3
    
    def test_mexican_cuisine_characteristics(self, cultural_db):
        """Test Mexican cuisine dish characteristics."""
        characteristics = cultural_db._get_mexican_characteristics("Tacos al Pastor", [
            "pork", "pineapple", "achiote", "corn tortillas", "onion", "cilantro"
        ])
        
        assert characteristics.authenticity_score >= 0.8
        assert "achiote" in characteristics.key_ingredients
        assert characteristics.regional_origin is not None

class TestAuthenticityScoring:
    """Test authenticity scoring algorithms."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_high_authenticity_thai_pad_thai(self, cultural_db, sample_thai_dish):
        """Test high authenticity scoring for authentic Pad Thai."""
        context = cultural_db.get_dish_cultural_context(
            sample_thai_dish.name,
            sample_thai_dish.ingredients,
            "thai"
        )
        
        assert context["authenticity_score"] >= 0.85
        assert "tamarind" in context["key_ingredients"]
        assert context["cultural_significance"] is not None
    
    def test_low_authenticity_fusion_dish(self, cultural_db):
        """Test lower authenticity for fusion/non-authentic dishes."""
        fusion_ingredients = ["pasta", "teriyaki sauce", "broccoli", "chicken"]
        
        context = cultural_db.get_dish_cultural_context(
            "Teriyaki Chicken Pasta",
            fusion_ingredients,
            "japanese"
        )
        
        assert context["authenticity_score"] < 0.6  # Lower for fusion
        assert "fusion" in context["preparation_notes"].lower()
    
    def test_authenticity_ingredient_penalties(self, cultural_db):
        """Test authenticity penalties for non-traditional ingredients."""
        # Traditional carbonara vs. cream-based "carbonara"
        traditional_carbonara = ["spaghetti", "guanciale", "pecorino romano", "eggs", "black pepper"]
        cream_carbonara = ["spaghetti", "bacon", "parmesan", "cream", "black pepper"]
        
        traditional_score = cultural_db.get_dish_cultural_context(
            "Carbonara", traditional_carbonara, "italian"
        )["authenticity_score"]
        
        cream_score = cultural_db.get_dish_cultural_context(
            "Carbonara", cream_carbonara, "italian"  
        )["authenticity_score"]
        
        assert traditional_score > cream_score + 0.2  # Significant penalty for cream
    
    def test_regional_authenticity_variations(self, cultural_db):
        """Test regional variations in authenticity scoring."""
        # Different ramen styles should have different authenticity profiles
        tonkotsu_ramen = ["ramen noodles", "pork bone broth", "chashu", "ajitsuke egg", "nori"]
        miso_ramen = ["ramen noodles", "miso paste", "corn", "butter", "scallions"]
        
        tonkotsu_context = cultural_db.get_dish_cultural_context(
            "Tonkotsu Ramen", tonkotsu_ramen, "japanese"
        )
        
        miso_context = cultural_db.get_dish_cultural_context(
            "Miso Ramen", miso_ramen, "japanese"
        )
        
        # Both should be authentic but with different characteristics
        assert tonkotsu_context["authenticity_score"] >= 0.8
        assert miso_context["authenticity_score"] >= 0.8
        assert tonkotsu_context["regional_origin"] != miso_context["regional_origin"]

class TestCulturalContext:
    """Test cultural context and significance."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_cultural_significance_content(self, cultural_db):
        """Test cultural significance provides meaningful content."""
        dishes_to_test = [
            ("Pad Thai", ["rice noodles", "tamarind"], "thai"),
            ("Sushi", ["rice", "nori", "fish"], "japanese"), 
            ("Paella", ["rice", "saffron", "seafood"], "spanish"),
            ("Coq au Vin", ["chicken", "wine", "mushrooms"], "french")
        ]
        
        for dish_name, ingredients, cuisine in dishes_to_test:
            context = cultural_db.get_dish_cultural_context(dish_name, ingredients, cuisine)
            
            assert len(context["cultural_significance"]) > 50  # Substantial content
            assert context["cultural_significance"] is not None
            assert context["preparation_notes"] is not None
    
    def test_dining_etiquette_guidance(self, cultural_db):
        """Test dining etiquette recommendations."""
        etiquette_tests = [
            ("Sushi", "japanese"),
            ("Dim Sum", "chinese"),
            ("Tapas", "spanish"),
            ("Fondue", "french")
        ]
        
        for dish, cuisine in etiquette_tests:
            context = cultural_db.get_dish_cultural_context(dish, [], cuisine)
            
            if "dining_etiquette" in context:
                assert len(context["dining_etiquette"]) > 20
                assert isinstance(context["dining_etiquette"], str)
    
    def test_preparation_method_insights(self, cultural_db):
        """Test preparation method cultural insights."""
        preparation_tests = [
            ("Ramen", ["noodles", "broth"], "japanese", "broth preparation"),
            ("Risotto", ["arborio rice", "stock"], "italian", "stirring technique"),
            ("Tandoori Chicken", ["chicken", "yogurt", "spices"], "indian", "clay oven"),
            ("Peking Duck", ["duck", "hoisin"], "chinese", "roasting process")
        ]
        
        for dish, ingredients, cuisine, expected_technique in preparation_tests:
            context = cultural_db.get_dish_cultural_context(dish, ingredients, cuisine)
            
            prep_notes = context["preparation_notes"].lower()
            assert any(word in prep_notes for word in expected_technique.split())

class TestSpiceComplexityAnalysis:
    """Test spice complexity analysis for various cuisines."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_indian_spice_complexity(self, cultural_db):
        """Test Indian cuisine spice complexity scoring."""
        complex_spices = ["garam masala", "turmeric", "cumin", "coriander", "fenugreek", "cardamom"]
        simple_spices = ["salt", "pepper"]
        
        complex_score = cultural_db._calculate_spice_complexity(complex_spices)
        simple_score = cultural_db._calculate_spice_complexity(simple_spices)
        
        assert complex_score > 0.8
        assert simple_score < 0.3
        assert complex_score > simple_score + 0.4
    
    def test_thai_spice_heat_levels(self, cultural_db):
        """Test Thai cuisine heat level assessment."""
        hot_ingredients = ["thai chilies", "bird's eye chili", "chili paste"]
        mild_ingredients = ["coconut milk", "lemongrass", "galangal"]
        
        hot_heat = cultural_db._assess_heat_level(hot_ingredients)
        mild_heat = cultural_db._assess_heat_level(mild_ingredients)
        
        assert hot_heat >= 4  # High heat
        assert mild_heat <= 2  # Low heat

class TestUmamiProfile:
    """Test umami profile analysis for Japanese cuisine."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_umami_rich_ingredients(self, cultural_db):
        """Test umami profile calculation for umami-rich dishes."""
        umami_ingredients = ["miso", "dashi", "kombu", "shiitake", "bonito flakes"]
        low_umami_ingredients = ["lettuce", "cucumber", "bread"]
        
        high_umami = cultural_db._calculate_umami_profile(umami_ingredients)
        low_umami = cultural_db._calculate_umami_profile(low_umami_ingredients)
        
        assert high_umami > 0.8
        assert low_umami < 0.3
    
    def test_fermentation_complexity(self, cultural_db):
        """Test fermentation complexity in umami calculation."""
        fermented_ingredients = ["miso", "soy sauce", "kimchi", "fish sauce"]
        
        fermentation_score = cultural_db._assess_fermentation_complexity(fermented_ingredients)
        assert fermentation_score > 0.7

class TestRegionalVariations:
    """Test regional variations and specialties."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_regional_dish_identification(self, cultural_db):
        """Test identification of regional dish origins."""
        regional_tests = [
            ("Carbonara", "italian", "Lazio"),
            ("Pad Thai", "thai", "Central Thailand"),
            ("Tonkotsu Ramen", "japanese", "Kyushu"),
            ("Mole Poblano", "mexican", "Puebla")
        ]
        
        for dish, cuisine, expected_region in regional_tests:
            context = cultural_db.get_dish_cultural_context(dish, [], cuisine)
            
            if "regional_origin" in context:
                assert expected_region.lower() in context["regional_origin"].lower()
    
    def test_seasonal_recommendations(self, cultural_db):
        """Test seasonal dish recommendations."""
        seasonal_dishes = [
            ("Gazpacho", "spanish", "summer"),
            ("Hot Pot", "chinese", "winter"), 
            ("Pumpkin Soup", "american", "autumn")
        ]
        
        for dish, cuisine, season in seasonal_dishes:
            context = cultural_db.get_dish_cultural_context(dish, [], cuisine)
            
            if "seasonal_notes" in context:
                assert season in context["seasonal_notes"].lower()

class TestDietaryAdaptations:
    """Test dietary adaptations and alternatives."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_vegetarian_adaptations(self, cultural_db):
        """Test vegetarian adaptation suggestions."""
        meat_dishes = ["Pad Thai", "Carbonara", "Ramen"]
        
        for dish in meat_dishes:
            context = cultural_db.get_dish_cultural_context(dish, [], "general")
            
            if "dietary_adaptations" in context:
                adaptations = context["dietary_adaptations"]
                assert "vegetarian" in adaptations.lower() or "vegan" in adaptations.lower()
    
    def test_gluten_free_alternatives(self, cultural_db):
        """Test gluten-free alternative recommendations."""
        gluten_dishes = ["Pad Thai", "Ramen", "Pasta"]
        
        for dish in gluten_dishes:
            context = cultural_db.get_dish_cultural_context(dish, ["wheat"], "general")
            
            if "dietary_adaptations" in context:
                assert "gluten" in context["dietary_adaptations"].lower()

class TestCulturalDiningIntegration:
    """Integration tests for cultural dining database."""
    
    @pytest.fixture
    def cultural_db(self):
        return CulturalDiningDatabase()
    
    def test_comprehensive_dish_analysis(self, cultural_db, sample_thai_dish):
        """Test comprehensive cultural analysis workflow."""
        context = cultural_db.get_dish_cultural_context(
            sample_thai_dish.name,
            sample_thai_dish.ingredients,
            "thai"
        )
        
        # Verify all expected context fields are present
        required_fields = [
            "authenticity_score", "cultural_significance", 
            "preparation_notes", "key_ingredients"
        ]
        
        for field in required_fields:
            assert field in context, f"Missing required field: {field}"
            assert context[field] is not None
        
        # Verify authenticity score is reasonable
        assert 0.0 <= context["authenticity_score"] <= 1.0
        
        # Verify content quality
        assert len(context["cultural_significance"]) > 30
        assert len(context["preparation_notes"]) > 20
    
    def test_multi_cuisine_comparison(self, cultural_db):
        """Test comparative analysis across multiple cuisines."""
        cuisines = ["thai", "italian", "japanese", "indian", "mexican"]
        dish_name = "Noodle Soup"  # Generic dish for comparison
        
        results = {}
        for cuisine in cuisines:
            context = cultural_db.get_dish_cultural_context(dish_name, [], cuisine)
            results[cuisine] = context
        
        # Each cuisine should provide distinct cultural context
        significance_texts = [results[c]["cultural_significance"] for c in cuisines]
        assert len(set(significance_texts)) == len(cuisines)  # All unique
    
    def test_database_performance(self, cultural_db):
        """Test database performance for rapid queries."""
        import time
        
        start_time = time.time()
        
        # Perform multiple rapid queries
        for i in range(100):
            cultural_db.get_dish_cultural_context(f"Test Dish {i}", ["ingredient1", "ingredient2"], "thai")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be very fast for production use
        assert avg_time < 0.01, f"Average query time {avg_time}s too slow"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
