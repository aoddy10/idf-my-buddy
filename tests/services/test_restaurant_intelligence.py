"""
Comprehensive tests for Restaurant Intelligence Service.
Tests core functionality, OCR processing, allergen detection, and voice integration.
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from app.services.restaurant_intelligence import RestaurantIntelligenceService
from app.schemas.restaurant import (
    MenuAnalysisRequest, MenuAnalysisResponse, DishExplanationRequest,
    VoiceExplanationRequest, VoiceCommandRequest
)
from app.schemas.common import LanguageCode
from app.models.entities.restaurant import MenuItemBase, MenuItem
import asyncio

class TestRestaurantIntelligenceService:
    """Test Restaurant Intelligence core functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return RestaurantIntelligenceService()
    
    @pytest.fixture
    def sample_menu_image(self):
        """Mock menu image data."""
        return b"fake_image_data"
    
    @pytest.fixture
    def sample_allergen_profile(self):
        """Sample allergen profile for testing."""
        return {
            "allergens": ["peanuts", "shellfish"],
            "severity": {"peanuts": "severe", "shellfish": "moderate"},
            "dietary_restrictions": ["vegetarian"]
        }
    
    @pytest.fixture
    def sample_menu_items(self):
        """Sample menu items for testing."""
        return [
            {
                "name": "Pad Thai",
                "description": "Traditional Thai stir-fried noodles", 
                "price": 12.99,
                "ingredients": ["rice noodles", "shrimp", "peanuts", "bean sprouts"],
                "allergens": ["peanuts", "shellfish"],
                "spice_level": 2
            },
            {
                "name": "Green Curry",
                "description": "Spicy Thai curry with coconut milk",
                "price": 14.99,
                "ingredients": ["green curry paste", "coconut milk", "chicken", "thai basil"],
                "allergens": ["fish"],
                "spice_level": 4
            }
        ]

class TestOCRProcessing:
    """Test OCR and menu processing functionality."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    @patch('app.services.restaurant_intelligence.OCRService')
    @patch('app.services.restaurant_intelligence.ImagePreprocessor')
    async def test_analyze_menu_success(self, mock_preprocessor, mock_ocr, service, sample_menu_image):
        """Test successful menu analysis with OCR."""
        # Mock preprocessing
        mock_preprocessor.return_value.preprocess_image.return_value = sample_menu_image
        
        # Mock OCR results
        mock_ocr.return_value.extract_text_async = AsyncMock(return_value=[
            "Pad Thai - $12.99",
            "Traditional Thai stir-fried noodles",
            "Green Curry - $14.99", 
            "Spicy Thai curry with coconut milk"
        ])
        
        request = MenuAnalysisRequest(
            user_language="en",
            target_currency="USD",
            user_allergens=["peanuts"]
        )
        
        result = await service.analyze_menu(request)
        
        assert isinstance(result, MenuAnalysisResponse)
        assert len(result.menu_items) >= 1
        assert result.processing_time < 5.0  # Performance requirement
    
    @patch('app.services.restaurant_intelligence.OCRService')
    async def test_ocr_multilingual_support(self, mock_ocr, service):
        """Test OCR with Thai language support."""
        mock_ocr.return_value.extract_text_async = AsyncMock(return_value=[
            "ผัดไทย - ฿150",
            "หอยทอด - ฿120"
        ])
        
        request = MenuAnalysisRequest(
            user_language="th",
            target_currency="THB",
            user_allergens=[]
        )
        
        result = await service.analyze_menu(request)
        assert result is not None
        mock_ocr.return_value.extract_text_async.assert_called_once()

class TestAllergenDetection:
    """Test allergen detection functionality."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    def test_detect_allergens_comprehensive(self, service):
        """Test comprehensive allergen detection."""
        ingredients = [
            "peanuts", "shrimp", "milk", "wheat flour", 
            "sesame oil", "egg whites", "fish sauce"
        ]
        
        detected = service._detect_allergens(ingredients)
        
        # Should detect all major allergens
        expected_allergens = ["peanuts", "shellfish", "milk", "gluten", "sesame", "eggs", "fish"]
        for allergen in expected_allergens:
            assert allergen in detected
    
    def test_allergen_variations_detection(self, service):
        """Test detection of allergen variations and synonyms."""
        test_cases = [
            (["groundnuts"], ["peanuts"]),
            (["prawns"], ["shellfish"]), 
            (["dairy"], ["milk"]),
            (["wheat"], ["gluten"]),
            (["tree nuts"], ["tree_nuts"]),
            (["soy sauce"], ["soy"])
        ]
        
        for ingredients, expected_allergens in test_cases:
            detected = service._detect_allergens(ingredients)
            for allergen in expected_allergens:
                assert allergen in detected, f"Failed to detect {allergen} from {ingredients}"
    
    def test_zero_false_negatives_requirement(self, service):
        """Test safety-critical requirement: zero false negatives."""
        # Test hidden allergens in common ingredients
        hidden_allergen_cases = [
            ("fish sauce", ["fish"]),
            ("worcestershire sauce", ["fish"]),
            ("mayonnaise", ["eggs"]),
            ("bread crumbs", ["gluten"]),
            ("imitation crab", ["fish", "eggs"])
        ]
        
        for ingredient, expected_allergens in hidden_allergen_cases:
            detected = service._detect_allergens([ingredient])
            for allergen in expected_allergens:
                assert allergen in detected, f"SAFETY CRITICAL: Missed {allergen} in {ingredient}"

class TestCulturalIntelligence:
    """Test cultural dining database and authenticity scoring."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    @patch('app.services.restaurant_intelligence.CulturalDiningDatabase')
    async def test_cultural_context_integration(self, mock_cultural_db, service):
        """Test cultural context integration in dish explanations."""
        # Mock cultural database
        mock_cultural_db.return_value.get_dish_cultural_context.return_value = {
            "authenticity_score": 0.85,
            "cultural_significance": "Traditional royal court dish",
            "preparation_notes": "Requires specific technique",
            "dining_etiquette": "Eaten with fork and spoon"
        }
        
        request = DishExplanationRequest(
            dish_name="Pad Thai",
            cuisine_type="thai",
            user_language="en",
            include_audio=False,
            cultural_context=None
        )
        
        result = await service.explain_dish(request)
        
        assert result.cultural_context is not None
        assert result.cultural_context.authenticity_score >= 0.8
        assert "Traditional" in result.cultural_context.cultural_significance
    
    async def test_authenticity_scoring(self, service):
        """Test dish authenticity scoring algorithms."""
        # Test with authentic Thai dish
        authentic_pad_thai = MenuItem(
            menu_id=1,
            name="Pad Thai",
            ingredients=["rice noodles", "tamarind", "fish sauce", "palm sugar", "peanuts"]
        )
        
        # Mock cultural database call
        with patch.object(service, '_get_cultural_context') as mock_cultural:
            mock_cultural.return_value = {"authenticity_score": 0.92}
            
            score = service._calculate_authenticity_score(authentic_pad_thai, "thai")
            assert score >= 0.9

class TestVoiceIntegration:
    """Test voice integration functionality."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    @patch('app.services.restaurant_intelligence.TTSService')
    async def test_voice_explanation_generation(self, mock_tts, service):
        """Test TTS generation for dish explanations."""
        mock_tts.return_value.generate_speech_async = AsyncMock(return_value=b"audio_data")
        
        request = VoiceExplanationRequest(
            dish_name="Pad Thai",
            language=LanguageCode.EN,
            voice_speed=1.0,
            include_cultural_context=True
        )
        
        result = await service.generate_voice_explanation(request)
        
        assert result.audio_data == b"audio_data"
        assert result.duration > 0
        mock_tts.return_value.generate_speech_async.assert_called_once()
    
    @patch('app.services.restaurant_intelligence.WhisperService')
    async def test_voice_command_processing(self, mock_whisper, service):
        """Test voice command processing with Whisper ASR."""
        mock_whisper.return_value.transcribe_async = AsyncMock(return_value={
            "text": "What ingredients are in pad thai?",
            "confidence": 0.95,
            "language": "en"
        })
        
        request = VoiceCommandRequest(
            command_language=LanguageCode.EN,
            context="restaurant menu analysis"
        )
        
        result = await service.process_voice_command(request)
        
        assert result.transcription == "What ingredients are in pad thai?"
        assert result.confidence >= 0.9
        assert result.command_type == "ingredient_query"
    
    async def test_multilingual_voice_support(self, service):
        """Test multilingual voice processing."""
        languages = ["en", "th", "es", "fr", "ja"]
        
        for lang in languages:
            with patch.object(service.tts_service, 'generate_speech_async') as mock_tts:
                mock_tts.return_value = b"audio_data"
                
                request = VoiceExplanationRequest(
                    dish_name="Test Dish",
                    explanation_text="Test explanation",
                    language=lang
                )
                
                result = await service.generate_voice_explanation(request)
                assert result.language == lang

class TestPerformanceRequirements:
    """Test performance benchmarks and requirements."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    @patch('app.services.restaurant_intelligence.OCRService')
    @patch('app.services.restaurant_intelligence.ImagePreprocessor')
    async def test_menu_processing_speed(self, mock_preprocessor, mock_ocr, service):
        """Test menu processing meets <3 second requirement."""
        import time
        
        # Mock fast responses
        mock_preprocessor.return_value.preprocess_image.return_value = b"processed_image"
        mock_ocr.return_value.extract_text_async = AsyncMock(return_value=["Pad Thai - $12.99"])
        
        request = MenuAnalysisRequest(
            image_data=b"large_menu_image",
            language="en"
        )
        
        start_time = time.time()
        result = await service.analyze_menu(request)
        processing_time = time.time() - start_time
        
        assert processing_time < 3.0, f"Processing took {processing_time}s, exceeds 3s requirement"
        assert result.processing_time < 3.0
    
    async def test_concurrent_processing(self, service):
        """Test concurrent menu processing capability."""
        # Mock dependencies
        with patch.object(service, 'analyze_menu') as mock_analyze:
            mock_analyze.return_value = MenuAnalysisResponse(
                menu_items=[],
                processing_time=1.5,
                allergen_warnings=[],
                cultural_insights=None
            )
            
            # Process multiple requests concurrently
            requests = [
                MenuAnalysisRequest(image_data=f"image_{i}".encode(), language="en")
                for i in range(5)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*[service.analyze_menu(req) for req in requests])
            total_time = time.time() - start_time
            
            assert len(results) == 5
            assert total_time < 10.0  # Should be much faster than 5x3s sequential

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    async def test_invalid_image_handling(self, service):
        """Test handling of invalid image data."""
        request = MenuAnalysisRequest(
            image_data=b"invalid_image_data",
            language="en"
        )
        
        with patch('app.services.restaurant_intelligence.ImagePreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.preprocess_image.side_effect = ValueError("Invalid image")
            
            with pytest.raises(ValueError):
                await service.analyze_menu(request)
    
    async def test_ocr_failure_graceful_degradation(self, service):
        """Test graceful degradation when OCR fails."""
        with patch('app.services.restaurant_intelligence.OCRService') as mock_ocr:
            mock_ocr.return_value.extract_text_async = AsyncMock(side_effect=Exception("OCR failed"))
            
            request = MenuAnalysisRequest(
                image_data=b"image_data",
                language="en"
            )
            
            # Should handle gracefully and return partial results
            result = await service.analyze_menu(request)
            assert result is not None
            assert len(result.menu_items) == 0  # No items extracted but no crash
    
    def test_allergen_detection_edge_cases(self, service):
        """Test allergen detection with edge cases."""
        edge_cases = [
            [],  # Empty ingredients
            ["unknown ingredient"],  # Unknown ingredients
            [""],  # Empty strings
            None  # None value
        ]
        
        for case in edge_cases:
            try:
                result = service._detect_allergens(case or [])
                assert isinstance(result, list)
            except Exception as e:
                pytest.fail(f"Allergen detection failed on edge case {case}: {e}")

# Integration test for full workflow
class TestRestaurantIntelligenceIntegration:
    """Integration tests for complete Restaurant Intelligence workflow."""
    
    @pytest.fixture
    def service(self):
        return RestaurantIntelligenceService()
    
    @patch('app.services.restaurant_intelligence.OCRService')
    @patch('app.services.restaurant_intelligence.TTSService') 
    @patch('app.services.restaurant_intelligence.WhisperService')
    @patch('app.services.restaurant_intelligence.CulturalDiningDatabase')
    async def test_complete_restaurant_intelligence_workflow(
        self, mock_cultural_db, mock_whisper, mock_tts, mock_ocr, service
    ):
        """Test complete workflow: OCR → Analysis → Cultural Context → Voice Response."""
        
        # Setup mocks
        mock_ocr.return_value.extract_text_async = AsyncMock(return_value=[
            "Pad Thai - $12.99",
            "Traditional stir-fried noodles with shrimp and peanuts"
        ])
        
        mock_cultural_db.return_value.get_dish_cultural_context.return_value = {
            "authenticity_score": 0.88,
            "cultural_significance": "National dish of Thailand",
            "preparation_notes": "Wok-fried at high heat"
        }
        
        mock_tts.return_value.generate_speech_async = AsyncMock(return_value=b"audio_explanation")
        
        # Step 1: Analyze menu
        menu_request = MenuAnalysisRequest(
            image_data=b"menu_image",
            language="en",
            allergen_profile=AllergenProfile(allergens=["peanuts"])
        )
        
        menu_result = await service.analyze_menu(menu_request)
        
        # Step 2: Get voice explanation for detected dish
        voice_request = VoiceExplanationRequest(
            dish_name="Pad Thai",
            explanation_text="Traditional Thai noodles with cultural significance",
            language="en"
        )
        
        voice_result = await service.generate_voice_explanation(voice_request)
        
        # Verify complete workflow
        assert len(menu_result.menu_items) > 0
        assert any("peanuts" in item.allergens for item in menu_result.menu_items)
        assert voice_result.audio_data == b"audio_explanation"
        
        # Verify all services were called
        mock_ocr.return_value.extract_text_async.assert_called()
        mock_tts.return_value.generate_speech_async.assert_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
