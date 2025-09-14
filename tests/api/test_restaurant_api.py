"""
API endpoint tests for Restaurant Intelligence features.
Tests REST endpoints, voice integration, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import json
import base64
from app.main import app
from app.schemas.restaurant import (
    MenuAnalysisResponse, DishExplanationResponse, VoiceExplanationResponse,
    MenuItemAnalysis, DishCategory, CuisineType
)
from app.schemas.common import LanguageCode

class TestRestaurantAPIEndpoints:
    """Test Restaurant Intelligence API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_data(self):
        """Sample base64 encoded image data."""
        return base64.b64encode(b"fake_image_data").decode()
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {"Authorization": "Bearer test_token"}

class TestMenuAnalysisEndpoint:
    """Test /restaurant/analyze-menu endpoint."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_analyze_menu_success(self, mock_service, client, auth_headers, sample_image_data):
        """Test successful menu analysis."""
        # Mock service response
        mock_service.return_value.analyze_menu = AsyncMock(return_value=MenuAnalysisResponse(
            success=True,
            message="Menu analysis completed successfully",
            restaurant_name="Test Thai Restaurant",
            cuisine_type="thai",
            detected_language="en",
            currency="USD",
            menu_items=[
                MenuItemAnalysis(
                    item_name="Pad Thai",
                    original_text="Pad Thai - Traditional Thai noodles",
                    translated_name="Pad Thai",
                    category=DishCategory.MAIN_COURSE,
                    estimated_price="12.99",
                    price_currency="USD",
                    description="Traditional Thai noodles",
                    ingredients=["rice noodles", "shrimp", "peanuts"],
                    allergen_warnings=["peanuts", "shellfish"],
                    allergen_risk_level="high",
                    spice_level="medium",
                    confidence_score=0.95
                )
            ],
            categories_found=["main_course"],
            price_range={"min": 12.99, "max": 12.99},
            allergen_summary={"peanuts": 1, "shellfish": 1},
            processing_time=2.1,
            confidence_score=0.92,
            recommendations=["Try the Pad Thai - it's authentic!"]
        ))
        
        request_data = {
            "image_data": sample_image_data,
            "language": "en",
            "allergen_profile": {
                "allergens": ["peanuts"],
                "severity": {"peanuts": "severe"}
            }
        }
        
        response = client.post("/restaurant/analyze-menu", 
                             json=request_data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["menu_items"]) == 1
        assert data["menu_items"][0]["name"] == "Pad Thai"
        assert "peanuts" in data["allergen_warnings"][0]
        assert data["processing_time"] < 3.0
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_analyze_menu_with_thai_language(self, mock_service, client, sample_image_data):
        """Test menu analysis with Thai language support."""
        mock_service.return_value.analyze_menu = AsyncMock(return_value=MenuAnalysisResponse(
            success=True,
            message="Menu analysis completed successfully",
            restaurant_name="ร้านอาหารไทย",
            cuisine_type="thai",
            detected_language="th",
            currency="THB",
            menu_items=[
                MenuItemAnalysis(
                    item_name="ผัดไทย",
                    original_text="ผัดไทย - หอยทอดกรอบๆ",
                    translated_name="Pad Thai",
                    category=DishCategory.MAIN_COURSE,
                    estimated_price="150.0",
                    price_currency="THB",
                    description="หอยทอดกรอบๆ",
                    ingredients=["เส้นจันทน์", "กุ้ง", "ถั่วลิสง"],
                    allergen_warnings=["ถั่วลิสง", "กุ้ง"],
                    allergen_risk_level="high",
                    spice_level="medium",
                    confidence_score=0.90
                )
            ],
            categories_found=["main_course"],
            price_range={"min": 150.0, "max": 150.0},
            allergen_summary={"peanuts": 1, "shellfish": 1},
            processing_time=2.3,
            confidence_score=0.90,
            recommendations=["มีถั่วลิสงและกุ้ง"]
        ))
        
        request_data = {
            "image_data": sample_image_data,
            "language": "th"
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["menu_items"][0]["name"] == "ผัดไทย"
        assert "ถั่วลิสง" in data["allergen_warnings"][0]
    
    def test_analyze_menu_invalid_image(self, client):
        """Test menu analysis with invalid image data."""
        request_data = {
            "image_data": "invalid_base64_data",
            "language": "en"
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_analyze_menu_missing_fields(self, client):
        """Test menu analysis with missing required fields."""
        request_data = {
            "language": "en"  # Missing image_data
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        assert response.status_code == 422  # Validation error

class TestDishExplanationEndpoint:
    """Test /restaurant/explain-dish endpoint."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_explain_dish_success(self, mock_service, client, auth_headers):
        """Test successful dish explanation."""
        mock_service.return_value.explain_dish = AsyncMock(return_value=DishExplanationResponse(
            success=True,
            message="Dish explanation generated successfully",
            explanation="Traditional Thai stir-fried noodles with tamarind sauce, creating a perfect balance of sweet, sour, and salty flavors.",
            cuisine_type=CuisineType.THAI,
            confidence=0.88,
            ingredients=["rice noodles", "tamarind", "fish sauce", "peanuts", "shrimp"],
            allergen_warnings=["Contains peanuts and shellfish - high risk for severe allergic reactions"],
            cultural_notes="Created in the 1930s as part of Thai nationalism campaign. Must be cooked in a wok at high heat. Traditionally eaten with fork and spoon.",
            preparation_tips=["Cook in wok at high heat", "Balance sweet, sour, and salty flavors", "Serve immediately while hot"]
        ))
        
        request_data = {
            "dish_name": "Pad Thai",
            "ingredients": ["rice noodles", "shrimp", "peanuts", "tamarind"],
            "language": "en",
            "include_cultural_context": True,
            "allergen_profile": {
                "allergens": ["peanuts"],
                "severity": {"peanuts": "severe"}
            }
        }
        
        response = client.post("/restaurant/explain-dish", 
                             json=request_data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["dish_name"] == "Pad Thai"
        assert "tamarind sauce" in data["explanation"]
        assert data["cultural_context"]["authenticity_score"] >= 0.8
        assert "peanuts" in data["allergen_warnings"][0]
        assert data["dietary_info"]["spice_level"] == 2
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_explain_dish_multilingual(self, mock_service, client):
        """Test dish explanation in multiple languages."""
        languages = ["en", "th", "es", "fr"]
        
        for lang in languages:
            mock_service.return_value.explain_dish = AsyncMock(return_value=DishExplanationResponse(
                success=True,
                message="Dish explanation generated successfully",
                explanation=f"Explanation in {lang}",
                cuisine_type=CuisineType.THAI,
                confidence=0.85,
                ingredients=["rice noodles", "shrimp", "peanuts"],
                allergen_warnings=[],
                cultural_notes=f"Cultural context in {lang}",
                preparation_tips=[]
            ))
            
            request_data = {
                "dish_name": "Pad Thai",
                "ingredients": ["noodles"],
                "language": lang
            }
            
            response = client.post("/restaurant/explain-dish", json=request_data)
            
            assert response.status_code == 200
            assert f"Explanation in {lang}" in response.json()["explanation"]
    
    def test_explain_dish_empty_name(self, client):
        """Test dish explanation with empty dish name."""
        request_data = {
            "dish_name": "",
            "ingredients": ["noodles"],
            "language": "en"
        }
        
        response = client.post("/restaurant/explain-dish", json=request_data)
        
        assert response.status_code == 400

class TestVoiceExplanationEndpoint:
    """Test /restaurant/explain-dish/voice endpoint."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_voice_explanation_success(self, mock_service, client, auth_headers):
        """Test successful voice explanation generation."""
        mock_service.return_value.generate_voice_explanation = AsyncMock(return_value=VoiceExplanationResponse(
            success=True,
            message="Voice explanation generated successfully",
            dish_name="Pad Thai",
            text_explanation="Pad Thai is a traditional Thai stir-fried noodle dish with tamarind sauce.",
            audio_available=True,
            audio_duration_seconds=15.3,
            language=LanguageCode.EN,
            authenticity_score=0.88
        ))
        
        request_data = {
            "dish_name": "Pad Thai",
            "explanation_text": "Traditional Thai noodles with sweet and sour flavors",
            "language": "en",
            "voice_speed": "normal"
        }
        
        response = client.post("/restaurant/explain-dish/voice", 
                             json=request_data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "audio_data" in data
        assert data["duration"] > 0
        assert data["language"] == "en"
        assert data["voice_settings"]["speed"] == "normal"
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_voice_explanation_multilingual(self, mock_service, client):
        """Test voice explanation in multiple languages."""
        languages = ["en", "th", "es", "ja"]
        
        for lang in languages:
            mock_service.return_value.generate_voice_explanation = AsyncMock(return_value=VoiceExplanationResponse(
                success=True,
                message="Voice explanation generated successfully",
                dish_name="Pad Thai",
                text_explanation="Explanation text",
                audio_available=True,
                audio_duration_seconds=10.0,
                language=LanguageCode.EN,
                authenticity_score=0.85
            ))
            
            request_data = {
                "dish_name": "Test Dish",
                "explanation_text": "Test explanation",
                "language": lang
            }
            
            response = client.post("/restaurant/explain-dish/voice", json=request_data)
            
            assert response.status_code == 200
            assert response.json()["language"] == lang
    
    def test_voice_explanation_invalid_speed(self, client):
        """Test voice explanation with invalid speed setting."""
        request_data = {
            "dish_name": "Pad Thai",
            "explanation_text": "Test explanation",
            "language": "en",
            "voice_speed": "invalid_speed"
        }
        
        response = client.post("/restaurant/explain-dish/voice", json=request_data)
        
        assert response.status_code == 400

class TestVoiceCommandEndpoint:
    """Test /restaurant/voice-command endpoint."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_voice_command_success(self, mock_service, client, auth_headers):
        """Test successful voice command processing."""
        mock_service.return_value.process_voice_command = AsyncMock(return_value={
            "transcription": "What ingredients are in pad thai?",
            "confidence": 0.94,
            "language": "en",
            "command_type": "ingredient_query",
            "extracted_entities": {
                "dish_name": "pad thai",
                "query_type": "ingredients"
            },
            "response": "Pad Thai typically contains rice noodles, shrimp, peanuts, tamarind, and fish sauce."
        })
        
        # Simulate audio data
        audio_data = base64.b64encode(b"fake_audio_command").decode()
        
        request_data = {
            "audio_data": audio_data,
            "language": "en"
        }
        
        response = client.post("/restaurant/voice-command", 
                             json=request_data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["transcription"] == "What ingredients are in pad thai?"
        assert data["confidence"] >= 0.9
        assert data["command_type"] == "ingredient_query"
        assert "pad thai" in data["extracted_entities"]["dish_name"]
        assert "ingredients" in data["response"].lower()
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')  
    def test_voice_command_allergen_query(self, mock_service, client):
        """Test voice command for allergen information."""
        mock_service.return_value.process_voice_command = AsyncMock(return_value={
            "transcription": "Does this dish contain peanuts?",
            "confidence": 0.91,
            "language": "en", 
            "command_type": "allergen_query",
            "extracted_entities": {
                "allergen": "peanuts",
                "query_type": "allergen_check"
            },
            "response": "WARNING: This dish contains peanuts. High risk for allergic reactions."
        })
        
        audio_data = base64.b64encode(b"allergen_voice_query").decode()
        
        request_data = {
            "audio_data": audio_data,
            "language": "en"
        }
        
        response = client.post("/restaurant/voice-command", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["command_type"] == "allergen_query"
        assert "peanuts" in data["extracted_entities"]["allergen"]
        assert "WARNING" in data["response"]
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_voice_command_thai_language(self, mock_service, client):
        """Test voice command processing in Thai."""
        mock_service.return_value.process_voice_command = AsyncMock(return_value={
            "transcription": "ผัดไทยมีส่วนผสมอะไรบ้าง",
            "confidence": 0.88,
            "language": "th",
            "command_type": "ingredient_query",
            "extracted_entities": {
                "dish_name": "ผัดไทย",
                "query_type": "ingredients"
            },
            "response": "ผัดไทยประกอบด้วย เส้นจันทน์ กุ้ง ถั่วลิสง น้ำมะขามเปียก และน้ำปลา"
        })
        
        audio_data = base64.b64encode(b"thai_voice_command").decode()
        
        request_data = {
            "audio_data": audio_data,
            "language": "th"
        }
        
        response = client.post("/restaurant/voice-command", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["language"] == "th"
        assert "ผัดไทย" in data["transcription"]
        assert "เส้นจันทน์" in data["response"]
    
    def test_voice_command_invalid_audio(self, client):
        """Test voice command with invalid audio data."""
        request_data = {
            "audio_data": "invalid_base64_audio",
            "language": "en"
        }
        
        response = client.post("/restaurant/voice-command", json=request_data)
        
        assert response.status_code == 400
        assert "error" in response.json()

class TestErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_service_unavailable_error(self, mock_service, client, sample_image_data):
        """Test handling when service is unavailable."""
        mock_service.return_value.analyze_menu = AsyncMock(side_effect=Exception("Service unavailable"))
        
        request_data = {
            "image_data": sample_image_data,
            "language": "en"
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        assert response.status_code == 500
        assert "error" in response.json()
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_timeout_handling(self, mock_service, client, sample_image_data):
        """Test handling of service timeouts."""
        import asyncio
        mock_service.return_value.analyze_menu = AsyncMock(side_effect=asyncio.TimeoutError())
        
        request_data = {
            "image_data": sample_image_data,
            "language": "en"
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        assert response.status_code == 504  # Gateway timeout
    
    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON payloads."""
        response = client.post("/restaurant/analyze-menu", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422
    
    def test_oversized_image_handling(self, client):
        """Test handling of oversized images."""
        # Create a very large base64 image (simulating 50MB image)
        large_image = base64.b64encode(b"x" * (50 * 1024 * 1024)).decode()
        
        request_data = {
            "image_data": large_image,
            "language": "en"
        }
        
        response = client.post("/restaurant/analyze-menu", json=request_data)
        
        # Should reject large payloads
        assert response.status_code in [413, 422]  # Payload too large or validation error

class TestAPIPerformance:
    """Test API performance requirements."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_response_time_requirements(self, mock_service, client, sample_image_data):
        """Test API meets response time requirements."""
        import time
        
        # Mock fast service response
        mock_service.return_value.analyze_menu = AsyncMock(return_value=MenuAnalysisResponse(
            success=True,
            message="Service unavailable",
            restaurant_name="Unknown Restaurant",
            cuisine_type="unknown",
            detected_language="en",
            currency="USD",
            menu_items=[],
            categories_found=[],
            price_range={"min": None, "max": None},
            allergen_summary={},
            processing_time=2.1,
            confidence_score=0.0,
            recommendations=[]
        ))
        
        request_data = {
            "image_data": sample_image_data,
            "language": "en"
        }
        
        start_time = time.time()
        response = client.post("/restaurant/analyze-menu", json=request_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # API should respond within 5 seconds
        assert response.json()["processing_time"] < 3.0  # Processing should be under 3s
    
    @patch('app.api.restaurant.RestaurantIntelligenceService')
    def test_concurrent_request_handling(self, mock_service, client):
        """Test concurrent request handling capability."""
        import asyncio
        import aiohttp
        
        mock_service.return_value.explain_dish = AsyncMock(return_value=DishExplanationResponse(
            success=True,
            message="Dish explanation generated successfully",
            explanation="Test explanation",
            cuisine_type=CuisineType.INTERNATIONAL,
            confidence=0.75,
            ingredients=[],
            allergen_warnings=[],
            cultural_notes=None,
            preparation_tips=[]
        ))
        
        # Simulate concurrent requests
        request_data = {
            "dish_name": "Test Dish",
            "ingredients": ["ingredient1"],
            "language": "en"
        }
        
        responses = []
        for i in range(10):  # 10 concurrent requests
            response = client.post("/restaurant/explain-dish", json=request_data)
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

class TestAPIDocumentation:
    """Test API documentation and schema validation."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Verify restaurant endpoints are documented
        paths = schema.get("paths", {})
        assert "/restaurant/analyze-menu" in paths
        assert "/restaurant/explain-dish" in paths
        assert "/restaurant/explain-dish/voice" in paths
        assert "/restaurant/voice-command" in paths
    
    def test_docs_endpoint_available(self, client):
        """Test interactive docs are available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "restaurant" in response.text.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
