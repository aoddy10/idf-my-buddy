"""Integration tests for AI services.

Tests the AI services with mock models and real processing logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from tests.utils import FileTestHelper, MockResponseBuilder


@pytest.mark.ai
class TestOCRServiceIntegration:
    """Integration tests for OCR service."""
    
    @pytest.fixture
    def ocr_service(self):
        """OCR service fixture with mocked engines."""
        from app.services.ocr import OCRService
        
        service = OCRService()
        # Mock the engine setup to avoid loading actual models
        service._tesseract_available = True
        service._easyocr_reader = Mock()
        service._paddle_ocr = Mock()
        
        return service
    
    async def test_extract_text_with_tesseract(self, ocr_service, file_test_helper):
        """Test text extraction with Tesseract engine."""
        # Create test image
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Mock Tesseract response
        mock_ocr_data = {
            'text': ['Hello', 'World', 'Test'],
            'conf': [95, 90, 88],
            'left': [10, 60, 120],
            'top': [10, 10, 10],
            'width': [45, 50, 40],
            'height': [20, 20, 20]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_ocr_data):
            result = await ocr_service.extract_text(
                test_image,
                language="en",
                engine="tesseract"
            )
        
        assert result["text"] == "Hello World Test"
        assert result["engine"] == "tesseract"
        assert len(result["words"]) == 3
        assert result["words"][0]["text"] == "Hello"
        assert result["words"][0]["confidence"] == 0.95
    
    async def test_extract_text_with_easyocr(self, ocr_service, file_test_helper):
        """Test text extraction with EasyOCR engine."""
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Mock EasyOCR response
        mock_results = [
            ([[10, 10], [60, 10], [60, 30], [10, 30]], "Hello", 0.95),
            ([[70, 10], [120, 10], [120, 30], [70, 30]], "World", 0.92)
        ]
        
        ocr_service._easyocr_reader.readtext = Mock(return_value=mock_results)
        
        result = await ocr_service.extract_text(
            test_image,
            language="en",
            engine="easyocr"
        )
        
        assert result["text"] == "Hello World"
        assert result["engine"] == "easyocr"
        assert len(result["words"]) == 2
        assert result["confidence"] > 0.9
    
    async def test_extract_text_with_paddleocr(self, ocr_service, file_test_helper):
        """Test text extraction with PaddleOCR engine."""
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Mock PaddleOCR response
        mock_results = [[
            [[[10, 10], [60, 10], [60, 30], [10, 30]], ("Hello", 0.95)],
            [[[70, 10], [120, 10], [120, 30], [70, 30]], ("World", 0.90)]
        ]]
        
        ocr_service._paddle_ocr.ocr = Mock(return_value=mock_results)
        
        result = await ocr_service.extract_text(
            test_image,
            language="en",
            engine="paddleocr"
        )
        
        assert result["text"] == "Hello World"
        assert result["engine"] == "paddleocr"
        assert len(result["words"]) == 2
    
    async def test_extract_structured_text_menu(self, ocr_service, file_test_helper):
        """Test structured text extraction for menu."""
        test_image = file_test_helper.create_test_image(300, 400)
        
        # Mock OCR result for menu
        with patch.object(ocr_service, 'extract_text') as mock_extract:
            mock_extract.return_value = {
                "text": "APPETIZERS\nBruschetta $8\nCalamari $12\nMAINS\nPizza Margherita $18\nPasta Carbonara $16",
                "words": [],
                "language": "en",
                "engine": "tesseract",
                "confidence": 0.9
            }
            
            result = await ocr_service.extract_structured_text(
                test_image,
                document_type="menu",
                language="en"
            )
        
        assert result["document_type"] == "menu"
        assert "structured" in result
        # Basic structure should be created (implementation specific)
    
    async def test_image_preprocessing(self, ocr_service, file_test_helper):
        """Test image preprocessing functionality."""
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Test with preprocessing enabled
        with patch('pytesseract.image_to_data') as mock_tesseract:
            mock_tesseract.return_value = {
                'text': ['Processed'], 'conf': [95],
                'left': [10], 'top': [10], 'width': [80], 'height': [20]
            }
            
            result = await ocr_service.extract_text(
                test_image,
                engine="tesseract",
                preprocess=True
            )
        
        assert result["text"] == "Processed"
        # Verify preprocessing was applied (mock would be called with processed image)
    
    async def test_fallback_engine_selection(self, ocr_service, file_test_helper):
        """Test fallback to different OCR engines."""
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Disable primary engines to test fallback
        ocr_service._tesseract_available = False
        
        # Mock EasyOCR as fallback
        ocr_service._easyocr_reader.readtext = Mock(return_value=[
            ([[10, 10], [60, 10], [60, 30], [10, 30]], "Fallback", 0.9)
        ])
        
        result = await ocr_service.extract_text(
            test_image,
            language="en"  # No engine specified, should fallback
        )
        
        assert result["text"] == "Fallback"
        assert result["engine"] == "easyocr"
    
    async def test_confidence_threshold_filtering(self, ocr_service, file_test_helper):
        """Test confidence threshold filtering."""
        test_image = file_test_helper.create_test_image(200, 100)
        
        # Mock results with varying confidence
        mock_ocr_data = {
            'text': ['High', 'Low', 'Medium'],
            'conf': [95, 40, 75],  # Low confidence should be filtered
            'left': [10, 60, 120],
            'top': [10, 10, 10],
            'width': [45, 50, 40],
            'height': [20, 20, 20]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_ocr_data):
            result = await ocr_service.extract_text(
                test_image,
                engine="tesseract",
                confidence_threshold=0.7  # 70% threshold
            )
        
        # Only "High" (95%) and "Medium" (75%) should pass
        assert "High" in result["text"]
        assert "Medium" in result["text"]
        assert "Low" not in result["text"]


@pytest.mark.ai
class TestWhisperServiceIntegration:
    """Integration tests for Whisper ASR service."""
    
    @pytest.fixture
    def whisper_service(self):
        """Whisper service fixture with mocked models."""
        from app.services.whisper import WhisperService
        
        service = WhisperService()
        # Mock model loading
        service.local_model = Mock()
        service.openai_client = Mock()
        
        return service
    
    async def test_transcribe_audio_local_model(self, whisper_service, file_test_helper):
        """Test audio transcription with local Whisper model."""
        test_audio = file_test_helper.create_test_audio(2.0)
        
        # Mock local model transcription
        mock_result = {
            "text": "Hello, this is a test transcription.",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello, this is a test transcription."}
            ]
        }
        
        whisper_service.local_model.transcribe = Mock(return_value=mock_result)
        
        result = await whisper_service.transcribe_audio(
            test_audio,
            language="en"
        )
        
        assert result["text"] == "Hello, this is a test transcription."
        assert result["language"] == "en"
        assert "confidence" in result
    
    async def test_transcribe_audio_openai_fallback(self, whisper_service, file_test_helper):
        """Test fallback to OpenAI Whisper API."""
        test_audio = file_test_helper.create_test_audio(2.0)
        
        # Mock local model failure
        whisper_service.local_model = None
        
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.text = "OpenAI transcription result"
        whisper_service.openai_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        
        result = await whisper_service.transcribe_audio(
            test_audio,
            language="en"
        )
        
        assert result["text"] == "OpenAI transcription result"
        assert result["source"] == "openai"
    
    async def test_transcribe_multilingual(self, whisper_service, file_test_helper):
        """Test multilingual transcription."""
        test_audio = file_test_helper.create_test_audio(3.0)
        
        # Mock multilingual result
        mock_result = {
            "text": "Bonjour, comment allez-vous?",
            "language": "fr",
            "segments": []
        }
        
        whisper_service.local_model.transcribe = Mock(return_value=mock_result)
        
        result = await whisper_service.transcribe_audio(
            test_audio,
            language="auto"  # Auto-detect language
        )
        
        assert result["text"] == "Bonjour, comment allez-vous?"
        assert result["language"] == "fr"
    
    async def test_audio_format_conversion(self, whisper_service):
        """Test audio format conversion."""
        # Mock non-WAV audio data
        mock_mp3_data = b"fake_mp3_audio_data"
        
        with patch('tempfile.NamedTemporaryFile'), \
             patch('subprocess.run') as mock_ffmpeg:
            
            mock_ffmpeg.return_value.returncode = 0
            
            # Test conversion
            converted = await whisper_service._convert_audio_format(
                mock_mp3_data,
                target_format="wav"
            )
        
        # Verify ffmpeg was called for conversion
        mock_ffmpeg.assert_called_once()


@pytest.mark.ai
class TestNLLBTranslationServiceIntegration:
    """Integration tests for NLLB translation service."""
    
    @pytest.fixture
    def translation_service(self):
        """Translation service fixture."""
        from app.services.nllb import NLLBTranslationService
        
        service = NLLBTranslationService()
        # Mock model components
        service.model = Mock()
        service.tokenizer = Mock()
        service.google_translator = Mock()
        
        return service
    
    async def test_translate_text_local_model(self, translation_service):
        """Test translation with local NLLB model."""
        # Mock tokenizer and model
        mock_inputs = Mock()
        mock_outputs = Mock()
        mock_outputs.sequences = [[1, 2, 3, 4]]  # Mock token IDs
        
        translation_service.tokenizer.encode = Mock(return_value=mock_inputs)
        translation_service.model.generate = Mock(return_value=mock_outputs)
        translation_service.tokenizer.decode = Mock(return_value="Hola mundo")
        
        result = await translation_service.translate(
            text="Hello world",
            source_language="en",
            target_language="es"
        )
        
        assert result["translated_text"] == "Hola mundo"
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert "confidence" in result
    
    async def test_translate_batch_processing(self, translation_service):
        """Test batch translation."""
        texts = [
            "Hello world",
            "Good morning",
            "Thank you"
        ]
        
        # Mock batch processing
        mock_outputs = Mock()
        mock_outputs.sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        translation_service.tokenizer.encode = Mock(return_value=Mock())
        translation_service.model.generate = Mock(return_value=mock_outputs)
        translation_service.tokenizer.decode = Mock(side_effect=[
            "Hola mundo", "Buenos días", "Gracias"
        ])
        
        results = await translation_service.translate_batch(
            texts=texts,
            source_language="en",
            target_language="es"
        )
        
        assert len(results) == 3
        assert results[0]["translated_text"] == "Hola mundo"
        assert results[1]["translated_text"] == "Buenos días"
        assert results[2]["translated_text"] == "Gracias"
    
    async def test_translate_google_fallback(self, translation_service):
        """Test fallback to Google Translate."""
        # Simulate local model failure
        translation_service.model = None
        
        # Mock Google Translate
        translation_service.google_translator.translate = AsyncMock(return_value={
            "translatedText": "Fallback translation",
            "detectedSourceLanguage": "en"
        })
        
        result = await translation_service.translate(
            text="Test text",
            source_language="en",
            target_language="es"
        )
        
        assert result["translated_text"] == "Fallback translation"
        assert result["source"] == "google"
    
    async def test_language_detection(self, translation_service):
        """Test automatic language detection."""
        # Mock language detection
        with patch('langdetect.detect', return_value="fr"):
            result = await translation_service.detect_language(
                "Bonjour le monde"
            )
        
        assert result["detected_language"] == "fr"
        assert "confidence" in result


@pytest.mark.ai
class TestTTSServiceIntegration:
    """Integration tests for TTS service."""
    
    @pytest.fixture
    def tts_service(self):
        """TTS service fixture."""
        from app.services.tts import TTSService
        
        service = TTSService()
        # Mock TTS engines
        service.openai_client = Mock()
        service.speechbrain_model = Mock()
        service.gtts_available = True
        service.pyttsx3_engine = Mock()
        
        return service
    
    async def test_synthesize_speech_openai(self, tts_service):
        """Test speech synthesis with OpenAI."""
        # Mock OpenAI TTS response
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        tts_service.openai_client.audio.speech.create = AsyncMock(return_value=mock_response)
        
        result = await tts_service.synthesize_speech(
            text="Hello world",
            language="en",
            voice="alloy",
            engine="openai"
        )
        
        assert result["audio_data"] == b"fake_audio_data"
        assert result["format"] == "mp3"
        assert "duration" in result
    
    async def test_synthesize_speech_gtts(self, tts_service):
        """Test speech synthesis with gTTS."""
        with patch('gtts.gTTS') as mock_gtts, \
             patch('io.BytesIO') as mock_io:
            
            # Mock gTTS
            mock_tts_instance = Mock()
            mock_gtts.return_value = mock_tts_instance
            
            # Mock audio data
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b"gtts_audio_data"
            mock_io.return_value = mock_buffer
            
            result = await tts_service.synthesize_speech(
                text="Hello world",
                language="en",
                engine="gtts"
            )
        
        assert result["audio_data"] == b"gtts_audio_data"
        assert result["engine"] == "gtts"
    
    async def test_voice_customization(self, tts_service):
        """Test voice customization options."""
        result = await tts_service.synthesize_speech(
            text="Hello world",
            language="en",
            voice="female",
            speed=1.2,
            pitch=1.1,
            engine="openai"
        )
        
        # Verify customization parameters were applied
        assert "speed" in result["metadata"]
        assert "pitch" in result["metadata"]
    
    async def test_multiple_language_support(self, tts_service):
        """Test TTS with multiple languages."""
        languages = ["en", "es", "fr", "de", "ja"]
        
        for lang in languages:
            with patch.object(tts_service, '_synthesize_with_engine') as mock_synthesize:
                mock_synthesize.return_value = {
                    "audio_data": f"audio_for_{lang}".encode(),
                    "format": "wav",
                    "language": lang
                }
                
                result = await tts_service.synthesize_speech(
                    text=f"Hello in {lang}",
                    language=lang
                )
            
            assert result["language"] == lang


@pytest.mark.integration
class TestAIServiceChaining:
    """Test chaining multiple AI services together."""
    
    async def test_voice_to_voice_translation(
        self, 
        mock_whisper_service,
        mock_nllb_service,
        mock_tts_service,
        file_test_helper
    ):
        """Test complete voice-to-voice translation pipeline."""
        # Create test audio
        test_audio = file_test_helper.create_test_audio(3.0)
        
        # Mock service chain
        mock_whisper_service.transcribe_audio.return_value = {
            "text": "Where is the nearest hospital?",
            "language": "en",
            "confidence": 0.95
        }
        
        mock_nllb_service.translate.return_value = {
            "translated_text": "¿Dónde está el hospital más cercano?",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.92
        }
        
        mock_tts_service.synthesize_speech.return_value = {
            "audio_data": b"spanish_audio_data",
            "format": "wav",
            "language": "es"
        }
        
        # Execute pipeline
        with patch('app.services.whisper.WhisperService', return_value=mock_whisper_service), \
             patch('app.services.nllb.NLLBTranslationService', return_value=mock_nllb_service), \
             patch('app.services.tts.TTSService', return_value=mock_tts_service):
            
            from app.services.voice_pipeline import VoicePipeline
            
            pipeline = VoicePipeline()
            result = await pipeline.voice_to_voice_translation(
                audio_data=test_audio,
                target_language="es"
            )
        
        assert result["original_text"] == "Where is the nearest hospital?"
        assert result["translated_text"] == "¿Dónde está el hospital más cercano?"
        assert result["output_audio"] == b"spanish_audio_data"
    
    async def test_image_text_to_speech(
        self,
        mock_ocr_service,
        mock_nllb_service,
        mock_tts_service,
        file_test_helper
    ):
        """Test image OCR to speech pipeline."""
        # Create test image
        test_image = file_test_helper.create_test_image(300, 200)
        
        # Mock service responses
        mock_ocr_service.extract_text.return_value = {
            "text": "Emergency Exit",
            "language": "en",
            "confidence": 0.9
        }
        
        mock_nllb_service.translate.return_value = {
            "translated_text": "Salida de Emergencia",
            "source_language": "en",
            "target_language": "es"
        }
        
        mock_tts_service.synthesize_speech.return_value = {
            "audio_data": b"exit_audio_spanish",
            "format": "wav"
        }
        
        # Execute pipeline
        with patch('app.services.ocr.OCRService', return_value=mock_ocr_service), \
             patch('app.services.nllb.NLLBTranslationService', return_value=mock_nllb_service), \
             patch('app.services.tts.TTSService', return_value=mock_tts_service):
            
            from app.services.multimodal_pipeline import MultimodalPipeline
            
            pipeline = MultimodalPipeline()
            result = await pipeline.image_to_speech(
                image_data=test_image,
                target_language="es"
            )
        
        assert result["extracted_text"] == "Emergency Exit"
        assert result["translated_text"] == "Salida de Emergencia"
        assert result["audio_data"] == b"exit_audio_spanish"
