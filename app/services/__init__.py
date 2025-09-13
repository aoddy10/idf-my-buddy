"""Services package for AI and external integrations."""

from .whisper import WhisperService
from .nllb import NLLBTranslationService
from .ocr import OCRService
from .tts import TTSService

__all__ = [
    "WhisperService",
    "NLLBTranslationService", 
    "OCRService",
    "TTSService"
]