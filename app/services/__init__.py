"""Services package for AI and external integrations."""

from .nllb import NLLBTranslationService
from .tts import TTSService
from .whisper import WhisperService

# Optional imports that may fail due to missing dependencies
try:
    from .ocr import OCRService
except ImportError:
    OCRService = None

__all__ = [
    "WhisperService",
    "NLLBTranslationService",
    "TTSService"
]

# Add OCRService to exports if available
if OCRService is not None:
    __all__.append("OCRService")
