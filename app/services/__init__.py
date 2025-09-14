"""Services package for AI and external integrations."""

# Core services with conditional imports for optional dependencies
try:
    from .nllb import NLLBTranslationService
except ImportError:
    NLLBTranslationService = None

try:
    from .tts import TTSService
except ImportError:
    TTSService = None

try:
    from .voice_navigation import VoiceNavigationService, VoiceInstructionTemplates
except ImportError:
    VoiceNavigationService = None
    VoiceInstructionTemplates = None

try:
    from .whisper import WhisperService
except ImportError:
    WhisperService = None

# Optional imports that may fail due to missing dependencies
try:
    from .maps import NavigationService
except ImportError:
    NavigationService = None

try:
    from .ocr import OCRService
except ImportError:
    OCRService = None

__all__ = [
    "WhisperService",
    "NLLBTranslationService", 
    "TTSService",
    "NavigationService",
    "VoiceNavigationService",
    "VoiceInstructionTemplates"
]

# Add OCRService to exports if available
if OCRService is not None:
    __all__.append("OCRService")
