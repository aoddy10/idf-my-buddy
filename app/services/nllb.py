"""NLLB Translation service.

This module provides multilingual translation using Meta's No Language Left Behind (NLLB) model
with support for edge computing and cloud translation fallbacks.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False


class NLLBTranslationService(LoggerMixin):
    """NLLB-based translation service.
    
    Supports both local NLLB models and cloud translation APIs with intelligent fallback.
    Optimized for edge computing with model caching and efficient batching.
    """
    
    def __init__(self):
        super().__init__()
        self._local_model = None
        self._tokenizer = None
        self._translator_pipeline = None
        self._google_translator = None
        self._device = None
        self._model_name = "facebook/nllb-200-distilled-600M"  # Default model
        
        # Language code mappings for NLLB
        self._nllb_codes = self._get_nllb_language_codes()
        
        # Initialize based on available resources
        self._setup_service()
    
    def _setup_service(self):
        """Setup the translation service based on available resources."""
        try:
            if TRANSFORMERS_AVAILABLE and settings.NLLB_USE_LOCAL:
                self._setup_local_model()
            
            if GOOGLE_TRANSLATE_AVAILABLE:
                self._setup_google_translator()
                
            if not self._local_model and not self._google_translator:
                self.logger.warning("No translation backend available - translation will be disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to setup translation service: {e}")
    
    def _setup_local_model(self):
        """Setup local NLLB model for edge computing."""
        try:
            # Determine optimal model size based on available resources
            self._model_name = self._get_optimal_model_size()
            
            # Setup device (CPU/GPU)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading NLLB model {self._model_name} on {self._device}")
            
            # Load tokenizer and model with caching
            cache_dir = settings.MODEL_CACHE_DIR
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=cache_dir
            )
            
            self._local_model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None
            )
            
            # Create translation pipeline
            self._translator_pipeline = pipeline(
                "translation",
                model=self._local_model,
                tokenizer=self._tokenizer,
                device=0 if self._device == "cuda" else -1
            )
            
            self.logger.info("Local NLLB model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load local NLLB model: {e}")
            self._local_model = None
    
    def _setup_google_translator(self):
        """Setup Google Translate as fallback."""
        try:
            self._google_translator = GoogleTranslator()
            self.logger.info("Google Translate fallback initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Google Translate: {e}")
            self._google_translator = None
    
    def _get_optimal_model_size(self) -> str:
        """Determine optimal NLLB model size based on system resources."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            # Choose model based on memory
            if available_gb >= 16:
                return "facebook/nllb-200-3.3B"
            elif available_gb >= 8:
                return "facebook/nllb-200-1.3B"
            elif available_gb >= 4:
                return "facebook/nllb-200-distilled-1.3B"
            else:
                return "facebook/nllb-200-distilled-600M"
                
        except ImportError:
            # Fallback if psutil not available
            return "facebook/nllb-200-distilled-600M"
    
    def _get_nllb_language_codes(self) -> Dict[str, str]:
        """Get NLLB language code mappings."""
        return {
            "en": "eng_Latn",  # English
            "es": "spa_Latn",  # Spanish
            "fr": "fra_Latn",  # French
            "de": "deu_Latn",  # German
            "it": "ita_Latn",  # Italian
            "pt": "por_Latn",  # Portuguese
            "ru": "rus_Cyrl",  # Russian
            "zh": "zho_Hans",  # Chinese (Simplified)
            "ja": "jpn_Jpan",  # Japanese
            "ko": "kor_Hang",  # Korean
            "ar": "arb_Arab",  # Arabic
            "hi": "hin_Deva",  # Hindi
            "th": "tha_Thai",  # Thai
            "vi": "vie_Latn",  # Vietnamese
            "tr": "tur_Latn",  # Turkish
            "pl": "pol_Latn",  # Polish
            "nl": "nld_Latn",  # Dutch
            "sv": "swe_Latn",  # Swedish
            "da": "dan_Latn",  # Danish
            "no": "nob_Latn",  # Norwegian
            "fi": "fin_Latn",  # Finnish
            "he": "heb_Hebr",  # Hebrew
            "cs": "ces_Latn",  # Czech
            "hu": "hun_Latn",  # Hungarian
            "ro": "ron_Latn",  # Romanian
            "bg": "bul_Cyrl",  # Bulgarian
            "hr": "hrv_Latn",  # Croatian
            "sk": "slk_Latn",  # Slovak
            "sl": "slv_Latn",  # Slovenian
            "et": "est_Latn",  # Estonian
            "lv": "lvs_Latn",  # Latvian
            "lt": "lit_Latn",  # Lithuanian
            "uk": "ukr_Cyrl",  # Ukrainian
            "be": "bel_Cyrl",  # Belarusian
            "mk": "mkd_Cyrl",  # Macedonian
            "sr": "srp_Cyrl",  # Serbian
            "bs": "bos_Latn",  # Bosnian
            "mt": "mlt_Latn",  # Maltese
            "ga": "gle_Latn",  # Irish
            "cy": "cym_Latn",  # Welsh
            "is": "isl_Latn",  # Icelandic
            "fo": "fao_Latn",  # Faroese
        }
    
    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (ISO 639-1)
            source_language: Source language code (None for auto-detection)
            max_length: Maximum output length
            
        Returns:
            Dict containing translation results
        """
        try:
            # Validate inputs
            if not text.strip():
                return {
                    "translated_text": "",
                    "source_language": "unknown",
                    "target_language": target_language,
                    "confidence": 0.0,
                    "backend": "none"
                }
            
            # Try local model first if available
            if self._local_model and target_language in self._nllb_codes:
                return await self._translate_local(
                    text, target_language, source_language, max_length
                )
            
            # Fallback to Google Translate
            elif self._google_translator:
                return await self._translate_google(
                    text, target_language, source_language
                )
            
            else:
                raise RuntimeError("No translation backend available")
                
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise
    
    async def _translate_local(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Translate using local NLLB model."""
        
        try:
            # Detect source language if not provided
            if source_language is None:
                source_language = await self.detect_language(text)
            
            # Get NLLB language codes
            source_code = self._nllb_codes.get(source_language, "eng_Latn")
            target_code = self._nllb_codes.get(target_language, "eng_Latn")
            
            # Skip translation if same language
            if source_code == target_code:
                return {
                    "translated_text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "confidence": 1.0,
                    "backend": "local_nllb"
                }
            
            # Set source language for tokenizer
            self._tokenizer.src_lang = source_code
            
            # Tokenize and translate
            inputs = self._tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate translation in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_translation,
                inputs, target_code, max_length
            )
            
            return {
                "translated_text": result,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 0.9,  # NLLB is generally high quality
                "backend": "local_nllb"
            }
            
        except Exception as e:
            self.logger.error(f"Local translation failed: {e}")
            raise
    
    def _generate_translation(self, inputs: Dict[str, Any], target_code: str, max_length: int) -> str:
        """Generate translation (blocking operation for thread pool)."""
        with torch.no_grad():
            generated_tokens = self._local_model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.lang_code_to_id[target_code],
                max_length=max_length,
                num_beams=4,
                do_sample=True,
                temperature=0.7
            )
            
            # Decode result
            result = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]
            
            return result.strip()
    
    async def _translate_google(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translate using Google Translate API."""
        
        try:
            # Run in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._google_translator.translate(
                    text,
                    dest=target_language,
                    src=source_language
                )
            )
            
            return {
                "translated_text": result.text,
                "source_language": result.src,
                "target_language": target_language,
                "confidence": getattr(result, 'confidence', 0.8),
                "backend": "google_translate"
            }
            
        except Exception as e:
            self.logger.error(f"Google Translate failed: {e}")
            raise
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            if self._google_translator:
                # Use Google Translate for language detection
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._google_translator.detect(text)
                )
                return result.lang
            
            else:
                # Simple heuristic-based detection as fallback
                return self._detect_language_heuristic(text)
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def _detect_language_heuristic(self, text: str) -> str:
        """Simple heuristic-based language detection."""
        # This is a very basic implementation
        # In a real system, you'd use a proper language detection library
        
        text_lower = text.lower()
        
        # Check for common words/patterns
        if any(word in text_lower for word in ["the", "and", "is", "to", "a"]):
            return "en"
        elif any(word in text_lower for word in ["el", "la", "de", "es", "en"]):
            return "es"
        elif any(word in text_lower for word in ["le", "de", "et", "à", "un"]):
            return "fr"
        elif any(word in text_lower for word in ["der", "die", "das", "und", "ist"]):
            return "de"
        else:
            return "en"  # Default fallback
    
    async def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """Translate multiple texts efficiently."""
        
        # For now, translate one by one
        # TODO: Implement proper batching for local model
        results = []
        
        for text in texts:
            try:
                result = await self.translate_text(
                    text, target_language, source_language, max_length
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch translation failed for text: {e}")
                results.append({
                    "translated_text": text,
                    "source_language": "unknown",
                    "target_language": target_language,
                    "confidence": 0.0,
                    "backend": "error",
                    "error": str(e)
                })
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self._nllb_codes.keys())
    
    def is_available(self) -> bool:
        """Check if translation service is available."""
        return self._local_model is not None or self._google_translator is not None
