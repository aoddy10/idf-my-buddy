"""OCR (Optical Character Recognition) service.

This module provides text extraction from images using various OCR engines
with support for multiple languages and document types.
"""

import logging
import asyncio
import tempfile
from typing import Optional, Dict, Any, List, Union, BinaryIO
from pathlib import Path
import io
import base64

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class OCRService(LoggerMixin):
    """OCR service with multiple backend support.
    
    Supports Tesseract, EasyOCR, and PaddleOCR with intelligent fallback.
    Optimized for edge computing with preprocessing and multi-language support.
    """
    
    def __init__(self):
        super().__init__()
        self._tesseract_available = False
        self._easyocr_reader = None
        self._paddle_ocr = None
        self._default_languages = ["en"]
        
        # Initialize available OCR engines
        self._setup_service()
    
    def _setup_service(self):
        """Setup OCR service with available engines."""
        try:
            # Setup Tesseract
            if TESSERACT_AVAILABLE:
                self._setup_tesseract()
            
            # Setup EasyOCR  
            if EASYOCR_AVAILABLE and settings.EASYOCR_ENABLED:
                self._setup_easyocr()
            
            # Setup PaddleOCR
            if PADDLEOCR_AVAILABLE and settings.PADDLEOCR_ENABLED:
                self._setup_paddleocr()
            
            if not any([self._tesseract_available, self._easyocr_reader, self._paddle_ocr]):
                self.logger.warning("No OCR backend available - text extraction will be disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to setup OCR service: {e}")
    
    def _setup_tesseract(self):
        """Setup Tesseract OCR engine."""
        try:
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            self.logger.info("Tesseract OCR initialized")
            
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {e}")
            self._tesseract_available = False
    
    def _setup_easyocr(self):
        """Setup EasyOCR engine."""
        try:
            # Initialize EasyOCR with default languages
            self._easyocr_reader = easyocr.Reader(
                self._default_languages,
                gpu=settings.USE_GPU_FOR_OCR,
                model_storage_directory=settings.MODEL_CACHE_DIR,
                download_enabled=True
            )
            self.logger.info("EasyOCR initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            self._easyocr_reader = None
    
    def _setup_paddleocr(self):
        """Setup PaddleOCR engine."""
        try:
            # Initialize PaddleOCR
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",  # Default language
                use_gpu=settings.USE_GPU_FOR_OCR,
                show_log=False
            )
            self.logger.info("PaddleOCR initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            self._paddle_ocr = None
    
    async def extract_text(
        self,
        image_data: Union[bytes, BinaryIO, Path, np.ndarray, str],
        language: str = "en",
        engine: Optional[str] = None,
        preprocess: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Extract text from image using OCR.
        
        Args:
            image_data: Image data, file object, path, numpy array, or base64 string
            language: Language code for OCR
            engine: Specific OCR engine to use ("tesseract", "easyocr", "paddleocr")
            preprocess: Apply image preprocessing
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Prepare image
            image = await self._prepare_image(image_data)
            
            # Apply preprocessing if requested
            if preprocess:
                image = self._preprocess_image(image)
            
            # Choose OCR engine
            if engine is None:
                engine = self._choose_best_engine(language)
            
            # Extract text based on engine
            if engine == "tesseract" and self._tesseract_available:
                return await self._extract_tesseract(image, language, confidence_threshold)
            elif engine == "easyocr" and self._easyocr_reader:
                return await self._extract_easyocr(image, language, confidence_threshold)
            elif engine == "paddleocr" and self._paddle_ocr:
                return await self._extract_paddleocr(image, language, confidence_threshold)
            else:
                # Try available engines in order
                for fallback_engine in ["easyocr", "paddleocr", "tesseract"]:
                    if self._is_engine_available(fallback_engine):
                        self.logger.info(f"Falling back to {fallback_engine}")
                        return await self.extract_text(
                            image, language, fallback_engine, False, confidence_threshold
                        )
                
                raise RuntimeError("No OCR engine available")
                
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise
    
    async def _prepare_image(self, image_data: Union[bytes, BinaryIO, Path, np.ndarray, str]) -> np.ndarray:
        """Prepare image for OCR processing."""
        
        if isinstance(image_data, np.ndarray):
            return image_data
        
        elif isinstance(image_data, Path):
            return cv2.imread(str(image_data))
        
        elif isinstance(image_data, str):
            # Handle base64 encoded images
            if image_data.startswith("data:image"):
                # Remove data URL prefix
                image_data = image_data.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            # File-like object
            image_bytes = image_data.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve OCR accuracy."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return processed
            
        except Exception as e:
            self.logger.warning(f"Preprocessing failed, using original: {e}")
            return image
    
    def _choose_best_engine(self, language: str) -> str:
        """Choose the best OCR engine for the given language."""
        # Language-specific engine preferences
        asian_languages = ["zh", "ja", "ko", "th", "vi"]
        
        if language in asian_languages and self._paddle_ocr:
            return "paddleocr"
        elif self._easyocr_reader:
            return "easyocr"
        elif self._tesseract_available:
            return "tesseract"
        else:
            # Return first available
            for engine in ["easyocr", "paddleocr", "tesseract"]:
                if self._is_engine_available(engine):
                    return engine
            raise RuntimeError("No OCR engine available")
    
    def _is_engine_available(self, engine: str) -> bool:
        """Check if OCR engine is available."""
        if engine == "tesseract":
            return self._tesseract_available
        elif engine == "easyocr":
            return self._easyocr_reader is not None
        elif engine == "paddleocr":
            return self._paddle_ocr is not None
        return False
    
    async def _extract_tesseract(
        self,
        image: np.ndarray,
        language: str,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Extract text using Tesseract OCR."""
        
        # Run OCR in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._tesseract_ocr,
            image, language, confidence_threshold
        )
        
        return result
    
    def _tesseract_ocr(self, image: np.ndarray, language: str, confidence_threshold: float) -> Dict[str, Any]:
        """Tesseract OCR processing (blocking operation)."""
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with confidence filtering
            words = []
            full_text = []
            
            for i in range(len(ocr_data["text"])):
                confidence = int(ocr_data["conf"][i])
                text = ocr_data["text"][i].strip()
                
                if text and confidence >= confidence_threshold * 100:
                    words.append({
                        "text": text,
                        "confidence": confidence / 100.0,
                        "bbox": {
                            "x": int(ocr_data["left"][i]),
                            "y": int(ocr_data["top"][i]),
                            "width": int(ocr_data["width"][i]),
                            "height": int(ocr_data["height"][i])
                        }
                    })
                    full_text.append(text)
            
            return {
                "text": " ".join(full_text),
                "words": words,
                "language": language,
                "engine": "tesseract",
                "confidence": sum(w["confidence"] for w in words) / len(words) if words else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    async def _extract_easyocr(
        self,
        image: np.ndarray,
        language: str,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        
        # Run OCR in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._easyocr_process,
            image, language, confidence_threshold
        )
        
        return result
    
    def _easyocr_process(self, image: np.ndarray, language: str, confidence_threshold: float) -> Dict[str, Any]:
        """EasyOCR processing (blocking operation)."""
        try:
            # Run EasyOCR
            results = self._easyocr_reader.readtext(image, detail=1)
            
            words = []
            full_text = []
            
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    # Convert bbox to standard format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    words.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": {
                            "x": int(min(x_coords)),
                            "y": int(min(y_coords)),
                            "width": int(max(x_coords) - min(x_coords)),
                            "height": int(max(y_coords) - min(y_coords))
                        }
                    })
                    full_text.append(text)
            
            return {
                "text": " ".join(full_text),
                "words": words,
                "language": language,
                "engine": "easyocr",
                "confidence": sum(w["confidence"] for w in words) / len(words) if words else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"EasyOCR processing failed: {e}")
            raise
    
    async def _extract_paddleocr(
        self,
        image: np.ndarray,
        language: str,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        
        # Run OCR in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._paddleocr_process,
            image, language, confidence_threshold
        )
        
        return result
    
    def _paddleocr_process(self, image: np.ndarray, language: str, confidence_threshold: float) -> Dict[str, Any]:
        """PaddleOCR processing (blocking operation)."""
        try:
            # Run PaddleOCR
            results = self._paddle_ocr.ocr(image, cls=True)
            
            words = []
            full_text = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    if confidence >= confidence_threshold:
                        # Convert bbox to standard format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        words.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": {
                                "x": int(min(x_coords)),
                                "y": int(min(y_coords)),
                                "width": int(max(x_coords) - min(x_coords)),
                                "height": int(max(y_coords) - min(y_coords))
                            }
                        })
                        full_text.append(text)
            
            return {
                "text": " ".join(full_text),
                "words": words,
                "language": language,
                "engine": "paddleocr",
                "confidence": sum(w["confidence"] for w in words) / len(words) if words else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"PaddleOCR processing failed: {e}")
            raise
    
    async def extract_structured_text(
        self,
        image_data: Union[bytes, BinaryIO, Path, np.ndarray, str],
        document_type: str = "general",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Extract structured text for specific document types."""
        
        # Extract basic text first
        ocr_result = await self.extract_text(image_data, language)
        
        # Apply document-specific processing
        if document_type == "menu":
            return self._structure_menu_text(ocr_result)
        elif document_type == "receipt":
            return self._structure_receipt_text(ocr_result)
        elif document_type == "sign":
            return self._structure_sign_text(ocr_result)
        else:
            return ocr_result
    
    def _structure_menu_text(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure menu text for better parsing."""
        # TODO: Implement menu-specific text structuring
        # - Identify sections (appetizers, mains, desserts)
        # - Extract prices
        # - Group items with descriptions
        
        ocr_result["document_type"] = "menu"
        ocr_result["structured"] = {
            "sections": [],
            "items": [],
            "prices": []
        }
        
        return ocr_result
    
    def _structure_receipt_text(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure receipt text for better parsing."""
        # TODO: Implement receipt-specific text structuring
        # - Extract items and prices
        # - Find total, tax, etc.
        # - Identify merchant info
        
        ocr_result["document_type"] = "receipt"
        ocr_result["structured"] = {
            "merchant": "",
            "items": [],
            "total": 0.0,
            "tax": 0.0,
            "date": ""
        }
        
        return ocr_result
    
    def _structure_sign_text(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure sign text for better parsing."""
        # TODO: Implement sign-specific text structuring
        # - Identify main message
        # - Extract directions/instructions
        # - Classify sign type
        
        ocr_result["document_type"] = "sign"
        ocr_result["structured"] = {
            "main_text": ocr_result["text"],
            "sign_type": "informational",
            "instructions": []
        }
        
        return ocr_result
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages by engine."""
        return {
            "tesseract": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
            "easyocr": ["en", "ch_sim", "ch_tra", "ja", "ko", "th", "vi", "ar", "fr", "de", "es"],
            "paddleocr": ["en", "ch", "ja", "ko", "ta", "te", "ka", "hi", "ar"]
        }
    
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return any([self._tesseract_available, self._easyocr_reader, self._paddle_ocr])
