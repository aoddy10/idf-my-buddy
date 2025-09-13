"""OCR (Optical Character Recognition) service.

This module provides text extraction from images using various OCR engines
with support for multiple languages and document types.
"""

import asyncio
import base64
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import settings
from app.core.logging import LoggerMixin

# Type imports for static analysis - removed to avoid unused import warnings


class OCRService(LoggerMixin):
    """OCR service with multiple backend support.

    Supports Tesseract, EasyOCR, and PaddleOCR with intelligent fallback.
    Optimized for edge computing with preprocessing and multi-language support.
    """

    def __init__(self):
        super().__init__()
        # Backend availability flags
        self._tesseract_available = False
        self._opencv_available = False
        self._easyocr_available = False
        self._paddleocr_available = False

        # Backend instances
        self._pytesseract = None
        self._cv2 = None
        self._np = None
        self._Image = None
        self._easyocr_reader = None
        self._paddle_ocr = None

        self._default_languages = ["en"]

        # Initialize available OCR engines
        self._setup_service()

    def _setup_service(self):
        """Setup OCR service with available engines."""
        try:
            # Check and setup OpenCV + NumPy
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
                self._cv2 = cv2
                self._np = np
                self._opencv_available = True
                self.logger.debug("OpenCV and NumPy available")
            except ImportError:
                self.logger.debug("OpenCV or NumPy not available")

            # Check and setup Tesseract
            try:
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore
                self._pytesseract = pytesseract
                self._Image = Image
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
                self.logger.info("Tesseract OCR initialized")
            except ImportError:
                self.logger.debug("Tesseract or PIL not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize Tesseract: {e}")

            # Setup EasyOCR
            if getattr(settings, 'easyocr_enabled', True):
                try:
                    import easyocr  # type: ignore
                    # Initialize EasyOCR with default languages
                    self._easyocr_reader = easyocr.Reader(
                        self._default_languages,
                        gpu=getattr(settings, 'use_gpu_for_ocr', False),
                        model_storage_directory=getattr(settings, 'model_cache_dir', './models'),
                        download_enabled=True
                    )
                    self._easyocr_available = True
                    self.logger.info("EasyOCR initialized")
                except ImportError:
                    self.logger.debug("EasyOCR not available")
                except Exception as e:
                    self.logger.error(f"Failed to initialize EasyOCR: {e}")

            # Setup PaddleOCR
            if getattr(settings, 'paddleocr_enabled', False):
                try:
                    from paddleocr import PaddleOCR  # type: ignore
                    # Initialize PaddleOCR
                    self._paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",  # Default language
                        use_gpu=getattr(settings, 'use_gpu_for_ocr', False),
                        show_log=False
                    )
                    self._paddleocr_available = True
                    self.logger.info("PaddleOCR initialized")
                except ImportError:
                    self.logger.debug("PaddleOCR not available")
                except Exception as e:
                    self.logger.error(f"Failed to initialize PaddleOCR: {e}")

            if not any([self._tesseract_available, self._easyocr_available, self._paddleocr_available]):
                self.logger.warning("No OCR backend available - text extraction will be disabled")

        except Exception as e:
            self.logger.error(f"Failed to setup OCR service: {e}")

    async def extract_text(
        self,
        image_data: bytes | BinaryIO | Path | Any | str,
        language: str = "en",
        engine: str | None = None,
        preprocess: bool = True,
        confidence_threshold: float = 0.5
    ) -> dict[str, Any]:
        """Extract text from image using OCR.

        Args:
            image_data: Image data, file object, path, numpy array, or base64 string
            language: Language code for OCR
            engine: Specific OCR engine to use ("tesseract", "easyocr", "paddleocr")
            preprocess: Whether to apply image preprocessing
            confidence_threshold: Minimum confidence for text detection

        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Check if any OCR backend is available
            if not any([self._tesseract_available, self._easyocr_available, self._paddleocr_available]):
                return {
                    "success": False,
                    "error": "No OCR backend available",
                    "text": "",
                    "confidence": 0.0,
                    "processing_time": 0.0
                }

            # Prepare image for processing
            image = await self._prepare_image(image_data)

            # Apply preprocessing if requested and OpenCV is available
            if preprocess and self._opencv_available and image is not None:
                image = self._preprocess_image(image)

            # Use specified engine or fallback to available ones
            if engine == "tesseract" and self._tesseract_available:
                return await self._extract_tesseract(image, language, confidence_threshold)
            elif engine == "easyocr" and self._easyocr_available:
                return await self._extract_easyocr(image, language, confidence_threshold)
            elif engine == "paddleocr" and self._paddleocr_available:
                return await self._extract_paddleocr(image, language, confidence_threshold)
            else:
                # Try available engines in order
                for fallback_engine in ["easyocr", "paddleocr", "tesseract"]:
                    if self._is_engine_available(fallback_engine):
                        self.logger.info(f"Falling back to {fallback_engine}")
                        return await self.extract_text(
                            image, language, fallback_engine, False, confidence_threshold
                        )

                return {
                    "success": False,
                    "error": "No suitable OCR engine available",
                    "text": "",
                    "confidence": 0.0,
                    "processing_time": 0.0
                }

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "processing_time": 0.0
            }

    async def _prepare_image(self, image_data: bytes | BinaryIO | Path | Any | str) -> Any:
        """Prepare image for OCR processing."""

        try:
            # Handle numpy array (if available)
            if hasattr(image_data, 'shape'):  # Check if it's numpy-like
                return image_data

            elif isinstance(image_data, Path):
                if self._opencv_available and self._cv2:
                    return self._cv2.imread(str(image_data))
                elif self._Image:
                    # Fallback to PIL if available
                    pil_image = self._Image.open(image_data)
                    return pil_image
                else:
                    raise RuntimeError("No image processing library available")

            elif isinstance(image_data, str):
                if not self._opencv_available or not self._cv2 or not self._np:
                    raise RuntimeError("OpenCV required for base64 image processing")

                # Handle base64 encoded images
                if image_data.startswith("data:image"):
                    # Remove data URL prefix
                    image_data = image_data.split(",")[1]

                # Decode base64
                image_bytes = base64.b64decode(image_data)
                nparr = self._np.frombuffer(image_bytes, self._np.uint8)
                return self._cv2.imdecode(nparr, self._cv2.IMREAD_COLOR)

            elif isinstance(image_data, bytes):
                if not self._opencv_available or not self._cv2 or not self._np:
                    raise RuntimeError("OpenCV required for bytes image processing")
                nparr = self._np.frombuffer(image_data, self._np.uint8)
                return self._cv2.imdecode(nparr, self._cv2.IMREAD_COLOR)

            else:
                # Handle file-like objects
                if hasattr(image_data, 'read') and callable(getattr(image_data, 'read', None)):
                    image_bytes = image_data.read()  # type: ignore
                    if not self._opencv_available or not self._cv2 or not self._np:
                        raise RuntimeError("OpenCV required for file image processing")
                    nparr = self._np.frombuffer(image_bytes, self._np.uint8)
                    return self._cv2.imdecode(nparr, self._cv2.IMREAD_COLOR)
                else:
                    raise ValueError(f"Unsupported image data type: {type(image_data)}")

        except Exception as e:
            self.logger.error(f"Image preparation failed: {e}")
            return None

    def _preprocess_image(self, image: Any) -> Any:
        """Apply preprocessing to improve OCR accuracy."""
        if not self._opencv_available or not self._cv2 or not self._np:
            self.logger.warning("OpenCV not available - skipping preprocessing")
            return image

        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply Gaussian blur to reduce noise
            blurred = self._cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive threshold
            _, thresh = self._cv2.threshold(blurred, 0, 255, self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU)

            # Apply morphological operations to clean up the image
            kernel = self._np.ones((1, 1), self._np.uint8)
            processed = self._cv2.morphologyEx(thresh, self._cv2.MORPH_CLOSE, kernel)
            processed = self._cv2.morphologyEx(processed, self._cv2.MORPH_OPEN, kernel)

            return processed

        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image

    def _is_engine_available(self, engine: str) -> bool:
        """Check if specified engine is available."""
        availability_map = {
            "tesseract": self._tesseract_available,
            "easyocr": self._easyocr_available,
            "paddleocr": self._paddleocr_available,
        }
        return availability_map.get(engine, False)

    async def _extract_tesseract(
        self,
        image: Any,
        language: str,
        confidence_threshold: float
    ) -> dict[str, Any]:
        """Extract text using Tesseract OCR."""
        if not self._tesseract_available or not self._pytesseract:
            raise RuntimeError("Tesseract not available")

        import time
        start_time = time.time()

        try:
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _run_tesseract():
                # Get OCR data with confidence scores
                data = self._pytesseract.image_to_data(  # type: ignore
                    image,
                    lang=language,
                    output_type=self._pytesseract.Output.DICT  # type: ignore
                )

                # Extract text and calculate average confidence
                texts = []
                confidences = []

                for i, conf in enumerate(data['conf']):
                    if int(conf) > confidence_threshold * 100:  # Tesseract uses 0-100 scale
                        text = data['text'][i].strip()
                        if text:
                            texts.append(text)
                            confidences.append(int(conf) / 100.0)  # Convert to 0-1 scale

                extracted_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return {
                    "success": True,
                    "text": extracted_text,
                    "confidence": avg_confidence,
                    "language": language,
                    "engine": "tesseract",
                    "processing_time": time.time() - start_time,
                    "word_count": len(texts),
                    "bounding_boxes": []
                }

            result = await loop.run_in_executor(None, _run_tesseract)
            return result

        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            raise

    async def _extract_easyocr(
        self,
        image: Any,
        language: str,
        confidence_threshold: float
    ) -> dict[str, Any]:
        """Extract text using EasyOCR."""
        if not self._easyocr_available or not self._easyocr_reader:
            raise RuntimeError("EasyOCR not available")

        import time
        start_time = time.time()

        try:
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _run_easyocr():
                results = self._easyocr_reader.readtext(image, detail=1)  # type: ignore

                # Process results
                texts = []
                confidences = []
                bounding_boxes = []

                for bbox, text, confidence in results:
                    if confidence >= confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)
                        bounding_boxes.append(bbox)

                extracted_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return {
                    "success": True,
                    "text": extracted_text,
                    "confidence": avg_confidence,
                    "language": language,
                    "engine": "easyocr",
                    "processing_time": time.time() - start_time,
                    "word_count": len(texts),
                    "bounding_boxes": bounding_boxes
                }

            result = await loop.run_in_executor(None, _run_easyocr)
            return result

        except Exception as e:
            self.logger.error(f"EasyOCR failed: {e}")
            raise

    async def _extract_paddleocr(
        self,
        image: Any,
        language: str,
        confidence_threshold: float
    ) -> dict[str, Any]:
        """Extract text using PaddleOCR."""
        if not self._paddleocr_available or not self._paddle_ocr:
            raise RuntimeError("PaddleOCR not available")

        import time
        start_time = time.time()

        try:
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _run_paddleocr():
                results = self._paddle_ocr.ocr(image, cls=True)  # type: ignore

                # Process results
                texts = []
                confidences = []
                bounding_boxes = []

                if results and results[0]:
                    for line in results[0]:
                        if line:
                            bbox, (text, confidence) = line
                            if confidence >= confidence_threshold:
                                texts.append(text)
                                confidences.append(confidence)
                                bounding_boxes.append(bbox)

                extracted_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return {
                    "success": True,
                    "text": extracted_text,
                    "confidence": avg_confidence,
                    "language": language,
                    "engine": "paddleocr",
                    "processing_time": time.time() - start_time,
                    "word_count": len(texts),
                    "bounding_boxes": bounding_boxes
                }

            result = await loop.run_in_executor(None, _run_paddleocr)
            return result

        except Exception as e:
            self.logger.error(f"PaddleOCR failed: {e}")
            raise

    async def get_available_languages(self, engine: str | None = None) -> list[str]:
        """Get list of supported languages for the specified engine."""
        languages = []

        try:
            if engine == "tesseract" and self._tesseract_available and self._pytesseract:
                try:
                    langs = self._pytesseract.get_languages()
                    languages.extend(langs)
                except Exception as e:
                    self.logger.error(f"Failed to get Tesseract languages: {e}")

            elif engine == "easyocr" and self._easyocr_available:
                # EasyOCR supported languages
                languages = ['en', 'ch_sim', 'ch_tra', 'th', 'ja', 'ko', 'vi', 'ar']

            elif engine == "paddleocr" and self._paddleocr_available:
                # PaddleOCR supported languages
                languages = ['en', 'ch', 'french', 'german', 'korean', 'japan']

            else:
                # Return all supported languages from available engines
                if self._tesseract_available:
                    languages.extend(await self.get_available_languages("tesseract"))
                if self._easyocr_available:
                    languages.extend(await self.get_available_languages("easyocr"))
                if self._paddleocr_available:
                    languages.extend(await self.get_available_languages("paddleocr"))

        except Exception as e:
            self.logger.error(f"Failed to get available languages: {e}")

        return list(set(languages)) if languages else ["en"]  # Default to English
