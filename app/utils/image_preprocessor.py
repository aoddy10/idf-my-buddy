"""Image preprocessing utilities for menu OCR enhancement.

This module provides specialized image preprocessing functions to improve OCR
accuracy for restaurant menus, handling various lighting conditions, angles,
and menu formats.
"""

import io
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.core.logging import LoggerMixin

# Optional imports for advanced preprocessing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class MenuImagePreprocessor(LoggerMixin):
    """Preprocessor for menu images to enhance OCR accuracy.
    
    Handles common menu photo issues:
    - Poor lighting conditions
    - Perspective distortion
    - Noise and blur
    - Format inconsistencies
    """

    def __init__(self):
        super().__init__()
        self.cv2_available = CV2_AVAILABLE
        
        if not self.cv2_available:
            self.logger.warning("OpenCV not available - using basic preprocessing only")

    def preprocess_menu_image(
        self, 
        image_data: bytes,
        enhance_contrast: bool = True,
        correct_perspective: bool = True,
        reduce_noise: bool = True
    ) -> bytes:
        """Preprocess menu image for optimal OCR results.
        
        Args:
            image_data: Raw image bytes
            enhance_contrast: Whether to enhance image contrast
            correct_perspective: Whether to apply perspective correction
            reduce_noise: Whether to apply noise reduction
            
        Returns:
            Preprocessed image bytes
        """
        try:
            # Load image from bytes
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Basic PIL preprocessing
            processed_image = self._basic_preprocessing(
                pil_image, enhance_contrast, reduce_noise
            )
            
            # Advanced OpenCV preprocessing if available
            if self.cv2_available and (correct_perspective or reduce_noise):
                processed_image = self._advanced_preprocessing(
                    processed_image, correct_perspective, reduce_noise
                )
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            processed_image.save(output_buffer, format='PNG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            return image_data  # Return original on failure

    def _basic_preprocessing(
        self, 
        image: Image.Image, 
        enhance_contrast: bool, 
        reduce_noise: bool
    ) -> Image.Image:
        """Apply basic PIL-based preprocessing."""
        
        # Resize if too large (limit to 2048px on longest side for performance)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_width = int(image.size[0] * ratio)
            new_height = int(image.size[1] * ratio)
            new_size = (new_width, new_height)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image to {new_size}")
        
        # Enhance contrast for better text visibility
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # 20% contrast boost
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.1)  # 10% sharpness boost
        
        # Basic noise reduction using PIL filters
        if reduce_noise:
            # Apply gentle noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
        return image

    def _advanced_preprocessing(
        self, 
        image: Image.Image, 
        correct_perspective: bool, 
        reduce_noise: bool
    ) -> Image.Image:
        """Apply advanced OpenCV-based preprocessing."""
        if not self.cv2_available or not CV2_AVAILABLE:
            return image
            
        try:
            import cv2  # Import in function scope to satisfy type checker
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply advanced noise reduction
            if reduce_noise:
                # Bilateral filter preserves edges while reducing noise
                cv_image = cv2.bilateralFilter(cv_image, 9, 75, 75)
                
                # Gaussian blur for additional noise reduction
                cv_image = cv2.GaussianBlur(cv_image, (3, 3), 0)
            
            # Perspective correction for angled menu photos
            if correct_perspective:
                cv_image = self._correct_perspective(cv_image)
            
            # Convert back to PIL format
            processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Advanced preprocessing failed: {str(e)}")
            return image

    def _correct_perspective(self, cv_image: np.ndarray) -> np.ndarray:
        """Attempt to correct perspective distortion in menu photos."""
        if not CV2_AVAILABLE:
            return cv_image
            
        try:
            import cv2  # Import in function scope to satisfy type checker
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Find edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest rectangular contour (likely the menu)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If we found a quadrilateral
                if len(approx) == 4:
                    # Get perspective transform
                    pts = approx.reshape(4, 2).astype(np.float32)
                    
                    # Order points: top-left, top-right, bottom-right, bottom-left
                    rect = self._order_points(pts)
                    
                    # Calculate dimensions for output rectangle
                    width = max(
                        float(np.linalg.norm(rect[1] - rect[0])),
                        float(np.linalg.norm(rect[2] - rect[3]))
                    )
                    height = max(
                        float(np.linalg.norm(rect[3] - rect[0])),
                        float(np.linalg.norm(rect[2] - rect[1]))
                    )
                    
                    # Define output rectangle
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)
                    
                    # Apply perspective transform
                    matrix = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(
                        cv_image, matrix, (int(width), int(height))
                    )
                    
                    self.logger.debug("Applied perspective correction")
                    return warped
                    
        except Exception as e:
            self.logger.debug(f"Perspective correction failed: {str(e)}")
            
        return cv_image

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left."""
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left has smallest sum, bottom-right has largest sum
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference, bottom-left has largest difference
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def validate_image_quality(self, image_data: bytes) -> dict:
        """Assess image quality for OCR suitability.
        
        Returns:
            Dictionary with quality metrics and recommendations
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Basic quality checks
            width, height = image.size
            total_pixels = width * height
            
            quality_report = {
                "resolution_adequate": total_pixels >= 300000,  # ~0.3MP minimum
                "aspect_ratio_reasonable": 0.3 <= width/height <= 3.0,
                "size_bytes": len(image_data),
                "dimensions": {"width": width, "height": height},
                "format": image.format,
                "mode": image.mode,
                "recommendations": []
            }
            
            # Add recommendations based on quality
            if total_pixels < 300000:
                quality_report["recommendations"].append("Image resolution too low - consider higher quality photo")
                
            if not (0.3 <= width/height <= 3.0):
                quality_report["recommendations"].append("Unusual aspect ratio - check image orientation")
                
            if len(image_data) > 10 * 1024 * 1024:  # 10MB
                quality_report["recommendations"].append("Image file very large - processing may be slow")
                
            if image.mode not in ['RGB', 'L']:
                quality_report["recommendations"].append("Consider converting to RGB format")
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Image quality validation failed: {str(e)}")
            return {
                "error": f"Cannot process image: {str(e)}",
                "resolution_adequate": False,
                "recommendations": ["Please provide a valid image file"]
            }
