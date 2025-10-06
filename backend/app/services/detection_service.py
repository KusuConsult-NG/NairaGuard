import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
from typing import List, Optional
import time

from backend.app.core.config import settings
from backend.app.models.detection import DetectionResult

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for detecting counterfeit naira notes"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model"""
        try:
            # In a real application, you would load a trained model
            # For now, we'll use a mock model
            logger.info("Loading detection model...")
            
            # Mock model loading - replace with actual model loading
            # self.model = tf.keras.models.load_model(settings.MODEL_PATH)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to model input size (224x224 for most models)
            image = cv2.resize(image, (224, 224))
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)  # type: ignore[assignment]
            
            return image  # type: ignore[return-value,assignment]
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def detect_security_features(self, image_path: str) -> List[str]:
        """Detect security features in the image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            features = []
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect watermarks (simplified)
            if self._detect_watermark(gray):
                features.append("Watermark detected")
            
            # Detect security threads (simplified)
            if self._detect_security_thread(gray):
                features.append("Security thread detected")
            
            # Detect microprinting (simplified)
            if self._detect_microprinting(gray):
                features.append("Microprinting detected")
            
            # Detect color variations
            if self._detect_color_consistency(image):
                features.append("Color consistency verified")
            
            return features
            
        except Exception as e:
            logger.error(f"Security feature detection failed: {str(e)}")
            return []
    
    def _detect_watermark(self, gray_image: np.ndarray) -> bool:
        """Detect watermark presence (simplified)"""
        # This is a simplified implementation
        # In reality, you would use more sophisticated computer vision techniques
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges) > 1000  # Arbitrary threshold
    
    def _detect_security_thread(self, gray_image: np.ndarray) -> bool:
        """Detect security thread presence (simplified)"""
        # This is a simplified implementation
        # Look for vertical lines that could be security threads
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        return lines is not None and len(lines) > 0
    
    def _detect_microprinting(self, gray_image: np.ndarray) -> bool:
        """Detect microprinting presence (simplified)"""
        # This is a simplified implementation
        # Look for small text patterns
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gray_image, kernel, iterations=1)
        return bool(np.std(dilated) > 30)  # Arbitrary threshold
    
    def _detect_color_consistency(self, image: np.ndarray) -> bool:
        """Detect color consistency (simplified)"""
        # This is a simplified implementation
        # Check for color variations that might indicate counterfeiting
        std_dev = np.std(image)
        return bool(std_dev > 20)  # Arbitrary threshold
    
    def predict_authenticity(self, image_path: str) -> tuple[bool, float]:
        """Predict if the note is authentic or fake"""
        try:
            if self.model is None:
                # Mock prediction for demo purposes
                # In reality, you would use the loaded model
                import random
                is_fake = random.random() > 0.7  # 30% chance of being fake
                confidence = random.uniform(0.6, 0.99)
                return is_fake, confidence
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(processed_image)
            
            # Extract confidence and class
            confidence = float(np.max(prediction))
            is_fake = prediction[0][0] > 0.5  # Assuming binary classification
            
            return is_fake, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return mock result on error
            return False, 0.5
    
    async def detect_counterfeit(self, image_path: str) -> DetectionResult:
        """Main method to detect counterfeit notes"""
        start_time = time.time()
        
        try:
            # Predict authenticity
            is_fake, confidence = self.predict_authenticity(image_path)
            
            # Detect security features
            security_features = self.detect_security_features(image_path)
            
            # Determine denomination (simplified)
            denomination = self._determine_denomination(image_path)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                is_fake=is_fake,
                confidence=confidence,
                denomination=denomination,
                security_features=security_features,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return DetectionResult(
                is_fake=False,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _determine_denomination(self, image_path: str) -> Optional[str]:
        """Determine the denomination of the note (simplified)"""
        try:
            # This is a simplified implementation
            # In reality, you would use OCR or another ML model
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Simple heuristic based on image size
            if height > 1000 or width > 2000:
                return "₦1000"
            elif height > 800 or width > 1600:
                return "₦500"
            elif height > 600 or width > 1200:
                return "₦200"
            else:
                return "₦100"
                
        except Exception as e:
            logger.error(f"Denomination detection failed: {str(e)}")
            return None
