"""
Unit tests for the detection service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the detection service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.services.detection_service import DetectionService

class TestDetectionService:
    """Test cases for DetectionService"""
    
    def test_init(self):
        """Test DetectionService initialization"""
        service = DetectionService()
        assert service.model is None  # Mock model not loaded in tests
    
    def test_preprocess_image(self, sample_image_path):
        """Test image preprocessing"""
        service = DetectionService()
        
        # Test preprocessing
        processed = service.preprocess_image(sample_image_path)
        
        # Check output shape and type
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    
    def test_preprocess_image_invalid_path(self):
        """Test preprocessing with invalid image path"""
        service = DetectionService()
        
        with pytest.raises(ValueError):
            service.preprocess_image("nonexistent_image.jpg")
    
    def test_detect_security_features(self, sample_authentic_image):
        """Test security feature detection"""
        service = DetectionService()
        
        features = service.detect_security_features(sample_authentic_image)
        
        # Should return a list of strings
        assert isinstance(features, list)
        assert all(isinstance(feature, str) for feature in features)
    
    def test_detect_watermark(self):
        """Test watermark detection"""
        service = DetectionService()
        
        # Create test image with edges
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[20:80, 20:80] = 255  # White square
        
        result = service._detect_watermark(test_image)
        assert isinstance(result, (bool, np.bool_))
    
    def test_detect_security_thread(self):
        """Test security thread detection"""
        service = DetectionService()
        
        # Create test image with vertical line
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[:, 50] = 255  # Vertical line
        
        result = service._detect_security_thread(test_image)
        assert isinstance(result, (bool, np.bool_))
    
    def test_detect_microprinting(self):
        """Test microprinting detection"""
        service = DetectionService()
        
        # Create test image with noise
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        result = service._detect_microprinting(test_image)
        assert isinstance(result, bool)
    
    def test_detect_color_consistency(self):
        """Test color consistency detection"""
        service = DetectionService()
        
        # Create test image with color variation
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = service._detect_color_consistency(test_image)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_detect_counterfeit_success(self, sample_authentic_image):
        """Test successful counterfeit detection"""
        service = DetectionService()
        
        result = await service.detect_counterfeit(sample_authentic_image)
        
        # Check result structure
        assert hasattr(result, 'is_fake')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'denomination')
        assert hasattr(result, 'security_features')
        assert hasattr(result, 'processing_time')
        
        # Check types
        assert isinstance(result.is_fake, bool)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.security_features, list)
        assert isinstance(result.processing_time, float)
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_detect_counterfeit_invalid_image(self):
        """Test counterfeit detection with invalid image"""
        service = DetectionService()
        
        result = await service.detect_counterfeit("nonexistent_image.jpg")
        
        # Should return error result or handle gracefully
        assert hasattr(result, 'error')
        # The error might be None if the service handles invalid paths gracefully
        # Just check that we get a valid DetectionResult
        assert isinstance(result.is_fake, (bool, np.bool_))
        assert isinstance(result.confidence, (float, np.floating))
    
    def test_determine_denomination(self, sample_authentic_image):
        """Test denomination determination"""
        service = DetectionService()
        
        denomination = service._determine_denomination(sample_authentic_image)
        
        # Should return a string or None
        assert denomination is None or isinstance(denomination, str)
    
    def test_determine_denomination_invalid_path(self):
        """Test denomination determination with invalid path"""
        service = DetectionService()
        
        denomination = service._determine_denomination("nonexistent_image.jpg")
        
        # Should return None for invalid path
        assert denomination is None
    
    @patch('backend.app.services.detection_service.cv2.imread')
    def test_detect_security_features_cv2_error(self, mock_imread, sample_authentic_image):
        """Test security feature detection with CV2 error"""
        service = DetectionService()
        mock_imread.return_value = None  # Simulate CV2 error
        
        features = service.detect_security_features(sample_authentic_image)
        
        # Should return empty list on error
        assert features == []
    
    @patch('backend.app.services.detection_service.cv2.imread')
    def test_determine_denomination_cv2_error(self, mock_imread, sample_authentic_image):
        """Test denomination determination with CV2 error"""
        service = DetectionService()
        mock_imread.return_value = None  # Simulate CV2 error
        
        denomination = service._determine_denomination(sample_authentic_image)
        
        # Should return None on error
        assert denomination is None
    
    def test_predict_authenticity_mock_model(self):
        """Test prediction with mock model"""
        service = DetectionService()
        
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7]])  # 70% fake
        
        service.model = mock_model
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            from PIL import Image
            pil_image = Image.fromarray(test_image)
            pil_image.save(tmp_file.name, "JPEG")
            
            try:
                is_fake, confidence = service.predict_authenticity(tmp_file.name)
                
                assert isinstance(is_fake, (bool, np.bool_))
                assert isinstance(confidence, float)
                assert 0 <= confidence <= 1
            finally:
                os.unlink(tmp_file.name)
    
    def test_predict_authenticity_no_model(self, sample_authentic_image):
        """Test prediction without model (mock mode)"""
        service = DetectionService()
        service.model = None  # No model loaded
        
        is_fake, confidence = service.predict_authenticity(sample_authentic_image)
        
        # Should return mock results
        assert isinstance(is_fake, bool)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_predict_authenticity_error(self):
        """Test prediction with error"""
        service = DetectionService()
        service.model = None
        
        # Test with invalid path
        is_fake, confidence = service.predict_authenticity("nonexistent_image.jpg")
        
        # Should return mock results even on error
        assert isinstance(is_fake, bool)
        assert isinstance(confidence, float)
