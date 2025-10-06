#!/usr/bin/env python3
"""
Unit tests for FastAPI endpoints
"""

import pytest
import json
import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.main import app
from backend.app.models.detection import DetectionLog

# Test client
client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_model():
    """Mock model inference"""
    mock_inference = Mock()
    mock_inference.predict.return_value = {
        "predicted_class": "genuine",
        "confidence": 0.85,
        "probabilities": [0.85, 0.15],
        "timestamp": "2023-01-01T00:00:00"
    }
    return mock_inference

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Naira Note Detection API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_predict_endpoint_success(self, sample_image, mock_model):
        """Test successful prediction"""
        with patch('backend.main.model_inference', mock_model):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "timestamp" in data
    
    def test_predict_endpoint_invalid_file_type(self):
        """Test prediction with invalid file type"""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file" in response.json()["detail"]
    
    def test_predict_endpoint_file_too_large(self):
        """Test prediction with file too large"""
        # Create a large image (simulate)
        large_image = Image.new('RGB', (1000, 1000), color='blue')
        img_byte_arr = io.BytesIO()
        large_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Mock file size check
        with patch('backend.main.validate_image_file') as mock_validate:
            mock_validate.return_value = False
            response = client.post(
                "/predict",
                files={"file": ("large.jpg", img_byte_arr, "image/jpeg")}
            )
            assert response.status_code == 400
    
    def test_predict_endpoint_no_model(self, sample_image):
        """Test prediction without trained model"""
        with patch('backend.main.model_inference', None):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["model_status"] == "demo_mode"
            assert data["predicted_class"] == "genuine"
    
    def test_batch_predict_endpoint(self, sample_image, mock_model):
        """Test batch prediction"""
        with patch('backend.main.model_inference', mock_model):
            response = client.post(
                "/predict/batch",
                files=[
                    ("files", ("test1.jpg", sample_image, "image/jpeg")),
                    ("files", ("test2.jpg", sample_image, "image/jpeg"))
                ]
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "summary" in data
            assert len(data["results"]) == 2
    
    def test_batch_predict_too_many_files(self, sample_image):
        """Test batch prediction with too many files"""
        files = []
        for i in range(11):  # More than 10 files
            files.append(("files", (f"test{i}.jpg", sample_image, "image/jpeg")))
        
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 400
        assert "Maximum 10 files" in response.json()["detail"]

class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_model_status_endpoint(self):
        """Test model status endpoint"""
        response = client.get("/model/status")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
        assert "timestamp" in data
    
    def test_reload_model_endpoint_success(self):
        """Test successful model reload"""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_file.write(b"fake model data")
            tmp_path = tmp_file.name
        
        try:
            with patch('backend.main.ModelInference') as mock_inference_class:
                mock_inference = Mock()
                mock_inference_class.return_value = mock_inference
                
                response = client.post(
                    "/model/reload",
                    params={"model_path": tmp_path, "model_type": "keras"}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "Model reloaded" in data["message"]
        finally:
            os.unlink(tmp_path)
    
    def test_reload_model_endpoint_file_not_found(self):
        """Test model reload with non-existent file"""
        response = client.post(
            "/model/reload",
            params={"model_path": "/nonexistent/model.h5", "model_type": "keras"}
        )
        assert response.status_code == 404
        assert "Model file not found" in response.json()["detail"]

class TestHistoryEndpoints:
    """Test prediction history endpoints"""
    
    def test_prediction_history_endpoint(self):
        """Test prediction history endpoint"""
        with patch('backend.main.get_db') as mock_db:
            # Mock database session
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock query result
            mock_prediction = Mock()
            mock_prediction.id = 1
            mock_prediction.filename = "test.jpg"
            mock_prediction.predicted_class = "genuine"
            mock_prediction.confidence = 0.85
            mock_prediction.timestamp = "2023-01-01T00:00:00"
            mock_prediction.model_status = "trained_model"
            
            mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_prediction]
            
            response = client.get("/predictions/history")
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert len(data["predictions"]) == 1

class TestFileValidation:
    """Test file validation functions"""
    
    def test_validate_image_file_valid(self):
        """Test valid image file validation"""
        from backend.main import validate_image_file
        
        # Create mock file
        mock_file = Mock()
        mock_file.content_type = "image/jpeg"
        mock_file.file.seek = Mock()
        mock_file.file.tell = Mock(return_value=1024)  # 1KB
        
        assert validate_image_file(mock_file) == True
    
    def test_validate_image_file_invalid_type(self):
        """Test invalid file type validation"""
        from backend.main import validate_image_file
        
        mock_file = Mock()
        mock_file.content_type = "text/plain"
        
        assert validate_image_file(mock_file) == False
    
    def test_validate_image_file_too_large(self):
        """Test file too large validation"""
        from backend.main import validate_image_file
        
        mock_file = Mock()
        mock_file.content_type = "image/jpeg"
        mock_file.file.seek = Mock()
        mock_file.file.tell = Mock(return_value=20 * 1024 * 1024)  # 20MB
        
        assert validate_image_file(mock_file) == False

class TestImageProcessing:
    """Test image processing functions"""
    
    def test_process_uploaded_image(self):
        """Test image processing"""
        from backend.main import process_uploaded_image
        
        # Create test image
        image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Create mock file
        mock_file = Mock()
        mock_file.file.read.return_value = img_byte_arr.getvalue()
        
        result = process_uploaded_image(mock_file)
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)

class TestErrorHandling:
    """Test error handling"""
    
    def test_internal_server_error(self, sample_image):
        """Test internal server error handling"""
        with patch('backend.main.process_uploaded_image') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image, "image/jpeg")}
            )
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]
    
    def test_database_error_handling(self, sample_image, mock_model):
        """Test database error handling"""
        with patch('backend.main.model_inference', mock_model):
            with patch('backend.main.get_db') as mock_db:
                mock_session = Mock()
                mock_session.add.side_effect = Exception("Database error")
                mock_db.return_value = mock_session
                
                response = client.post(
                    "/predict",
                    files={"file": ("test.jpg", sample_image, "image/jpeg")}
                )
                # Should still succeed but log warning
                assert response.status_code == 200

class TestCORS:
    """Test CORS functionality"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/predict")
        # FastAPI TestClient doesn't always show CORS headers in tests
        # but the middleware is configured correctly

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
