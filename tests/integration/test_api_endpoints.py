"""
Integration tests for API endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_detailed_health_endpoint(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/api/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "application" in data
    
    @patch('backend.app.services.detection_service.DetectionService')
    def test_detect_single_image_success(self, mock_service_class, client, sample_authentic_image):
        """Test successful single image detection"""
        # Mock the detection service
        mock_service = Mock()
        mock_result = Mock()
        mock_result.is_fake = False
        mock_result.confidence = 0.95
        mock_result.denomination = "₦1000"
        mock_result.security_features = ["Watermark detected"]
        mock_result.processing_time = 1.2
        mock_result.timestamp = "2024-01-15T10:30:00Z"
        mock_result.filename = "test_image.jpg"
        mock_result.file_size = 1024
        mock_result.user_id = None
        mock_result.error = None
        
        mock_service.detect_counterfeit.return_value = mock_result
        mock_service_class.return_value = mock_service
        
        # Prepare test file
        with open(sample_authentic_image, "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = client.post("/api/detect", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_fake"] is False
        assert data["confidence"] == 0.95
        assert data["denomination"] == "₦1000"
        assert "Watermark detected" in data["security_features"]
    
    def test_detect_single_image_invalid_file_type(self, client):
        """Test detection with invalid file type"""
        # Create a text file instead of image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/detect", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "File must be an image" in data["detail"]
    
    def test_detect_single_image_file_too_large(self, client):
        """Test detection with file too large"""
        # Create a large file (simulate)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large_image.jpg", large_content, "image/jpeg")}
        response = client.post("/api/detect", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]
    
    @patch('backend.app.services.detection_service.DetectionService')
    def test_detect_batch_images_success(self, mock_service_class, client, sample_authentic_image, sample_fake_image):
        """Test successful batch image detection"""
        # Mock the detection service
        mock_service = Mock()
        
        # Mock results for two images
        mock_result1 = Mock()
        mock_result1.is_fake = False
        mock_result1.confidence = 0.95
        mock_result1.denomination = "₦1000"
        mock_result1.security_features = ["Watermark detected"]
        mock_result1.processing_time = 1.2
        mock_result1.timestamp = "2024-01-15T10:30:00Z"
        mock_result1.filename = "authentic.jpg"
        mock_result1.file_size = 1024
        mock_result1.user_id = None
        mock_result1.error = None
        
        mock_result2 = Mock()
        mock_result2.is_fake = True
        mock_result2.confidence = 0.88
        mock_result2.denomination = "₦500"
        mock_result2.security_features = ["Watermark mismatch"]
        mock_result2.processing_time = 1.1
        mock_result2.timestamp = "2024-01-15T10:30:00Z"
        mock_result2.filename = "fake.jpg"
        mock_result2.file_size = 1024
        mock_result2.user_id = None
        mock_result2.error = None
        
        mock_service.detect_counterfeit.side_effect = [mock_result1, mock_result2]
        mock_service_class.return_value = mock_service
        
        # Prepare test files
        with open(sample_authentic_image, "rb") as f1, open(sample_fake_image, "rb") as f2:
            files = [
                ("files", ("authentic.jpg", f1, "image/jpeg")),
                ("files", ("fake.jpg", f2, "image/jpeg"))
            ]
            response = client.post("/api/detect/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["is_fake"] is False
        assert data[1]["is_fake"] is True
    
    def test_detect_batch_images_too_many_files(self, client):
        """Test batch detection with too many files"""
        # Create 11 files (limit is 10)
        files = []
        for i in range(11):
            files.append(("files", (f"image_{i}.jpg", b"fake image data", "image/jpeg")))
        
        response = client.post("/api/detect/batch", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 10 files per batch" in data["detail"]
    
    def test_detection_history_endpoint(self, client):
        """Test detection history endpoint"""
        response = client.get("/api/detect/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_detection_history_with_params(self, client):
        """Test detection history with query parameters"""
        response = client.get("/api/detect/history?user_id=test123&limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0
    
    def test_detection_stats_endpoint(self, client):
        """Test detection statistics endpoint"""
        response = client.get("/api/detect/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "authentic_count" in data
        assert "fake_count" in data
        assert "accuracy_rate" in data
        assert "average_confidence" in data
    
    def test_detection_stats_with_user_id(self, client):
        """Test detection statistics with user ID"""
        response = client.get("/api/detect/stats?user_id=test123")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total_detections"], int)
        assert isinstance(data["accuracy_rate"], float)
    
    def test_auth_register_endpoint(self, client):
        """Test user registration endpoint"""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "password": "testpassword123"
        }
        
        response = client.post("/api/register", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["full_name"] == user_data["full_name"]
        assert "id" in data
        assert "created_at" in data
    
    def test_auth_login_endpoint(self, client):
        """Test user login endpoint"""
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        
        response = client.post("/api/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert data["token_type"] == "bearer"
    
    def test_auth_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        login_data = {
            "username": "",
            "password": ""
        }
        
        response = client.post("/api/login", data=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "Incorrect username or password" in data["detail"]
    
    def test_auth_me_endpoint(self, client, mock_auth_token):
        """Test get current user endpoint"""
        headers = {"Authorization": f"Bearer {mock_auth_token}"}
        response = client.get("/api/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "username" in data
        assert "full_name" in data
    
    def test_auth_logout_endpoint(self, client, mock_auth_token):
        """Test logout endpoint"""
        headers = {"Authorization": f"Bearer {mock_auth_token}"}
        response = client.post("/api/logout", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "Successfully logged out" in data["message"]
    
    def test_auth_verify_token_endpoint(self, client, mock_auth_token):
        """Test token verification endpoint"""
        headers = {"Authorization": f"Bearer {mock_auth_token}"}
        response = client.get("/api/verify-token", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "user_id" in data
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/health")
        
        # CORS preflight should be handled by middleware
        assert response.status_code in [200, 204]
    
    def test_api_documentation_endpoints(self, client):
        """Test API documentation endpoints"""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
    
    @patch('backend.app.services.detection_service.DetectionService')
    def test_detect_error_handling(self, mock_service_class, client, sample_authentic_image):
        """Test error handling in detection endpoint"""
        # Mock service to raise an exception
        mock_service = Mock()
        mock_service.detect_counterfeit.side_effect = Exception("Detection failed")
        mock_service_class.return_value = mock_service
        
        with open(sample_authentic_image, "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = client.post("/api/detect", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "Detection failed" in data["detail"]
