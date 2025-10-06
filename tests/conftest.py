"""
Pytest configuration and fixtures for the Fake Naira Detection project
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import application components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.main import app
from backend.app.core.database import Base, get_db
from backend.app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_fake_naira_detection.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test function."""
    # Create test database
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield TestingSessionLocal()
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_image_path(temp_dir):
    """Create a sample image file for testing."""
    # Create a simple test image
    image_path = os.path.join(temp_dir, "test_image.jpg")
    
    # Create a simple RGB image (224x224x3)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Save as JPEG
    from PIL import Image
    pil_image = Image.fromarray(image)
    pil_image.save(image_path, "JPEG")
    
    return image_path

@pytest.fixture(scope="function")
def sample_authentic_image(temp_dir):
    """Create a sample authentic naira note image."""
    image_path = os.path.join(temp_dir, "authentic_note.jpg")
    
    # Create an image that looks more like a naira note
    image = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background
    
    # Add some patterns to make it look like a note
    image[50:150, 50:150] = [200, 200, 255]  # Light blue rectangle
    image[100:120, 20:200] = [255, 255, 0]   # Yellow line
    
    from PIL import Image
    pil_image = Image.fromarray(image)
    pil_image.save(image_path, "JPEG")
    
    return image_path

@pytest.fixture(scope="function")
def sample_fake_image(temp_dir):
    """Create a sample fake naira note image."""
    image_path = os.path.join(temp_dir, "fake_note.jpg")
    
    # Create an image that looks different from authentic
    image = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add different patterns
    image[50:150, 50:150] = [255, 200, 200]  # Light red rectangle
    image[100:120, 20:200] = [200, 255, 200] # Light green line
    
    from PIL import Image
    pil_image = Image.fromarray(image)
    pil_image.save(image_path, "JPEG")
    
    return image_path

@pytest.fixture(scope="function")
def mock_detection_result():
    """Mock detection result for testing."""
    return {
        "is_fake": False,
        "confidence": 0.95,
        "denomination": "₦1000",
        "security_features": ["Watermark detected", "Security thread detected"],
        "processing_time": 1.2,
        "timestamp": "2024-01-15T10:30:00Z"
    }

@pytest.fixture(scope="function")
def mock_user_data():
    """Mock user data for testing."""
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
        "created_at": "2024-01-15T10:00:00Z"
    }

@pytest.fixture(scope="function")
def mock_auth_token():
    """Mock authentication token for testing."""
    return "mock-jwt-token-12345"

@pytest.fixture(scope="function")
def sample_dataset(temp_dir):
    """Create a sample dataset for testing."""
    dataset_path = os.path.join(temp_dir, "sample_dataset")
    
    # Create directory structure
    for split in ["train", "validation", "test"]:
        for class_name in ["authentic", "fake"]:
            class_dir = os.path.join(dataset_path, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create sample images
            for i in range(5):  # 5 images per class per split
                image_path = os.path.join(class_dir, f"image_{i:03d}.jpg")
                
                # Create different images for authentic vs fake
                if class_name == "authentic":
                    image = np.ones((224, 224, 3), dtype=np.uint8) * 255
                    image[50:150, 50:150] = [200, 200, 255]
                else:
                    image = np.ones((224, 224, 3), dtype=np.uint8) * 200
                    image[50:150, 50:150] = [255, 200, 200]
                
                from PIL import Image
                pil_image = Image.fromarray(image)
                pil_image.save(image_path, "JPEG")
    
    return dataset_path

@pytest.fixture(scope="function")
def mock_model():
    """Mock ML model for testing."""
    class MockModel:
        def __init__(self):
            self.input_shape = (224, 224, 3)
            self.output_shape = (2,)
        
        def predict(self, X):
            # Return random predictions
            batch_size = X.shape[0] if len(X.shape) > 3 else 1
            predictions = np.random.rand(batch_size, 2)
            # Normalize to probabilities
            predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
            return predictions
        
        def save(self, filepath):
            # Mock save operation
            pass
    
    return MockModel()

@pytest.fixture(scope="function")
def mock_detection_service():
    """Mock detection service for testing."""
    class MockDetectionService:
        def __init__(self):
            self.model = None
        
        async def detect_counterfeit(self, image_path):
            # Mock detection result
            import random
            from backend.app.models.detection import DetectionResult
            
            is_fake = random.random() > 0.7
            confidence = random.uniform(0.6, 0.99)
            
            return DetectionResult(
                is_fake=is_fake,
                confidence=confidence,
                denomination="₦1000",
                security_features=["Watermark detected"] if not is_fake else ["Watermark mismatch"],
                processing_time=1.0
            )
    
    return MockDetectionService()

@pytest.fixture(scope="function")
def mock_file_upload():
    """Mock file upload for testing."""
    class MockFileUpload:
        def __init__(self, content: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
            self.content = content
            self.filename = filename
            self.content_type = content_type
            self.size = len(content)
        
        def read(self):
            return self.content
    
    return MockFileUpload

@pytest.fixture(scope="function")
def test_config():
    """Test configuration settings."""
    return {
        "DATABASE_URL": TEST_DATABASE_URL,
        "SECRET_KEY": "test-secret-key",
        "MODEL_PATH": "test_model.h5",
        "UPLOAD_DIR": "test_uploads",
        "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
        "ALLOWED_EXTENSIONS": [".jpg", ".jpeg", ".png"],
        "CONFIDENCE_THRESHOLD": 0.8
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for tests that take longer
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
