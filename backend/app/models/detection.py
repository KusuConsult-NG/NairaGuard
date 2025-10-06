from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DetectionRequest(BaseModel):
    """Request model for detection"""
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image

class DetectionResponse(BaseModel):
    """Response model for detection"""
    predicted_class: str
    confidence: float
    probabilities: List[float]
    timestamp: str
    model_status: Optional[str] = None

class DetectionLog(Base):
    """Database model for detection logs"""
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=True)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=True)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_status = Column(String(50), nullable=True)
    user_id = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)

class DetectionResult(BaseModel):
    """Detection result model"""
    filename: Optional[str] = None
    is_fake: bool
    confidence: float
    denomination: Optional[str] = None
    security_features: Optional[List[str]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None

class BatchDetectionRequest(BaseModel):
    """Batch detection request model"""
    images: List[str]  # List of base64 encoded images
    user_id: Optional[str] = None

class DetectionHistory(BaseModel):
    """Detection history model"""
    id: str
    user_id: str
    filename: str
    is_fake: bool
    confidence: float
    timestamp: str
    file_size: int

class DetectionStats(BaseModel):
    """Detection statistics model"""
    total_detections: int
    authentic_count: int
    fake_count: int
    accuracy_rate: float
    average_confidence: float
    last_updated: str