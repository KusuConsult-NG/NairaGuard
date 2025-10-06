from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
import io
import json

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from app.routers import detection, health, auth
    from app.core.config import settings
    from app.core.database import get_db, engine, Base
    from app.models.detection import DetectionRequest, DetectionResponse, DetectionLog
except ImportError:
    # Fallback for when running from project root
    from backend.app.routers import detection, health, auth
    from backend.app.core.config import settings
    from backend.app.core.database import get_db, engine, Base
    from backend.app.models.detection import DetectionRequest, DetectionResponse, DetectionLog
from models.model_inference import ModelInference
from models.preprocess import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Naira Note Detection API",
    description="AI-powered counterfeit naira note detection system using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global model instance
model_inference: Optional[ModelInference] = None
preprocessor: Optional[ImagePreprocessor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_inference, preprocessor
    
    try:
        logger.info("Starting Naira Note Detection API...")
        
        # Initialize preprocessor
        global preprocessor
        preprocessor = ImagePreprocessor(model_type="mobilenet")
        logger.info("Image preprocessor initialized")
        
        # Load ML model if available
        model_path = Path("models/saved/mobilenet_best_fixed.h5")
        logger.info(f"Checking for model at: {model_path}")
        logger.info(f"Model exists: {model_path.exists()}")
        
        if model_path.exists():
            try:
                global model_inference
                logger.info("Creating ModelInference instance...")
                model_inference = ModelInference(str(model_path), "keras")
                logger.info(f"ModelInference created, loaded: {model_inference.loaded}")
                if model_inference.loaded:
                    logger.info(f"ML model loaded from {model_path}")
                else:
                    logger.warning("Model file exists but failed to load. Using mock predictions.")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("Using mock predictions.")
        else:
            logger.warning("No trained model found. Using mock predictions.")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Naira Note Detection API...")

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(detection.router, prefix="/api/v1", tags=["detection"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])

def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file
        
    Returns:
        True if valid, False otherwise
    """
    # Check file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        return False
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        return False
    
    return True

def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Process uploaded image file
    
    Args:
        file: Uploaded file
        
    Returns:
        Processed image as numpy array
    """
    # Read file content
    content = file.file.read()
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(content))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    return image_array

@app.post("/predict", response_model=DetectionResponse)
async def predict_note_authenticity(
    file: UploadFile = File(...),
    db = Depends(get_db)
) -> DetectionResponse:
    """
    Predict authenticity of uploaded naira note image
    
    Args:
        file: Uploaded image file
        db: Database dependency
        
    Returns:
        Detection result with confidence score
    """
    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file. Please upload a JPEG, PNG, or WebP image under 10MB."
            )
        
        # Process image
        image_array = process_uploaded_image(file)
        
        # Check if model is available
        if model_inference is None:
            # Return mock prediction for demo
            logger.warning("No trained model available, returning mock prediction")
            result = {
                "predicted_class": "genuine",
                "confidence": 0.85,
                "probabilities": [0.85, 0.15],
                "timestamp": datetime.now().isoformat(),
                "model_status": "demo_mode"
            }
        else:
            # Make prediction using trained model
            result = model_inference.predict(image_array)
            result["model_status"] = "trained_model"
        
        # Log prediction to database
        try:
            detection_log = DetectionLog(
                filename=file.filename,
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                timestamp=datetime.now(),
                model_status=result["model_status"]
            )
            db.add(detection_log)
            db.commit()
            logger.info(f"Prediction logged to database: {result['predicted_class']}")
        except Exception as e:
            logger.warning(f"Failed to log prediction to database: {str(e)}")
        
        return DetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Predict authenticity for multiple images
    
    Args:
        files: List of uploaded image files
        db: Database dependency
        
    Returns:
        Batch prediction results
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 files allowed per batch"
            )
        
        results = []
        processed_count = 0
        error_count = 0
        
        for file in files:
            try:
                # Validate file
                if not validate_image_file(file):
                    results.append({
                        "filename": file.filename,
                        "error": "Invalid file format or size"
                    })
                    error_count += 1
                    continue
                
                # Process image
                image_array = process_uploaded_image(file)
                
                # Make prediction
                if model_inference is None:
                    result = {
                        "predicted_class": "genuine",
                        "confidence": 0.85,
                        "probabilities": [0.85, 0.15],
                        "model_status": "demo_mode"
                    }
                else:
                    result = model_inference.predict(image_array)
                    result["model_status"] = "trained_model"
                
                result["filename"] = file.filename
                results.append(result)
                processed_count += 1
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                error_count += 1
        
        return {
            "results": results,
            "summary": {
                "total_files": len(files),
                "processed": processed_count,
                "errors": error_count,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get current model status
    
    Returns:
        Model status information
    """
    status_info = {
        "model_loaded": model_inference is not None and hasattr(model_inference, 'loaded') and model_inference.loaded,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if model_inference:
        status_info["model_type"] = model_inference.model_type
        status_info["model_path"] = str(model_inference.model_path)
        # Check if model is actually working by testing a prediction
        if hasattr(model_inference, 'model') and model_inference.model is not None:
            status_info["model_loaded"] = True
        else:
            status_info["model_loaded"] = False
    
    return status_info

@app.post("/model/reload")
async def reload_model(
    model_path: str,
    model_type: str = "keras"
) -> Dict[str, Any]:
    """
    Reload ML model
    
    Args:
        model_path: Path to model file
        model_type: Type of model (keras, tflite, onnx, pickle)
        
    Returns:
        Reload status
    """
    global model_inference
    
    try:
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {model_path}"
            )
        
        model_inference = ModelInference(model_path, model_type)
        
        return {
            "status": "success",
            "message": f"Model reloaded from {model_path}",
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )

@app.get("/predictions/history")
async def get_prediction_history(
    limit: int = 100,
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get prediction history
    
    Args:
        limit: Maximum number of records to return
        db: Database dependency
        
    Returns:
        Prediction history
    """
    try:
        # Query recent predictions
        predictions = db.query(DetectionLog).order_by(
            DetectionLog.timestamp.desc()
        ).limit(limit).all()
        
        results = []
        for pred in predictions:
            results.append({
                "id": pred.id,
                "filename": pred.filename,
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "timestamp": pred.timestamp.isoformat(),
                "model_status": pred.model_status
            })
        
        return {
            "predictions": results,
            "total_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction history: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Naira Note Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_status": "/model/status",
            "reload_model": "/model/reload",
            "prediction_history": "/predictions/history",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )