import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.models.detection import (
    BatchDetectionRequest,
    DetectionRequest,
    DetectionResult,
)
from backend.app.services.detection_service import DetectionService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize detection service
detection_service = DetectionService()


@router.post("/detect", response_model=DetectionResult)
async def detect_single_image(
    file: UploadFile = File(...), user_id: Optional[str] = None
):
    """
    Detect if a single naira note image is authentic or fake
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process image
        result = await detection_service.detect_counterfeit(file_path)

        # Add metadata
        result.filename = file.filename
        result.file_size = file.size
        result.timestamp = datetime.utcnow().isoformat()
        result.user_id = user_id

        # Clean up uploaded file
        os.remove(file_path)

        return result

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect/batch", response_model=List[DetectionResult])
async def detect_batch_images(
    files: List[UploadFile] = File(...), user_id: Optional[str] = None
):
    """
    Detect multiple naira note images in batch
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

        results = []

        for file in files:
            try:
                # Validate file
                if not file.content_type.startswith("image/"):
                    results.append(
                        DetectionResult(
                            filename=file.filename,
                            is_fake=False,
                            confidence=0.0,
                            error="Invalid file type",
                            timestamp=datetime.utcnow().isoformat(),
                        )
                    )
                    continue

                if file.size > settings.MAX_FILE_SIZE:
                    results.append(
                        DetectionResult(
                            filename=file.filename,
                            is_fake=False,
                            confidence=0.0,
                            error="File too large",
                            timestamp=datetime.utcnow().isoformat(),
                        )
                    )
                    continue

                # Generate unique filename
                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

                # Ensure upload directory exists
                os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

                # Save uploaded file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                # Process image
                result = await detection_service.detect_counterfeit(file_path)
                result.filename = file.filename
                result.file_size = file.size
                result.timestamp = datetime.utcnow().isoformat()
                result.user_id = user_id

                results.append(result)

                # Clean up uploaded file
                os.remove(file_path)

            except Exception as e:
                logger.error(f"Batch detection error for {file.filename}: {str(e)}")
                results.append(
                    DetectionResult(
                        filename=file.filename,
                        is_fake=False,
                        confidence=0.0,
                        error=str(e),
                        timestamp=datetime.utcnow().isoformat(),
                    )
                )

        return results

    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@router.get("/detect/history")
async def get_detection_history(
    user_id: Optional[str] = None, limit: int = 50, offset: int = 0
):
    """
    Get detection history for a user
    """
    try:
        # This would typically query a database
        # For now, return mock data
        return {"detections": [], "total": 0, "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.get("/detect/stats")
async def get_detection_stats(user_id: Optional[str] = None):
    """
    Get detection statistics
    """
    try:
        # This would typically query a database
        # For now, return mock data
        return {
            "total_detections": 0,
            "authentic_count": 0,
            "fake_count": 0,
            "accuracy_rate": 0.0,
            "average_confidence": 0.0,
        }
    except Exception as e:
        logger.error(f"Stats retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")
