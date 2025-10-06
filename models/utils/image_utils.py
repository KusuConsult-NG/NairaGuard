"""
Image processing utilities for naira note detection
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def detect_edges(image: np.ndarray) -> np.ndarray:
    """Detect edges in the image using Canny edge detection"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # type: ignore[assignment]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # type: ignore[assignment]
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)  # type: ignore[assignment]
        
        return edges
    except Exception as e:
        logger.error(f"Error detecting edges: {str(e)}")
        raise

def detect_corners(image: np.ndarray) -> np.ndarray:
    """Detect corners in the image using Harris corner detection"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # type: ignore[assignment]
        
        # Apply Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)  # type: ignore[assignment]
        corners = cv2.dilate(corners, None)  # type: ignore[assignment]
        
        return corners
    except Exception as e:
        logger.error(f"Error detecting corners: {str(e)}")
        raise

def extract_text_regions(image: np.ndarray) -> List[np.ndarray]:
    """Extract text regions from the image using OCR preprocessing"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # type: ignore[assignment]
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # type: ignore[assignment]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for text-like regions
            if area > 100 and 0.1 < aspect_ratio < 10:
                text_region = image[y:y+h, x:x+w]
                text_regions.append(text_region)
        
        return text_regions
    except Exception as e:
        logger.error(f"Error extracting text regions: {str(e)}")
        return []

def detect_watermark(image: np.ndarray) -> bool:
    """Detect watermark presence in the image"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # type: ignore[assignment]
        
        # Apply edge detection
        edges = detect_edges(gray)
        
        # Look for watermark patterns (simplified)
        # In reality, this would be more sophisticated
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        return edge_density > 0.1  # Arbitrary threshold
    except Exception as e:
        logger.error(f"Error detecting watermark: {str(e)}")
        return False

def detect_security_thread(image: np.ndarray) -> bool:
    """Detect security thread presence in the image"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image  # type: ignore[assignment]
        
        # Apply edge detection
        edges = detect_edges(gray)
        
        # Look for vertical lines (security threads)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Check if any lines are roughly vertical
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15 or abs(angle - 180) < 15:  # Vertical lines
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error detecting security thread: {str(e)}")
        return False

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better detection"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def load_and_preprocess_images(data_path: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess images from directory"""
    try:
        images = []
        labels = []
        
        data_path = Path(data_path)  # type: ignore[assignment]
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(data_path.glob(f'**/*{ext}'))  # type: ignore[attr-defined]
            image_files.extend(data_path.glob(f'**/*{ext.upper()}'))  # type: ignore[attr-defined]
        
        logger.info(f"Found {len(image_files)} image files")
        
        for image_file in image_files:
            try:
                # Load image
                image = load_image(str(image_file))
                
                # Preprocess image
                processed_image = preprocess_image(image, target_size)
                
                # Determine label from directory structure
                label = 0 if 'authentic' in str(image_file).lower() else 1
                
                images.append(processed_image)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"Skipping {image_file}: {str(e)}")
                continue
        
        return np.array(images), np.array(labels)
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}")
        raise

def create_augmented_dataset(images: np.ndarray, labels: np.ndarray, 
                           augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented dataset"""
    try:
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Create augmented versions
            for _ in range(augmentation_factor):
                # Random rotation
                angle = np.random.uniform(-15, 15)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
                
                # Random brightness adjustment
                brightness = np.random.uniform(0.8, 1.2)
                brightened = np.clip(rotated * brightness, 0, 1)
                
                # Random noise
                noise = np.random.normal(0, 0.01, brightened.shape)
                noisy = np.clip(brightened + noise, 0, 1)
                
                augmented_images.append(noisy)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    except Exception as e:
        logger.error(f"Error creating augmented dataset: {str(e)}")
        raise
