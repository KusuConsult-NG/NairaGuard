#!/usr/bin/env python3
"""
Image Preprocessing Module for Naira Note Detection
Handles normalization, resizing, and augmentation for ML models
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for naira note detection"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 model_type: str = "mobilenet"):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
            model_type: Type of model ("mobilenet", "efficientnet", "custom")
        """
        self.target_size = target_size
        self.model_type = model_type
        
        # Define preprocessing functions for different models
        self.preprocess_functions = {
            "mobilenet": mobilenet_preprocess,
            "efficientnet": efficientnet_preprocess,
            "custom": self._custom_preprocess
        }
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),
        ])
        
        # Define validation pipeline (no augmentation)
        self.validation_pipeline = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, str, Path], 
                        augment: bool = False) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Image as numpy array, file path, or Path object
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Apply augmentation or validation pipeline
        pipeline = self.augmentation_pipeline if augment else self.validation_pipeline
        
        # Apply transformations
        transformed = pipeline(image=image)
        processed_image = transformed["image"]
        
        # Apply model-specific preprocessing
        preprocess_func = self.preprocess_functions.get(self.model_type, self._custom_preprocess)
        processed_image = preprocess_func(processed_image)
        
        return processed_image
    
    def preprocess_batch(self, images: List[Union[np.ndarray, str, Path]], 
                        augment: bool = False) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of images
            augment: Whether to apply augmentation
            
        Returns:
            Batch of preprocessed images
        """
        processed_batch = []
        
        for image in images:
            processed_image = self.preprocess_image(image, augment)
            processed_batch.append(processed_image)
        
        return np.array(processed_batch)
    
    def create_data_generator(self, data_dir: str, batch_size: int = 32, 
                             augment: bool = True, shuffle: bool = True) -> tf.keras.utils.Sequence:
        """
        Create a data generator for training/validation
        
        Args:
            data_dir: Directory containing images
            batch_size: Batch size
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
            
        Returns:
            Data generator
        """
        # Define preprocessing function
        if self.model_type == "mobilenet":
            preprocessing_function = mobilenet_preprocess
        elif self.model_type == "efficientnet":
            preprocessing_function = efficientnet_preprocess
        else:
            preprocessing_function = None
        
        # Create ImageDataGenerator
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            horizontal_flip=augment,
            rotation_range=15 if augment else 0,
            width_shift_range=0.1 if augment else 0,
            height_shift_range=0.1 if augment else 0,
            brightness_range=[0.8, 1.2] if augment else None,
            zoom_range=0.1 if augment else 0,
            fill_mode='nearest'
        )
        
        # Generate batches
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=shuffle
        )
        
        return generator
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            return image.astype(np.float32)
    
    def resize_image(self, image: np.ndarray, 
                    interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                         interpolation=interpolation)
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various techniques
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_image, -1, kernel)
        
        # Blend original and sharpened image
        enhanced_image = cv2.addWeighted(enhanced_image, 0.7, sharpened, 0.3, 0)
        
        return enhanced_image
    
    def detect_and_crop_note(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop naira note from image
        
        Args:
            image: Input image
            
        Returns:
            Cropped image containing the note
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the note)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            
            return cropped
        
        return image
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _custom_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Custom preprocessing function"""
        # Normalize to [0, 1]
        image = self.normalize_image(image)
        
        # Apply custom normalization (mean centering)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = (image - mean) / std
        
        return image
    
    def get_preprocessing_stats(self, data_dir: str) -> dict:
        """
        Get statistics about the dataset for preprocessing
        
        Args:
            data_dir: Directory containing images
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            "total_images": 0,
            "image_sizes": [],
            "aspect_ratios": [],
            "mean_brightness": [],
            "mean_contrast": []
        }
        
        data_path = Path(data_dir)
        
        # Count images and collect statistics
        for image_path in data_path.rglob("*.jpg"):
            try:
                image = self._load_image(image_path)
                
                stats["total_images"] += 1
                stats["image_sizes"].append(image.shape[:2])
                stats["aspect_ratios"].append(image.shape[1] / image.shape[0])
                
                # Calculate brightness and contrast
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                stats["mean_brightness"].append(np.mean(gray))
                stats["mean_contrast"].append(np.std(gray))
                
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {str(e)}")
        
        # Calculate summary statistics
        if stats["image_sizes"]:
            stats["avg_size"] = np.mean(stats["image_sizes"], axis=0)
            stats["min_size"] = np.min(stats["image_sizes"], axis=0)
            stats["max_size"] = np.max(stats["image_sizes"], axis=0)
            stats["avg_aspect_ratio"] = np.mean(stats["aspect_ratios"])
            stats["avg_brightness"] = np.mean(stats["mean_brightness"])
            stats["avg_contrast"] = np.mean(stats["mean_contrast"])
        
        return stats

def main():
    """Test the preprocessor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test image preprocessor')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--model-type', type=str, default='mobilenet',
                       choices=['mobilenet', 'efficientnet', 'custom'],
                       help='Model type for preprocessing')
    parser.add_argument('--augment', action='store_true',
                       help='Apply augmentation')
    parser.add_argument('--output-path', type=str,
                       help='Output path for processed image')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(model_type=args.model_type)
    
    # Process image
    try:
        processed_image = preprocessor.preprocess_image(args.image_path, args.augment)
        
        print(f"Original image shape: {processed_image.shape}")
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Image dtype: {processed_image.dtype}")
        print(f"Image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # Save processed image if output path provided
        if args.output_path:
            # Convert back to uint8 for saving
            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output_path, processed_image_bgr)
            print(f"Processed image saved to: {args.output_path}")
        
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
