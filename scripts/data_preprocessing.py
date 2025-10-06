#!/usr/bin/env python3
"""
Data Preprocessing Script for Naira Note Images
Applies data augmentation and preprocessing to collected images
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NairaDataPreprocessor:
    """Data preprocessor for naira note images"""
    
    def __init__(self, input_dir: str = "datasets/raw", output_dir: str = "datasets/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self._create_output_structure()
        
        # Define augmentation pipelines
        self._setup_augmentation_pipelines()
        
        # Quality thresholds
        self.min_resolution = (224, 224)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_file_size = 10 * 1024  # 10KB
    
    def _create_output_structure(self):
        """Create output directory structure"""
        for split in ["train", "validation", "test"]:
            for authenticity in ["genuine", "fake"]:
                (self.output_dir / split / authenticity).mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Created output structure in {self.output_dir}")
    
    def _setup_augmentation_pipelines(self):
        """Setup different augmentation pipelines for different purposes"""
        
        # Basic augmentation (for all images)
        self.basic_augmentation = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Training augmentation (more aggressive)
        self.training_augmentation = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation augmentation (minimal)
        self.validation_augmentation = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Test augmentation (no augmentation)
        self.test_augmentation = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: Path, output_path: Path, 
                        augmentation_pipeline: A.Compose, 
                        quality_enhancement: bool = True) -> bool:
        """Preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Quality enhancement
            if quality_enhancement:
                image = self._enhance_image_quality(image)
            
            # Apply augmentation
            augmented = augmentation_pipeline(image=image)
            processed_image = augmented['image']
            
            # Convert tensor back to numpy array for saving
            if hasattr(processed_image, 'numpy'):
                processed_image = processed_image.numpy()
            elif hasattr(processed_image, 'cpu'):
                processed_image = processed_image.cpu().numpy()
            
            # Convert back to PIL Image for saving
            if processed_image.shape[0] == 3:  # CHW format
                processed_image = np.transpose(processed_image, (1, 2, 0))
            
            # Denormalize for saving
            processed_image = self._denormalize_image(processed_image)
            
            # Convert to uint8
            processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            
            # Save processed image
            pil_image = Image.fromarray(processed_image)
            pil_image.save(output_path, "JPEG", quality=95)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality using various techniques"""
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Enhance color
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(1.05)
            
            # Convert back to numpy
            return np.array(pil_image)
            
        except Exception as e:
            logger.error(f"Error enhancing image quality: {str(e)}")
            return image
    
    def _denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image from normalized format"""
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Denormalize
        image = image * std + mean
        return np.clip(image, 0, 1)
    
    def process_dataset(self, split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                       augmentation_factor: int = 3):
        """Process entire dataset with train/validation/test split"""
        logger.info("Starting dataset processing...")
        
        # Collect all images
        all_images = self._collect_all_images()
        
        # Split dataset
        train_images, val_images, test_images = self._split_dataset(all_images, split_ratios)
        
        # Process each split
        self._process_split(train_images, "train", self.training_augmentation, augmentation_factor)
        self._process_split(val_images, "validation", self.validation_augmentation, 1)
        self._process_split(test_images, "test", self.test_augmentation, 1)
        
        # Generate processing report
        self._generate_processing_report()
        
        logger.info("Dataset processing complete!")
    
    def _collect_all_images(self) -> List[Dict]:
        """Collect all images from input directory"""
        all_images = []
        
        for authenticity in ["genuine", "fake"]:
            for denomination in ["100", "200", "500", "1000"]:
                denomination_dir = self.input_dir / authenticity / f"naira_{denomination}"
                if denomination_dir.exists():
                    for condition_dir in denomination_dir.iterdir():
                        if condition_dir.is_dir():
                            for image_file in condition_dir.glob("*.jpg"):
                                all_images.append({
                                    "path": image_file,
                                    "authenticity": authenticity,
                                    "denomination": denomination,
                                    "condition": condition_dir.name
                                })
        
        logger.info(f"Collected {len(all_images)} images for processing")
        return all_images
    
    def _split_dataset(self, images: List[Dict], 
                      split_ratios: Tuple[float, float, float]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/validation/test"""
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total_images = len(images)
        train_end = int(total_images * split_ratios[0])
        val_end = train_end + int(total_images * split_ratios[1])
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        logger.info(f"Dataset split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        return train_images, val_images, test_images
    
    def _process_split(self, images: List[Dict], split_name: str, 
                      augmentation_pipeline: A.Compose, augmentation_factor: int):
        """Process a single split of the dataset"""
        logger.info(f"Processing {split_name} split with {len(images)} images...")
        
        processed_count = 0
        failed_count = 0
        
        for image_info in images:
            # Process original image
            output_filename = f"{image_info['authenticity']}_{image_info['denomination']}_{image_info['condition']}_{image_info['path'].stem}.jpg"
            output_path = self.output_dir / split_name / image_info['authenticity'] / output_filename
            
            if self.preprocess_image(image_info['path'], output_path, augmentation_pipeline):
                processed_count += 1
            else:
                failed_count += 1
            
            # Generate augmented versions
            for i in range(augmentation_factor - 1):
                aug_filename = f"{image_info['authenticity']}_{image_info['denomination']}_{image_info['condition']}_{image_info['path'].stem}_aug_{i:02d}.jpg"
                aug_output_path = self.output_dir / split_name / image_info['authenticity'] / aug_filename
                
                if self.preprocess_image(image_info['path'], aug_output_path, augmentation_pipeline):
                    processed_count += 1
                else:
                    failed_count += 1
        
        logger.info(f"{split_name} split complete: {processed_count} successful, {failed_count} failed")
    
    def _generate_processing_report(self):
        """Generate processing statistics report"""
        report = {
            "processing_date": datetime.now().isoformat(),
            "total_processed": 0,
            "by_split": {},
            "by_authenticity": {},
            "quality_metrics": {}
        }
        
        # Count processed images
        for split in ["train", "validation", "test"]:
            report["by_split"][split] = 0
            for authenticity in ["genuine", "fake"]:
                if authenticity not in report["by_authenticity"]:
                    report["by_authenticity"][authenticity] = 0
                
                split_dir = self.output_dir / split / authenticity
                if split_dir.exists():
                    image_count = len(list(split_dir.glob("*.jpg")))
                    report["by_split"][split] += image_count
                    report["by_authenticity"][authenticity] += image_count
                    report["total_processed"] += image_count
        
        # Calculate quality metrics
        report["quality_metrics"] = self._calculate_quality_metrics()
        
        # Save report
        report_path = self.output_dir / "metadata" / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved to {report_path}")
        return report
    
    def _calculate_quality_metrics(self) -> Dict:
        """Calculate quality metrics for processed images"""
        metrics = {
            "average_file_size": 0,
            "average_resolution": [0, 0],
            "corrupted_images": 0,
            "quality_distribution": {}
        }
        
        total_size = 0
        total_resolution = [0, 0]
        corrupted_count = 0
        quality_scores = []
        
        for split in ["train", "validation", "test"]:
            for authenticity in ["genuine", "fake"]:
                split_dir = self.output_dir / split / authenticity
                if split_dir.exists():
                    for image_file in split_dir.glob("*.jpg"):
                        try:
                            # File size
                            file_size = image_file.stat().st_size
                            total_size += file_size
                            
                            # Image resolution
                            image = cv2.imread(str(image_file))
                            if image is not None:
                                height, width = image.shape[:2]
                                total_resolution[0] += height
                                total_resolution[1] += width
                                
                                # Quality score (Laplacian variance)
                                quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                                quality_scores.append(quality_score)
                            else:
                                corrupted_count += 1
                                
                        except Exception as e:
                            corrupted_count += 1
                            logger.error(f"Error calculating metrics for {image_file}: {str(e)}")
        
        # Calculate averages
        total_images = sum(len(list((self.output_dir / split / auth).glob("*.jpg"))) 
                          for split in ["train", "validation", "test"] 
                          for auth in ["genuine", "fake"] 
                          if (self.output_dir / split / auth).exists())
        
        if total_images > 0:
            metrics["average_file_size"] = total_size / total_images
            metrics["average_resolution"] = [total_resolution[0] / total_images, 
                                           total_resolution[1] / total_images]
            metrics["corrupted_images"] = corrupted_count
            
            if quality_scores:
                metrics["average_quality_score"] = np.mean(quality_scores)
                metrics["quality_distribution"] = {
                    "excellent": len([s for s in quality_scores if s > 1000]),
                    "good": len([s for s in quality_scores if 500 < s <= 1000]),
                    "fair": len([s for s in quality_scores if 100 < s <= 500]),
                    "poor": len([s for s in quality_scores if s <= 100])
                }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Preprocess naira note images')
    parser.add_argument('--input-dir', type=str, default='datasets/raw',
                       help='Input directory with raw images')
    parser.add_argument('--output-dir', type=str, default='datasets/processed',
                       help='Output directory for processed images')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--augmentation-factor', type=int, default=3,
                       help='Number of augmented versions per original image')
    parser.add_argument('--no-quality-enhancement', action='store_true',
                       help='Disable quality enhancement')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("Split ratios must sum to 1.0")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = NairaDataPreprocessor(args.input_dir, args.output_dir)
    
    # Process dataset
    preprocessor.process_dataset(
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        augmentation_factor=args.augmentation_factor
    )

if __name__ == '__main__':
    main()
