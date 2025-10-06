"""
Data loading utilities for naira note detection dataset
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class NairaDatasetLoader:
    """Dataset loader for naira note detection"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.authentic_path = self.data_root / "authentic"
        self.fake_path = self.data_root / "fake"
        
    def load_dataset(self, split: str = "train", 
                    target_size: Tuple[int, int] = (224, 224)) -> Dict[str, np.ndarray]:
        """
        Load dataset for specified split
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            target_size: Target image size (height, width)
            
        Returns:
            Dictionary containing images and labels
        """
        try:
            # Determine data path based on split
            if split in ["train", "validation", "test"]:
                data_path = self.data_root / split
            else:
                raise ValueError(f"Invalid split: {split}")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset split not found: {data_path}")
            
            # Load images and labels
            images, labels = self._load_images_and_labels(data_path, target_size)
            
            logger.info(f"Loaded {len(images)} images for {split} split")
            logger.info(f"Class distribution: {np.bincount(labels)}")
            
            return {
                'images': images,
                'labels': labels,
                'class_names': ['authentic', 'fake']
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _load_images_and_labels(self, data_path: Path, 
                               target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Load images and labels from directory structure"""
        images = []
        labels = []
        
        # Load authentic images
        authentic_path = data_path / "authentic"
        if authentic_path.exists():
            authentic_images, authentic_labels = self._load_class_images(
                authentic_path, target_size, class_label=0
            )
            images.extend(authentic_images)
            labels.extend(authentic_labels)
        
        # Load fake images
        fake_path = data_path / "fake"
        if fake_path.exists():
            fake_images, fake_labels = self._load_class_images(
                fake_path, target_size, class_label=1
            )
            images.extend(fake_images)
            labels.extend(fake_labels)
        
        return np.array(images), np.array(labels)
    
    def _load_class_images(self, class_path: Path, 
                          target_size: Tuple[int, int], 
                          class_label: int) -> Tuple[List[np.ndarray], List[int]]:
        """Load images for a specific class"""
        images = []
        labels = []
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_path.glob(f'**/*{ext}'))
            image_files.extend(class_path.glob(f'**/*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {class_path}")
        
        for image_file in image_files:
            try:
                # Load and preprocess image
                image = self._load_and_preprocess_image(image_file, target_size)
                images.append(image)
                labels.append(class_label)
                
            except Exception as e:
                logger.warning(f"Skipping {image_file}: {str(e)}")
                continue
        
        return images, labels
    
    def _load_and_preprocess_image(self, image_path: Path, 
                                  target_size: Tuple[int, int]) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def get_class_distribution(self, split: str = "train") -> Dict[str, int]:
        """Get class distribution for a dataset split"""
        try:
            data = self.load_dataset(split)
            labels = data['labels']
            class_names = data['class_names']
            
            distribution = {}
            for i, class_name in enumerate(class_names):
                count = np.sum(labels == i)
                distribution[class_name] = int(count)
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting class distribution: {str(e)}")
            raise
    
    def split_dataset(self, test_size: float = 0.2, 
                     validation_size: float = 0.1, 
                     random_state: int = 42) -> None:
        """
        Split raw dataset into train/validation/test sets
        
        Args:
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
        """
        try:
            # Load all data
            all_images, all_labels = self._load_images_and_labels(
                self.data_root / "raw", target_size=(224, 224)
            )
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_images, all_labels, 
                test_size=test_size, 
                random_state=random_state,
                stratify=all_labels
            )
            
            # Second split: separate train and validation
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=random_state,
                stratify=y_temp
            )
            
            # Save splits
            self._save_split(X_train, y_train, "train")
            self._save_split(X_val, y_val, "validation")
            self._save_split(X_test, y_test, "test")
            
            logger.info("Dataset split completed successfully")
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise
    
    def _save_split(self, images: np.ndarray, labels: np.ndarray, split_name: str) -> None:
        """Save dataset split to appropriate directories"""
        try:
            split_path = self.data_root / split_name
            split_path.mkdir(exist_ok=True)
            
            # Create class directories
            (split_path / "authentic").mkdir(exist_ok=True)
            (split_path / "fake").mkdir(exist_ok=True)
            
            # Save images to appropriate class directories
            for i, (image, label) in enumerate(zip(images, labels)):
                class_name = "authentic" if label == 0 else "fake"
                image_path = split_path / class_name / f"image_{i:06d}.jpg"
                
                # Convert back to uint8 and save
                image_uint8 = (image * 255).astype(np.uint8)
                cv2.imwrite(str(image_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Saved {len(images)} images to {split_name} split")
            
        except Exception as e:
            logger.error(f"Error saving split: {str(e)}")
            raise

def load_dataset(data_root: str, split: str = "train", 
                target_size: Tuple[int, int] = (224, 224)) -> Dict[str, np.ndarray]:
    """
    Convenience function to load dataset
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split to load
        target_size: Target image size
        
    Returns:
        Dictionary containing images and labels
    """
    loader = NairaDatasetLoader(data_root)
    return loader.load_dataset(split, target_size)

def create_sample_dataset(data_root: str, num_samples: int = 100) -> None:
    """
    Create a sample dataset for testing purposes
    
    Args:
        data_root: Root directory for the dataset
        num_samples: Number of samples per class
    """
    try:
        data_path = Path(data_root)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split in ["train", "validation", "test"]:
            for class_name in ["authentic", "fake"]:
                (data_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create sample images (random noise for demo)
        for split in ["train", "validation", "test"]:
            for class_name in ["authentic", "fake"]:
                split_path = data_path / split / class_name
                
                # Create sample images
                for i in range(num_samples // 3):  # Divide samples across splits
                    # Create random image
                    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    
                    # Add some structure to make it look more like a note
                    if class_name == "authentic":
                        # Add some patterns for authentic notes
                        image[50:150, 50:150] = [255, 255, 255]  # White rectangle
                    else:
                        # Add different patterns for fake notes
                        image[50:150, 50:150] = [200, 200, 200]  # Gray rectangle
                    
                    # Save image
                    image_path = split_path / f"sample_{i:03d}.jpg"
                    cv2.imwrite(str(image_path), image)
        
        logger.info(f"Created sample dataset with {num_samples} samples per class")
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {str(e)}")
        raise
