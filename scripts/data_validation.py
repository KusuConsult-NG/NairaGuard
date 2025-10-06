#!/usr/bin/env python3
"""
Data Quality Validation Script for Naira Note Dataset
Validates dataset quality, completeness, and consistency
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import cv2
import numpy as np
from PIL import Image
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates naira note dataset quality and completeness"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.validation_results = {
            "validation_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "total_images": 0,
            "quality_issues": [],
            "completeness_issues": [],
            "consistency_issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        # Quality thresholds
        self.min_resolution = (224, 224)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_file_size = 10 * 1024  # 10KB
        self.min_quality_score = 100  # Laplacian variance
        self.max_corruption_threshold = 0.05  # 5% corrupted images allowed
    
    def validate_dataset(self) -> Dict:
        """Run complete dataset validation"""
        logger.info(f"Starting validation of dataset: {self.dataset_path}")
        
        # Check dataset structure
        self._validate_structure()
        
        # Validate image quality
        self._validate_image_quality()
        
        # Check dataset completeness
        self._validate_completeness()
        
        # Validate consistency
        self._validate_consistency()
        
        # Generate statistics
        self._generate_statistics()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save validation report
        self._save_validation_report()
        
        logger.info("Dataset validation complete")
        return self.validation_results
    
    def _validate_structure(self):
        """Validate dataset directory structure"""
        logger.info("Validating dataset structure...")
        
        expected_structure = {
            "raw": ["genuine", "fake"],
            "processed": ["train", "validation", "test"]
        }
        
        structure_issues = []
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            structure_issues.append(f"Dataset path does not exist: {self.dataset_path}")
            self.validation_results["structure_valid"] = False
            return
        
        # Check raw data structure
        raw_path = self.dataset_path / "raw"
        if raw_path.exists():
            for authenticity in expected_structure["raw"]:
                auth_path = raw_path / authenticity
                if not auth_path.exists():
                    structure_issues.append(f"Missing authenticity directory: {auth_path}")
                else:
                    # Check denominations
                    denominations = ["100", "200", "500", "1000"]
                    for denom in denominations:
                        denom_path = auth_path / f"naira_{denom}"
                        if not denom_path.exists():
                            structure_issues.append(f"Missing denomination directory: {denom_path}")
        
        # Check processed data structure
        processed_path = self.dataset_path / "processed"
        if processed_path.exists():
            for split in expected_structure["processed"]:
                split_path = processed_path / split
                if not split_path.exists():
                    structure_issues.append(f"Missing split directory: {split_path}")
                else:
                    # Check authenticity directories in each split
                    for authenticity in expected_structure["raw"]:
                        auth_path = split_path / authenticity
                        if not auth_path.exists():
                            structure_issues.append(f"Missing authenticity directory in {split}: {auth_path}")
        
        if structure_issues:
            self.validation_results["structure_issues"] = structure_issues
            self.validation_results["structure_valid"] = False
        else:
            self.validation_results["structure_valid"] = True
        
        logger.info(f"Structure validation: {'PASSED' if not structure_issues else 'FAILED'}")
    
    def _validate_image_quality(self):
        """Validate individual image quality"""
        logger.info("Validating image quality...")
        
        quality_issues = []
        image_stats = {
            "total_images": 0,
            "valid_images": 0,
            "corrupted_images": 0,
            "low_resolution_images": 0,
            "oversized_images": 0,
            "undersized_images": 0,
            "low_quality_images": 0,
            "resolution_distribution": defaultdict(int),
            "file_size_distribution": defaultdict(int),
            "quality_score_distribution": []
        }
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.dataset_path.rglob(f'*{ext}'))
            image_files.extend(self.dataset_path.rglob(f'*{ext.upper()}'))
        
        image_stats["total_images"] = len(image_files)
        self.validation_results["total_images"] = len(image_files)
        
        # Validate each image
        for image_file in image_files:
            try:
                # Check file size
                file_size = image_file.stat().st_size
                image_stats["file_size_distribution"][f"{file_size // (1024*1024)}MB"] += 1
                
                if file_size > self.max_file_size:
                    quality_issues.append(f"Oversized file: {image_file} ({file_size / (1024*1024):.1f}MB)")
                    image_stats["oversized_images"] += 1
                elif file_size < self.min_file_size:
                    quality_issues.append(f"Undersized file: {image_file} ({file_size / 1024:.1f}KB)")
                    image_stats["undersized_images"] += 1
                
                # Load and validate image
                image = cv2.imread(str(image_file))
                if image is None:
                    quality_issues.append(f"Corrupted image: {image_file}")
                    image_stats["corrupted_images"] += 1
                    continue
                
                # Check resolution
                height, width = image.shape[:2]
                resolution_key = f"{width}x{height}"
                image_stats["resolution_distribution"][resolution_key] += 1
                
                if height < self.min_resolution[0] or width < self.min_resolution[1]:
                    quality_issues.append(f"Low resolution: {image_file} ({width}x{height})")
                    image_stats["low_resolution_images"] += 1
                
                # Check quality score
                quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                image_stats["quality_score_distribution"].append(quality_score)
                
                if quality_score < self.min_quality_score:
                    quality_issues.append(f"Low quality: {image_file} (score: {quality_score:.1f})")
                    image_stats["low_quality_images"] += 1
                
                image_stats["valid_images"] += 1
                
            except Exception as e:
                quality_issues.append(f"Error processing {image_file}: {str(e)}")
                image_stats["corrupted_images"] += 1
        
        # Calculate quality metrics
        if image_stats["quality_score_distribution"]:
            image_stats["average_quality_score"] = np.mean(image_stats["quality_score_distribution"])
            image_stats["median_quality_score"] = np.median(image_stats["quality_score_distribution"])
            image_stats["quality_score_std"] = np.std(image_stats["quality_score_distribution"])
        
        self.validation_results["quality_issues"] = quality_issues
        self.validation_results["image_statistics"] = image_stats
        
        corruption_rate = image_stats["corrupted_images"] / image_stats["total_images"] if image_stats["total_images"] > 0 else 0
        self.validation_results["corruption_rate"] = corruption_rate
        self.validation_results["quality_valid"] = corruption_rate <= self.max_corruption_threshold
        
        logger.info(f"Quality validation: {'PASSED' if self.validation_results['quality_valid'] else 'FAILED'}")
        logger.info(f"Valid images: {image_stats['valid_images']}/{image_stats['total_images']}")
    
    def _validate_completeness(self):
        """Validate dataset completeness"""
        logger.info("Validating dataset completeness...")
        
        completeness_issues = []
        completeness_stats = {
            "by_authenticity": defaultdict(int),
            "by_denomination": defaultdict(int),
            "by_condition": defaultdict(int),
            "by_split": defaultdict(int)
        }
        
        # Expected structure
        denominations = ["100", "200", "500", "1000"]
        conditions = ["good_lighting", "poor_lighting", "bright_lighting", 
                     "new_condition", "worn_condition", "damaged_condition",
                     "front_view", "back_view", "angled_view"]
        splits = ["train", "validation", "test"]
        
        # Count images by category
        for authenticity in ["genuine", "fake"]:
            for denomination in denominations:
                for condition in conditions:
                    # Check raw data
                    raw_path = self.dataset_path / "raw" / authenticity / f"naira_{denomination}" / condition
                    if raw_path.exists():
                        image_count = len(list(raw_path.glob("*.jpg")))
                        completeness_stats["by_authenticity"][authenticity] += image_count
                        completeness_stats["by_denomination"][denomination] += image_count
                        completeness_stats["by_condition"][condition] += image_count
                        
                        if image_count == 0:
                            completeness_issues.append(f"No images in {raw_path}")
        
        # Check processed data splits
        processed_path = self.dataset_path / "processed"
        if processed_path.exists():
            for split in splits:
                for authenticity in ["genuine", "fake"]:
                    split_path = processed_path / split / authenticity
                    if split_path.exists():
                        image_count = len(list(split_path.glob("*.jpg")))
                        completeness_stats["by_split"][split] += image_count
        
        # Check for balanced classes
        genuine_count = completeness_stats["by_authenticity"]["genuine"]
        fake_count = completeness_stats["by_authenticity"]["fake"]
        
        if genuine_count > 0 and fake_count > 0:
            balance_ratio = min(genuine_count, fake_count) / max(genuine_count, fake_count)
            if balance_ratio < 0.5:  # Less than 50% balance
                completeness_issues.append(f"Class imbalance: {genuine_count} genuine vs {fake_count} fake (ratio: {balance_ratio:.2f})")
        
        # Check minimum requirements
        min_images_per_class = 100
        for authenticity in ["genuine", "fake"]:
            count = completeness_stats["by_authenticity"][authenticity]
            if count < min_images_per_class:
                completeness_issues.append(f"Insufficient {authenticity} images: {count} (minimum: {min_images_per_class})")
        
        self.validation_results["completeness_issues"] = completeness_issues
        self.validation_results["completeness_statistics"] = completeness_stats
        self.validation_results["completeness_valid"] = len(completeness_issues) == 0
        
        logger.info(f"Completeness validation: {'PASSED' if self.validation_results['completeness_valid'] else 'FAILED'}")
    
    def _validate_consistency(self):
        """Validate dataset consistency"""
        logger.info("Validating dataset consistency...")
        
        consistency_issues = []
        
        # Check file naming consistency
        naming_patterns = defaultdict(int)
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            filename = image_file.stem
            # Extract pattern (e.g., "genuine_500_good_lighting_001")
            parts = filename.split('_')
            if len(parts) >= 4:
                pattern = '_'.join(parts[:4])  # First 4 parts
                naming_patterns[pattern] += 1
            else:
                consistency_issues.append(f"Inconsistent naming: {image_file}")
        
        # Check for duplicate files
        file_hashes = defaultdict(list)
        for image_file in image_files:
            try:
                file_hash = self._calculate_file_hash(image_file)
                file_hashes[file_hash].append(image_file)
            except Exception as e:
                consistency_issues.append(f"Error calculating hash for {image_file}: {str(e)}")
        
        duplicates = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
        if duplicates:
            for hash_val, files in duplicates.items():
                consistency_issues.append(f"Duplicate files found: {files}")
        
        # Check metadata consistency
        metadata_files = list(self.dataset_path.rglob("*.json"))
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Check required fields
                    required_fields = ["total_images", "collection_date"]
                    for field in required_fields:
                        if field not in metadata:
                            consistency_issues.append(f"Missing field '{field}' in {metadata_file}")
            except Exception as e:
                consistency_issues.append(f"Error reading metadata {metadata_file}: {str(e)}")
        
        self.validation_results["consistency_issues"] = consistency_issues
        self.validation_results["naming_patterns"] = dict(naming_patterns)
        self.validation_results["duplicate_files"] = len(duplicates)
        self.validation_results["consistency_valid"] = len(consistency_issues) == 0
        
        logger.info(f"Consistency validation: {'PASSED' if self.validation_results['consistency_valid'] else 'FAILED'}")
    
    def _generate_statistics(self):
        """Generate comprehensive dataset statistics"""
        logger.info("Generating dataset statistics...")
        
        stats = {
            "total_images": self.validation_results["total_images"],
            "dataset_size_mb": 0,
            "average_image_size_mb": 0,
            "resolution_distribution": {},
            "quality_distribution": {},
            "class_distribution": {},
            "split_distribution": {}
        }
        
        # Calculate dataset size
        total_size = 0
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            total_size += image_file.stat().st_size
        
        stats["dataset_size_mb"] = total_size / (1024 * 1024)
        stats["average_image_size_mb"] = stats["dataset_size_mb"] / len(image_files) if image_files else 0
        
        # Add existing statistics
        if "image_statistics" in self.validation_results:
            stats.update(self.validation_results["image_statistics"])
        
        if "completeness_statistics" in self.validation_results:
            stats.update(self.validation_results["completeness_statistics"])
        
        self.validation_results["statistics"] = stats
    
    def _generate_recommendations(self):
        """Generate recommendations for dataset improvement"""
        recommendations = []
        
        # Quality recommendations
        if not self.validation_results.get("quality_valid", True):
            recommendations.append("Improve image quality by removing corrupted or low-quality images")
            recommendations.append("Ensure all images meet minimum resolution requirements (224x224)")
        
        # Completeness recommendations
        if not self.validation_results.get("completeness_valid", True):
            recommendations.append("Collect more images to meet minimum requirements per class")
            recommendations.append("Balance genuine and fake image counts")
            recommendations.append("Ensure coverage of all denominations and conditions")
        
        # Consistency recommendations
        if not self.validation_results.get("consistency_valid", True):
            recommendations.append("Standardize file naming conventions")
            recommendations.append("Remove duplicate images")
            recommendations.append("Ensure metadata files are complete and consistent")
        
        # General recommendations
        if self.validation_results["total_images"] < 1000:
            recommendations.append("Consider collecting more images for better model performance")
        
        corruption_rate = self.validation_results.get("corruption_rate", 0)
        if corruption_rate > 0.01:  # More than 1% corruption
            recommendations.append("Investigate and fix image corruption issues")
        
        self.validation_results["recommendations"] = recommendations
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_validation_report(self):
        """Save validation report to file"""
        report_path = self.dataset_path / "metadata" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def generate_visualization(self, output_dir: str = "reports"):
        """Generate visualization of dataset statistics"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create quality score distribution plot
        if "image_statistics" in self.validation_results:
            quality_scores = self.validation_results["image_statistics"].get("quality_score_distribution", [])
            if quality_scores:
                plt.figure(figsize=(10, 6))
                plt.hist(quality_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(self.min_quality_score, color='red', linestyle='--', label=f'Minimum Quality ({self.min_quality_score})')
                plt.xlabel('Quality Score (Laplacian Variance)')
                plt.ylabel('Number of Images')
                plt.title('Image Quality Score Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_path / 'quality_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Create class distribution plot
        if "completeness_statistics" in self.validation_results:
            class_dist = self.validation_results["completeness_statistics"]["by_authenticity"]
            if class_dist:
                plt.figure(figsize=(8, 6))
                plt.bar(class_dist.keys(), class_dist.values(), color=['green', 'red'], alpha=0.7)
                plt.xlabel('Authenticity')
                plt.ylabel('Number of Images')
                plt.title('Class Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Validate naira note dataset quality')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports and visualizations')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DatasetValidator(args.dataset_path)
    
    # Run validation
    results = validator.validate_dataset()
    
    # Generate visualizations if requested
    if args.generate_plots:
        validator.generate_visualization(args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET VALIDATION SUMMARY")
    print("="*50)
    print(f"Total Images: {results['total_images']}")
    print(f"Structure Valid: {'✓' if results.get('structure_valid', False) else '✗'}")
    print(f"Quality Valid: {'✓' if results.get('quality_valid', False) else '✗'}")
    print(f"Completeness Valid: {'✓' if results.get('completeness_valid', False) else '✗'}")
    print(f"Consistency Valid: {'✓' if results.get('consistency_valid', False) else '✗'}")
    
    if results.get('recommendations'):
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print("\nDetailed report saved to dataset metadata directory.")

if __name__ == '__main__':
    main()
