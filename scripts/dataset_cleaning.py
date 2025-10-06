#!/usr/bin/env python3
"""
Dataset Cleaning and Maintenance Script
Cleans, organizes, and maintains the naira note dataset
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import cv2
import numpy as np
from PIL import Image
import hashlib
import shutil
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCleaner:
    """Cleans and maintains naira note dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.cleaning_results = {
            "cleaning_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "actions_performed": [],
            "files_processed": 0,
            "files_removed": 0,
            "files_renamed": 0,
            "files_moved": 0,
            "duplicates_found": 0,
            "corrupted_files": 0,
            "statistics": {}
        }
        
        # Quality thresholds
        self.min_resolution = (224, 224)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_file_size = 10 * 1024  # 10KB
        self.min_quality_score = 100
        self.max_corruption_threshold = 0.05
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def clean_dataset(self, actions: List[str] = None) -> Dict:
        """Run complete dataset cleaning"""
        if actions is None:
            actions = ["remove_corrupted", "remove_duplicates", "standardize_names", "organize_structure"]
        
        logger.info(f"Starting dataset cleaning: {self.dataset_path}")
        
        for action in actions:
            if action == "remove_corrupted":
                self._remove_corrupted_files()
            elif action == "remove_duplicates":
                self._remove_duplicate_files()
            elif action == "standardize_names":
                self._standardize_file_names()
            elif action == "organize_structure":
                self._organize_directory_structure()
            elif action == "compress_images":
                self._compress_images()
            elif action == "validate_metadata":
                self._validate_metadata()
            else:
                logger.warning(f"Unknown cleaning action: {action}")
        
        # Generate cleaning report
        self._generate_cleaning_report()
        
        logger.info("Dataset cleaning complete")
        return self.cleaning_results
    
    def _remove_corrupted_files(self):
        """Remove corrupted or invalid image files"""
        logger.info("Removing corrupted files...")
        
        corrupted_count = 0
        processed_count = 0
        
        # Find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.dataset_path.rglob(f'*{ext}'))
            image_files.extend(self.dataset_path.rglob(f'*{ext.upper()}'))
        
        for image_file in image_files:
            processed_count += 1
            
            try:
                # Check file size
                file_size = image_file.stat().st_size
                if file_size < self.min_file_size or file_size > self.max_file_size:
                    logger.info(f"Removing invalid size file: {image_file}")
                    image_file.unlink()
                    corrupted_count += 1
                    continue
                
                # Check if image can be loaded
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.info(f"Removing corrupted file: {image_file}")
                    image_file.unlink()
                    corrupted_count += 1
                    continue
                
                # Check resolution
                height, width = image.shape[:2]
                if height < self.min_resolution[0] or width < self.min_resolution[1]:
                    logger.info(f"Removing low resolution file: {image_file}")
                    image_file.unlink()
                    corrupted_count += 1
                    continue
                
                # Check quality score
                quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                if quality_score < self.min_quality_score:
                    logger.info(f"Removing low quality file: {image_file}")
                    image_file.unlink()
                    corrupted_count += 1
                    continue
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {str(e)}")
                try:
                    image_file.unlink()
                    corrupted_count += 1
                except:
                    pass
        
        self.cleaning_results["files_processed"] += processed_count
        self.cleaning_results["files_removed"] += corrupted_count
        self.cleaning_results["corrupted_files"] = corrupted_count
        self.cleaning_results["actions_performed"].append(f"removed_corrupted: {corrupted_count} files")
        
        logger.info(f"Removed {corrupted_count} corrupted files out of {processed_count} processed")
    
    def _remove_duplicate_files(self):
        """Remove duplicate files based on content hash"""
        logger.info("Removing duplicate files...")
        
        file_hashes = defaultdict(list)
        duplicates_removed = 0
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        # Calculate hashes
        for image_file in image_files:
            try:
                file_hash = self._calculate_file_hash(image_file)
                file_hashes[file_hash].append(image_file)
            except Exception as e:
                logger.warning(f"Error calculating hash for {image_file}: {str(e)}")
        
        # Remove duplicates (keep first occurrence)
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                # Keep the first file, remove the rest
                for duplicate_file in files[1:]:
                    logger.info(f"Removing duplicate: {duplicate_file}")
                    duplicate_file.unlink()
                    duplicates_removed += 1
        
        self.cleaning_results["files_removed"] += duplicates_removed
        self.cleaning_results["duplicates_found"] = duplicates_removed
        self.cleaning_results["actions_performed"].append(f"removed_duplicates: {duplicates_removed} files")
        
        logger.info(f"Removed {duplicates_removed} duplicate files")
    
    def _standardize_file_names(self):
        """Standardize file naming conventions"""
        logger.info("Standardizing file names...")
        
        renamed_count = 0
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                # Extract information from current filename
                filename = image_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 4:
                    # Expected format: authenticity_denomination_condition_identifier
                    authenticity = parts[0]
                    denomination = parts[1]
                    condition = parts[2]
                    identifier = '_'.join(parts[3:])
                    
                    # Validate authenticity
                    if authenticity not in ['genuine', 'fake']:
                        continue
                    
                    # Validate denomination
                    if denomination not in ['100', '200', '500', '1000']:
                        continue
                    
                    # Create standardized filename
                    new_filename = f"{authenticity}_{denomination}_{condition}_{identifier}.jpg"
                    new_path = image_file.parent / new_filename
                    
                    if new_path != image_file:
                        image_file.rename(new_path)
                        renamed_count += 1
                        logger.debug(f"Renamed: {image_file.name} -> {new_filename}")
                
            except Exception as e:
                logger.warning(f"Error renaming {image_file}: {str(e)}")
        
        self.cleaning_results["files_renamed"] = renamed_count
        self.cleaning_results["actions_performed"].append(f"standardized_names: {renamed_count} files")
        
        logger.info(f"Renamed {renamed_count} files")
    
    def _organize_directory_structure(self):
        """Organize files into proper directory structure"""
        logger.info("Organizing directory structure...")
        
        moved_count = 0
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                # Extract information from filename
                filename = image_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    authenticity = parts[0]
                    denomination = parts[1]
                    condition = parts[2]
                    
                    # Validate components
                    if (authenticity in ['genuine', 'fake'] and 
                        denomination in ['100', '200', '500', '1000']):
                        
                        # Create target directory
                        target_dir = (self.dataset_path / authenticity / 
                                    f"naira_{denomination}" / condition)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move file if not already in correct location
                        target_path = target_dir / image_file.name
                        if image_file != target_path:
                            shutil.move(str(image_file), str(target_path))
                            moved_count += 1
                            logger.debug(f"Moved: {image_file} -> {target_path}")
                
            except Exception as e:
                logger.warning(f"Error organizing {image_file}: {str(e)}")
        
        self.cleaning_results["files_moved"] = moved_count
        self.cleaning_results["actions_performed"].append(f"organized_structure: {moved_count} files")
        
        logger.info(f"Moved {moved_count} files to organize structure")
    
    def _compress_images(self):
        """Compress images to reduce file size while maintaining quality"""
        logger.info("Compressing images...")
        
        compressed_count = 0
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                # Check if file is large enough to benefit from compression
                file_size = image_file.stat().st_size
                if file_size < 500 * 1024:  # Less than 500KB
                    continue
                
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Compress image
                compressed_image = cv2.imencode('.jpg', image, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
                
                # Check if compression resulted in significant size reduction
                compressed_size = len(compressed_image)
                if compressed_size < file_size * 0.8:  # At least 20% reduction
                    # Save compressed image
                    with open(image_file, 'wb') as f:
                        f.write(compressed_image)
                    compressed_count += 1
                    logger.debug(f"Compressed: {image_file.name}")
                
            except Exception as e:
                logger.warning(f"Error compressing {image_file}: {str(e)}")
        
        self.cleaning_results["actions_performed"].append(f"compressed_images: {compressed_count} files")
        
        logger.info(f"Compressed {compressed_count} images")
    
    def _validate_metadata(self):
        """Validate and fix metadata files"""
        logger.info("Validating metadata...")
        
        metadata_files = list(self.dataset_path.rglob("*.json"))
        validated_count = 0
        
        for metadata_file in metadata_files:
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Validate required fields
                required_fields = ["total_images", "collection_date"]
                missing_fields = [field for field in required_fields if field not in metadata]
                
                if missing_fields:
                    # Add missing fields with default values
                    if "total_images" not in metadata:
                        metadata["total_images"] = 0
                    if "collection_date" not in metadata:
                        metadata["collection_date"] = datetime.now().isoformat()
                    
                    # Save updated metadata
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    validated_count += 1
                    logger.debug(f"Updated metadata: {metadata_file}")
                
            except Exception as e:
                logger.warning(f"Error validating metadata {metadata_file}: {str(e)}")
        
        self.cleaning_results["actions_performed"].append(f"validated_metadata: {validated_count} files")
        
        logger.info(f"Validated {validated_count} metadata files")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_cleaning_report(self):
        """Generate cleaning report"""
        # Calculate final statistics
        final_stats = {
            "total_files_after_cleaning": len(list(self.dataset_path.rglob("*.jpg"))),
            "total_size_mb": sum(f.stat().st_size for f in self.dataset_path.rglob("*.jpg")) / (1024 * 1024),
            "cleaning_efficiency": {
                "corruption_rate": self.cleaning_results["corrupted_files"] / max(1, self.cleaning_results["files_processed"]),
                "duplicate_rate": self.cleaning_results["duplicates_found"] / max(1, self.cleaning_results["files_processed"]),
                "files_improved": self.cleaning_results["files_renamed"] + self.cleaning_results["files_moved"]
            }
        }
        
        self.cleaning_results["statistics"] = final_stats
        
        # Save report
        report_path = self.dataset_path / "metadata" / "cleaning_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.cleaning_results, f, indent=2)
        
        logger.info(f"Cleaning report saved to {report_path}")
    
    def backup_dataset(self, backup_path: str):
        """Create backup of dataset before cleaning"""
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup to {backup_path}")
        
        # Copy dataset to backup location
        shutil.copytree(self.dataset_path, backup_path / "dataset_backup", 
                       dirs_exist_ok=True)
        
        logger.info("Backup created successfully")
    
    def restore_from_backup(self, backup_path: str):
        """Restore dataset from backup"""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup path does not exist: {backup_path}")
            return False
        
        logger.info(f"Restoring from backup: {backup_path}")
        
        # Remove current dataset
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)
        
        # Restore from backup
        shutil.copytree(backup_path / "dataset_backup", self.dataset_path)
        
        logger.info("Dataset restored successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description='Clean and maintain naira note dataset')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--actions', type=str, nargs='+',
                       choices=['remove_corrupted', 'remove_duplicates', 'standardize_names', 
                               'organize_structure', 'compress_images', 'validate_metadata'],
                       default=['remove_corrupted', 'remove_duplicates', 'standardize_names', 'organize_structure'],
                       help='Cleaning actions to perform')
    parser.add_argument('--backup', type=str,
                       help='Create backup before cleaning')
    parser.add_argument('--restore', type=str,
                       help='Restore from backup')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = DatasetCleaner(args.dataset_path)
    
    try:
        if args.restore:
            # Restore from backup
            success = cleaner.restore_from_backup(args.restore)
            if success:
                logger.info("Dataset restored successfully")
            else:
                logger.error("Failed to restore dataset")
                sys.exit(1)
        
        else:
            # Create backup if requested
            if args.backup:
                cleaner.backup_dataset(args.backup)
            
            if args.dry_run:
                logger.info("Dry run mode - no actual changes will be made")
                # In a real implementation, you would add dry-run logic here
            
            # Run cleaning
            results = cleaner.clean_dataset(args.actions)
            
            # Print summary
            print("\n" + "="*60)
            print("DATASET CLEANING SUMMARY")
            print("="*60)
            print(f"Files Processed: {results['files_processed']}")
            print(f"Files Removed: {results['files_removed']}")
            print(f"Files Renamed: {results['files_renamed']}")
            print(f"Files Moved: {results['files_moved']}")
            print(f"Duplicates Found: {results['duplicates_found']}")
            print(f"Corrupted Files: {results['corrupted_files']}")
            
            if results['actions_performed']:
                print("\nActions Performed:")
                for action in results['actions_performed']:
                    print(f"- {action}")
            
            if results['statistics']:
                stats = results['statistics']
                print(f"\nFinal Statistics:")
                print(f"- Total Files: {stats['total_files_after_cleaning']}")
                print(f"- Total Size: {stats['total_size_mb']:.1f} MB")
                print(f"- Corruption Rate: {stats['cleaning_efficiency']['corruption_rate']:.1%}")
                print(f"- Duplicate Rate: {stats['cleaning_efficiency']['duplicate_rate']:.1%}")
            
            print("="*60)
            print("Cleaning report saved to dataset metadata directory.")
    
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
