#!/usr/bin/env python3
"""
Data Collection Script for Naira Note Images
Gathers and organizes genuine and counterfeit naira note images
"""

import os
import sys
import argparse
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json
import hashlib
from urllib.parse import urlparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.utils.data_loader import NairaDatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NairaDataCollector:
    """Data collector for naira note images"""
    
    def __init__(self, output_dir: str = "datasets/raw"):
        self.output_dir = Path(output_dir)
        self.denominations = ["100", "200", "500", "1000"]
        self.conditions = [
            "good_lighting", "poor_lighting", "bright_lighting",
            "new_condition", "worn_condition", "damaged_condition",
            "front_view", "back_view", "angled_view"
        ]
        self.authenticity = ["genuine", "fake"]
        
        # Create directory structure
        self._create_directory_structure()
        
        # Image sources (you would replace these with actual sources)
        self.image_sources = {
            "genuine": {
                "cbn_official": "https://cbn.gov.ng/currency/",
                "bank_images": "https://banks.com/naira-notes/",
                "museum_collections": "https://museums.com/currency/"
            },
            "fake": {
                "seized_evidence": "https://lawenforcement.gov.ng/evidence/",
                "bank_reports": "https://banks.com/counterfeit-reports/",
                "training_materials": "https://training.gov.ng/counterfeit/"
            }
        }
    
    def _create_directory_structure(self):
        """Create organized directory structure for data collection"""
        for authenticity in self.authenticity:
            for denomination in self.denominations:
                for condition in self.conditions:
                    dir_path = self.output_dir / authenticity / f"naira_{denomination}" / condition
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Created directory structure in {self.output_dir}")
    
    def collect_from_urls(self, urls: List[str], authenticity: str, 
                         denomination: str, condition: str = "good_lighting"):
        """Collect images from provided URLs"""
        logger.info(f"Collecting {authenticity} {denomination} naira images from {len(urls)} URLs")
        
        collected_count = 0
        failed_count = 0
        
        for i, url in enumerate(urls):
            try:
                # Download image
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Generate filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{authenticity}_{denomination}_{condition}_{url_hash}_{i:03d}.jpg"
                
                # Save image
                output_path = self.output_dir / authenticity / f"naira_{denomination}" / condition / filename
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # Validate image
                if self._validate_image(output_path):
                    collected_count += 1
                    logger.info(f"✓ Collected: {filename}")
                else:
                    os.remove(output_path)
                    failed_count += 1
                    logger.warning(f"✗ Invalid image: {filename}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                failed_count += 1
                logger.error(f"✗ Failed to collect from {url}: {str(e)}")
        
        logger.info(f"Collection complete: {collected_count} successful, {failed_count} failed")
        return collected_count, failed_count
    
    def collect_from_camera(self, authenticity: str, denomination: str, 
                           num_images: int = 10, condition: str = "good_lighting"):
        """Collect images using camera (for manual collection)"""
        logger.info(f"Starting camera collection for {authenticity} {denomination} naira")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return 0, 0
        
        collected_count = 0
        failed_count = 0
        
        print(f"\nPress SPACE to capture image, 'q' to quit")
        print(f"Target: {num_images} images of {authenticity} {denomination} naira")
        
        while collected_count < num_images:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                break
            
            # Display frame
            cv2.imshow('Naira Note Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Save image
                filename = f"{authenticity}_{denomination}_{condition}_camera_{collected_count:03d}.jpg"
                output_path = self.output_dir / authenticity / f"naira_{denomination}" / condition / filename
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save as PIL Image
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(output_path, "JPEG", quality=95)
                
                if self._validate_image(output_path):
                    collected_count += 1
                    print(f"✓ Captured {collected_count}/{num_images}: {filename}")
                else:
                    os.remove(output_path)
                    failed_count += 1
                    print(f"✗ Invalid image, retrying...")
                
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Camera collection complete: {collected_count} successful, {failed_count} failed")
        return collected_count, failed_count
    
    def collect_from_directory(self, source_dir: str, authenticity: str, 
                              denomination: str, condition: str = "good_lighting"):
        """Collect images from local directory"""
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return 0, 0
        
        logger.info(f"Collecting images from {source_dir}")
        
        collected_count = 0
        failed_count = 0
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for image_file in source_path.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                try:
                    # Generate new filename
                    filename = f"{authenticity}_{denomination}_{condition}_{image_file.stem}.jpg"
                    output_path = self.output_dir / authenticity / f"naira_{denomination}" / condition / filename
                    
                    # Copy and convert image
                    image = Image.open(image_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.save(output_path, "JPEG", quality=95)
                    
                    if self._validate_image(output_path):
                        collected_count += 1
                        logger.info(f"✓ Collected: {filename}")
                    else:
                        os.remove(output_path)
                        failed_count += 1
                        logger.warning(f"✗ Invalid image: {filename}")
                
                except Exception as e:
                    failed_count += 1
                    logger.error(f"✗ Failed to process {image_file}: {str(e)}")
        
        logger.info(f"Directory collection complete: {collected_count} successful, {failed_count} failed")
        return collected_count, failed_count
    
    def _validate_image(self, image_path: Path) -> bool:
        """Validate image quality and format"""
        try:
            # Check file size (minimum 10KB, maximum 10MB)
            file_size = image_path.stat().st_size
            if file_size < 10240 or file_size > 10485760:
                return False
            
            # Load and validate image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Check dimensions (minimum 224x224)
            height, width = image.shape[:2]
            if height < 224 or width < 224:
                return False
            
            # Check for corruption
            if cv2.Laplacian(image, cv2.CV_64F).var() < 100:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return False
    
    def generate_collection_report(self):
        """Generate collection statistics report"""
        report = {
            "collection_date": datetime.now().isoformat(),
            "total_images": 0,
            "by_authenticity": {},
            "by_denomination": {},
            "by_condition": {},
            "quality_metrics": {}
        }
        
        # Count images by category
        for authenticity in self.authenticity:
            report["by_authenticity"][authenticity] = 0
            for denomination in self.denominations:
                if denomination not in report["by_denomination"]:
                    report["by_denomination"][denomination] = 0
                for condition in self.conditions:
                    if condition not in report["by_condition"]:
                        report["by_condition"][condition] = 0
                    
                    dir_path = self.output_dir / authenticity / f"naira_{denomination}" / condition
                    if dir_path.exists():
                        image_count = len(list(dir_path.glob("*.jpg")))
                        report["by_authenticity"][authenticity] += image_count
                        report["by_denomination"][denomination] += image_count
                        report["by_condition"][condition] += image_count
                        report["total_images"] += image_count
        
        # Save report
        report_path = self.output_dir / "metadata" / "collection_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Collection report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Collect naira note images for dataset')
    parser.add_argument('--output-dir', type=str, default='datasets/raw',
                       help='Output directory for collected images')
    parser.add_argument('--method', type=str, choices=['urls', 'camera', 'directory'],
                       required=True, help='Collection method')
    parser.add_argument('--authenticity', type=str, choices=['genuine', 'fake'],
                       required=True, help='Image authenticity')
    parser.add_argument('--denomination', type=str, choices=['100', '200', '500', '1000'],
                       required=True, help='Naira denomination')
    parser.add_argument('--condition', type=str, default='good_lighting',
                       help='Image condition')
    parser.add_argument('--urls', type=str, nargs='+',
                       help='URLs to collect images from')
    parser.add_argument('--source-dir', type=str,
                       help='Source directory for local images')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to collect (camera method)')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = NairaDataCollector(args.output_dir)
    
    # Collect images based on method
    if args.method == 'urls':
        if not args.urls:
            logger.error("URLs must be provided for URL collection method")
            sys.exit(1)
        collector.collect_from_urls(args.urls, args.authenticity, args.denomination, args.condition)
    
    elif args.method == 'camera':
        collector.collect_from_camera(args.authenticity, args.denomination, 
                                    args.num_images, args.condition)
    
    elif args.method == 'directory':
        if not args.source_dir:
            logger.error("Source directory must be provided for directory collection method")
            sys.exit(1)
        collector.collect_from_directory(args.source_dir, args.authenticity, 
                                       args.denomination, args.condition)
    
    # Generate report
    collector.generate_collection_report()

if __name__ == '__main__':
    main()
