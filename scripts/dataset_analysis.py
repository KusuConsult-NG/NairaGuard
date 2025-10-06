#!/usr/bin/env python3
"""
Dataset Statistics and Analysis Script
Provides detailed analysis and statistics for the naira note dataset
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzes naira note dataset and generates comprehensive statistics"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.analysis_results = {
            "analysis_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "statistics": {},
            "visualizations": [],
            "insights": [],
            "recommendations": []
        }
        
        # Image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def analyze_dataset(self) -> Dict:
        """Run complete dataset analysis"""
        logger.info(f"Starting analysis of dataset: {self.dataset_path}")
        
        # Basic statistics
        self._analyze_basic_statistics()
        
        # Image quality analysis
        self._analyze_image_quality()
        
        # Class distribution analysis
        self._analyze_class_distribution()
        
        # Resolution analysis
        self._analyze_resolution_distribution()
        
        # File size analysis
        self._analyze_file_size_distribution()
        
        # Color analysis
        self._analyze_color_distribution()
        
        # Generate insights
        self._generate_insights()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save analysis report
        self._save_analysis_report()
        
        logger.info("Dataset analysis complete")
        return self.analysis_results
    
    def _analyze_basic_statistics(self):
        """Analyze basic dataset statistics"""
        logger.info("Analyzing basic statistics...")
        
        stats = {
            "total_images": 0,
            "total_size_mb": 0,
            "average_file_size_mb": 0,
            "dataset_structure": {},
            "file_types": defaultdict(int),
            "directories": []
        }
        
        # Find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.dataset_path.rglob(f'*{ext}'))
            image_files.extend(self.dataset_path.rglob(f'*{ext.upper()}'))
        
        stats["total_images"] = len(image_files)
        
        # Calculate total size
        total_size = 0
        for image_file in image_files:
            try:
                file_size = image_file.stat().st_size
                total_size += file_size
                stats["file_types"][image_file.suffix.lower()] += 1
            except Exception as e:
                logger.warning(f"Could not get size for {image_file}: {str(e)}")
        
        stats["total_size_mb"] = total_size / (1024 * 1024)
        stats["average_file_size_mb"] = stats["total_size_mb"] / stats["total_images"] if stats["total_images"] > 0 else 0
        
        # Analyze directory structure
        for item in self.dataset_path.rglob('*'):
            if item.is_dir():
                stats["directories"].append(str(item.relative_to(self.dataset_path)))
        
        self.analysis_results["statistics"]["basic"] = stats
    
    def _analyze_image_quality(self):
        """Analyze image quality metrics"""
        logger.info("Analyzing image quality...")
        
        quality_stats = {
            "quality_scores": [],
            "corrupted_images": 0,
            "low_quality_images": 0,
            "average_quality_score": 0,
            "quality_distribution": defaultdict(int)
        }
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    quality_stats["corrupted_images"] += 1
                    continue
                
                # Calculate quality score (Laplacian variance)
                quality_score = cv2.Laplacian(image, cv2.CV_64F).var()
                quality_stats["quality_scores"].append(quality_score)
                
                # Categorize quality
                if quality_score < 100:
                    quality_stats["low_quality_images"] += 1
                    quality_stats["quality_distribution"]["poor"] += 1
                elif quality_score < 500:
                    quality_stats["quality_distribution"]["fair"] += 1
                elif quality_score < 1000:
                    quality_stats["quality_distribution"]["good"] += 1
                else:
                    quality_stats["quality_distribution"]["excellent"] += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing quality for {image_file}: {str(e)}")
                quality_stats["corrupted_images"] += 1
        
        # Calculate average quality score
        if quality_stats["quality_scores"]:
            quality_stats["average_quality_score"] = np.mean(quality_stats["quality_scores"])
            quality_stats["median_quality_score"] = np.median(quality_stats["quality_scores"])
            quality_stats["quality_score_std"] = np.std(quality_stats["quality_scores"])
        
        self.analysis_results["statistics"]["quality"] = quality_stats
    
    def _analyze_class_distribution(self):
        """Analyze class distribution"""
        logger.info("Analyzing class distribution...")
        
        class_stats = {
            "by_authenticity": defaultdict(int),
            "by_denomination": defaultdict(int),
            "by_condition": defaultdict(int),
            "by_split": defaultdict(int),
            "class_balance": {}
        }
        
        # Analyze by authenticity
        for authenticity in ["genuine", "fake"]:
            auth_files = list(self.dataset_path.rglob(f"*{authenticity}*"))
            class_stats["by_authenticity"][authenticity] = len(auth_files)
        
        # Analyze by denomination
        denominations = ["100", "200", "500", "1000"]
        for denom in denominations:
            denom_files = list(self.dataset_path.rglob(f"*{denom}*"))
            class_stats["by_denomination"][denom] = len(denom_files)
        
        # Analyze by condition
        conditions = ["good_lighting", "poor_lighting", "bright_lighting", 
                     "new_condition", "worn_condition", "damaged_condition",
                     "front_view", "back_view", "angled_view"]
        for condition in conditions:
            condition_files = list(self.dataset_path.rglob(f"*{condition}*"))
            class_stats["by_condition"][condition] = len(condition_files)
        
        # Analyze by split
        splits = ["train", "validation", "test"]
        for split in splits:
            split_files = list(self.dataset_path.rglob(f"*{split}*"))
            class_stats["by_split"][split] = len(split_files)
        
        # Calculate class balance
        genuine_count = class_stats["by_authenticity"]["genuine"]
        fake_count = class_stats["by_authenticity"]["fake"]
        
        if genuine_count > 0 and fake_count > 0:
            total_count = genuine_count + fake_count
            class_stats["class_balance"] = {
                "genuine_percentage": (genuine_count / total_count) * 100,
                "fake_percentage": (fake_count / total_count) * 100,
                "balance_ratio": min(genuine_count, fake_count) / max(genuine_count, fake_count)
            }
        
        self.analysis_results["statistics"]["class_distribution"] = class_stats
    
    def _analyze_resolution_distribution(self):
        """Analyze image resolution distribution"""
        logger.info("Analyzing resolution distribution...")
        
        resolution_stats = {
            "resolutions": defaultdict(int),
            "widths": [],
            "heights": [],
            "aspect_ratios": [],
            "common_resolutions": []
        }
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                resolution = f"{width}x{height}"
                
                resolution_stats["resolutions"][resolution] += 1
                resolution_stats["widths"].append(width)
                resolution_stats["heights"].append(height)
                resolution_stats["aspect_ratios"].append(width / height)
                
            except Exception as e:
                logger.warning(f"Error analyzing resolution for {image_file}: {str(e)}")
        
        # Find common resolutions
        sorted_resolutions = sorted(resolution_stats["resolutions"].items(), 
                                  key=lambda x: x[1], reverse=True)
        resolution_stats["common_resolutions"] = sorted_resolutions[:10]
        
        # Calculate statistics
        if resolution_stats["widths"]:
            resolution_stats["average_width"] = np.mean(resolution_stats["widths"])
            resolution_stats["average_height"] = np.mean(resolution_stats["heights"])
            resolution_stats["average_aspect_ratio"] = np.mean(resolution_stats["aspect_ratios"])
        
        self.analysis_results["statistics"]["resolution"] = resolution_stats
    
    def _analyze_file_size_distribution(self):
        """Analyze file size distribution"""
        logger.info("Analyzing file size distribution...")
        
        size_stats = {
            "file_sizes": [],
            "size_categories": defaultdict(int),
            "average_size_mb": 0,
            "median_size_mb": 0
        }
        
        # Find all image files
        image_files = list(self.dataset_path.rglob("*.jpg"))
        
        for image_file in image_files:
            try:
                file_size = image_file.stat().st_size
                size_mb = file_size / (1024 * 1024)
                
                size_stats["file_sizes"].append(size_mb)
                
                # Categorize by size
                if size_mb < 0.1:
                    size_stats["size_categories"]["< 0.1MB"] += 1
                elif size_mb < 0.5:
                    size_stats["size_categories"]["0.1-0.5MB"] += 1
                elif size_mb < 1.0:
                    size_stats["size_categories"]["0.5-1.0MB"] += 1
                elif size_mb < 2.0:
                    size_stats["size_categories"]["1.0-2.0MB"] += 1
                else:
                    size_stats["size_categories"]["> 2.0MB"] += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing file size for {image_file}: {str(e)}")
        
        # Calculate statistics
        if size_stats["file_sizes"]:
            size_stats["average_size_mb"] = np.mean(size_stats["file_sizes"])
            size_stats["median_size_mb"] = np.median(size_stats["file_sizes"])
            size_stats["min_size_mb"] = np.min(size_stats["file_sizes"])
            size_stats["max_size_mb"] = np.max(size_stats["file_sizes"])
        
        self.analysis_results["statistics"]["file_size"] = size_stats
    
    def _analyze_color_distribution(self):
        """Analyze color distribution in images"""
        logger.info("Analyzing color distribution...")
        
        color_stats = {
            "average_colors": [],
            "dominant_colors": [],
            "color_channels": {
                "red": [],
                "green": [],
                "blue": []
            },
            "brightness_levels": []
        }
        
        # Sample images for color analysis (to avoid memory issues)
        image_files = list(self.dataset_path.rglob("*.jpg"))
        sample_size = min(100, len(image_files))
        sample_files = np.random.choice(image_files, sample_size, replace=False)
        
        for image_file in sample_files:
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Calculate average color
                avg_color = np.mean(image_rgb, axis=(0, 1))
                color_stats["average_colors"].append(avg_color)
                
                # Calculate channel statistics
                color_stats["color_channels"]["red"].append(np.mean(image_rgb[:, :, 0]))
                color_stats["color_channels"]["green"].append(np.mean(image_rgb[:, :, 1]))
                color_stats["color_channels"]["blue"].append(np.mean(image_rgb[:, :, 2]))
                
                # Calculate brightness
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                color_stats["brightness_levels"].append(brightness)
                
            except Exception as e:
                logger.warning(f"Error analyzing color for {image_file}: {str(e)}")
        
        # Calculate statistics
        if color_stats["average_colors"]:
            color_stats["overall_average_color"] = np.mean(color_stats["average_colors"], axis=0).tolist()
            color_stats["average_brightness"] = np.mean(color_stats["brightness_levels"])
        
        self.analysis_results["statistics"]["color"] = color_stats
    
    def _generate_insights(self):
        """Generate insights from analysis"""
        insights = []
        
        # Basic insights
        basic_stats = self.analysis_results["statistics"]["basic"]
        insights.append(f"Dataset contains {basic_stats['total_images']} images totaling {basic_stats['total_size_mb']:.1f}MB")
        
        # Quality insights
        quality_stats = self.analysis_results["statistics"]["quality"]
        if quality_stats["quality_scores"]:
            insights.append(f"Average image quality score: {quality_stats['average_quality_score']:.1f}")
            insights.append(f"Corruption rate: {(quality_stats['corrupted_images'] / basic_stats['total_images']) * 100:.1f}%")
        
        # Class balance insights
        class_stats = self.analysis_results["statistics"]["class_distribution"]
        if "class_balance" in class_stats:
            balance = class_stats["class_balance"]
            insights.append(f"Class balance: {balance['genuine_percentage']:.1f}% genuine, {balance['fake_percentage']:.1f}% fake")
            insights.append(f"Balance ratio: {balance['balance_ratio']:.2f}")
        
        # Resolution insights
        resolution_stats = self.analysis_results["statistics"]["resolution"]
        if resolution_stats["common_resolutions"]:
            most_common = resolution_stats["common_resolutions"][0]
            insights.append(f"Most common resolution: {most_common[0]} ({most_common[1]} images)")
        
        self.analysis_results["insights"] = insights
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Quality recommendations
        quality_stats = self.analysis_results["statistics"]["quality"]
        corruption_rate = quality_stats["corrupted_images"] / self.analysis_results["statistics"]["basic"]["total_images"]
        
        if corruption_rate > 0.05:
            recommendations.append("High corruption rate detected. Consider removing corrupted images.")
        
        if quality_stats["average_quality_score"] < 200:
            recommendations.append("Low average quality score. Consider improving image quality.")
        
        # Class balance recommendations
        class_stats = self.analysis_results["statistics"]["class_distribution"]
        if "class_balance" in class_stats:
            balance_ratio = class_stats["class_balance"]["balance_ratio"]
            if balance_ratio < 0.5:
                recommendations.append("Significant class imbalance detected. Consider collecting more images for minority class.")
        
        # Resolution recommendations
        resolution_stats = self.analysis_results["statistics"]["resolution"]
        if resolution_stats["common_resolutions"]:
            most_common_res = resolution_stats["common_resolutions"][0][0]
            if most_common_res != "224x224":
                recommendations.append(f"Most common resolution ({most_common_res}) differs from target (224x224). Consider resizing.")
        
        # File size recommendations
        size_stats = self.analysis_results["statistics"]["file_size"]
        if size_stats["average_size_mb"] > 2.0:
            recommendations.append("Large average file size detected. Consider compression to reduce storage requirements.")
        
        self.analysis_results["recommendations"] = recommendations
    
    def _save_analysis_report(self):
        """Save analysis report to file"""
        report_path = self.dataset_path / "metadata" / "analysis_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        logger.info(f"Analysis report saved to {report_path}")
    
    def generate_visualizations(self, output_dir: str = "reports"):
        """Generate visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Quality Score Distribution
        quality_stats = self.analysis_results["statistics"]["quality"]
        if quality_stats["quality_scores"]:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(quality_stats["quality_scores"], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(quality_stats["average_quality_score"], color='red', linestyle='--', 
                       label=f'Average: {quality_stats["average_quality_score"]:.1f}')
            plt.xlabel('Quality Score (Laplacian Variance)')
            plt.ylabel('Number of Images')
            plt.title('Image Quality Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Class Distribution
            plt.subplot(2, 2, 2)
            class_stats = self.analysis_results["statistics"]["class_distribution"]
            auth_counts = list(class_stats["by_authenticity"].values())
            auth_labels = list(class_stats["by_authenticity"].keys())
            colors = ['green', 'red']
            bars = plt.bar(auth_labels, auth_counts, color=colors, alpha=0.7)
            plt.xlabel('Authenticity')
            plt.ylabel('Number of Images')
            plt.title('Class Distribution')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, auth_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            # 3. Resolution Distribution
            plt.subplot(2, 2, 3)
            resolution_stats = self.analysis_results["statistics"]["resolution"]
            if resolution_stats["common_resolutions"]:
                resolutions = [res[0] for res in resolution_stats["common_resolutions"][:10]]
                counts = [res[1] for res in resolution_stats["common_resolutions"][:10]]
                plt.bar(range(len(resolutions)), counts, alpha=0.7, color='orange')
                plt.xlabel('Resolution')
                plt.ylabel('Number of Images')
                plt.title('Top 10 Resolutions')
                plt.xticks(range(len(resolutions)), resolutions, rotation=45)
                plt.grid(True, alpha=0.3)
            
            # 4. File Size Distribution
            plt.subplot(2, 2, 4)
            size_stats = self.analysis_results["statistics"]["file_size"]
            if size_stats["file_sizes"]:
                plt.hist(size_stats["file_sizes"], bins=30, alpha=0.7, color='purple', edgecolor='black')
                plt.axvline(size_stats["average_size_mb"], color='red', linestyle='--',
                           label=f'Average: {size_stats["average_size_mb"]:.2f}MB')
                plt.xlabel('File Size (MB)')
                plt.ylabel('Number of Images')
                plt.title('File Size Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Denomination Distribution
        class_stats = self.analysis_results["statistics"]["class_distribution"]
        if class_stats["by_denomination"]:
            plt.figure(figsize=(10, 6))
            denominations = list(class_stats["by_denomination"].keys())
            counts = list(class_stats["by_denomination"].values())
            
            bars = plt.bar(denominations, counts, alpha=0.7, color='lightblue', edgecolor='black')
            plt.xlabel('Denomination (₦)')
            plt.ylabel('Number of Images')
            plt.title('Distribution by Denomination')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            plt.savefig(output_path / 'denomination_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze naira note dataset')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports and visualizations')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.dataset_path)
    
    # Run analysis
    results = analyzer.analyze_dataset()
    
    # Generate visualizations if requested
    if args.generate_plots:
        analyzer.generate_visualizations(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET ANALYSIS SUMMARY")
    print("="*60)
    
    basic_stats = results["statistics"]["basic"]
    print(f"Total Images: {basic_stats['total_images']}")
    print(f"Total Size: {basic_stats['total_size_mb']:.1f} MB")
    print(f"Average File Size: {basic_stats['average_file_size_mb']:.2f} MB")
    
    quality_stats = results["statistics"]["quality"]
    if quality_stats["quality_scores"]:
        print(f"Average Quality Score: {quality_stats['average_quality_score']:.1f}")
        print(f"Corruption Rate: {(quality_stats['corrupted_images'] / basic_stats['total_images']) * 100:.1f}%")
    
    class_stats = results["statistics"]["class_distribution"]
    if "class_balance" in class_stats:
        balance = class_stats["class_balance"]
        print(f"Class Balance: {balance['genuine_percentage']:.1f}% genuine, {balance['fake_percentage']:.1f}% fake")
    
    if results["insights"]:
        print("\nINSIGHTS:")
        for i, insight in enumerate(results["insights"], 1):
            print(f"{i}. {insight}")
    
    if results["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"{i}. {rec}")
    
    print("="*60)
    print("Detailed analysis report saved to dataset metadata directory.")

if __name__ == '__main__':
    main()
