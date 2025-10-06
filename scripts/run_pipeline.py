#!/usr/bin/env python3
"""
Complete Data Pipeline Script for Naira Note Dataset
Orchestrates the entire data collection, preprocessing, and validation pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import pipeline components
from scripts.data_collection import NairaDataCollector
from scripts.data_preprocessing import NairaDataPreprocessor
from scripts.data_validation import DatasetValidator
from scripts.cloud_storage import CloudStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NairaDataPipeline:
    """Complete data pipeline for naira note dataset"""
    
    def __init__(self, config_path: str = "config/pipeline_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.collector = NairaDataCollector(self.config["paths"]["raw_data"])
        self.preprocessor = NairaDataPreprocessor(
            self.config["paths"]["raw_data"],
            self.config["paths"]["processed_data"]
        )
        self.validator = DatasetValidator(self.config["paths"]["processed_data"])
        self.cloud_manager = None
        
        # Pipeline state
        self.pipeline_state = {
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "statistics": {}
        }
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "paths": {
                    "raw_data": "datasets/raw",
                    "processed_data": "datasets/processed",
                    "reports": "reports"
                },
                "collection": {
                    "target_images_per_class": 1000,
                    "denominations": ["100", "200", "500", "1000"],
                    "conditions": ["good_lighting", "poor_lighting", "bright_lighting", 
                                 "new_condition", "worn_condition", "damaged_condition",
                                 "front_view", "back_view", "angled_view"],
                    "authenticity": ["genuine", "fake"]
                },
                "preprocessing": {
                    "train_ratio": 0.7,
                    "val_ratio": 0.2,
                    "test_ratio": 0.1,
                    "augmentation_factor": 3,
                    "quality_enhancement": True
                },
                "validation": {
                    "min_resolution": [224, 224],
                    "max_file_size_mb": 10,
                    "min_file_size_kb": 10,
                    "min_quality_score": 100,
                    "max_corruption_rate": 0.05
                },
                "cloud_storage": {
                    "enabled": False,
                    "provider": "aws",
                    "bucket_name": "",
                    "region": "us-east-1",
                    "sync_enabled": True
                }
            }
    
    def _save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_complete_pipeline(self, steps: List[str] = None) -> Dict:
        """Run the complete data pipeline"""
        if steps is None:
            steps = ["collect", "preprocess", "validate", "upload"]
        
        logger.info("Starting complete naira dataset pipeline")
        self.pipeline_state["start_time"] = datetime.now()
        
        try:
            # Step 1: Data Collection
            if "collect" in steps:
                self._run_collection_step()
            
            # Step 2: Data Preprocessing
            if "preprocess" in steps:
                self._run_preprocessing_step()
            
            # Step 3: Data Validation
            if "validate" in steps:
                self._run_validation_step()
            
            # Step 4: Cloud Upload
            if "upload" in steps and self.config["cloud_storage"]["enabled"]:
                self._run_upload_step()
            
            # Generate final report
            self._generate_pipeline_report()
            
            self.pipeline_state["end_time"] = datetime.now()
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.pipeline_state["failed_steps"].append({
                "step": self.pipeline_state["current_step"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
        
        return self.pipeline_state
    
    def _run_collection_step(self):
        """Run data collection step"""
        self.pipeline_state["current_step"] = "collection"
        logger.info("Starting data collection step...")
        
        collection_stats = {
            "total_collected": 0,
            "by_authenticity": {},
            "by_denomination": {},
            "by_condition": {}
        }
        
        # Collect images for each combination
        for authenticity in self.config["collection"]["authenticity"]:
            collection_stats["by_authenticity"][authenticity] = 0
            
            for denomination in self.config["collection"]["denominations"]:
                collection_stats["by_denomination"][denomination] = 0
                
                for condition in self.config["collection"]["conditions"]:
                    collection_stats["by_condition"][condition] = 0
                    
                    # Simulate collection (replace with actual collection logic)
                    logger.info(f"Collecting {authenticity} {denomination} naira images ({condition})")
                    
                    # This would be replaced with actual collection calls
                    # collector.collect_from_camera(authenticity, denomination, 10, condition)
                    # collector.collect_from_directory(source_dir, authenticity, denomination, condition)
                    
                    # For demo purposes, create sample data
                    self._create_sample_collection_data(authenticity, denomination, condition)
                    
                    # Update stats
                    sample_count = 10  # Simulated count
                    collection_stats["total_collected"] += sample_count
                    collection_stats["by_authenticity"][authenticity] += sample_count
                    collection_stats["by_denomination"][denomination] += sample_count
                    collection_stats["by_condition"][condition] += sample_count
        
        self.pipeline_state["statistics"]["collection"] = collection_stats
        self.pipeline_state["completed_steps"].append("collection")
        logger.info(f"Collection step completed: {collection_stats['total_collected']} images collected")
    
    def _run_preprocessing_step(self):
        """Run data preprocessing step"""
        self.pipeline_state["current_step"] = "preprocessing"
        logger.info("Starting data preprocessing step...")
        
        # Run preprocessing
        self.preprocessor.process_dataset(
            split_ratios=(
                self.config["preprocessing"]["train_ratio"],
                self.config["preprocessing"]["val_ratio"],
                self.config["preprocessing"]["test_ratio"]
            ),
            augmentation_factor=self.config["preprocessing"]["augmentation_factor"]
        )
        
        # Collect preprocessing statistics
        preprocessing_stats = self._collect_preprocessing_stats()
        
        self.pipeline_state["statistics"]["preprocessing"] = preprocessing_stats
        self.pipeline_state["completed_steps"].append("preprocessing")
        logger.info("Preprocessing step completed")
    
    def _run_validation_step(self):
        """Run data validation step"""
        self.pipeline_state["current_step"] = "validation"
        logger.info("Starting data validation step...")
        
        # Run validation
        validation_results = self.validator.validate_dataset()
        
        # Generate visualizations
        self.validator.generate_visualization(self.config["paths"]["reports"])
        
        self.pipeline_state["statistics"]["validation"] = validation_results
        self.pipeline_state["completed_steps"].append("validation")
        logger.info("Validation step completed")
    
    def _run_upload_step(self):
        """Run cloud upload step"""
        self.pipeline_state["current_step"] = "upload"
        logger.info("Starting cloud upload step...")
        
        # Initialize cloud manager
        cloud_config = self.config["cloud_storage"]
        self.cloud_manager = CloudStorageManager(
            provider=cloud_config["provider"],
            bucket_name=cloud_config["bucket_name"],
            region=cloud_config["region"]
        )
        
        # Create bucket if it doesn't exist
        self.cloud_manager.create_bucket()
        
        # Upload dataset
        upload_stats = self.cloud_manager.upload_dataset(
            self.config["paths"]["processed_data"],
            "datasets/"
        )
        
        self.pipeline_state["statistics"]["upload"] = upload_stats
        self.pipeline_state["completed_steps"].append("upload")
        logger.info("Cloud upload step completed")
    
    def _create_sample_collection_data(self, authenticity: str, denomination: str, condition: str):
        """Create sample collection data for demonstration"""
        # This is a placeholder - in real implementation, this would collect actual images
        output_dir = Path(self.config["paths"]["raw_data"]) / authenticity / f"naira_{denomination}" / condition
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple placeholder file to represent collected data
        placeholder_file = output_dir / f"{authenticity}_{denomination}_{condition}_placeholder.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"Sample data for {authenticity} {denomination} naira ({condition})\n")
            f.write(f"Collection timestamp: {datetime.now().isoformat()}\n")
    
    def _collect_preprocessing_stats(self) -> Dict:
        """Collect preprocessing statistics"""
        processed_path = Path(self.config["paths"]["processed_data"])
        
        stats = {
            "total_processed": 0,
            "by_split": {},
            "by_authenticity": {}
        }
        
        for split in ["train", "validation", "test"]:
            stats["by_split"][split] = 0
            for authenticity in ["genuine", "fake"]:
                if authenticity not in stats["by_authenticity"]:
                    stats["by_authenticity"][authenticity] = 0
                
                split_dir = processed_path / split / authenticity
                if split_dir.exists():
                    image_count = len(list(split_dir.glob("*.jpg")))
                    stats["by_split"][split] += image_count
                    stats["by_authenticity"][authenticity] += image_count
                    stats["total_processed"] += image_count
        
        return stats
    
    def _generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        report = {
            "pipeline_execution": self.pipeline_state,
            "configuration": self.config,
            "summary": {
                "total_duration": None,
                "success_rate": 0,
                "total_images_processed": 0
            }
        }
        
        # Calculate duration
        if self.pipeline_state["start_time"] and self.pipeline_state["end_time"]:
            duration = self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
            report["summary"]["total_duration"] = str(duration)
        
        # Calculate success rate
        total_steps = len(self.pipeline_state["completed_steps"]) + len(self.pipeline_state["failed_steps"])
        if total_steps > 0:
            report["summary"]["success_rate"] = len(self.pipeline_state["completed_steps"]) / total_steps
        
        # Calculate total images processed
        if "preprocessing" in self.pipeline_state["statistics"]:
            report["summary"]["total_images_processed"] = self.pipeline_state["statistics"]["preprocessing"]["total_processed"]
        
        # Save report
        report_path = Path(self.config["paths"]["reports"]) / "pipeline_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report saved to {report_path}")
    
    def run_individual_step(self, step: str) -> Dict:
        """Run individual pipeline step"""
        if step == "collect":
            self._run_collection_step()
        elif step == "preprocess":
            self._run_preprocessing_step()
        elif step == "validate":
            self._run_validation_step()
        elif step == "upload":
            self._run_upload_step()
        else:
            raise ValueError(f"Unknown step: {step}")
        
        return self.pipeline_state
    
    def setup_cloud_storage(self, provider: str, bucket_name: str, 
                           region: str = "us-east-1", credentials_path: str = None):
        """Setup cloud storage configuration"""
        self.config["cloud_storage"] = {
            "enabled": True,
            "provider": provider,
            "bucket_name": bucket_name,
            "region": region,
            "credentials_path": credentials_path,
            "sync_enabled": True
        }
        
        self._save_config()
        logger.info(f"Cloud storage configured: {provider}://{bucket_name}")

def main():
    parser = argparse.ArgumentParser(description='Run naira dataset pipeline')
    parser.add_argument('--config', type=str, default='config/pipeline_config.json',
                       help='Pipeline configuration file')
    parser.add_argument('--steps', type=str, nargs='+', 
                       choices=['collect', 'preprocess', 'validate', 'upload'],
                       help='Specific steps to run')
    parser.add_argument('--step', type=str, choices=['collect', 'preprocess', 'validate', 'upload'],
                       help='Run single step')
    parser.add_argument('--setup-cloud', action='store_true',
                       help='Setup cloud storage configuration')
    parser.add_argument('--provider', type=str, choices=['aws', 'azure', 'gcp'],
                       help='Cloud storage provider')
    parser.add_argument('--bucket', type=str, help='Cloud storage bucket name')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='Cloud storage region')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NairaDataPipeline(args.config)
    
    try:
        if args.setup_cloud:
            if not args.provider or not args.bucket:
                logger.error("Provider and bucket name required for cloud setup")
                sys.exit(1)
            pipeline.setup_cloud_storage(args.provider, args.bucket, args.region)
            logger.info("Cloud storage setup complete")
        
        elif args.step:
            # Run single step
            result = pipeline.run_individual_step(args.step)
            logger.info(f"Step '{args.step}' completed")
        
        else:
            # Run complete pipeline
            steps = args.steps if args.steps else None
            result = pipeline.run_complete_pipeline(steps)
            
            # Print summary
            print("\n" + "="*60)
            print("NAIRA DATASET PIPELINE SUMMARY")
            print("="*60)
            print(f"Duration: {result['summary']['total_duration']}")
            print(f"Success Rate: {result['summary']['success_rate']:.1%}")
            print(f"Images Processed: {result['summary']['total_images_processed']}")
            print(f"Completed Steps: {', '.join(result['completed_steps'])}")
            if result['failed_steps']:
                print(f"Failed Steps: {', '.join([s['step'] for s in result['failed_steps']])}")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
