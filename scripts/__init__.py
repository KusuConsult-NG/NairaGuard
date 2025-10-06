#!/usr/bin/env python3
"""
Data Collection and Preprocessing Pipeline for Naira Note Detection

This script implements a complete data pipeline for collecting, preprocessing,
and validating naira note images for counterfeit detection.

Usage:
    # Run complete pipeline
    python scripts/run_pipeline.py

    # Run specific steps
    python scripts/run_pipeline.py --steps collect preprocess validate

    # Run single step
    python scripts/run_pipeline.py --step collect

    # Setup cloud storage
    python scripts/run_pipeline.py --setup-cloud --provider aws --bucket my-bucket

    # Individual scripts
    python scripts/data_collection.py --method camera --authenticity genuine --denomination 500
    python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed
    python scripts/data_validation.py --dataset-path datasets/processed --generate-plots
    python scripts/cloud_storage.py --provider aws --bucket my-bucket --action upload --local-path datasets/processed
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Main entry point for the data pipeline"""
    print("Naira Note Detection - Data Pipeline")
    print("=" * 50)
    print()
    print("Available commands:")
    print("1. Complete Pipeline: python scripts/run_pipeline.py")
    print("2. Data Collection: python scripts/data_collection.py --help")
    print("3. Data Preprocessing: python scripts/data_preprocessing.py --help")
    print("4. Data Validation: python scripts/data_validation.py --help")
    print("5. Cloud Storage: python scripts/cloud_storage.py --help")
    print()
    print("For detailed usage, run any script with --help")

if __name__ == '__main__':
    main()
