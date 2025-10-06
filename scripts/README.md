# 📊 Data Collection & Preprocessing Pipeline

## 🎯 Overview

This directory contains a comprehensive data pipeline for collecting, preprocessing, and validating naira note images for counterfeit detection. The pipeline handles the entire data lifecycle from collection to cloud storage.

## 📁 Structure

```
scripts/
├── data_collection.py      # Image collection from various sources
├── data_preprocessing.py    # Image preprocessing and augmentation
├── data_validation.py      # Dataset quality validation
├── cloud_storage.py        # Cloud storage integration
├── run_pipeline.py         # Complete pipeline orchestration
└── __init__.py            # Package initialization
```

## 🚀 Quick Start

### 1. Complete Pipeline
```bash
# Run the entire pipeline
python scripts/run_pipeline.py

# Run specific steps
python scripts/run_pipeline.py --steps collect preprocess validate
```

### 2. Individual Steps

#### Data Collection
```bash
# Collect from camera
python scripts/data_collection.py --method camera --authenticity genuine --denomination 500 --num-images 50

# Collect from URLs
python scripts/data_collection.py --method urls --authenticity fake --denomination 1000 --urls "https://example.com/image1.jpg" "https://example.com/image2.jpg"

# Collect from local directory
python scripts/data_collection.py --method directory --authenticity genuine --denomination 200 --source-dir /path/to/images
```

#### Data Preprocessing
```bash
# Process with default settings (70/20/10 split, 3x augmentation)
python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed

# Custom split ratios
python scripts/data_preprocessing.py --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05 --augmentation-factor 5
```

#### Data Validation
```bash
# Validate dataset quality
python scripts/data_validation.py --dataset-path datasets/processed --generate-plots

# Generate detailed reports
python scripts/data_validation.py --dataset-path datasets/processed --output-dir reports --generate-plots
```

#### Cloud Storage
```bash
# Upload to AWS S3
python scripts/cloud_storage.py --provider aws --bucket my-dataset-bucket --action upload --local-path datasets/processed

# Download from cloud
python scripts/cloud_storage.py --provider aws --bucket my-dataset-bucket --action download --local-path datasets/downloaded

# Sync with cloud (upload only changed files)
python scripts/cloud_storage.py --provider aws --bucket my-dataset-bucket --action sync --local-path datasets/processed
```

## 📋 Pipeline Steps

### 1. Data Collection (`data_collection.py`)

**Purpose**: Gather naira note images from various sources

**Features**:
- Camera collection for manual capture
- URL-based collection from web sources
- Local directory collection
- Automatic directory structure creation
- Image validation and quality checks
- Collection statistics and reporting

**Supported Sources**:
- Central Bank of Nigeria (CBN) official images
- Bank training materials
- Law enforcement seized evidence
- Museum collections
- Manual camera capture

### 2. Data Preprocessing (`data_preprocessing.py`)

**Purpose**: Process raw images into ML-ready format

**Features**:
- Train/validation/test splitting (70/20/10)
- Data augmentation (rotation, brightness, contrast, noise)
- Image quality enhancement
- Resolution standardization (224x224)
- Format normalization (JPEG)
- Batch processing

**Augmentation Techniques**:
- Rotation (±15°)
- Brightness/Contrast adjustment (±20%)
- Hue/Saturation/Value shifts
- Gaussian noise addition
- Motion blur simulation
- Random shadows
- Coarse dropout

### 3. Data Validation (`data_validation.py`)

**Purpose**: Ensure dataset quality and completeness

**Features**:
- Image quality validation
- Dataset completeness checks
- Consistency validation
- Duplicate detection
- Statistical analysis
- Visualization generation
- Quality recommendations

**Validation Criteria**:
- Minimum resolution: 224x224
- File size limits: 10KB - 10MB
- Quality score threshold: 100 (Laplacian variance)
- Maximum corruption rate: 5%
- Class balance requirements

### 4. Cloud Storage (`cloud_storage.py`)

**Purpose**: Manage dataset in cloud storage

**Features**:
- Multi-cloud support (AWS S3, Azure Blob, GCP Storage)
- Incremental sync
- Duplicate detection
- Public URL generation
- Encryption support
- Cost optimization

**Supported Providers**:
- AWS S3
- Azure Blob Storage
- Google Cloud Storage

## ⚙️ Configuration

### Pipeline Configuration (`config/pipeline_config.json`)

```json
{
  "paths": {
    "raw_data": "datasets/raw",
    "processed_data": "datasets/processed",
    "reports": "reports"
  },
  "collection": {
    "target_images_per_class": 1000,
    "denominations": ["100", "200", "500", "1000"],
    "conditions": ["good_lighting", "poor_lighting", "bright_lighting"],
    "authenticity": ["genuine", "fake"]
  },
  "preprocessing": {
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "augmentation_factor": 3
  },
  "validation": {
    "min_resolution": [224, 224],
    "max_file_size_mb": 10,
    "min_quality_score": 100
  }
}
```

## 📊 Dataset Structure

### Raw Data Structure
```
datasets/raw/
├── genuine/
│   ├── naira_100/
│   │   ├── good_lighting/
│   │   ├── poor_lighting/
│   │   └── bright_lighting/
│   ├── naira_200/
│   ├── naira_500/
│   └── naira_1000/
└── fake/
    ├── naira_100/
    ├── naira_200/
    ├── naira_500/
    └── naira_1000/
```

### Processed Data Structure
```
datasets/processed/
├── train/
│   ├── genuine/
│   └── fake/
├── validation/
│   ├── genuine/
│   └── fake/
└── test/
    ├── genuine/
    └── fake/
```

## 🔧 Requirements

### Python Dependencies
```bash
pip install opencv-python pillow albumentations numpy pandas matplotlib seaborn boto3 azure-storage-blob google-cloud-storage requests
```

### System Requirements
- Python 3.8+
- OpenCV 4.0+
- PIL/Pillow
- Camera access (for manual collection)
- Internet connection (for URL collection)
- Cloud storage credentials (for cloud integration)

## 📈 Quality Metrics

### Image Quality Standards
- **Resolution**: Minimum 224x224 pixels
- **Format**: JPEG with 95% quality
- **File Size**: 10KB - 10MB
- **Quality Score**: Minimum 100 (Laplacian variance)
- **Corruption Rate**: Maximum 5%

### Dataset Balance
- **Class Balance**: Genuine vs Fake ratio within 50%
- **Denomination Coverage**: All denominations (₦100, ₦200, ₦500, ₦1000)
- **Condition Coverage**: Multiple lighting and wear conditions
- **Minimum Images**: 100+ per class

## 🚨 Troubleshooting

### Common Issues

1. **Camera Access Denied**
   ```bash
   # Grant camera permissions
   sudo chmod 666 /dev/video0
   ```

2. **Cloud Storage Authentication**
   ```bash
   # AWS
   aws configure
   
   # Azure
   az login
   
   # GCP
   gcloud auth application-default login
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   python scripts/data_preprocessing.py --augmentation-factor 1
   ```

4. **File Permission Errors**
   ```bash
   # Fix permissions
   chmod -R 755 datasets/
   ```

## 📝 Best Practices

### Data Collection
- Collect images under various lighting conditions
- Include different wear levels (new, used, damaged)
- Capture multiple angles (front, back, angled)
- Maintain consistent naming conventions
- Validate images immediately after collection

### Preprocessing
- Use appropriate augmentation techniques
- Maintain class balance during splitting
- Preserve original images as backup
- Document augmentation parameters
- Test preprocessing on small samples first

### Validation
- Run validation after each major change
- Monitor quality metrics over time
- Address issues before proceeding
- Generate visual reports for review
- Maintain validation logs

### Cloud Storage
- Use incremental sync for large datasets
- Enable encryption for sensitive data
- Monitor storage costs
- Implement backup strategies
- Use appropriate access controls

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review configuration files
- Run scripts with `--help` for detailed options
- Check logs in the `logs/` directory
- Generate validation reports for analysis

## 🔄 Updates

The pipeline is designed to be extensible:
- Add new collection sources in `data_collection.py`
- Implement new augmentation techniques in `data_preprocessing.py`
- Add validation criteria in `data_validation.py`
- Support additional cloud providers in `cloud_storage.py`
- Extend configuration options in `pipeline_config.json`
