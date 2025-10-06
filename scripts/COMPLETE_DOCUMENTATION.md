# 📊 Complete Data Pipeline Documentation

## 🎯 Overview

This directory contains a comprehensive data pipeline for collecting, preprocessing, validating, and managing naira note images for counterfeit detection. The pipeline handles the entire data lifecycle from collection to cloud storage.

## 📁 Scripts Overview

### Core Pipeline Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `data_collection.py` | Image collection from various sources | Camera capture, URL collection, directory scanning, quality validation |
| `data_preprocessing.py` | Image preprocessing and augmentation | Data splitting, augmentation, quality enhancement, format standardization |
| `data_validation.py` | Dataset quality validation | Structure validation, quality checks, completeness analysis, consistency validation |
| `cloud_storage.py` | Cloud storage integration | Multi-cloud support, incremental sync, encryption, public URLs |
| `run_pipeline.py` | Complete pipeline orchestration | End-to-end automation, step-by-step control, configuration management |

### Utility Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `dataset_analysis.py` | Dataset statistics and analysis | Quality analysis, class distribution, resolution analysis, visualizations |
| `dataset_cleaning.py` | Dataset cleaning and maintenance | Remove corrupted files, remove duplicates, standardize names, organize structure |
| `dataset_export_import.py` | Dataset export/import in various formats | ZIP/TAR archives, CSV/JSON metadata, pickle format, directory copying |

## 🚀 Quick Start Guide

### 1. Complete Pipeline Execution

```bash
# Run entire pipeline
python scripts/run_pipeline.py

# Run specific steps
python scripts/run_pipeline.py --steps collect preprocess validate

# Run single step
python scripts/run_pipeline.py --step collect
```

### 2. Individual Operations

#### Data Collection
```bash
# Collect from camera
python scripts/data_collection.py --method camera --authenticity genuine --denomination 500 --num-images 50

# Collect from URLs
python scripts/data_collection.py --method urls --authenticity fake --denomination 1000 --urls "https://example.com/image1.jpg"

# Collect from local directory
python scripts/data_collection.py --method directory --authenticity genuine --denomination 200 --source-dir /path/to/images
```

#### Data Preprocessing
```bash
# Process with default settings
python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed

# Custom configuration
python scripts/data_preprocessing.py --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05 --augmentation-factor 5
```

#### Data Validation
```bash
# Validate dataset
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

# Sync with cloud
python scripts/cloud_storage.py --provider aws --bucket my-dataset-bucket --action sync --local-path datasets/processed
```

### 3. Dataset Management

#### Analysis
```bash
# Analyze dataset
python scripts/dataset_analysis.py --dataset-path datasets/processed --generate-plots

# Generate comprehensive statistics
python scripts/dataset_analysis.py --dataset-path datasets/processed --output-dir reports --generate-plots
```

#### Cleaning
```bash
# Clean dataset
python scripts/dataset_cleaning.py --dataset-path datasets/raw --actions remove_corrupted remove_duplicates standardize_names

# Create backup before cleaning
python scripts/dataset_cleaning.py --dataset-path datasets/raw --backup backups/pre_cleaning --actions remove_corrupted
```

#### Export/Import
```bash
# Export to ZIP
python scripts/dataset_export_import.py --action export --dataset-path datasets/processed --output-path exports/dataset.zip --format zip

# Export to CSV
python scripts/dataset_export_import.py --action export --dataset-path datasets/processed --output-path exports/metadata.csv --format csv

# Import from ZIP
python scripts/dataset_export_import.py --action import --dataset-path datasets/imported --source-path imports/dataset.zip --format zip
```

## 📋 Detailed Script Documentation

### 1. Data Collection (`data_collection.py`)

**Purpose**: Gather naira note images from various sources

**Features**:
- **Camera Collection**: Manual capture using webcam
- **URL Collection**: Download images from web sources
- **Directory Collection**: Process existing image collections
- **Quality Validation**: Real-time image quality checks
- **Automatic Organization**: Create proper directory structure
- **Collection Reporting**: Statistics and progress tracking

**Usage Examples**:
```bash
# Collect genuine naira notes from camera
python scripts/data_collection.py --method camera --authenticity genuine --denomination 1000 --num-images 100 --condition good_lighting

# Collect fake naira notes from URLs
python scripts/data_collection.py --method urls --authenticity fake --denomination 500 --urls "https://example.com/fake1.jpg" "https://example.com/fake2.jpg"

# Process existing image collection
python scripts/data_collection.py --method directory --authenticity genuine --denomination 200 --source-dir /path/to/existing/images
```

### 2. Data Preprocessing (`data_preprocessing.py`)

**Purpose**: Process raw images into ML-ready format

**Features**:
- **Data Splitting**: Train/validation/test (70/20/10)
- **Data Augmentation**: 8+ augmentation techniques
- **Quality Enhancement**: Contrast, sharpness, color improvements
- **Format Standardization**: Consistent 224x224 JPEG output
- **Batch Processing**: Efficient processing of large datasets

**Augmentation Techniques**:
- Rotation (±15°)
- Brightness/Contrast adjustment (±20%)
- Hue/Saturation/Value shifts
- Gaussian noise addition
- Motion blur simulation
- Random shadows
- Coarse dropout

**Usage Examples**:
```bash
# Process with default settings
python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed

# Custom configuration
python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05 --augmentation-factor 5

# Disable quality enhancement
python scripts/data_preprocessing.py --input-dir datasets/raw --output-dir datasets/processed --no-quality-enhancement
```

### 3. Data Validation (`data_validation.py`)

**Purpose**: Ensure dataset quality and completeness

**Features**:
- **Structure Validation**: Directory organization checks
- **Quality Validation**: Resolution, file size, corruption detection
- **Completeness Validation**: Class balance, coverage analysis
- **Consistency Validation**: Naming conventions, duplicates, metadata
- **Statistical Analysis**: Distributions, metrics, trends
- **Visual Reporting**: Charts, graphs, plots

**Validation Criteria**:
- Minimum resolution: 224x224
- File size limits: 10KB - 10MB
- Quality score threshold: 100 (Laplacian variance)
- Maximum corruption rate: 5%
- Class balance requirements

**Usage Examples**:
```bash
# Basic validation
python scripts/data_validation.py --dataset-path datasets/processed

# Generate visualizations
python scripts/data_validation.py --dataset-path datasets/processed --generate-plots

# Custom output directory
python scripts/data_validation.py --dataset-path datasets/processed --output-dir reports --generate-plots
```

### 4. Cloud Storage (`cloud_storage.py`)

**Purpose**: Manage dataset in cloud storage

**Features**:
- **Multi-Cloud Support**: AWS S3, Azure Blob, Google Cloud Storage
- **Incremental Sync**: Upload only changed files
- **Duplicate Detection**: MD5 hash comparison
- **Public URL Generation**: Presigned URLs for sharing
- **Encryption Support**: AES256 encryption
- **Cost Optimization**: Efficient transfers and storage

**Usage Examples**:
```bash
# Upload to AWS S3
python scripts/cloud_storage.py --provider aws --bucket my-dataset-bucket --action upload --local-path datasets/processed

# Download from Azure
python scripts/cloud_storage.py --provider azure --bucket my-container --action download --local-path datasets/downloaded

# Sync with Google Cloud
python scripts/cloud_storage.py --provider gcp --bucket my-bucket --action sync --local-path datasets/processed
```

### 5. Pipeline Orchestration (`run_pipeline.py`)

**Purpose**: Orchestrate complete data pipeline

**Features**:
- **Complete Automation**: End-to-end pipeline execution
- **Step-by-Step Control**: Run individual or multiple steps
- **Configuration Management**: JSON-based configuration
- **Progress Tracking**: Real-time status and statistics
- **Error Handling**: Robust error recovery and reporting

**Usage Examples**:
```bash
# Run complete pipeline
python scripts/run_pipeline.py

# Run specific steps
python scripts/run_pipeline.py --steps collect preprocess validate

# Run single step
python scripts/run_pipeline.py --step collect

# Setup cloud storage
python scripts/run_pipeline.py --setup-cloud --provider aws --bucket my-bucket --region us-east-1
```

### 6. Dataset Analysis (`dataset_analysis.py`)

**Purpose**: Analyze dataset statistics and generate insights

**Features**:
- **Quality Analysis**: Image quality metrics and distribution
- **Class Distribution**: Authenticity, denomination, condition analysis
- **Resolution Analysis**: Resolution distribution and trends
- **File Size Analysis**: Size distribution and optimization opportunities
- **Color Analysis**: Color distribution and characteristics
- **Visualizations**: Charts, graphs, and plots

**Usage Examples**:
```bash
# Basic analysis
python scripts/dataset_analysis.py --dataset-path datasets/processed

# Generate visualizations
python scripts/dataset_analysis.py --dataset-path datasets/processed --generate-plots

# Custom output directory
python scripts/dataset_analysis.py --dataset-path datasets/processed --output-dir reports --generate-plots
```

### 7. Dataset Cleaning (`dataset_cleaning.py`)

**Purpose**: Clean and maintain dataset

**Features**:
- **Remove Corrupted Files**: Invalid images, low quality, wrong size
- **Remove Duplicates**: Content-based duplicate detection
- **Standardize Names**: Consistent naming conventions
- **Organize Structure**: Proper directory organization
- **Compress Images**: Reduce file size while maintaining quality
- **Validate Metadata**: Check and fix metadata files

**Usage Examples**:
```bash
# Clean dataset
python scripts/dataset_cleaning.py --dataset-path datasets/raw --actions remove_corrupted remove_duplicates standardize_names

# Create backup before cleaning
python scripts/dataset_cleaning.py --dataset-path datasets/raw --backup backups/pre_cleaning --actions remove_corrupted

# Restore from backup
python scripts/dataset_cleaning.py --dataset-path datasets/raw --restore backups/pre_cleaning
```

### 8. Dataset Export/Import (`dataset_export_import.py`)

**Purpose**: Export and import dataset in various formats

**Features**:
- **Multiple Formats**: ZIP, TAR, CSV, JSON, pickle, directory
- **Metadata Export**: Comprehensive metadata in structured formats
- **Compression**: Optional compression for archives
- **Import Support**: Import from various sources
- **Format Detection**: Auto-detect format for imports

**Usage Examples**:
```bash
# Export to ZIP
python scripts/dataset_export_import.py --action export --dataset-path datasets/processed --output-path exports/dataset.zip --format zip

# Export metadata to CSV
python scripts/dataset_export_import.py --action export --dataset-path datasets/processed --output-path exports/metadata.csv --format csv

# Import from ZIP
python scripts/dataset_export_import.py --action import --dataset-path datasets/imported --source-path imports/dataset.zip --format zip
```

## ⚙️ Configuration

### Pipeline Configuration (`config/pipeline_config.json`)

The pipeline uses a comprehensive JSON configuration file:

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
  },
  "cloud_storage": {
    "enabled": false,
    "provider": "aws",
    "bucket_name": "",
    "region": "us-east-1"
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

## 🔄 Updates and Extensions

The pipeline is designed to be extensible:

- **New Collection Sources**: Add to `data_collection.py`
- **New Augmentation Techniques**: Extend `data_preprocessing.py`
- **New Validation Criteria**: Add to `data_validation.py`
- **New Cloud Providers**: Extend `cloud_storage.py`
- **New Export Formats**: Add to `dataset_export_import.py`
- **Configuration Options**: Extend `pipeline_config.json`

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review configuration files
- Run scripts with `--help` for detailed options
- Check logs in the `logs/` directory
- Generate validation reports for analysis

## 🎯 Next Steps

1. **Install Dependencies**: Set up Python environment
2. **Configure Pipeline**: Customize `pipeline_config.json`
3. **Start Collection**: Begin gathering naira note images
4. **Process Dataset**: Run preprocessing pipeline
5. **Validate Quality**: Ensure dataset meets standards
6. **Deploy to Cloud**: Upload to cloud storage
7. **Monitor and Maintain**: Regular cleaning and updates

The complete data pipeline is now ready for production use! 🚀
