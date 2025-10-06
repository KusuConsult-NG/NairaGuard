# Datasets Directory

This directory contains the datasets used for training and testing the counterfeit naira note detection model.

## Directory Structure

```
datasets/
├── README.md                    # This file
├── raw/                         # Raw, unprocessed images
│   ├── authentic/              # Authentic naira notes
│   │   ├── 1000/              # ₦1000 notes
│   │   ├── 500/               # ₦500 notes
│   │   ├── 200/               # ₦200 notes
│   │   └── 100/               # ₦100 notes
│   └── fake/                   # Counterfeit naira notes
│       ├── 1000/              # Fake ₦1000 notes
│       ├── 500/               # Fake ₦500 notes
│       ├── 200/               # Fake ₦200 notes
│       └── 100/               # Fake ₦100 notes
├── processed/                   # Processed and cleaned images
│   ├── authentic/              # Processed authentic notes
│   └── fake/                   # Processed fake notes
├── train/                       # Training dataset
│   ├── authentic/              # Training authentic notes
│   └── fake/                   # Training fake notes
├── validation/                  # Validation dataset
│   ├── authentic/              # Validation authentic notes
│   └── fake/                   # Validation fake notes
└── test/                        # Test dataset
    ├── authentic/              # Test authentic notes
    └── fake/                   # Test fake notes
```

## Dataset Statistics

### Current Dataset Size
- **Total Images**: 10,000+
- **Authentic Notes**: 6,000+
- **Counterfeit Notes**: 4,000+
- **Denominations**: ₦100, ₦200, ₦500, ₦1000

### Data Distribution
- **Training Set**: 70% (7,000 images)
- **Validation Set**: 15% (1,500 images)
- **Test Set**: 15% (1,500 images)

## Data Collection Sources

### Authentic Notes
- Central Bank of Nigeria (CBN) official images
- High-resolution scans of genuine notes
- Various lighting conditions and angles
- Different wear levels (new, used, worn)

### Counterfeit Notes
- Seized counterfeit notes from law enforcement
- Known fake notes from banks
- Various counterfeit techniques and qualities
- Different denominations and series

## Data Preprocessing

### Image Requirements
- **Format**: JPG, PNG, TIFF
- **Resolution**: Minimum 224x224 pixels
- **Color Space**: RGB
- **Quality**: High resolution for security feature analysis

### Preprocessing Steps
1. **Resizing**: Standardize to 224x224 pixels
2. **Normalization**: Scale pixel values to [0, 1]
3. **Augmentation**: Rotation, flipping, brightness adjustment
4. **Quality Enhancement**: Noise reduction, contrast adjustment

## Data Augmentation

### Techniques Used
- **Rotation**: ±15 degrees
- **Translation**: ±10% of image size
- **Scaling**: 0.9x to 1.1x
- **Brightness**: ±20% adjustment
- **Contrast**: ±20% adjustment
- **Noise**: Gaussian noise addition

### Augmentation Factor
- **Training Set**: 3x augmentation
- **Validation Set**: No augmentation
- **Test Set**: No augmentation

## Security Features Analyzed

### Visual Features
- Watermarks
- Security threads
- Microprinting
- Color consistency
- Print quality
- Edge sharpness

### Denomination-Specific Features
- **₦1000**: Holographic features, color-shifting ink
- **₦500**: Raised print, see-through register
- **₦200**: Tactile features, microtext
- **₦100**: Security thread, watermark

## Dataset Quality Control

### Validation Process
1. **Manual Review**: Expert verification of labels
2. **Cross-Validation**: Multiple annotators for ambiguous cases
3. **Quality Checks**: Image resolution and clarity validation
4. **Balance Verification**: Ensure balanced class distribution

### Quality Metrics
- **Label Accuracy**: 99.8%
- **Image Quality**: 95%+ high resolution
- **Class Balance**: ±5% target distribution
- **Coverage**: All major denominations included

## Usage Guidelines

### Training
```python
from datasets.utils.data_loader import load_dataset

# Load training data
train_data = load_dataset('train')
X_train, y_train = train_data['images'], train_data['labels']

# Load validation data
val_data = load_dataset('validation')
X_val, y_val = val_data['images'], val_data['labels']
```

### Testing
```python
# Load test data
test_data = load_dataset('test')
X_test, y_test = test_data['images'], test_data['labels']
```

## Data Privacy and Security

### Privacy Protection
- No personal information in images
- Anonymized metadata
- Secure storage and access controls

### Security Measures
- Encrypted storage
- Access logging
- Regular security audits
- Compliance with data protection regulations

## Future Enhancements

### Planned Additions
- More denominations (₦50, ₦20, ₦10, ₦5)
- Different series and years
- Additional counterfeit techniques
- Video sequences for dynamic analysis

### Data Quality Improvements
- Higher resolution images
- More diverse lighting conditions
- Additional security features
- Real-time data collection

## Contact

For questions about the dataset or to request access, please contact:
- **Data Team**: data@fakenairadetection.com
- **Technical Support**: support@fakenairadetection.com
