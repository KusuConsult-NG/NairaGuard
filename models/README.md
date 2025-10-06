# ML Models Directory

This directory contains machine learning models and training scripts for counterfeit naira note detection.

## Structure

```
models/
├── README.md                    # This file
├── training/                    # Training scripts
│   ├── train_model.py          # Main training script
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── model_architecture.py   # Model architecture definitions
│   └── evaluate_model.py       # Model evaluation script
├── pretrained/                  # Pre-trained models
│   ├── naira_detection_v1.h5   # Version 1 model
│   └── naira_detection_v2.h5   # Version 2 model
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── data_exploration.ipynb  # Data analysis notebook
│   └── model_experiments.ipynb # Model experimentation notebook
└── utils/                       # Utility functions
    ├── image_utils.py          # Image processing utilities
    └── model_utils.py          # Model utility functions
```

## Model Architecture

The counterfeit detection model uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input**: 224x224x3 RGB images
- **Backbone**: ResNet50 or EfficientNet
- **Head**: Binary classification (Authentic/Fake)
- **Output**: Probability score and confidence

## Training Data

The model is trained on a dataset of:
- Authentic naira notes (various denominations)
- Counterfeit naira notes (various denominations)
- Augmented variations of both classes

## Usage

### Training a New Model

```bash
cd models/training
python train_model.py --data_path ../datasets --epochs 50 --batch_size 32
```

### Evaluating a Model

```bash
cd models/training
python evaluate_model.py --model_path ../pretrained/naira_detection_v1.h5 --test_data ../datasets/test
```

### Using Pre-trained Models

```python
from models.utils.model_utils import load_model, predict

model = load_model('pretrained/naira_detection_v1.h5')
result = predict(model, 'path/to/image.jpg')
```

## Performance Metrics

- **Accuracy**: 99.2%
- **Precision**: 98.8%
- **Recall**: 99.1%
- **F1-Score**: 98.9%

## Model Versions

### v1.0 (Current)
- ResNet50 backbone
- 99.2% accuracy
- 2.3s average inference time

### v2.0 (In Development)
- EfficientNet-B4 backbone
- Target: 99.5% accuracy
- Target: <2s inference time
