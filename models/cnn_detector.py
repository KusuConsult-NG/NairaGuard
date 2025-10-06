#!/usr/bin/env python3
"""
CNN Feature Extraction and Model Training for Naira Note Detection
Implements MobileNetV2, EfficientNet, and baseline models with comprehensive evaluation
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.preprocess import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNAutoDetector:
    """CNN-based naira note detection system"""
    
    def __init__(self, model_type: str = "mobilenet", input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize CNN detector
        
        Args:
            model_type: Type of CNN model ("mobilenet", "efficientnet")
            input_shape: Input image shape
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.preprocessor = ImagePreprocessor(model_type=model_type)
        
        # Model configurations
        self.model_configs = {
            "mobilenet": {
                "base_model": MobileNetV2,
                "preprocess": mobilenet_preprocess,
                "weights": "imagenet"
            },
            "efficientnet": {
                "base_model": EfficientNetB0,
                "preprocess": efficientnet_preprocess,
                "weights": "imagenet"
            }
        }
    
    def build_model(self, num_classes: int = 2, dropout_rate: float = 0.5) -> Model:
        """
        Build CNN model for naira note detection
        
        Args:
            num_classes: Number of classes (2 for genuine/fake)
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        config = self.model_configs[self.model_type]
        
        # Load base model
        base_model = config["base_model"](
            input_shape=self.input_shape,
            include_top=False,
            weights=config["weights"]
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"Built {self.model_type} model with {model.count_params()} parameters")
        
        return model
    
    def train_model(self, train_dir: str, val_dir: str, 
                   epochs: int = 50, batch_size: int = 32,
                   fine_tune_epochs: int = 10) -> Dict:
        """
        Train the CNN model
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            epochs: Number of training epochs
            batch_size: Batch size
            fine_tune_epochs: Epochs for fine-tuning
            
        Returns:
            Training history and metrics
        """
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_generator = self.preprocessor.create_data_generator(
            train_dir, batch_size=batch_size, augment=True, shuffle=True
        )
        
        val_generator = self.preprocessor.create_data_generator(
            val_dir, batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(
                f'models/{self.model_type}_best.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Phase 1: Train only the classification head
        logger.info("Phase 1: Training classification head...")
        history1 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune the entire model
        logger.info("Phase 2: Fine-tuning entire model...")
        self.model.layers[0].trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        history2 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        combined_history = self._combine_histories(history1, history2)
        self.history = combined_history
        
        return combined_history
    
    def evaluate_model(self, test_dir: str, batch_size: int = 32) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            test_dir: Test data directory
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create test generator
        test_generator = self.preprocessor.create_data_generator(
            test_dir, batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Evaluate model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_generator, verbose=1
        )
        
        # Get predictions for detailed metrics
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate additional metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        return metrics
    
    def extract_features(self, data_dir: str, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using the trained model
        
        Args:
            data_dir: Data directory
            batch_size: Batch size
            
        Returns:
            Features and labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create feature extractor (remove classification head)
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.layers[-4].output  # Before final dense layer
        )
        
        # Create data generator
        generator = self.preprocessor.create_data_generator(
            data_dir, batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Extract features
        features = feature_extractor.predict(generator, verbose=1)
        labels = generator.classes
        
        return features, labels
    
    def _combine_histories(self, history1, history2) -> Dict:
        """Combine two training histories"""
        combined = {}
        
        for key in history1.history.keys():
            combined[key] = history1.history[key] + history2.history[key]
        
        return combined

class BaselineDetector:
    """Baseline models for comparison (SVM, Logistic Regression, Random Forest)"""
    
    def __init__(self, model_type: str = "svm"):
        """
        Initialize baseline detector
        
        Args:
            model_type: Type of baseline model ("svm", "logistic", "random_forest")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.preprocessor = ImagePreprocessor(model_type="custom")
        
        # Model configurations
        self.model_configs = {
            "svm": SVC(kernel='rbf', probability=True, random_state=42),
            "logistic": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
    
    def extract_features(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using traditional computer vision methods
        
        Args:
            data_dir: Data directory
            
        Returns:
            Features and labels
        """
        features = []
        labels = []
        
        data_path = Path(data_dir)
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_label = 0 if class_dir.name == "genuine" else 1
            
            for image_path in class_dir.rglob("*.jpg"):
                try:
                    # Load and preprocess image
                    image = self.preprocessor._load_image(image_path)
                    
                    # Extract traditional features
                    feature_vector = self._extract_traditional_features(image)
                    features.append(feature_vector)
                    labels.append(class_label)
                    
                except Exception as e:
                    logger.warning(f"Error processing {image_path}: {str(e)}")
        
        return np.array(features), np.array(labels)
    
    def train_model(self, train_dir: str, val_dir: str) -> Dict:
        """
        Train baseline model
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            
        Returns:
            Training metrics
        """
        # Extract features
        logger.info("Extracting features from training data...")
        X_train, y_train = self.extract_features(train_dir)
        
        logger.info("Extracting features from validation data...")
        X_val, y_val = self.extract_features(val_dir)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model = self.model_configs[self.model_type]
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_scaled)),
            'val_accuracy': accuracy_score(y_val, y_pred),
            'val_precision': precision_score(y_val, y_pred),
            'val_recall': recall_score(y_val, y_pred),
            'val_f1': f1_score(y_val, y_pred)
        }
        
        return metrics
    
    def evaluate_model(self, test_dir: str) -> Dict:
        """
        Evaluate baseline model
        
        Args:
            test_dir: Test data directory
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract features
        X_test, y_test = self.extract_features(test_dir)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def _extract_traditional_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract traditional computer vision features
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten())
        
        # Texture features (LBP-like)
        # Calculate local binary pattern
        lbp = self._calculate_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Color features (if RGB)
        if len(image.shape) == 3:
            # Mean and std for each channel
            for i in range(3):
                features.extend([np.mean(image[:, :, i]), np.std(image[:, :, i])])
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            features.extend([area, perimeter])
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize evaluator
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
    
    def compare_models(self, results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple models
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Comparison summary
        """
        comparison = {
            'models': list(results.keys()),
            'metrics': {},
            'best_model': None,
            'best_accuracy': 0
        }
        
        # Extract metrics for comparison
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        for metric in metrics:
            comparison['metrics'][metric] = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    comparison['metrics'][metric][model_name] = model_results[metric]
        
        # Find best model
        if 'test_accuracy' in comparison['metrics']:
            best_model = max(
                comparison['metrics']['test_accuracy'].items(),
                key=lambda x: x[1]
            )
            comparison['best_model'] = best_model[0]
            comparison['best_accuracy'] = best_model[1]
        
        return comparison
    
    def plot_training_history(self, history: Dict, model_name: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training History', fontsize=16)
        
        # Plot accuracy
        axes[0, 0].plot(history['accuracy'], label='Training')
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history['loss'], label='Training')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        axes[1, 0].plot(history['precision'], label='Training')
        axes[1, 0].plot(history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        axes[1, 1].plot(history['recall'], label='Training')
        axes[1, 1].plot(history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, class_names: List[str] = None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Genuine', 'Fake']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, comparison: Dict):
        """Plot model comparison"""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            if metric in comparison['metrics']:
                model_names = list(comparison['metrics'][metric].keys())
                values = list(comparison['metrics'][metric].values())
                
                bars = ax.bar(model_names, values, alpha=0.7)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, Dict], comparison: Dict):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(results),
                'best_model': comparison['best_model'],
                'best_accuracy': comparison['best_accuracy']
            },
            'detailed_results': results,
            'comparison': comparison
        }
        
        # Save report
        report_path = self.output_dir / 'model_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report

def main():
    """Main training and evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate naira note detection models')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='Validation data directory')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Test data directory')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['mobilenet', 'efficientnet', 'svm', 'logistic', 'random_forest'],
                       default=['mobilenet', 'efficientnet', 'svm'],
                       help='Models to train and evaluate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Results storage
    results = {}
    
    # Train and evaluate models
    for model_name in args.models:
        logger.info(f"Training and evaluating {model_name}...")
        
        try:
            if model_name in ['mobilenet', 'efficientnet']:
                # CNN models
                detector = CNNAutoDetector(model_type=model_name)
                
                # Train model
                history = detector.train_model(
                    args.train_dir, args.val_dir,
                    epochs=args.epochs, batch_size=args.batch_size
                )
                
                # Evaluate model
                metrics = detector.evaluate_model(args.test_dir, args.batch_size)
                
                # Plot training history
                evaluator.plot_training_history(history, model_name)
                
                # Plot confusion matrix
                cm = np.array(metrics['confusion_matrix'])
                evaluator.plot_confusion_matrix(cm, model_name)
                
                results[model_name] = metrics
                
            else:
                # Baseline models
                detector = BaselineDetector(model_type=model_name)
                
                # Train model
                train_metrics = detector.train_model(args.train_dir, args.val_dir)
                
                # Evaluate model
                test_metrics = detector.evaluate_model(args.test_dir)
                
                # Plot confusion matrix
                cm = np.array(test_metrics['confusion_matrix'])
                evaluator.plot_confusion_matrix(cm, model_name)
                
                results[model_name] = test_metrics
                
            logger.info(f"{model_name} completed successfully")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Compare models
    comparison = evaluator.compare_models(results)
    
    # Plot comparison
    evaluator.plot_model_comparison(comparison)
    
    # Generate report
    report = evaluator.generate_report(results, comparison)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Best Model: {comparison['best_model']}")
    print(f"Best Accuracy: {comparison['best_accuracy']:.3f}")
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"  Precision: {metrics['test_precision']:.3f}")
        print(f"  Recall: {metrics['test_recall']:.3f}")
        print(f"  F1-Score: {metrics['test_f1']:.3f}")
    print("="*60)

if __name__ == '__main__':
    main()
