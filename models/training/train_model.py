#!/usr/bin/env python3
"""
Training script for counterfeit naira note detection model
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_utils import load_and_preprocess_images
from utils.model_utils import save_model, plot_training_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NairaDetectionModel:
    """Counterfeit naira note detection model"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, backbone='resnet50'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the model architecture"""
        logger.info(f"Building model with {self.backbone} backbone")
        
        # Load pre-trained backbone
        if self.backbone == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.backbone == 'efficientnet':
            base_model = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
        return self.model
    
    def prepare_data(self, data_path, batch_size=32, validation_split=0.2):
        """Prepare training and validation data"""
        logger.info("Preparing data generators")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50, batch_size=32):
        """Train the model"""
        logger.info("Starting model training")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def fine_tune(self, train_generator, val_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        logger.info("Starting fine-tuning")
        
        # Unfreeze some layers
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=5),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ],
            verbose=1
        )
        
        logger.info("Fine-tuning completed")
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate the model"""
        logger.info("Evaluating model")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_names
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }

def main():
    parser = argparse.ArgumentParser(description='Train counterfeit naira detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet'], help='Backbone architecture')
    parser.add_argument('--output_dir', type=str, default='../pretrained', 
                       help='Output directory for trained model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model_wrapper = NairaDetectionModel(backbone=args.backbone)
    
    # Build model
    model_wrapper.build_model()
    
    # Prepare data
    train_gen, val_gen = model_wrapper.prepare_data(
        args.data_path, 
        batch_size=args.batch_size
    )
    
    # Train model
    model_wrapper.train(train_gen, val_gen, epochs=args.epochs)
    
    # Fine-tune
    model_wrapper.fine_tune(train_gen, val_gen, epochs=20)
    
    # Evaluate
    results = model_wrapper.evaluate(val_gen)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'naira_detection_{args.backbone}.h5')
    save_model(model_wrapper.model, model_path)
    
    # Plot training history
    plot_training_history(model_wrapper.history, save_path='training_history.png')
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Final accuracy: {results['accuracy']:.4f}")

if __name__ == '__main__':
    main()
