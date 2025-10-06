#!/usr/bin/env python3
"""
Model Setup Script for NairaGuard
This script creates the required ML model files for the application.
Run this script after cloning the repository to set up the AI model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model():
    """Create a simple CNN model for naira note detection"""
    logger.info("Creating simple CNN model...")
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: genuine, fake
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model created successfully")
    return model

def create_mock_data():
    """Create mock training data"""
    logger.info("Creating mock training data...")
    
    # Generate random images (224x224x3)
    X_train = np.random.random((100, 224, 224, 3)).astype(np.float32)
    X_val = np.random.random((20, 224, 224, 3)).astype(np.float32)
    
    # Generate random labels (one-hot encoded)
    y_train = np.random.randint(0, 2, (100, 2)).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 2)).astype(np.float32)
    
    logger.info(f"Created mock data: {X_train.shape[0]} train, {X_val.shape[0]} val")
    return X_train, X_val, y_train, y_val

def train_model():
    """Train the simple model"""
    logger.info("Starting model training...")
    
    # Create model
    model = create_simple_model()
    
    # Create mock data
    X_train, X_val, y_train, y_val = create_mock_data()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # Quick training for demo
        batch_size=16,
        verbose=1
    )
    
    logger.info("Model training completed")
    return model, history

def save_model():
    """Save the trained model"""
    logger.info("Training and saving model...")
    
    # Create saved directory
    saved_dir = Path("models/saved")
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and train model
    model, history = train_model()
    
    # Save model in the format expected by the backend
    model_path = saved_dir / "mobilenet_best_fixed.h5"
    model.save(str(model_path))
    
    logger.info(f"Model saved to {model_path}")
    
    # Print model summary
    model.summary()
    
    return str(model_path)

if __name__ == "__main__":
    try:
        model_path = save_model()
        print(f"✅ Model created and saved successfully: {model_path}")
        print("🚀 You can now start the backend server!")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
