"""
Model utility functions for naira note detection
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def save_model(model, filepath: str, metadata: Optional[Dict[str, Any]] = None):
    """Save model and metadata"""
    try:
        # Save model
        model.save(filepath)
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str):
    """Load model from file"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def save_scaler(scaler, filepath: str):
    """Save data scaler"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving scaler: {str(e)}")
        raise

def load_scaler(filepath: str):
    """Load data scaler"""
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {filepath}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise

def plot_training_history(history, save_path: Optional[str] = None):
    """Plot training history"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_confusion_matrix(cm, class_names, save_path: Optional[str] = None):
    """Plot confusion matrix"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_class_distribution(labels, class_names, save_path: Optional[str] = None):
    """Plot class distribution"""
    try:
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar([class_names[i] for i in unique], counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting class distribution: {str(e)}")
        raise

def calculate_model_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive model metrics"""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert to binary if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_binary = np.argmax(y_pred, axis=1)
        else:
            y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, average='weighted')
        recall = recall_score(y_true, y_pred_binary, average='weighted')
        f1 = f1_score(y_true, y_pred_binary, average='weighted')
        
        # Calculate ROC AUC if binary classification
        if len(np.unique(y_true)) == 2:
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                roc_auc = roc_auc_score(y_true, y_pred[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_pred)
        else:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Model metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def create_model_summary(model, save_path: Optional[str] = None):
    """Create and save model summary"""
    try:
        # Get model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = '\n'.join(summary_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary_text)
            logger.info(f"Model summary saved to {save_path}")
        
        return summary_text
    except Exception as e:
        logger.error(f"Error creating model summary: {str(e)}")
        raise

def export_model_for_production(model, output_dir: str, model_name: str = "naira_detection"):
    """Export model for production deployment"""
    try:
        import tensorflow as tf
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model in SavedModel format
        saved_model_path = os.path.join(output_dir, f"{model_name}_savedmodel")
        model.save(saved_model_path, save_format='tf')
        
        # Save model in H5 format
        h5_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(h5_path)
        
        # Create metadata file
        metadata = {
            "model_name": model_name,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_parameters": model.count_params(),
            "architecture": "CNN with ResNet50 backbone"
        }
        
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model exported for production to {output_dir}")
        return {
            "savedmodel_path": saved_model_path,
            "h5_path": h5_path,
            "metadata_path": metadata_path
        }
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise

def benchmark_model(model, test_data, batch_size: int = 32, num_runs: int = 10):
    """Benchmark model performance"""
    try:
        import time
        
        # Warm up
        _ = model.predict(test_data[:batch_size], verbose=0)
        
        # Benchmark inference time
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(test_data[:batch_size], verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput
        throughput = batch_size / avg_time
        
        benchmark_results = {
            "average_inference_time": avg_time,
            "std_inference_time": std_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "throughput_images_per_second": throughput
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results
    except Exception as e:
        logger.error(f"Error benchmarking model: {str(e)}")
        raise
