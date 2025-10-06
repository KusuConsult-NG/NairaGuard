#!/usr/bin/env python3
"""
Model Saving and Inference Module for Naira Note Detection
Handles model serialization, loading, and real-time prediction
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import tensorflow_model_optimization as tfmot

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.preprocess import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSaver:
    """Handles model saving in multiple formats"""
    
    def __init__(self, model: Model, model_name: str = "naira_detector"):
        """
        Initialize model saver
        
        Args:
            model: Trained Keras model
            model_name: Name for saved models
        """
        self.model = model
        self.model_name = model_name
        self.output_dir = Path("models/saved")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_keras_model(self, format: str = "h5") -> str:
        """
        Save model in Keras format
        
        Args:
            format: Format ("h5", "tf")
            
        Returns:
            Path to saved model
        """
        if format == "h5":
            model_path = self.output_dir / f"{self.model_name}.h5"
            self.model.save(model_path)
        elif format == "tf":
            model_path = self.output_dir / f"{self.model_name}_tf"
            self.model.save(model_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Keras model saved to {model_path}")
        return str(model_path)
    
    def save_tflite_model(self, quantize: bool = True) -> str:
        """
        Save model in TensorFlow Lite format
        
        Args:
            quantize: Whether to apply quantization
            
        Returns:
            Path to saved TFLite model
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        model_path = self.output_dir / f"{self.model_name}.tflite"
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TensorFlow Lite model saved to {model_path}")
        return str(model_path)
    
    def save_onnx_model(self) -> str:
        """
        Save model in ONNX format
        
        Returns:
            Path to saved ONNX model
        """
        try:
            import tf2onnx
            
            model_path = self.output_dir / f"{self.model_name}.onnx"
            
            spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            output, _ = tf2onnx.convert.from_keras(
                self.model, 
                input_signature=spec,
                opset=13,
                output_path=str(model_path)
            )
            
            logger.info(f"ONNX model saved to {model_path}")
            return str(model_path)
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            raise
    
    def save_pickle_model(self, scaler: Optional[object] = None, 
                         feature_extractor: Optional[object] = None) -> str:
        """
        Save model components in pickle format
        
        Args:
            scaler: Fitted scaler (for baseline models)
            feature_extractor: Feature extractor (for baseline models)
            
        Returns:
            Path to saved pickle file
        """
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'scaler': scaler,
            'feature_extractor': feature_extractor
        }
        
        model_path = self.output_dir / f"{self.model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Pickle model saved to {model_path}")
        return str(model_path)
    
    def save_model_metadata(self, metrics: Dict, training_history: Optional[Dict] = None) -> str:
        """
        Save model metadata
        
        Args:
            metrics: Model evaluation metrics
            training_history: Training history
            
        Returns:
            Path to saved metadata
        """
        metadata = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': self.model.to_json(),
            'metrics': metrics,
            'training_history': training_history,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }
        
        metadata_path = self.output_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        return str(metadata_path)
    
    def save_all_formats(self, metrics: Dict, training_history: Optional[Dict] = None) -> Dict[str, str]:
        """
        Save model in all supported formats
        
        Args:
            metrics: Model evaluation metrics
            training_history: Training history
            
        Returns:
            Dictionary of saved model paths
        """
        saved_models = {}
        
        # Save Keras formats
        saved_models['keras_h5'] = self.save_keras_model("h5")
        saved_models['keras_tf'] = self.save_keras_model("tf")
        
        # Save TensorFlow Lite
        saved_models['tflite'] = self.save_tflite_model(quantize=True)
        saved_models['tflite_unquantized'] = self.save_tflite_model(quantize=False)
        
        # Save ONNX
        try:
            saved_models['onnx'] = self.save_onnx_model()
        except Exception as e:
            logger.warning(f"ONNX save failed: {str(e)}")
        
        # Save pickle
        saved_models['pickle'] = self.save_pickle_model()
        
        # Save metadata
        saved_models['metadata'] = self.save_model_metadata(metrics, training_history)
        
        return saved_models

class ModelInference:
    """Handles model loading and inference"""
    
    def __init__(self, model_path: str, model_type: str = "keras"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model
            model_type: Type of model ("keras", "tflite", "onnx", "pickle")
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.preprocessor = ImagePreprocessor()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model based on type"""
        try:
            if self.model_type == "keras":
                if self.model_path.suffix == ".h5":
                    self.model = load_model(self.model_path)
                else:
                    self.model = tf.saved_model.load(str(self.model_path))
                    
            elif self.model_type == "tflite":
                self.model = tf.lite.Interpreter(model_path=str(self.model_path))
                self.model.allocate_tensors()
                
            elif self.model_type == "onnx":
                try:
                    import onnxruntime as ort
                    self.model = ort.InferenceSession(str(self.model_path))
                except ImportError:
                    logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
                    raise
                    
            elif self.model_type == "pickle":
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data.get('scaler')
                    self.feature_extractor = model_data.get('feature_extractor')
            
            logger.info(f"Loaded {self.model_type} model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image: Union[np.ndarray, str, Path], 
                return_probabilities: bool = True) -> Dict:
        """
        Predict authenticity of naira note
        
        Args:
            image: Input image (numpy array, file path, or Path)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image, augment=False)
        
        # Make prediction based on model type
        if self.model_type == "keras":
            prediction = self._predict_keras(processed_image)
            
        elif self.model_type == "tflite":
            prediction = self._predict_tflite(processed_image)
            
        elif self.model_type == "onnx":
            prediction = self._predict_onnx(processed_image)
            
        elif self.model_type == "pickle":
            prediction = self._predict_pickle(processed_image)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Format results
        results = {
            'predicted_class': 'genuine' if prediction['class'] == 0 else 'fake',
            'confidence': prediction['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        
        if return_probabilities:
            results['probabilities'] = prediction['probabilities']
        
        return results
    
    def _predict_keras(self, image: np.ndarray) -> Dict:
        """Predict using Keras model"""
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        probabilities = self.model.predict(image_batch, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def _predict_tflite(self, image: np.ndarray) -> Dict:
        """Predict using TensorFlow Lite model"""
        # Get input and output details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Set input
        self.model.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))
        
        # Run inference
        self.model.invoke()
        
        # Get output
        probabilities = self.model.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def _predict_onnx(self, image: np.ndarray) -> Dict:
        """Predict using ONNX model"""
        # Prepare input
        input_name = self.model.get_inputs()[0].name
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Make prediction
        probabilities = self.model.run(None, {input_name: input_data})[0][0]
        predicted_class = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def _predict_pickle(self, image: np.ndarray) -> Dict:
        """Predict using pickle model (baseline models)"""
        # Extract features if feature extractor available
        if hasattr(self, 'feature_extractor') and self.feature_extractor:
            features = self.feature_extractor.extract_features([image])
        else:
            # Use traditional feature extraction
            features = self._extract_traditional_features(image)
        
        # Scale features if scaler available
        if hasattr(self, 'scaler') and self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
        else:
            prediction = self.model.predict(features)[0]
            probabilities = [1 - prediction, prediction]
        
        predicted_class = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def _extract_traditional_features(self, image: np.ndarray) -> np.ndarray:
        """Extract traditional computer vision features"""
        import cv2
        
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten())
        
        # Texture features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Color features
        for i in range(3):
            features.extend([np.mean(image[:, :, i]), np.std(image[:, :, i])])
        
        return np.array(features)
    
    def batch_predict(self, images: List[Union[np.ndarray, str, Path]]) -> List[Dict]:
        """
        Predict authenticity for multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error predicting image: {str(e)}")
                results.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def benchmark_inference(self, test_images: List[np.ndarray], 
                           num_runs: int = 100) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            test_images: List of test images
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        logger.info(f"Benchmarking inference speed ({num_runs} runs)...")
        
        # Warm up
        for _ in range(10):
            self.predict(test_images[0])
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for image in test_images:
                self.predict(image)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        results = {
            'average_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'throughput_fps': len(test_images) / avg_time,
            'num_runs': num_runs,
            'num_images': len(test_images)
        }
        
        logger.info(f"Benchmark completed: {results['average_time_ms']:.2f}ms average")
        
        return results

def main():
    """Test model inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--model-type', type=str, default='keras',
                       choices=['keras', 'tflite', 'onnx', 'pickle'],
                       help='Type of model')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark test')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ModelInference(args.model_path, args.model_type)
    
    # Test single prediction
    try:
        result = inference.predict(args.image_path)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)
    
    # Run benchmark if requested
    if args.benchmark:
        # Create dummy test images
        test_images = [np.random.random((224, 224, 3)) for _ in range(10)]
        
        benchmark_results = inference.benchmark_inference(test_images)
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Average Time: {benchmark_results['average_time_ms']:.2f}ms")
        print(f"Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
        print(f"Min Time: {benchmark_results['min_time_ms']:.2f}ms")
        print(f"Max Time: {benchmark_results['max_time_ms']:.2f}ms")
        print("="*50)

if __name__ == '__main__':
    main()
