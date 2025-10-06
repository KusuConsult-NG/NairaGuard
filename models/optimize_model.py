#!/usr/bin/env python3
"""
Model Optimization Module for Naira Note Detection
Implements quantization, pruning, and model compression for deployment
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import numpy as np
import json
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_model_optimization as tfmot

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.preprocess import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Model optimization for deployment"""
    
    def __init__(self, model_path: str, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize optimizer
        
        Args:
            model_path: Path to trained model
            input_shape: Input image shape
        """
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.original_model = None
        self.optimized_model = None
        self.preprocessor = ImagePreprocessor()
        
        # Load original model
        self._load_model()
    
    def _load_model(self):
        """Load the original model"""
        try:
            self.original_model = load_model(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def quantize_model(self, representative_dataset: Optional[np.ndarray] = None) -> Model:
        """
        Apply post-training quantization
        
        Args:
            representative_dataset: Representative data for calibration
            
        Returns:
            Quantized model
        """
        logger.info("Applying post-training quantization...")
        
        # Convert to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Use representative dataset if provided
        if representative_dataset is not None:
            def representative_data_gen():
                for data in representative_dataset:
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save quantized model
        quantized_path = self.model_path.parent / f"{self.model_path.stem}_quantized.tflite"
        with open(quantized_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Quantized model saved to {quantized_path}")
        
        # Load quantized model for evaluation
        interpreter = tf.lite.Interpreter(model_path=str(quantized_path))
        interpreter.allocate_tensors()
        
        return interpreter
    
    def prune_model(self, pruning_schedule: str = "polynomial", 
                   final_sparsity: float = 0.5) -> Model:
        """
        Apply magnitude-based pruning
        
        Args:
            pruning_schedule: Pruning schedule ("polynomial", "constant")
            final_sparsity: Final sparsity level
            
        Returns:
            Pruned model
        """
        logger.info(f"Applying magnitude-based pruning with {final_sparsity} sparsity...")
        
        # Define pruning schedule
        if pruning_schedule == "polynomial":
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=final_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
        else:
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                    target_sparsity=final_sparsity,
                    begin_step=0
                )
            }
        
        # Apply pruning
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruned_model = prune_low_magnitude(self.original_model, **pruning_params)
        
        # Compile pruned model
        pruned_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune pruned model
        logger.info("Fine-tuning pruned model...")
        
        # Create dummy data for fine-tuning (in practice, use real data)
        dummy_data = np.random.random((100, *self.input_shape))
        dummy_labels = np.random.randint(0, 2, (100, 2))
        
        pruned_model.fit(
            dummy_data, dummy_labels,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Strip pruning wrappers
        stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        # Save pruned model
        pruned_path = self.model_path.parent / f"{self.model_path.stem}_pruned.h5"
        stripped_model.save(pruned_path)
        
        logger.info(f"Pruned model saved to {pruned_path}")
        
        self.optimized_model = stripped_model
        return stripped_model
    
    def cluster_weights(self, num_clusters: int = 16) -> Model:
        """
        Apply weight clustering
        
        Args:
            num_clusters: Number of clusters
            
        Returns:
            Clustered model
        """
        logger.info(f"Applying weight clustering with {num_clusters} clusters...")
        
        # Apply clustering
        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        
        clustering_params = {
            'number_of_clusters': num_clusters,
            'cluster_centroids_init': CentroidInitialization.LINEAR
        }
        
        clustered_model = cluster_weights(self.original_model, **clustering_params)
        
        # Compile clustered model
        clustered_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune clustered model
        logger.info("Fine-tuning clustered model...")
        
        # Create dummy data for fine-tuning
        dummy_data = np.random.random((100, *self.input_shape))
        dummy_labels = np.random.randint(0, 2, (100, 2))
        
        clustered_model.fit(
            dummy_data, dummy_labels,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Strip clustering wrappers
        stripped_model = tfmot.clustering.keras.strip_clustering(clustered_model)
        
        # Save clustered model
        clustered_path = self.model_path.parent / f"{self.model_path.stem}_clustered.h5"
        stripped_model.save(clustered_path)
        
        logger.info(f"Clustered model saved to {clustered_path}")
        
        self.optimized_model = stripped_model
        return stripped_model
    
    def convert_to_onnx(self, output_path: Optional[str] = None) -> str:
        """
        Convert model to ONNX format
        
        Args:
            output_path: Output path for ONNX model
            
        Returns:
            Path to ONNX model
        """
        try:
            import tf2onnx
            logger.info("Converting model to ONNX format...")
            
            if output_path is None:
                output_path = self.model_path.parent / f"{self.model_path.stem}.onnx"
            
            # Convert to ONNX
            spec = (tf.TensorSpec((None, *self.input_shape), tf.float32, name="input"),)
            output, _ = tf2onnx.convert.from_keras(
                self.original_model, 
                input_signature=spec,
                opset=13,
                output_path=str(output_path)
            )
            
            logger.info(f"ONNX model saved to {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            raise
        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
            raise
    
    def optimize_for_mobile(self, target_size_mb: float = 5.0) -> Dict[str, str]:
        """
        Optimize model for mobile deployment
        
        Args:
            target_size_mb: Target model size in MB
            
        Returns:
            Dictionary of optimized model paths
        """
        logger.info(f"Optimizing model for mobile deployment (target: {target_size_mb}MB)...")
        
        optimized_models = {}
        
        # 1. Apply pruning
        pruned_model = self.prune_model(final_sparsity=0.7)
        optimized_models['pruned'] = str(self.model_path.parent / f"{self.model_path.stem}_pruned.h5")
        
        # 2. Apply quantization
        quantized_interpreter = self.quantize_model()
        optimized_models['quantized'] = str(self.model_path.parent / f"{self.model_path.stem}_quantized.tflite")
        
        # 3. Apply clustering
        clustered_model = self.cluster_weights(num_clusters=8)
        optimized_models['clustered'] = str(self.model_path.parent / f"{self.model_path.stem}_clustered.h5")
        
        # 4. Convert to ONNX
        try:
            onnx_path = self.convert_to_onnx()
            optimized_models['onnx'] = onnx_path
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {str(e)}")
        
        # 5. Create ultra-lightweight version
        ultra_light_path = self._create_ultra_lightweight_model()
        optimized_models['ultra_light'] = ultra_light_path
        
        return optimized_models
    
    def _create_ultra_lightweight_model(self) -> str:
        """Create ultra-lightweight model for edge deployment"""
        logger.info("Creating ultra-lightweight model...")
        
        # Create a very small model
        ultra_light_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        ultra_light_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save ultra-lightweight model
        ultra_light_path = self.model_path.parent / f"{self.model_path.stem}_ultra_light.h5"
        ultra_light_model.save(ultra_light_path)
        
        logger.info(f"Ultra-lightweight model saved to {ultra_light_path}")
        return str(ultra_light_path)
    
    def benchmark_models(self, test_data: np.ndarray, 
                        model_paths: Dict[str, str]) -> Dict[str, Dict]:
        """
        Benchmark different optimized models
        
        Args:
            test_data: Test data for benchmarking
            model_paths: Dictionary of model paths
            
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking optimized models...")
        
        results = {}
        
        for model_name, model_path in model_paths.items():
            try:
                if model_path.endswith('.tflite'):
                    # Benchmark TensorFlow Lite model
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    # Warm up
                    for _ in range(10):
                        interpreter.set_tensor(input_details[0]['index'], test_data[:1])
                        interpreter.invoke()
                    
                    # Benchmark
                    import time
                    start_time = time.time()
                    
                    for i in range(len(test_data)):
                        interpreter.set_tensor(input_details[0]['index'], test_data[i:i+1])
                        interpreter.invoke()
                    
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) / len(test_data)
                    
                    # Get model size
                    model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
                    
                    results[model_name] = {
                        'inference_time_ms': inference_time * 1000,
                        'model_size_mb': model_size,
                        'throughput_fps': 1 / inference_time
                    }
                    
                else:
                    # Benchmark Keras model
                    model = load_model(model_path)
                    
                    # Warm up
                    model.predict(test_data[:10])
                    
                    # Benchmark
                    import time
                    start_time = time.time()
                    
                    predictions = model.predict(test_data)
                    
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) / len(test_data)
                    
                    # Get model size
                    model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
                    
                    results[model_name] = {
                        'inference_time_ms': inference_time * 1000,
                        'model_size_mb': model_size,
                        'throughput_fps': 1 / inference_time
                    }
                
                logger.info(f"{model_name}: {results[model_name]['inference_time_ms']:.2f}ms, {results[model_name]['model_size_mb']:.2f}MB")
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {str(e)}")
                continue
        
        return results
    
    def generate_optimization_report(self, benchmark_results: Dict[str, Dict]) -> Dict:
        """
        Generate optimization report
        
        Args:
            benchmark_results: Benchmark results
            
        Returns:
            Optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_model': str(self.model_path),
            'optimization_summary': {
                'total_optimized_models': len(benchmark_results),
                'fastest_model': min(benchmark_results.items(), key=lambda x: x[1]['inference_time_ms'])[0],
                'smallest_model': min(benchmark_results.items(), key=lambda x: x[1]['model_size_mb'])[0],
                'highest_throughput': max(benchmark_results.items(), key=lambda x: x[1]['throughput_fps'])[0]
            },
            'detailed_results': benchmark_results,
            'recommendations': self._generate_recommendations(benchmark_results)
        }
        
        # Save report
        report_path = self.model_path.parent / 'optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_path}")
        
        return report
    
    def _generate_recommendations(self, benchmark_results: Dict[str, Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Find best models for different criteria
        fastest = min(benchmark_results.items(), key=lambda x: x[1]['inference_time_ms'])
        smallest = min(benchmark_results.items(), key=lambda x: x[1]['model_size_mb'])
        highest_throughput = max(benchmark_results.items(), key=lambda x: x[1]['throughput_fps'])
        
        recommendations.append(f"Fastest inference: {fastest[0]} ({fastest[1]['inference_time_ms']:.2f}ms)")
        recommendations.append(f"Smallest model: {smallest[0]} ({smallest[1]['model_size_mb']:.2f}MB)")
        recommendations.append(f"Highest throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput_fps']:.1f} FPS)")
        
        # Mobile deployment recommendation
        mobile_candidates = [name for name, results in benchmark_results.items() 
                           if results['model_size_mb'] < 10 and results['inference_time_ms'] < 100]
        
        if mobile_candidates:
            best_mobile = min(mobile_candidates, key=lambda x: benchmark_results[x]['model_size_mb'])
            recommendations.append(f"Best for mobile deployment: {best_mobile}")
        else:
            recommendations.append("Consider further optimization for mobile deployment")
        
        # Edge deployment recommendation
        edge_candidates = [name for name, results in benchmark_results.items() 
                          if results['model_size_mb'] < 5 and results['inference_time_ms'] < 50]
        
        if edge_candidates:
            best_edge = min(edge_candidates, key=lambda x: benchmark_results[x]['model_size_mb'])
            recommendations.append(f"Best for edge deployment: {best_edge}")
        else:
            recommendations.append("Consider ultra-lightweight model for edge deployment")
        
        return recommendations

def main():
    """Main optimization pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize naira note detection model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test-data', type=str,
                       help='Path to test data for benchmarking')
    parser.add_argument('--target-size-mb', type=float, default=5.0,
                       help='Target model size in MB')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for optimized models')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.model_path)
    
    # Optimize for mobile deployment
    optimized_models = optimizer.optimize_for_mobile(args.target_size_mb)
    
    # Benchmark models if test data provided
    if args.test_data:
        # Load test data
        test_data = np.load(args.test_data)
        
        # Add original model to benchmark
        optimized_models['original'] = args.model_path
        
        # Benchmark all models
        benchmark_results = optimizer.benchmark_models(test_data, optimized_models)
        
        # Generate report
        report = optimizer.generate_optimization_report(benchmark_results)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Fastest Model: {report['optimization_summary']['fastest_model']}")
        print(f"Smallest Model: {report['optimization_summary']['smallest_model']}")
        print(f"Highest Throughput: {report['optimization_summary']['highest_throughput']}")
        print("\nDetailed Results:")
        for model_name, results in benchmark_results.items():
            print(f"{model_name}:")
            print(f"  Inference Time: {results['inference_time_ms']:.2f}ms")
            print(f"  Model Size: {results['model_size_mb']:.2f}MB")
            print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        print("="*60)
    
    else:
        print("Optimization completed. Use --test-data to benchmark models.")

if __name__ == '__main__':
    main()
