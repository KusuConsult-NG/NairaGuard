import React, { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { 
  Upload, 
  Camera, 
  FileImage, 
  AlertCircle, 
  CheckCircle,
  Shield,
  Zap,
  Users,
  BarChart3,
  ArrowRight,
  X,
  RotateCcw
} from 'lucide-react';
import { useApp, appActions } from '../context/AppContext';
import { ApiService, validateImageFile, createImagePreview, handleApiError } from '../services/api';
import toast from 'react-hot-toast';
import LoadingSpinner from '../components/LoadingSpinner';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();
  const [showCamera, setShowCamera] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const webcamRef = useRef<Webcam>(null);

  // Handle file drop
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file
    const validation = validateImageFile(file);
    if (!validation.valid) {
      toast.error(validation.error!);
      return;
    }

    try {
      // Create preview
      const preview = await createImagePreview(file);
      dispatch(appActions.setUploadedImage(file, preview));
      
      // Start detection
      dispatch(appActions.setLoading(true));
      const result = await ApiService.predictImage(file);
      
      dispatch(appActions.setDetectionResult(result));
      dispatch(appActions.addToHistory(result));
      
      toast.success('Detection completed successfully!');
      navigate('/result');
    } catch (error) {
      const errorMessage = handleApiError(error);
      dispatch(appActions.setError(errorMessage));
      toast.error(errorMessage);
    } finally {
      dispatch(appActions.setLoading(false));
    }
  }, [dispatch, navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  // Handle camera capture
  const capturePhoto = useCallback(async () => {
    if (!webcamRef.current) return;

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        throw new Error('Failed to capture photo');
      }

      // Convert data URL to File
      const response = await fetch(imageSrc);
      const blob = await response.blob();
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });

      // Validate and process
      const validation = validateImageFile(file);
      if (!validation.valid) {
        toast.error(validation.error!);
        return;
      }

      dispatch(appActions.setUploadedImage(file, imageSrc));
      
      // Start detection
      dispatch(appActions.setLoading(true));
      const result = await ApiService.predictImage(file);
      
      dispatch(appActions.setDetectionResult(result));
      dispatch(appActions.addToHistory(result));
      
      toast.success('Detection completed successfully!');
      navigate('/result');
    } catch (error) {
      const errorMessage = handleApiError(error);
      dispatch(appActions.setError(errorMessage));
      toast.error(errorMessage);
    } finally {
      dispatch(appActions.setLoading(false));
      setShowCamera(false);
    }
  }, [dispatch, navigate]);

  // Handle camera error
  const handleCameraError = (error: string | DOMException) => {
    console.error('Camera error:', error);
    setCameraError('Unable to access camera. Please check permissions.');
  };

  const features = [
    {
      icon: Shield,
      title: 'AI-Powered Detection',
      description: 'Advanced machine learning algorithms detect counterfeit notes with 99%+ accuracy'
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get detection results in seconds with detailed confidence scores'
    },
    {
      icon: Users,
      title: 'Trusted by Many',
      description: 'Used by banks, businesses, and individuals across Nigeria'
    },
    {
      icon: BarChart3,
      title: 'Real-time Analytics',
      description: 'Track detection patterns and get insights into your verification process'
    }
  ];

  const securityFeatures = [
    'Watermark verification',
    'Security thread analysis',
    'Color-shifting ink detection',
    'Microprinting validation',
    'Holographic element check',
    'Paper quality assessment'
  ];

  return (
    <div className="min-h-screen pt-16">
      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Detect Fake Naira Notes
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                with AI Precision
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Upload an image or take a photo to instantly verify the authenticity of Nigerian currency. 
              Our advanced AI technology provides accurate detection in seconds.
            </p>
          </motion.div>

          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="max-w-2xl mx-auto mb-16"
          >
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Upload Naira Note Image
                </h2>
                <p className="text-gray-600">
                  Drag & drop an image or click to browse. You can also use your camera.
                </p>
              </div>

              {/* Upload Area */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="space-y-4"
                >
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                    <Upload className="w-8 h-8 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      {isDragActive ? 'Drop the image here' : 'Drag & drop your image here'}
                    </p>
                    <p className="text-gray-500 mt-2">
                      or click to browse files
                    </p>
                  </div>
                  <p className="text-sm text-gray-400">
                    Supports JPEG, PNG, WebP (max 10MB)
                  </p>
                </motion.div>
              </div>

              {/* Camera Button */}
              <div className="mt-6">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setShowCamera(true)}
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <Camera className="w-5 h-5" />
                  <span>Take Photo with Camera</span>
                </motion.button>
              </div>

              {/* Loading State */}
              {state.isLoading && (
                <div className="mt-6">
                  <LoadingSpinner text="Analyzing image..." />
                </div>
              )}

              {/* Error State */}
              {state.error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-3"
                >
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <p className="text-red-700">{state.error}</p>
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Features Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16"
          >
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                  className="text-center p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-200"
                >
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 text-sm">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </motion.div>

          {/* Security Features Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white"
          >
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold text-center mb-8">
                Advanced Security Features Detected
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {securityFeatures.map((feature, index) => (
                  <motion.div
                    key={feature}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 * index }}
                    className="flex items-center space-x-3"
                  >
                    <CheckCircle className="w-5 h-5 text-green-300" />
                    <span className="text-sm">{feature}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Camera Modal */}
      <AnimatePresence>
        {showCamera && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl p-6 max-w-md w-full"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Take Photo</h3>
                <button
                  onClick={() => setShowCamera(false)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                {cameraError ? (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-700 text-sm">{cameraError}</p>
                  </div>
                ) : (
                  <div className="relative">
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      onUserMediaError={handleCameraError}
                      className="w-full h-64 object-cover rounded-lg"
                    />
                  </div>
                )}

                <div className="flex space-x-3">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={capturePhoto}
                    disabled={!!cameraError}
                    className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    <Camera className="w-4 h-4" />
                    <span>Capture Photo</span>
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setShowCamera(false)}
                    className="px-4 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50"
                  >
                    Cancel
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default HomePage;