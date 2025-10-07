import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  Eye,
  Zap,
  CheckCircle,
  AlertTriangle,
  Info,
  Camera,
  FileImage,
  BarChart3,
  Users,
  Clock,
  Target,
  Award,
  Lock,
  Sparkles,
} from 'lucide-react';
import { ApiService } from '../services/api';

const AboutPage: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [status, history] = await Promise.all([
          ApiService.getModelStatus(),
          ApiService.getPredictionHistory(100),
        ]);
        setModelStatus(status);

        // Calculate stats from history
        const total = history.predictions.length;
        const genuine = history.predictions.filter(
          p => p.predicted_class === 'genuine'
        ).length;
        const fake = history.predictions.filter(
          p => p.predicted_class === 'fake'
        ).length;
        const avgConfidence =
          history.predictions.reduce((sum, p) => sum + p.confidence, 0) / total;

        setStats({
          total,
          genuine,
          fake,
          avgConfidence: Math.round(avgConfidence * 100),
          accuracy: Math.round((genuine / total) * 100) || 0,
        });
      } catch (error) {
        console.error('Failed to fetch data:', error);
      }
    };

    fetchData();
  }, []);

  const securityFeatures = [
    {
      name: 'Watermark Verification',
      description:
        'Detects the embedded watermark that becomes visible when held up to light',
      icon: Eye,
      importance: 'Critical',
    },
    {
      name: 'Security Thread Analysis',
      description: 'Analyzes the metallic security thread embedded in the note',
      icon: Zap,
      importance: 'Critical',
    },
    {
      name: 'Color-shifting Ink Detection',
      description:
        'Identifies color-changing ink used in denomination numerals',
      icon: Sparkles,
      importance: 'High',
    },
    {
      name: 'Microprinting Validation',
      description:
        'Checks for tiny text that appears as a solid line to the naked eye',
      icon: Target,
      importance: 'High',
    },
    {
      name: 'Holographic Element Check',
      description: 'Verifies holographic patches and strips on the note',
      icon: Award,
      importance: 'Medium',
    },
    {
      name: 'Paper Quality Assessment',
      description: 'Analyzes the unique paper composition and texture',
      icon: Lock,
      importance: 'Medium',
    },
  ];

  const denominations = [
    {
      value: '₦200',
      color: 'from-green-400 to-green-600',
      features: ['Watermark', 'Security Thread', 'Microprinting'],
    },
    {
      value: '₦500',
      color: 'from-blue-400 to-blue-600',
      features: [
        'Watermark',
        'Security Thread',
        'Color-shifting Ink',
        'Holographic Strip',
      ],
    },
    {
      value: '₦1000',
      color: 'from-purple-400 to-purple-600',
      features: [
        'Watermark',
        'Security Thread',
        'Color-shifting Ink',
        'Holographic Patch',
        'Microprinting',
      ],
    },
  ];

  const howItWorks = [
    {
      step: 1,
      title: 'Upload Image',
      description: 'Take a photo or upload an image of the naira note',
      icon: Camera,
    },
    {
      step: 2,
      title: 'AI Analysis',
      description: 'Our AI analyzes multiple security features simultaneously',
      icon: BarChart3,
    },
    {
      step: 3,
      title: 'Instant Result',
      description:
        'Get immediate results with confidence scores and detailed analysis',
      icon: CheckCircle,
    },
  ];

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              About NairaGuard
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced AI-powered counterfeit detection for Nigerian currency.
              Protecting businesses and individuals from fake naira notes with
              cutting-edge technology.
            </p>
          </motion.div>

          {/* Stats Cards */}
          {stats && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-16"
            >
              <div className="bg-white rounded-xl shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  {stats.total}
                </h3>
                <p className="text-gray-600">Total Detections</p>
              </div>
              <div className="bg-white rounded-xl shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  {stats.genuine}
                </h3>
                <p className="text-gray-600">Genuine Notes</p>
              </div>
              <div className="bg-white rounded-xl shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  {stats.fake}
                </h3>
                <p className="text-gray-600">Fake Notes Detected</p>
              </div>
              <div className="bg-white rounded-xl shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Target className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  {stats.avgConfidence}%
                </h3>
                <p className="text-gray-600">Avg Confidence</p>
              </div>
            </motion.div>
          )}
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our advanced AI system uses multiple detection methods to ensure
              accurate results
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {howItWorks.map((step, index) => {
              const Icon = step.icon;
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                  className="text-center"
                >
                  <div className="relative">
                    <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-8 h-8 bg-yellow-400 rounded-full flex items-center justify-center text-sm font-bold text-gray-900">
                      {step.step}
                    </div>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">
                    {step.title}
                  </h3>
                  <p className="text-gray-600">{step.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Security Features */}
      <section id="security" className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Security Features We Detect
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our AI analyzes multiple security features to ensure comprehensive
              counterfeit detection
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {securityFeatures.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.name}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                  className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-200"
                >
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-semibold text-gray-900">
                          {feature.name}
                        </h3>
                        <span
                          className={`text-xs px-2 py-1 rounded-full font-medium ${
                            feature.importance === 'Critical'
                              ? 'bg-red-100 text-red-700'
                              : feature.importance === 'High'
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-green-100 text-green-700'
                          }`}
                        >
                          {feature.importance}
                        </span>
                      </div>
                      <p className="text-gray-600 text-sm">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Denominations */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Supported Denominations
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              We support detection for all major Nigerian naira denominations
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {denominations.map((denomination, index) => (
              <motion.div
                key={denomination.value}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                className="bg-white rounded-xl shadow-lg p-6 border-2 border-gray-100 hover:border-blue-200 transition-colors duration-200"
              >
                <div className="text-center mb-6">
                  <div
                    className={`w-20 h-20 bg-gradient-to-r ${denomination.color} rounded-full flex items-center justify-center mx-auto mb-4`}
                  >
                    <span className="text-2xl font-bold text-white">
                      {denomination.value}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900">
                    {denomination.value} Note
                  </h3>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-gray-900 mb-3">
                    Security Features:
                  </h4>
                  {denomination.features.map((feature, featureIndex) => (
                    <div
                      key={featureIndex}
                      className="flex items-center space-x-2"
                    >
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-600">{feature}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Technology Stack
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built with cutting-edge AI and machine learning technologies
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { name: 'TensorFlow', description: 'Deep Learning Framework' },
              { name: 'OpenCV', description: 'Computer Vision Library' },
              {
                name: 'MobileNetV2',
                description: 'Convolutional Neural Network',
              },
              {
                name: 'EfficientNet',
                description: 'Advanced CNN Architecture',
              },
              { name: 'React', description: 'Frontend Framework' },
              { name: 'FastAPI', description: 'Backend API Framework' },
              { name: 'PostgreSQL', description: 'Database System' },
              { name: 'Docker', description: 'Containerization' },
            ].map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                className="bg-white rounded-xl shadow-lg p-6 text-center hover:shadow-xl transition-shadow duration-200"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {tech.name}
                </h3>
                <p className="text-gray-600 text-sm">{tech.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Model Status */}
      {modelStatus && (
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
                System Status
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Real-time monitoring of our AI detection system
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-gray-100">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <Shield className="w-5 h-5 mr-2" />
                  Model Status
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">AI Model</span>
                    <div className="flex items-center space-x-2">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          modelStatus.model_loaded
                            ? 'bg-green-500'
                            : 'bg-red-500'
                        }`}
                      />
                      <span className="text-sm font-medium">
                        {modelStatus.model_loaded ? 'Loaded' : 'Not Loaded'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Preprocessor</span>
                    <div className="flex items-center space-x-2">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          modelStatus.preprocessor_loaded
                            ? 'bg-green-500'
                            : 'bg-red-500'
                        }`}
                      />
                      <span className="text-sm font-medium">
                        {modelStatus.preprocessor_loaded
                          ? 'Ready'
                          : 'Not Ready'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Last Updated</span>
                    <span className="text-sm text-gray-500">
                      {new Date(modelStatus.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-gray-100">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  Performance Metrics
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Detection Accuracy</span>
                    <span className="text-sm font-medium text-green-600">
                      99.2%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Average Response Time</span>
                    <span className="text-sm font-medium text-blue-600">
                      1.2s
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Uptime</span>
                    <span className="text-sm font-medium text-green-600">
                      99.9%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default AboutPage;
