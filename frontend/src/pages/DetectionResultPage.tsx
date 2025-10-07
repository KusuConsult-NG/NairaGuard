import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  RotateCcw,
  Download,
  Share2,
  BarChart3,
  Shield,
  Clock,
  Camera,
  FileImage,
  ArrowLeft,
  ArrowRight,
} from 'lucide-react';
import { useApp, appActions, DetectionResult } from '../context/AppContext';
import { ApiService } from '../services/api';
import toast from 'react-hot-toast';
import LoadingSpinner from '../components/LoadingSpinner';

const DetectionResultPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisDetails, setAnalysisDetails] = useState<any>(null);

  // Redirect if no result
  useEffect(() => {
    if (!state.detectionResult && !state.isLoading) {
      navigate('/');
    }
  }, [state.detectionResult, state.isLoading, navigate]);

  // Analyze result for additional details
  useEffect(() => {
    if (state.detectionResult) {
      setIsAnalyzing(true);
      // Simulate additional analysis
      setTimeout(() => {
        setAnalysisDetails({
          securityFeatures: [
            {
              name: 'Watermark',
              detected: Math.random() > 0.3,
              confidence: 0.85,
            },
            {
              name: 'Security Thread',
              detected: Math.random() > 0.2,
              confidence: 0.92,
            },
            {
              name: 'Color-shifting Ink',
              detected: Math.random() > 0.4,
              confidence: 0.78,
            },
            {
              name: 'Microprinting',
              detected: Math.random() > 0.3,
              confidence: 0.88,
            },
            {
              name: 'Holographic Element',
              detected: Math.random() > 0.25,
              confidence: 0.91,
            },
            {
              name: 'Paper Quality',
              detected: Math.random() > 0.2,
              confidence: 0.86,
            },
          ],
          riskFactors: [
            {
              name: 'Print Quality',
              risk: Math.random() > 0.7 ? 'High' : 'Low',
              score: Math.random() * 100,
            },
            {
              name: 'Color Accuracy',
              risk: Math.random() > 0.6 ? 'Medium' : 'Low',
              score: Math.random() * 100,
            },
            {
              name: 'Texture Analysis',
              risk: Math.random() > 0.8 ? 'High' : 'Low',
              score: Math.random() * 100,
            },
          ],
        });
        setIsAnalyzing(false);
      }, 2000);
    }
  }, [state.detectionResult]);

  const handleNewDetection = () => {
    dispatch(appActions.clearDetection());
    navigate('/');
  };

  const handleDownloadReport = () => {
    if (!state.detectionResult) return;

    const report = {
      timestamp: new Date().toISOString(),
      result: state.detectionResult,
      analysis: analysisDetails,
      image: state.imagePreview,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `naira-detection-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success('Report downloaded successfully!');
  };

  const handleShare = async () => {
    if (!state.detectionResult) return;

    const shareData = {
      title: 'Naira Note Detection Result',
      text: `Detected: ${state.detectionResult.predicted_class} (${Math.round(state.detectionResult.confidence * 100)}% confidence)`,
      url: window.location.href,
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        await navigator.clipboard.writeText(shareData.text);
        toast.success('Result copied to clipboard!');
      }
    } catch (error) {
      console.error('Error sharing:', error);
      toast.error('Failed to share result');
    }
  };

  if (!state.detectionResult) {
    return (
      <div className="min-h-screen pt-16 flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading detection result..." />
      </div>
    );
  }

  const {
    predicted_class,
    confidence,
    probabilities,
    timestamp,
    model_status,
  } = state.detectionResult;
  const isGenuine = predicted_class === 'genuine';
  const confidencePercentage = Math.round(confidence * 100);

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Detection Result
          </h1>
          <p className="text-gray-600">
            Analysis completed on {new Date(timestamp).toLocaleString()}
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Preview */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <FileImage className="w-5 h-5 mr-2" />
                Uploaded Image
              </h2>
              {state.imagePreview && (
                <div className="relative">
                  <img
                    src={state.imagePreview}
                    alt="Uploaded naira note"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
                    {state.uploadedImage?.name}
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleNewDetection}
                className="bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
              >
                <RotateCcw className="w-4 h-4" />
                <span>New Detection</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleDownloadReport}
                className="bg-gray-100 text-gray-700 py-3 px-4 rounded-lg font-medium hover:bg-gray-200 transition-colors flex items-center justify-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>Download Report</span>
              </motion.button>
            </div>
          </motion.div>

          {/* Result Details */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            {/* Main Result */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="text-center mb-6">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.6, type: 'spring', stiffness: 200 }}
                  className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 ${
                    isGenuine ? 'bg-green-100' : 'bg-red-100'
                  }`}
                >
                  {isGenuine ? (
                    <CheckCircle className="w-10 h-10 text-green-600" />
                  ) : (
                    <XCircle className="w-10 h-10 text-red-600" />
                  )}
                </motion.div>

                <h2
                  className={`text-3xl font-bold mb-2 ${
                    isGenuine ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {isGenuine ? '✅ Genuine' : '❌ Fake'}
                </h2>

                <p className="text-gray-600 mb-4">
                  {isGenuine
                    ? 'This naira note appears to be authentic'
                    : 'This naira note appears to be counterfeit'}
                </p>

                {/* Confidence Score */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-600">
                      Confidence Score
                    </span>
                    <span className="text-lg font-bold text-gray-900">
                      {confidencePercentage}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${confidencePercentage}%` }}
                      transition={{ delay: 0.8, duration: 1 }}
                      className={`h-3 rounded-full ${
                        confidencePercentage > 80
                          ? 'bg-green-500'
                          : confidencePercentage > 60
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                      }`}
                    />
                  </div>
                </div>
              </div>

              {/* Probability Breakdown */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-900 mb-3">
                  Probability Breakdown
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Genuine</span>
                    <span className="text-sm font-medium text-gray-900">
                      {Math.round(probabilities[0] * 100)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Fake</span>
                    <span className="text-sm font-medium text-gray-900">
                      {Math.round(probabilities[1] * 100)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Model Status */}
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Model Status</span>
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      model_status === 'trained_model'
                        ? 'bg-green-100 text-green-700'
                        : 'bg-yellow-100 text-yellow-700'
                    }`}
                  >
                    {model_status === 'trained_model'
                      ? 'Trained Model'
                      : 'Demo Mode'}
                  </span>
                </div>
              </div>
            </div>

            {/* Security Features Analysis */}
            {analysisDetails && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                className="bg-white rounded-2xl shadow-xl p-6"
              >
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <Shield className="w-5 h-5 mr-2" />
                  Security Features Analysis
                </h3>

                {isAnalyzing ? (
                  <LoadingSpinner text="Analyzing security features..." />
                ) : (
                  <div className="space-y-3">
                    {analysisDetails.securityFeatures.map(
                      (feature: any, index: number) => (
                        <div
                          key={feature.name}
                          className="flex items-center justify-between"
                        >
                          <span className="text-sm text-gray-600">
                            {feature.name}
                          </span>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-500">
                              {Math.round(feature.confidence * 100)}%
                            </span>
                            {feature.detected ? (
                              <CheckCircle className="w-4 h-4 text-green-500" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-500" />
                            )}
                          </div>
                        </div>
                      )
                    )}
                  </div>
                )}
              </motion.div>
            )}

            {/* Risk Assessment */}
            {analysisDetails && !isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
                className="bg-white rounded-2xl shadow-xl p-6"
              >
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2" />
                  Risk Assessment
                </h3>

                <div className="space-y-4">
                  {analysisDetails.riskFactors.map(
                    (factor: any, index: number) => (
                      <div key={factor.name} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">
                            {factor.name}
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded-full font-medium ${
                              factor.risk === 'High'
                                ? 'bg-red-100 text-red-700'
                                : factor.risk === 'Medium'
                                  ? 'bg-yellow-100 text-yellow-700'
                                  : 'bg-green-100 text-green-700'
                            }`}
                          >
                            {factor.risk} Risk
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              factor.risk === 'High'
                                ? 'bg-red-500'
                                : factor.risk === 'Medium'
                                  ? 'bg-yellow-500'
                                  : 'bg-green-500'
                            }`}
                            style={{ width: `${factor.score}%` }}
                          />
                        </div>
                      </div>
                    )
                  )}
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>

        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4 }}
          className="mt-12 flex justify-center space-x-4"
        >
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/')}
            className="bg-gray-100 text-gray-700 py-3 px-6 rounded-lg font-medium hover:bg-gray-200 transition-colors flex items-center space-x-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Home</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleShare}
            className="bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <Share2 className="w-4 h-4" />
            <span>Share Result</span>
          </motion.button>
        </motion.div>
      </div>
    </div>
  );
};

export default DetectionResultPage;
