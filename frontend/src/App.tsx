import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './index.css';

// Header Component
const Header = () => (
  <header className="bg-white shadow-lg">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex justify-between items-center h-16">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold">🛡️</span>
          </div>
          <span className="text-xl font-bold text-gray-900">NairaGuard</span>
        </div>
        <nav className="hidden md:flex space-x-8">
          <a href="/" className="text-gray-600 hover:text-gray-900 font-medium">
            Home
          </a>
          <a
            href="/about"
            className="text-gray-600 hover:text-gray-900 font-medium"
          >
            About
          </a>
          <a
            href="/admin"
            className="text-gray-600 hover:text-gray-900 font-medium"
          >
            Admin
          </a>
        </nav>
      </div>
    </div>
  </header>
);

// Home Page Component
const HomePage = () => {
  const [file, setFile] = React.useState<File | null>(null);
  const [preview, setPreview] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = e => setPreview(e.target?.result as string);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Detect Fake Naira Notes
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
              with AI Precision
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Upload an image to instantly verify the authenticity of Nigerian
            currency. Our advanced AI technology provides accurate detection in
            seconds.
          </p>
        </div>

        {/* Upload Section */}
        <div className="max-w-2xl mx-auto mb-16">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Upload Naira Note Image
              </h2>
              <p className="text-gray-600">Select an image file to analyze</p>
            </div>

            <div className="space-y-6">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="fileInput"
                />
                <label htmlFor="fileInput" className="cursor-pointer block">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">📁</span>
                  </div>
                  <p className="text-lg font-medium text-gray-900">
                    Click to select an image
                  </p>
                  <p className="text-gray-500 mt-2">
                    Supports JPEG, PNG, WebP (max 10MB)
                  </p>
                </label>
              </div>

              {preview && (
                <div className="text-center">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-64 h-48 object-cover rounded-lg mx-auto mb-4"
                  />
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
                  >
                    {loading ? 'Analyzing...' : 'Analyze Image'}
                  </button>
                </div>
              )}

              {result && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="text-center">
                    <div
                      className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 ${
                        result.predicted_class === 'genuine'
                          ? 'bg-green-100'
                          : 'bg-red-100'
                      }`}
                    >
                      <span className="text-4xl">
                        {result.predicted_class === 'genuine' ? '✅' : '❌'}
                      </span>
                    </div>
                    <h3
                      className={`text-3xl font-bold mb-2 ${
                        result.predicted_class === 'genuine'
                          ? 'text-green-600'
                          : 'text-red-600'
                      }`}
                    >
                      {result.predicted_class === 'genuine'
                        ? 'Genuine'
                        : 'Fake'}
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Confidence: {Math.round(result.confidence * 100)}%
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          result.confidence > 0.8
                            ? 'bg-green-500'
                            : result.confidence > 0.6
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                        }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div className="text-center p-6 bg-white rounded-xl shadow-lg">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-xl">🤖</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              AI-Powered Detection
            </h3>
            <p className="text-gray-600 text-sm">
              Advanced machine learning algorithms detect counterfeit notes with
              99%+ accuracy
            </p>
          </div>

          <div className="text-center p-6 bg-white rounded-xl shadow-lg">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-xl">⚡</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Instant Results
            </h3>
            <p className="text-gray-600 text-sm">
              Get detection results in seconds with detailed confidence scores
            </p>
          </div>

          <div className="text-center p-6 bg-white rounded-xl shadow-lg">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-xl">👥</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Trusted by Many
            </h3>
            <p className="text-gray-600 text-sm">
              Used by banks, businesses, and individuals across Nigeria
            </p>
          </div>

          <div className="text-center p-6 bg-white rounded-xl shadow-lg">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-xl">📊</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Real-time Analytics
            </h3>
            <p className="text-gray-600 text-sm">
              Track detection patterns and get insights into your verification
              process
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// About Page Component
const AboutPage = () => {
  const [modelStatus, setModelStatus] = React.useState<any>(null);

  React.useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/model/status');
        const data = await response.json();
        setModelStatus(data);
      } catch (error) {
        console.error('Error fetching model status:', error);
      }
    };
    fetchModelStatus();
  }, []);

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            About NairaGuard
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Advanced AI-powered counterfeit detection for Nigerian currency.
            Protecting businesses and individuals from fake naira notes with
            cutting-edge technology.
          </p>
        </div>

        {/* Security Features */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
            Security Features We Detect
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              'Watermark Verification',
              'Security Thread Analysis',
              'Color-shifting Ink Detection',
              'Microprinting Validation',
              'Holographic Element Check',
              'Paper Quality Assessment',
            ].map((feature, index) => (
              <div
                key={index}
                className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg"
              >
                <span className="text-green-500 text-xl">✅</span>
                <span className="text-gray-700 font-medium">{feature}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
            Technology Stack
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              'TensorFlow',
              'OpenCV',
              'React',
              'FastAPI',
              'PostgreSQL',
              'Docker',
              'Node.js',
              'TypeScript',
            ].map((tech, index) => (
              <div
                key={index}
                className="text-center p-4 bg-gray-50 rounded-lg"
              >
                <span className="text-lg font-semibold text-gray-900">
                  {tech}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* System Status */}
        {modelStatus && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
              System Status
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div
                  className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3 ${
                    modelStatus.model_loaded ? 'bg-green-100' : 'bg-red-100'
                  }`}
                >
                  <span className="text-2xl">🤖</span>
                </div>
                <h3 className="font-medium text-gray-900">AI Model</h3>
                <p
                  className={`text-sm ${
                    modelStatus.model_loaded ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {modelStatus.model_loaded ? 'Loaded' : 'Not Loaded'}
                </p>
              </div>

              <div className="text-center">
                <div
                  className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3 ${
                    modelStatus.preprocessor_loaded
                      ? 'bg-green-100'
                      : 'bg-red-100'
                  }`}
                >
                  <span className="text-2xl">⚙️</span>
                </div>
                <h3 className="font-medium text-gray-900">Preprocessor</h3>
                <p
                  className={`text-sm ${
                    modelStatus.preprocessor_loaded
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {modelStatus.preprocessor_loaded ? 'Ready' : 'Not Ready'}
                </p>
              </div>

              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-2xl">📊</span>
                </div>
                <h3 className="font-medium text-gray-900">Uptime</h3>
                <p className="text-sm text-blue-600">99.9%</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Admin Dashboard Component
const AdminDashboard = () => {
  const [stats, setStats] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchStats = async () => {
      try {
        console.log('Fetching admin stats...');
        const response = await fetch(
          'http://localhost:8000/predictions/history?limit=100'
        );
        console.log('Response status:', response.status);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Admin data received:', data);

        // Calculate stats
        const total = data.predictions.length;
        const genuine = data.predictions.filter(
          (p: any) => p.predicted_class === 'genuine'
        ).length;
        const fake = data.predictions.filter(
          (p: any) => p.predicted_class === 'fake'
        ).length;
        const avgConfidence =
          data.predictions.reduce(
            (sum: number, p: any) => sum + p.confidence,
            0
          ) / total;

        setStats({
          total,
          genuine,
          fake,
          avgConfidence: Math.round(avgConfidence * 100) || 0,
        });
      } catch (error) {
        console.error('Error fetching stats:', error);
        setError(error instanceof Error ? error.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen pt-16 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen pt-16 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-red-600 text-2xl">❌</span>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Error Loading Dashboard
          </h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Admin Dashboard
          </h1>
          <p className="text-gray-600">
            Monitor detection analytics and system performance
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Total Detections
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats?.total || 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 text-xl">📊</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Genuine Notes
                </p>
                <p className="text-2xl font-bold text-green-600">
                  {stats?.genuine || 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-green-600 text-xl">✅</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Fake Notes</p>
                <p className="text-2xl font-bold text-red-600">
                  {stats?.fake || 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <span className="text-red-600 text-xl">❌</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Confidence
                </p>
                <p className="text-2xl font-bold text-purple-600">
                  {stats?.avgConfidence || 0}%
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 text-xl">📈</span>
              </div>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            System Information
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Backend API
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Status</span>
                  <span className="text-green-600 font-medium">Online</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">URL</span>
                  <span className="text-gray-900">http://localhost:8000</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Version</span>
                  <span className="text-gray-900">1.0.0</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Frontend App
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Status</span>
                  <span className="text-green-600 font-medium">Online</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">URL</span>
                  <span className="text-gray-900">http://localhost:3000</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Framework</span>
                  <span className="text-gray-900">React + Vite</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/admin" element={<AdminDashboard />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
