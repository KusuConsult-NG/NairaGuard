import React from 'react';
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Download,
} from 'lucide-react';

const ResultsPage: React.FC = () => {
  // Mock data - in real app, this would come from API/state
  const mockResults = {
    totalProcessed: 1247,
    authenticCount: 892,
    fakeCount: 355,
    accuracyRate: 99.2,
    recentDetections: [
      {
        id: 1,
        image: '/api/placeholder/100/100',
        denomination: '₦1000',
        result: 'authentic',
        confidence: 98.5,
        timestamp: '2024-01-15 14:30:22',
      },
      {
        id: 2,
        image: '/api/placeholder/100/100',
        denomination: '₦500',
        result: 'fake',
        confidence: 94.2,
        timestamp: '2024-01-15 14:28:15',
      },
      {
        id: 3,
        image: '/api/placeholder/100/100',
        denomination: '₦200',
        result: 'authentic',
        confidence: 99.1,
        timestamp: '2024-01-15 14:25:08',
      },
    ],
  };

  const stats = [
    {
      title: 'Total Processed',
      value: mockResults.totalProcessed.toLocaleString(),
      icon: BarChart3,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      title: 'Authentic Notes',
      value: mockResults.authenticCount.toLocaleString(),
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      title: 'Fake Detected',
      value: mockResults.fakeCount.toLocaleString(),
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
    },
    {
      title: 'Accuracy Rate',
      value: `${mockResults.accuracyRate}%`,
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  ];

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Detection Results
          </h1>
          <p className="text-xl text-gray-600">
            Overview of your naira note detection history and statistics
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <div key={index} className="card">
              <div className="flex items-center">
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">
                    {stat.title}
                  </p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stat.value}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Detection Trend Chart */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Detection Trend (Last 7 Days)
            </h3>
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500">
                  Chart visualization would go here
                </p>
                <p className="text-sm text-gray-400">
                  Integration with Chart.js or similar
                </p>
              </div>
            </div>
          </div>

          {/* Accuracy Distribution */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Accuracy Distribution
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">90-95%</span>
                <div className="flex-1 mx-4">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-yellow-400 h-2 rounded-full"
                      style={{ width: '15%' }}
                    ></div>
                  </div>
                </div>
                <span className="text-sm font-medium text-gray-900">15%</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">95-98%</span>
                <div className="flex-1 mx-4">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-orange-400 h-2 rounded-full"
                      style={{ width: '25%' }}
                    ></div>
                  </div>
                </div>
                <span className="text-sm font-medium text-gray-900">25%</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">98-100%</span>
                <div className="flex-1 mx-4">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-400 h-2 rounded-full"
                      style={{ width: '60%' }}
                    ></div>
                  </div>
                </div>
                <span className="text-sm font-medium text-gray-900">60%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Detections */}
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Recent Detections
            </h3>
            <button className="btn-secondary flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Image
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Denomination
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Result
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {mockResults.recentDetections.map(detection => (
                  <tr key={detection.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center">
                        <img
                          src={detection.image}
                          alt="Note preview"
                          className="w-full h-full object-cover rounded-lg"
                        />
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {detection.denomination}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          detection.result === 'authentic'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {detection.result === 'authentic' ? (
                          <>
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Authentic
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="w-3 h-3 mr-1" />
                            Fake
                          </>
                        )}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {detection.confidence}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {detection.timestamp}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
