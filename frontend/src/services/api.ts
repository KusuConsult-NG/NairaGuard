import axios, { AxiosResponse } from 'axios';
import { DetectionResult } from '../context/AppContext';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API service class
export class ApiService {
  // Health check
  static async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response: AxiosResponse = await api.get('/api/v1/health');
    return response.data;
  }

  // Get model status
  static async getModelStatus(): Promise<{
    model_loaded: boolean;
    preprocessor_loaded: boolean;
    timestamp: string;
    model_type?: string;
    model_path?: string;
  }> {
    const response: AxiosResponse = await api.get('/model/status');
    return response.data;
  }

  // Predict single image
  static async predictImage(file: File): Promise<DetectionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response: AxiosResponse<DetectionResult> = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  // Predict multiple images
  static async predictBatch(files: File[]): Promise<{
    results: DetectionResult[];
    summary: {
      total_files: number;
      processed: number;
      errors: number;
      timestamp: string;
    };
  }> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response: AxiosResponse = await api.post('/predict/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  // Get prediction history
  static async getPredictionHistory(limit: number = 100): Promise<{
    predictions: Array<{
      id: number;
      filename: string;
      predicted_class: string;
      confidence: number;
      timestamp: string;
      model_status: string;
    }>;
    total_count: number;
    timestamp: string;
  }> {
    const response: AxiosResponse = await api.get(`/predictions/history?limit=${limit}`);
    return response.data;
  }

  // Reload model
  static async reloadModel(modelPath: string, modelType: string = 'keras'): Promise<{
    status: string;
    message: string;
    model_type: string;
    timestamp: string;
  }> {
    const response: AxiosResponse = await api.post('/model/reload', null, {
      params: { model_path: modelPath, model_type: modelType },
    });
    return response.data;
  }

  // Get API info
  static async getApiInfo(): Promise<{
    message: string;
    version: string;
    status: string;
    timestamp: string;
    endpoints: Record<string, string>;
  }> {
    const response: AxiosResponse = await api.get('/');
    return response.data;
  }
}

// Error handling utilities
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Error handler
export const handleApiError = (error: any): string => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    return data?.detail || `Server error: ${status}`;
  } else if (error.request) {
    // Request was made but no response received
    return 'Network error: Unable to connect to server';
  } else {
    // Something else happened
    return error.message || 'An unexpected error occurred';
  }
};

// File validation utilities
export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  // Check file type
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: 'Invalid file type. Please upload a JPEG, PNG, or WebP image.',
    };
  }

  // Check file size (10MB limit)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    return {
      valid: false,
      error: 'File too large. Please upload an image smaller than 10MB.',
    };
  }

  return { valid: true };
};

// Image processing utilities
export const createImagePreview = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        resolve(e.target.result as string);
      } else {
        reject(new Error('Failed to create image preview'));
      }
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
};

// Batch file validation
export const validateBatchFiles = (files: File[]): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  if (files.length === 0) {
    errors.push('No files selected');
  }
  
  if (files.length > 10) {
    errors.push('Maximum 10 files allowed per batch');
  }
  
  files.forEach((file, index) => {
    const validation = validateImageFile(file);
    if (!validation.valid) {
      errors.push(`File ${index + 1} (${file.name}): ${validation.error}`);
    }
  });
  
  return {
    valid: errors.length === 0,
    errors,
  };
};

export default ApiService;
