import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, Image as ImageIcon, AlertCircle, CheckCircle } from 'lucide-react'

interface UploadedFile {
  file: File
  id: string
  preview: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  result?: {
    isFake: boolean
    confidence: number
    details: string[]
  }
}

const UploadPage: React.FC = () => {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      preview: URL.createObjectURL(file),
      status: 'pending'
    }))
    setFiles(prev => [...prev, ...newFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    multiple: true
  })

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(file => file.id !== id))
  }

  const processFiles = async () => {
    setIsUploading(true)
    
    // Simulate API call
    for (const file of files) {
      if (file.status === 'pending') {
        setFiles(prev => prev.map(f => 
          f.id === file.id ? { ...f, status: 'processing' } : f
        ))
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        // Simulate result
        const isFake = Math.random() > 0.7
        const confidence = Math.random() * 40 + 60 // 60-100%
        
        setFiles(prev => prev.map(f => 
          f.id === file.id ? {
            ...f,
            status: 'completed',
            result: {
              isFake,
              confidence,
              details: isFake 
                ? ['Watermark mismatch', 'Security thread anomaly', 'Color variation detected']
                : ['All security features verified', 'Watermark authentic', 'Security thread intact']
            }
          } : f
        ))
      }
    }
    
    setIsUploading(false)
  }

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'processing':
        return <div className="w-4 h-4 border-2 border-naira-600 border-t-transparent rounded-full animate-spin" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-600" />
      default:
        return null
    }
  }

  const getStatusColor = (file: UploadedFile) => {
    if (file.status === 'completed' && file.result) {
      return file.result.isFake ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'
    }
    if (file.status === 'processing') {
      return 'border-naira-200 bg-naira-50'
    }
    return 'border-gray-200 bg-white'
  }

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Upload Naira Note Images
          </h1>
          <p className="text-xl text-gray-600">
            Drag and drop your images or click to browse. We support JPG, PNG, and other image formats.
          </p>
        </div>

        {/* Upload Area */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-naira-400 bg-naira-50'
              : 'border-gray-300 hover:border-naira-400 hover:bg-naira-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-lg text-gray-600 mb-2">
            {isDragActive
              ? 'Drop the files here...'
              : 'Drag & drop naira note images here, or click to select files'
            }
          </p>
          <p className="text-sm text-gray-500">
            Supports JPG, PNG, GIF, BMP, and WebP formats
          </p>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                Uploaded Files ({files.length})
              </h2>
              <button
                onClick={processFiles}
                disabled={isUploading || files.every(f => f.status !== 'pending')}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? 'Processing...' : 'Process All Files'}
              </button>
            </div>

            <div className="space-y-4">
              {files.map((file) => (
                <div
                  key={file.id}
                  className={`card ${getStatusColor(file)}`}
                >
                  <div className="flex items-center space-x-4">
                    <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
                      <img
                        src={file.preview}
                        alt={file.file.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {file.file.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(file.file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      
                      {file.result && (
                        <div className="mt-2">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(file.status)}
                            <span className={`text-sm font-medium ${
                              file.result.isFake ? 'text-red-600' : 'text-green-600'
                            }`}>
                              {file.result.isFake ? 'FAKE DETECTED' : 'AUTHENTIC'}
                            </span>
                            <span className="text-sm text-gray-500">
                              ({file.result.confidence.toFixed(1)}% confidence)
                            </span>
                          </div>
                          
                          <div className="mt-2">
                            <ul className="text-xs text-gray-600 space-y-1">
                              {file.result.details.map((detail, index) => (
                                <li key={index} className="flex items-center space-x-1">
                                  <div className={`w-1 h-1 rounded-full ${
                                    file.result!.isFake ? 'bg-red-400' : 'bg-green-400'
                                  }`} />
                                  <span>{detail}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <button
                      onClick={() => removeFile(file.id)}
                      className="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadPage
