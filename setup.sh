#!/bin/bash

# NairaGuard Setup Script
echo "🚀 Setting up NairaGuard AI Detection System..."

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Please run this script from the fake-naira-detection directory"
    echo "   cd /Users/mac/Downloads/fake-naira-detection"
    echo "   ./setup.sh"
    exit 1
fi

echo "✅ Found project files"

# Check Python dependencies
echo "🔧 Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install fastapi uvicorn tensorflow opencv-python pillow numpy pandas scikit-learn
else
    echo "✅ Python dependencies are installed"
fi

# Start backend
echo "🚀 Starting backend server..."
cd backend
python3 main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
curl -s http://localhost:8000/ > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend is running on http://localhost:8000"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Open frontend in browser
echo "🌐 Opening frontend in browser..."
open frontend/index.html

echo ""
echo "🎉 NairaGuard is now running!"
echo ""
echo "📱 Frontend: Open in your browser (should open automatically)"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the backend, press Ctrl+C or run: kill $BACKEND_PID"
echo ""
echo "🧪 Test the app by uploading an image!"
