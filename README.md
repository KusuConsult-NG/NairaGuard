# 💻 Fake Naira Currency Detection Web App

## 🎯 Objective
Build a **web-based AI platform** that allows users (banks, traders, and the public) to upload images of naira notes for instant counterfeit detection using machine learning and computer vision.

## 🏗️ Project Structure
```
fake-naira-detection/
├── frontend/           # Next.js (Vite + TailwindCSS)
├── backend/           # FastAPI (Python)
├── models/            # ML models + training scripts
├── datasets/          # Naira note images (real/fake)
├── tests/             # Unit & integration tests
├── .env               # Environment variables
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites
- Node.js (v18+)
- Python (v3.8+)
- Git

### 1. Clone and Setup
```bash
git clone https://github.com/KusuConsult-NG/NairaGuard.git
cd NairaGuard
```

### 2. Setup AI Model
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Create the AI model
python setup_model.py
```

### 3. Start Backend
```bash
cd backend
python main.py
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Access the App
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🧠 Features
- **Image Upload**: Drag & drop interface for naira note images
- **AI Detection**: Real-time counterfeit detection using ML models
- **Confidence Scores**: Percentage confidence in detection results
- **Batch Processing**: Upload multiple images at once
- **API Integration**: RESTful API for third-party integrations
- **User Management**: Role-based access (banks, traders, public)

## 🔧 Tech Stack
- **Frontend**: Next.js, Vite, TailwindCSS, TypeScript
- **Backend**: FastAPI, Python, OpenCV, TensorFlow/PyTorch
- **ML**: Computer Vision, Image Classification
- **Database**: PostgreSQL (optional)
- **Deployment**: Docker, Vercel/Heroku

## 📊 API Endpoints
- `POST /api/upload` - Upload image for detection
- `GET /api/health` - Health check
- `POST /api/batch-upload` - Batch image processing
- `GET /api/history` - Detection history (authenticated)

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License
MIT License - see LICENSE file for details
