from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
from typing import Dict, Any

app = FastAPI(title="NairaGuard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NairaGuard API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("VERCEL_ENV", "development")
    }

@app.post("/predict")
async def predict_note_authenticity():
    """Mock prediction endpoint for Vercel"""
    return {
        "predicted_class": "genuine",
        "confidence": 0.85,
        "probabilities": [0.85, 0.15],
        "timestamp": datetime.now().isoformat(),
        "model_status": "mock_vercel",
        "message": "Mock prediction - ML model not available on Vercel"
    }

@app.get("/model/status")
async def get_model_status():
    """Model status endpoint"""
    return {
        "status": "unloaded",
        "model_path": "N/A",
        "model_type": "N/A",
        "message": "ML model not available on Vercel serverless environment",
        "environment": "vercel"
    }

@app.get("/predictions/history")
async def get_prediction_history():
    """Mock prediction history"""
    return {
        "predictions": [
            {
                "id": 1,
                "filename": "sample_note.jpg",
                "predicted_class": "genuine",
                "confidence": 0.85,
                "probabilities": [0.85, 0.15],
                "timestamp": datetime.now().isoformat(),
                "model_status": "mock_vercel",
                "user_id": None,
                "ip_address": "127.0.0.1"
            }
        ],
        "total": 1,
        "message": "Mock data - database not available on Vercel"
    }

# Vercel serverless function handler
def handler(request):
    """Vercel serverless function handler"""
    return app(request.scope, request.receive, request.send)
