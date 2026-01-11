"""
FastAPI server for emotion-aware AI tutor system.
Handles real-time emotion inference requests from Streamlit dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import asyncio
import json
import os
from dotenv import load_dotenv

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference_face import FaceEmotionInference
from backend.inference_audio import AudioEmotionInference
from backend.engagement import EngagementTracker
from backend.tutor import AdaptiveTutor
from backend.utils.logger import setup_logger

load_dotenv()

app = FastAPI(title="Emotion-Aware AI Tutor API", version="1.0.0")

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger = setup_logger(__name__)
face_inference = None
audio_inference = None
engagement_tracker = EngagementTracker()
tutor = AdaptiveTutor()

# WebSocket connections pool
active_connections: List[WebSocket] = []


class EmotionRequest(BaseModel):
    """Request model for emotion inference"""
    frame_data: Optional[str] = None  # Base64 encoded image
    audio_data: Optional[str] = None  # Base64 encoded audio bytes
    timestamp: float


class EmotionResponse(BaseModel):
    """Response model for emotion inference"""
    face_emotions: Optional[Dict[str, float]] = None
    audio_emotions: Optional[Dict[str, float]] = None
    engagement_score: float
    confusion_level: float
    tutor_response: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global face_inference, audio_inference
    
    try:
        face_model_path = os.getenv("FACE_MODEL_PATH", "models/face_emotion_model.pth")
        audio_model_path = os.getenv("AUDIO_MODEL_PATH", "models/speech_emotion_model.pth")
        
        logger.info("Loading face emotion model...")
        face_inference = FaceEmotionInference(model_path=face_model_path)
        
        logger.info("Loading audio emotion model (4-class speech emotion)...")
        audio_inference = AudioEmotionInference(model_path=audio_model_path)
        
        logger.info("API server started successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Emotion-Aware AI Tutor API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "face_model_loaded": face_inference is not None,
        "audio_model_loaded": audio_inference is not None,
        "active_connections": len(active_connections)
    }


@app.post("/infer/emotions", response_model=EmotionResponse)
async def infer_emotions(request: EmotionRequest):
    """
    Process emotion inference from face and/or audio data.
    
    Args:
        request: EmotionRequest with frame_data and/or audio_data
        
    Returns:
        EmotionResponse with emotions, engagement, and tutor response
    """
    try:
        face_emotions = None
        audio_emotions = None
        
        # Process face emotion if frame data provided
        if request.frame_data and face_inference:
            face_emotions = await asyncio.to_thread(
                face_inference.predict_from_base64,
                request.frame_data
            )
        
        # Process audio emotion if audio data provided
        if request.audio_data and audio_inference:
            # Decode base64 audio data
            import base64 as b64
            audio_bytes = b64.b64decode(request.audio_data)
            audio_emotions = await asyncio.to_thread(
                audio_inference.predict_from_bytes,
                audio_bytes
            )
        
        # Compute engagement and confusion
        engagement_data = engagement_tracker.compute_engagement(
            face_emotions=face_emotions,
            audio_emotions=audio_emotions,
            timestamp=request.timestamp
        )
        
        # Get adaptive tutor response
        tutor_response = None
        if engagement_data["confusion_level"] > 0.5:
            tutor_response = tutor.generate_response(
                emotions=face_emotions or audio_emotions or {},
                engagement=engagement_data["engagement_score"],
                confusion=engagement_data["confusion_level"]
            )
        
        return EmotionResponse(
            face_emotions=face_emotions,
            audio_emotions=audio_emotions,
            engagement_score=engagement_data["engagement_score"],
            confusion_level=engagement_data["confusion_level"],
            tutor_response=tutor_response
        )
    
    except Exception as e:
        logger.error(f"Error in emotion inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/emotions")
async def websocket_emotions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion streaming.
    Streams emotion data continuously from client.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connection established. Total: {len(active_connections)}")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process emotions
            request = EmotionRequest(**data)
            response = await infer_emotions(request)
            
            # Send response back
            await websocket.send_json(response.dict())
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/tutor/chat")
async def tutor_chat(message: str, context: Optional[Dict] = None):
    """
    Get adaptive tutor response based on chat message and emotional context.
    
    Args:
        message: Student's chat message
        context: Optional emotional context (emotions, engagement, confusion)
        
    Returns:
        Tutor's adaptive response
    """
    try:
        response = tutor.generate_chat_response(
            message=message,
            context=context or {}
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in tutor chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/analytics")
async def get_session_analytics():
    """
    Get aggregated session analytics from engagement tracker.
    
    Returns:
        Session statistics (average engagement, emotion distribution, etc.)
    """
    try:
        analytics = engagement_tracker.get_session_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/reset")
async def reset_session():
    """Reset engagement tracker session data"""
    try:
        engagement_tracker.reset_session()
        return {"status": "session_reset"}
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

