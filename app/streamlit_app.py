"""
Main Streamlit dashboard for emotion-aware AI tutor.
Real-time visualization of emotions, engagement, and adaptive tutoring.
"""

import streamlit as st
import cv2
import numpy as np
import requests
from requests.exceptions import Timeout, ConnectionError
import base64
import time
from typing import Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image

from components.emotion_meter import EmotionMeter
from components.voice_gauge import VoiceGauge
from components.engagement_bar import EngagementBar
from components.tutor_chatbox import TutorChatbox

# Page configuration
st.set_page_config(
    page_title="Emotion-Aware AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="FastAPI backend URL"
)

# Initialize session state
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
if "engagement_history" not in st.session_state:
    st.session_state.engagement_history = []
if "confusion_alerts" not in st.session_state:
    st.session_state.confusion_alerts = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "audio_emotions" not in st.session_state:
    st.session_state.audio_emotions = None


def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def process_frame(frame: np.ndarray) -> Dict:
    """
    Process frame and get emotion predictions from API.
    
    Args:
        frame: Video frame (BGR)
        
    Returns:
        Emotion and engagement data
    """
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send to API
    try:
        response = requests.post(
            f"{API_URL}/infer/emotions",
            json={
                "frame_data": frame_base64,
                "timestamp": time.time()
            },
            timeout=1
        )
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    
    return None


def process_audio(audio_file) -> Dict:
    """
    Process audio and get emotion predictions from API.
    
    Args:
        audio_file: Audio file from st.audio_input (UploadedFile or bytes)
        
    Returns:
        Emotion and engagement data
    """
    # Read audio bytes from file if it's an UploadedFile
    if hasattr(audio_file, 'read'):
        audio_bytes = audio_file.read()
    else:
        audio_bytes = audio_file
    
    # Encode audio to base64 for JSON transmission
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Send to API
    try:
        response = requests.post(
            f"{API_URL}/infer/emotions",
            json={
                "audio_data": audio_base64,
                "timestamp": time.time()
            },
            timeout=30  # Increased timeout for audio processing (can take longer)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned status {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json()
                    st.error(f"Error details: {error_detail.get('detail', 'Unknown error')}")
                except:
                    pass
    except requests.exceptions.Timeout:
        st.error("⏱️ Audio processing timed out. The audio file might be too long or the server is busy. Try recording a shorter audio clip.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Make sure the backend API is running on port 8000.")
    except Exception as e:
        st.error(f"API Error: {e}")
    
    return None


def main():
    """Main dashboard application"""
    
    st.title("🎓 Emotion-Aware AI Tutor")
    st.markdown("Real-time emotion detection and adaptive tutoring system")
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("⚠️ API server is not available. Please start the FastAPI backend.")
        st.info("Run: `uvicorn backend.api:app --reload`")
        return
    
    st.success("✅ Connected to API")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Webcam selection
    camera_index = st.sidebar.selectbox(
        "Camera",
        options=[0, 1, 2],
        index=0
    )
    
    # Start/Stop button
    start_detection = st.sidebar.button("▶️ Start Detection", type="primary")
    stop_detection = st.sidebar.button("⏹️ Stop Detection")
    
    if stop_detection:
        st.session_state.detection_active = False
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Webcam Feed")
        
        # Webcam placeholder
        webcam_placeholder = st.empty()
        
        # Emotion visualization
        st.subheader("😊 Face Emotions")
        emotion_placeholder = st.empty()
        
        # Engagement visualization
        st.subheader("📊 Engagement & Confusion")
        engagement_placeholder = st.empty()
    
    with col2:
        st.subheader("🎤 Voice Emotions")
        
        # Audio input recorder
        st.markdown("**Record Audio:**")
        audio_bytes = st.audio_input("Speak to detect emotions", key="audio_recorder")
        
        voice_placeholder = st.empty()
        
        # Process audio if recorded
        if audio_bytes:
            # Clear previous audio emotions when new recording starts
            st.session_state.audio_emotions = None
            voice_placeholder.empty()  # Clear previous display
            
            with st.spinner("Processing audio emotions..."):
                audio_result = process_audio(audio_bytes)
                if audio_result and audio_result.get("audio_emotions"):
                    st.session_state.audio_emotions = audio_result["audio_emotions"]
                    st.success("✅ Audio processed!")
                    # Playback recorded audio
                    st.audio(audio_bytes, format="audio/wav")
                    # Display voice emotions
                    with voice_placeholder.container():
                        VoiceGauge.display(st.session_state.audio_emotions, frame_id=999999)
                else:
                    st.warning("⚠️ No emotions detected in audio. Please try again.")
        else:
            # Clear display when no audio is recorded
            if st.session_state.audio_emotions:
                st.session_state.audio_emotions = None
            voice_placeholder.empty()
        
        st.subheader("💬 Tutor Chat")
        chat_placeholder = st.empty()
        
        # Session analytics
        with st.expander("📈 Session Analytics"):
            if st.button("View Analytics"):
                try:
                    response = requests.get(f"{API_URL}/session/analytics")
                    if response.status_code == 200:
                        analytics = response.json()
                        st.json(analytics)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Detection loop
    if start_detection:
        st.session_state.detection_active = True
    
    if st.session_state.get("detection_active", False):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Failed to open camera {camera_index}")
            return
        
        frame_count = 0
        
        while st.session_state.get("detection_active", False):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame to reduce API load
            if frame_count % 5 == 0:
                # Process frame
                result = process_frame(frame)
                
                if result:
                    # Update history
                    if result.get("face_emotions"):
                        st.session_state.emotion_history.append({
                            "timestamp": time.time(),
                            "emotions": result["face_emotions"]
                        })
                    
                    if result.get("engagement_score") is not None:
                        st.session_state.engagement_history.append({
                            "timestamp": time.time(),
                            "engagement": result["engagement_score"],
                            "confusion": result.get("confusion_level", 0.0)
                        })
                    
                    # Check for confusion alerts
                    if result.get("confusion_level", 0.0) > 0.7:
                        alert = {
                            "timestamp": time.time(),
                            "confusion": result["confusion_level"],
                            "message": result.get("tutor_response", "High confusion detected")
                        }
                        if alert not in st.session_state.confusion_alerts:
                            st.session_state.confusion_alerts.append(alert)
                            st.warning(f"⚠️ Confusion Alert: {alert['message']}")
                    
                    # Display emotion meter
                    if result.get("face_emotions"):
                        with emotion_placeholder.container():
                            EmotionMeter.display(result["face_emotions"], frame_id=frame_count)
                    
                    # Display engagement bar
                    if result.get("engagement_score") is not None:
                        with engagement_placeholder.container():
                            EngagementBar.display(
                                result["engagement_score"],
                                result.get("confusion_level", 0.0),
                                frame_id=frame_count
                            )
                    
                    # Display voice gauge (if audio available from video processing)
                    if result.get("audio_emotions"):
                        st.session_state.audio_emotions = result["audio_emotions"]
                        with voice_placeholder.container():
                            VoiceGauge.display(result["audio_emotions"], frame_id=frame_count)
            
            # Display webcam feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            time.sleep(0.1)  # Control frame rate
        
        cap.release()
    
    # Display tutor chatbox
    with chat_placeholder.container():
        TutorChatbox.display(API_URL)
    
    # Confusion alerts sidebar
    if st.session_state.confusion_alerts:
        st.sidebar.subheader("⚠️ Confusion Alerts")
        for alert in st.session_state.confusion_alerts[-5:]:  # Show last 5
            st.sidebar.warning(f"{alert['message']}")
    
    # Reset session button
    if st.sidebar.button("🔄 Reset Session"):
        try:
            response = requests.post(f"{API_URL}/session/reset")
            if response.status_code == 200:
                st.session_state.emotion_history = []
                st.session_state.engagement_history = []
                st.session_state.confusion_alerts = []
                st.success("Session reset!")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

