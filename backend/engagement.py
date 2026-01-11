"""
Engagement and confusion tracking using MediaPipe face landmarks.
Computes blink rate, gaze direction, and emotion trends.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
import time
from dataclasses import dataclass

from backend.utils.face_detector import FaceDetector
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EngagementMetrics:
    """Container for engagement metrics"""
    engagement_score: float  # 0-100
    confusion_level: float  # 0-1
    blink_rate: float  # blinks per minute
    gaze_attention: float  # 0-1
    emotion_trend: float  # -1 to 1 (negative to positive)


class EngagementTracker:
    """Tracks student engagement and confusion in real-time"""
    
    def __init__(
        self,
        blink_threshold: float = 0.25,  # Eye aspect ratio threshold
        window_size: int = 30,  # Frames to track
        normal_blink_rate: Tuple[float, float] = (15.0, 20.0)  # Normal range (blinks/min)
    ):
        """
        Initialize engagement tracker.
        
        Args:
            blink_threshold: EAR threshold for blink detection
            window_size: Number of frames to track for trends
            normal_blink_rate: Normal blink rate range (min, max) per minute
        """
        self.blink_threshold = blink_threshold
        self.window_size = window_size
        self.normal_blink_rate = normal_blink_rate
        
        # Face detector for landmarks
        self.face_detector = FaceDetector()
        
        # Tracking buffers
        self.ear_history: deque = deque(maxlen=window_size)  # Eye aspect ratio
        self.emotion_history: deque = deque(maxlen=window_size)
        self.gaze_history: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        
        # Blink detection
        self.blink_count = 0
        self.last_blink_time = None
        self.consecutive_frames_below_threshold = 0
        
        # Session statistics
        self.session_start_time = time.time()
        self.total_frames = 0
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) from eye landmarks.
        
        Args:
            eye_landmarks: Array of 6 eye landmark points
            
        Returns:
            EAR value
        """
        # Vertical distances
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def detect_blink(self, left_ear: float, right_ear: float) -> bool:
        """
        Detect blink from eye aspect ratios.
        
        Args:
            left_ear: Left eye EAR
            right_ear: Right eye EAR
            
        Returns:
            True if blink detected
        """
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to history
        self.ear_history.append(avg_ear)
        
        # Blink detection: EAR drops below threshold
        if avg_ear < self.blink_threshold:
            self.consecutive_frames_below_threshold += 1
        else:
            # If we were below threshold and now above, it's a blink
            if self.consecutive_frames_below_threshold > 2:  # Minimum frames for valid blink
                self.blink_count += 1
                self.last_blink_time = time.time()
            self.consecutive_frames_below_threshold = 0
        
        return self.consecutive_frames_below_threshold > 2
    
    def estimate_gaze_direction(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Estimate gaze direction from face landmarks.
        Returns attention score (0-1) based on head pose.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Gaze attention score (0-1, higher = more attentive)
        """
        # Get key facial points
        nose_tip = face_landmarks[4]  # Nose tip
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        chin = face_landmarks[18]
        
        # Calculate head pose angles (simplified)
        # Horizontal angle (yaw)
        eye_center = (left_eye + right_eye) / 2
        horizontal_offset = abs(nose_tip[0] - eye_center[0]) / frame_width
        yaw_score = 1.0 - min(horizontal_offset * 2, 1.0)
        
        # Vertical angle (pitch)
        vertical_offset = abs(nose_tip[1] - eye_center[1]) / frame_height
        pitch_score = 1.0 - min(vertical_offset * 2, 1.0)
        
        # Combined attention score
        attention = (yaw_score + pitch_score) / 2.0
        return attention
    
    def compute_emotion_trend(self, current_emotions: Dict[str, float]) -> float:
        """
        Compute emotion trend from history.
        Positive emotions increase trend, negative decrease.
        
        Args:
            current_emotions: Current emotion probabilities
            
        Returns:
            Emotion trend score (-1 to 1)
        """
        # Positive emotions
        positive_emotions = ["happy", "surprise", "calm"]
        # Negative emotions
        negative_emotions = ["sad", "angry", "fear", "fearful", "disgust"]
        
        positive_score = sum(current_emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(current_emotions.get(emotion, 0) for emotion in negative_emotions)
        
        # Normalize to -1 to 1
        trend = positive_score - negative_score
        return np.clip(trend, -1.0, 1.0)
    
    def compute_engagement(
        self,
        face_emotions: Optional[Dict[str, float]] = None,
        audio_emotions: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
        face_landmarks: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        Compute overall engagement score and confusion level.
        
        Args:
            face_emotions: Face emotion probabilities
            audio_emotions: Audio emotion probabilities
            timestamp: Current timestamp
            face_landmarks: MediaPipe face landmarks (optional)
            frame_shape: (width, height) of frame (optional)
            
        Returns:
            Dictionary with engagement_score, confusion_level, and metrics
        """
        timestamp = timestamp or time.time()
        self.timestamps.append(timestamp)
        self.total_frames += 1
        
        # Combine emotions (prioritize face if available)
        emotions = face_emotions or audio_emotions or {}
        self.emotion_history.append(emotions)
        
        # Compute blink rate
        blink_rate = 0.0
        if len(self.timestamps) > 1:
            time_window = self.timestamps[-1] - self.timestamps[0]
            if time_window > 0:
                blink_rate = (self.blink_count / time_window) * 60.0  # blinks per minute
        
        # Normalize blink rate deviation
        normal_min, normal_max = self.normal_blink_rate
        if normal_min <= blink_rate <= normal_max:
            blink_score = 1.0
        elif blink_rate > normal_max:
            # High blink rate (confusion/fatigue)
            deviation = (blink_rate - normal_max) / normal_max
            blink_score = max(0.0, 1.0 - deviation)
        else:
            # Low blink rate (disengagement)
            deviation = (normal_min - blink_rate) / normal_min
            blink_score = max(0.0, 1.0 - deviation)
        
        # Compute gaze attention
        gaze_attention = 0.7  # Default if no landmarks
        if face_landmarks is not None and frame_shape is not None:
            gaze_attention = self.estimate_gaze_direction(
                face_landmarks,
                frame_shape[0],
                frame_shape[1]
            )
        self.gaze_history.append(gaze_attention)
        
        # Average gaze over window
        avg_gaze = np.mean(list(self.gaze_history)) if self.gaze_history else 0.7
        
        # Compute emotion trend
        emotion_trend = self.compute_emotion_trend(emotions)
        
        # Normalize emotion trend to 0-1
        emotion_score = (emotion_trend + 1.0) / 2.0
        
        # Compute engagement score (weighted combination)
        engagement_score = (
            0.3 * blink_score +
            0.3 * avg_gaze +
            0.4 * emotion_score
        ) * 100.0
        
        # Compute confusion level
        confusion_indicators = []
        
        # High blink rate
        if blink_rate > normal_max * 1.5:
            confusion_indicators.append(0.3)
        
        # Negative emotions
        negative_emotion_score = sum(
            emotions.get(emotion, 0)
            for emotion in ["sad", "angry", "fear", "fearful", "disgust"]
        )
        if negative_emotion_score > 0.5:
            confusion_indicators.append(0.4)
        
        # Low gaze attention
        if avg_gaze < 0.5:
            confusion_indicators.append(0.3)
        
        confusion_level = min(1.0, sum(confusion_indicators))
        
        return {
            "engagement_score": float(np.clip(engagement_score, 0.0, 100.0)),
            "confusion_level": float(confusion_level),
            "blink_rate": float(blink_rate),
            "gaze_attention": float(avg_gaze),
            "emotion_trend": float(emotion_trend),
            "blink_score": float(blink_score),
            "emotion_score": float(emotion_score)
        }
    
    def get_session_analytics(self) -> Dict:
        """Get aggregated session statistics"""
        session_duration = time.time() - self.session_start_time
        
        # Average engagement over session
        avg_engagement = np.mean([
            self.compute_engagement().get("engagement_score", 50.0)
            for _ in range(min(10, len(self.emotion_history)))
        ]) if self.emotion_history else 50.0
        
        # Emotion distribution
        emotion_dist = {}
        for emotions in self.emotion_history:
            for emotion, prob in emotions.items():
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + prob
        
        # Normalize
        total = sum(emotion_dist.values())
        if total > 0:
            emotion_dist = {k: v / total for k, v in emotion_dist.items()}
        
        return {
            "session_duration_seconds": session_duration,
            "total_frames": self.total_frames,
            "average_engagement": float(avg_engagement),
            "total_blinks": self.blink_count,
            "average_blink_rate": float(self.blink_count / (session_duration / 60.0)) if session_duration > 0 else 0.0,
            "emotion_distribution": emotion_dist
        }
    
    def reset_session(self):
        """Reset session tracking data"""
        self.ear_history.clear()
        self.emotion_history.clear()
        self.gaze_history.clear()
        self.timestamps.clear()
        self.blink_count = 0
        self.last_blink_time = None
        self.consecutive_frames_below_threshold = 0
        self.session_start_time = time.time()
        self.total_frames = 0
        logger.info("Session reset")

