"""
Real-time audio emotion inference using trained Speech Emotion Recognition model.
Uses CNN+BiLSTM+Attention architecture with 4 classes.
Extracts comprehensive features: 40 MFCCs + Chroma + Mel + ZCR
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from typing import Dict, Optional
import io
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.speech_emotion_model import SpeechEmotionCNNBiLSTMAttention

# Emotion class labels (4 classes - learning-centric)
EMOTION_LABELS = {
    0: "happy",
    1: "neutral",
    2: "confused",
    3: "frustrated"
}

# Target sample rate (matches training)
TARGET_SR = 16000


def trim_silence(audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Trim silence from audio using librosa."""
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20, frame_length=frame_length, hop_length=hop_length)
    return audio_trimmed


def extract_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract comprehensive features from audio.
    Returns: 40 MFCCs + Chroma + Mel spectrogram + ZCR (181 dimensions)
    """
    features_list = []
    
    # 1. MFCCs (40 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    mfccs_mean = np.mean(mfccs, axis=1)  # Average over time
    features_list.append(mfccs_mean)
    
    # 2. Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)  # Average over time
    features_list.append(chroma_mean)
    
    # 3. Mel spectrogram (mean)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_mean = np.mean(mel_spec, axis=1)  # Average over time
    features_list.append(mel_spec_mean)
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
    zcr_mean = np.mean(zcr)
    features_list.append(np.array([zcr_mean]))
    
    # Concatenate all features
    combined_features = np.concatenate(features_list)
    
    return combined_features


class AudioEmotionInference:
    """Real-time audio emotion inference engine (4-class model)"""
    
    def __init__(
        self,
        model_path: str = "models/speech_emotion_model.pth",
        device: Optional[str] = None,
        sample_rate: int = TARGET_SR
    ):
        """
        Initialize audio emotion inference.
        
        Args:
            model_path: Path to trained PyTorch model
            device: 'cuda' or 'cpu' (auto-detected if None)
            sample_rate: Target audio sample rate (16kHz)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        
        print(f"Initializing audio emotion inference (device: {self.device})...")
        
        # Load emotion model
        input_dim = 181  # 40 MFCCs + 12 Chroma + 128 Mel + 1 ZCR
        num_classes = 4
        
        self.model = SpeechEmotionCNNBiLSTMAttention(
            input_dim=input_dim,
            num_classes=num_classes
        )
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded audio emotion model from {model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("Using untrained model.")
        else:
            print(f"⚠️  Model not found at {model_path}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio for inference.
        
        Args:
            audio: Audio signal array
            sr: Original sample rate
            
        Returns:
            Preprocessed audio at target sample rate
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to target sample rate if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Trim silence
        audio = trim_silence(audio, self.sample_rate)
        
        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / (np.abs(audio).max() + 1e-8)
        
        return audio
    
    def predict(self, audio: np.ndarray, sr: int = None) -> Dict[str, float]:
        """
        Predict emotions from audio signal.
        
        Args:
            audio: Audio signal array (1D)
            sr: Sample rate (defaults to self.sample_rate)
            
        Returns:
            Dictionary of emotion probabilities
        """
        if sr is None:
            sr = self.sample_rate
        
        # Preprocess audio
        audio = self.preprocess_audio(audio, sr)
        
        # Skip if audio is too short
        if len(audio) < self.sample_rate * 0.5:  # At least 0.5 seconds
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}
        
        # Extract features
        features = extract_features(audio, self.sample_rate)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Convert to dictionary
        emotion_dict = {
            EMOTION_LABELS[i]: float(probs[i])
            for i in range(len(EMOTION_LABELS))
        }
        
        return emotion_dict
    
    def predict_from_bytes(self, audio_bytes: bytes) -> Dict[str, float]:
        """
        Predict emotions from audio bytes (WAV format).
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Dictionary of emotion probabilities
        """
        try:
            # Load audio from bytes
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            return self.predict(audio, sr)
        except Exception as e:
            print(f"Error processing audio bytes: {type(e).__name__}: {e}")
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}
    
    def predict_from_base64(self, base64_string: str) -> Dict[str, float]:
        """
        Predict emotions from base64-encoded audio string.
        
        Args:
            base64_string: Base64-encoded audio string
            
        Returns:
            Dictionary of emotion probabilities
        """
        try:
            import base64 as b64
            audio_bytes = b64.b64decode(base64_string)
            return self.predict_from_bytes(audio_bytes)
        except Exception as e:
            print(f"Error processing base64 audio: {type(e).__name__}: {e}")
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}
    
    def predict_from_file(self, file_path: str) -> Dict[str, float]:
        """
        Predict emotions from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary of emotion probabilities
        """
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            return self.predict(audio, sr)
        except Exception as e:
            print(f"Error loading audio file: {type(e).__name__}: {e}")
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}


if __name__ == "__main__":
    # Test inference
    print("Testing audio emotion inference...")
    
    inference = AudioEmotionInference(
        model_path="models/speech_emotion_model.pth"
    )
    
    print("\n✅ Inference engine initialized successfully!")
    print(f"   Model classes: {list(EMOTION_LABELS.values())}")
    print(f"   Feature dimension: 181")
    print(f"   Sample rate: {TARGET_SR} Hz")
