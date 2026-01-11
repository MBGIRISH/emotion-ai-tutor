"""
Real-time audio emotion inference using trained RAVDESS model.
Extracts MFCC features and runs LSTM/CNN classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, Optional, Tuple
import io
import os
from dotenv import load_dotenv

from backend.utils.audio_utils import AudioProcessor
from backend.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

# Emotion class labels (RAVDESS)
EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}


class AudioEmotionLSTM(nn.Module):
    """LSTM architecture for audio emotion recognition (RAVDESS)"""
    
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=8):
        super(AudioEmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        x = torch.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AudioEmotionInference:
    """Real-time audio emotion inference engine"""
    
    def __init__(
        self,
        model_path: str = "models/audio_emotion_model.pth",
        device: Optional[str] = None,
        sample_rate: int = 22050,
        n_mfcc: int = 13
    ):
        """
        Initialize audio emotion inference.
        
        Args:
            model_path: Path to trained PyTorch model
            device: 'cuda' or 'cpu' (auto-detected if None)
            sample_rate: Target audio sample rate
            n_mfcc: Number of MFCC coefficients
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate, n_mfcc=n_mfcc)
        
        # Load emotion model
        self.model = AudioEmotionLSTM(
            input_size=n_mfcc,
            hidden_size=128,
            num_layers=2,
            num_classes=8
        )
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded audio emotion model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Audio signal array
            
        Returns:
            MFCC features tensor (1, sequence_length, n_mfcc)
        """
        # Ensure correct sample rate
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        if len(audio) == 0:
            # Return zero features if empty
            return torch.zeros(1, 100, self.n_mfcc).to(self.device)
        
        # Resample if needed
        if len(audio) > 0:
            audio = librosa.resample(audio, orig_sr=len(audio), target_sr=self.sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        # Transpose to (time, features)
        mfccs = mfccs.T
        
        # Pad or truncate to fixed length (100 frames ~2-3 seconds)
        target_length = 100
        if mfccs.shape[0] < target_length:
            pad_length = target_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_length), (0, 0)), mode='constant')
        elif mfccs.shape[0] > target_length:
            mfccs = mfccs[:target_length]
        
        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        # Convert to tensor
        tensor = torch.from_numpy(mfccs).float().unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions from audio signal.
        
        Args:
            audio: Audio signal array (1D or 2D)
            
        Returns:
            Dictionary of emotion probabilities
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features)
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
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return self.predict(audio)
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
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
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return self.predict(audio)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}


if __name__ == "__main__":
    # Test inference with sample file
    model_path = os.getenv("AUDIO_MODEL_PATH", "models/audio_emotion_model.pth")
    inference = AudioEmotionInference(model_path=model_path)
    
    # Example: predict from file (if available)
    test_file = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    if os.path.exists(test_file):
        emotions = inference.predict_from_file(test_file)
        print("Predicted emotions:", emotions)
    else:
        print("Test file not found. Please provide a valid audio file path.")

