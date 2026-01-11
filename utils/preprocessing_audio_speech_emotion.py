"""
Preprocessing utilities for CREMA-D, TESS, and SAVEE datasets.
Speech Emotion Recognition pipeline with 4 learning-centric classes.
"""

import numpy as np
import os
import glob
import librosa
import soundfile as sf
from typing import Tuple, List, Dict
import re
from scipy import signal
import argparse


# Target sample rate
TARGET_SR = 16000

# Emotion label mapping to 4 learning-centric classes
EMOTION_TO_CLASS = {
    # Happy
    'happy': 0,
    'happiness': 0,
    'hap': 0,
    
    # Neutral
    'neutral': 1,
    'neu': 1,
    
    # Confused (Sad + Neutral-like confusion)
    'sad': 2,
    'sadness': 2,
    
    # Frustrated (Anger + Fear)
    'angry': 3,
    'anger': 3,
    'ang': 3,
    'fear': 3,
    'fearful': 3,
    'disgust': 3,  # Map disgust to frustrated
}

CLASS_LABELS = {
    0: 'happy',
    1: 'neutral',
    2: 'confused',
    3: 'frustrated'
}


def trim_silence(audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Trim silence from audio using librosa.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        
    Returns:
        Trimmed audio signal
    """
    # Trim leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20, frame_length=frame_length, hop_length=hop_length)
    return audio_trimmed


def load_crema_files(crema_dir: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load CREMA-D audio files.
    Format: ActorXX_Emotion_Statement_Sentence_Intensity.wav
    Emotions: ANG, DIS, FEA, HAP, NEU, SAD
    """
    audio_files = glob.glob(os.path.join(crema_dir, "*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {crema_dir}")
    
    features = []
    labels = []
    file_paths = []
    
    emotion_map = {
        'HAP': 0,  # happy
        'NEU': 1,  # neutral
        'SAD': 2,  # confused (sad)
        'ANG': 3,  # frustrated (angry)
        'DIS': 3,  # frustrated (disgust)
        'FEA': 3,  # frustrated (fear)
    }
    
    print(f"Loading CREMA-D dataset from {crema_dir}...")
    for filepath in audio_files:
        try:
            filename = os.path.basename(filepath)
            # Extract emotion from filename (second part after underscore)
            parts = filename.split('_')
            if len(parts) >= 2:
                emotion_code = parts[1].upper()
                if emotion_code in emotion_map:
                    # Load and process audio
                    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
                    
                    # Trim silence
                    audio = trim_silence(audio, sr)
                    
                    # Skip if audio is too short after trimming
                    if len(audio) < sr * 0.5:  # At least 0.5 seconds
                        continue
                    
                    features.append(audio)
                    labels.append(emotion_map[emotion_code])
                    file_paths.append(filepath)
        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue
    
    print(f"Loaded {len(features)} samples from CREMA-D")
    return features, labels, file_paths


def load_tess_files(tess_dir: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load TESS dataset.
    Format: Can be in subdirectories (YAF_disgust/, OAF_happy/, etc.) or flat structure
    Emotions: angry, disgust, fear, happy, neutral, pleasant_surprise, sad
    """
    # Try subdirectory structure first
    audio_files = []
    for root, dirs, files in os.walk(tess_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {tess_dir}")
    
    features = []
    labels = []
    file_paths = []
    
    print(f"Loading TESS dataset from {tess_dir}...")
    for filepath in audio_files:
        try:
            filename = os.path.basename(filepath)
            filename_lower = filename.lower()
            path_lower = filepath.lower()
            
            # Extract emotion from filename or path
            emotion_label = None
            if 'happy' in filename_lower or 'happy' in path_lower:
                emotion_label = 0  # happy
            elif 'neutral' in filename_lower or 'neutral' in path_lower:
                emotion_label = 1  # neutral
            elif 'sad' in filename_lower or 'sad' in path_lower:
                emotion_label = 2  # confused (sad)
            elif 'angry' in filename_lower or 'angry' in path_lower:
                emotion_label = 3  # frustrated
            elif 'disgust' in filename_lower or 'disgust' in path_lower:
                emotion_label = 3  # frustrated
            elif 'fear' in filename_lower or 'fear' in path_lower:
                emotion_label = 3  # frustrated
            
            if emotion_label is not None:
                # Load and process audio
                audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
                
                # Trim silence
                audio = trim_silence(audio, sr)
                
                # Skip if audio is too short after trimming
                if len(audio) < sr * 0.5:
                    continue
                
                features.append(audio)
                labels.append(emotion_label)
                file_paths.append(filepath)
        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue
    
    print(f"Loaded {len(features)} samples from TESS")
    return features, labels, file_paths


def load_savee_files(savee_dir: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load SAVEE dataset.
    Format: DC_a01.wav, JE_h01.wav, etc.
    Emotions: a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise
    """
    audio_files = glob.glob(os.path.join(savee_dir, "*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {savee_dir}")
    
    features = []
    labels = []
    file_paths = []
    
    print(f"Loading SAVEE dataset from {savee_dir}...")
    for filepath in audio_files:
        try:
            filename = os.path.basename(filepath)
            # Extract emotion code (letter(s) before underscore)
            parts = filename.split('_')
            if len(parts) >= 2:
                emotion_code = parts[1][0:2].lower()
                
                emotion_label = None
                if emotion_code.startswith('h'):
                    emotion_label = 0  # happy
                elif emotion_code.startswith('n'):
                    emotion_label = 1  # neutral
                elif emotion_code.startswith('sa'):
                    emotion_label = 2  # confused (sad)
                elif emotion_code.startswith('a') or emotion_code.startswith('d') or emotion_code.startswith('f'):
                    emotion_label = 3  # frustrated
                
                if emotion_label is not None:
                    # Load and process audio
                    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
                    
                    # Trim silence
                    audio = trim_silence(audio, sr)
                    
                    # Skip if audio is too short after trimming
                    if len(audio) < sr * 0.5:
                        continue
                    
                    features.append(audio)
                    labels.append(emotion_label)
                    file_paths.append(filepath)
        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue
    
    print(f"Loaded {len(features)} samples from SAVEE")
    return features, labels, file_paths


def add_background_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add random Gaussian noise to audio."""
    noise = np.random.randn(len(audio)) * noise_factor
    return audio + noise


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Time stretch audio by rate."""
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_audio(audio: np.ndarray, sr: int) -> List[np.ndarray]:
    """
    Apply data augmentation to audio.
    Returns list of augmented audio samples (original + augmentations).
    """
    augmented = [audio]  # Include original
    
    # Add background noise
    augmented.append(add_background_noise(audio, noise_factor=0.005))
    
    # Pitch shift (+/- 2 semitones)
    augmented.append(pitch_shift(audio, sr, n_steps=2.0))
    augmented.append(pitch_shift(audio, sr, n_steps=-2.0))
    
    # Time stretch (0.9-1.1)
    augmented.append(time_stretch(audio, rate=0.9))
    augmented.append(time_stretch(audio, rate=1.1))
    
    return augmented


def extract_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract comprehensive features from audio.
    Returns: 40 MFCCs + Chroma + Mel spectrogram + ZCR (flattened and concatenated)
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


def load_combined_datasets(
    crema_dir: str,
    tess_dir: str,
    savee_dir: str,
    apply_augmentation: bool = True,
    cache_dir: str = "data/processed"
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load and combine CREMA-D, TESS, and SAVEE datasets.
    
    Args:
        crema_dir: Directory containing CREMA-D audio files
        tess_dir: Directory containing TESS audio files
        savee_dir: Directory containing SAVEE audio files
        apply_augmentation: Whether to apply data augmentation
        cache_dir: Directory to cache processed data
        
    Returns:
        Tuple of (features_list, labels_list)
    """
    cache_file = os.path.join(cache_dir, "speech_emotion_4class_processed.npz")
    
    # Check for cached data
    if os.path.exists(cache_file) and not apply_augmentation:
        print(f"Loading cached processed data from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'].tolist(), data['labels'].tolist()
    
    print("=" * 70)
    print("Loading Speech Emotion Recognition Datasets")
    print("=" * 70)
    
    all_features = []
    all_labels = []
    
    # Load CREMA-D
    if os.path.exists(crema_dir):
        crema_features, crema_labels, _ = load_crema_files(crema_dir)
        all_features.extend(crema_features)
        all_labels.extend(crema_labels)
    
    # Load TESS
    if os.path.exists(tess_dir):
        tess_features, tess_labels, _ = load_tess_files(tess_dir)
        all_features.extend(tess_features)
        all_labels.extend(tess_labels)
    
    # Load SAVEE
    if os.path.exists(savee_dir):
        savee_features, savee_labels, _ = load_savee_files(savee_dir)
        all_features.extend(savee_features)
        all_labels.extend(savee_labels)
    
    if not all_features:
        raise ValueError("No datasets found. Please provide at least one dataset directory.")
    
    print(f"\nTotal samples before augmentation: {len(all_features)}")
    
    # Apply augmentation if requested
    if apply_augmentation:
        print("\nApplying data augmentation...")
        augmented_features = []
        augmented_labels = []
        
        for audio, label in zip(all_features, all_labels):
            augmented_samples = augment_audio(audio, TARGET_SR)
            augmented_features.extend(augmented_samples)
            augmented_labels.extend([label] * len(augmented_samples))
        
        all_features = augmented_features
        all_labels = augmented_labels
        print(f"Total samples after augmentation: {len(all_features)}")
    
    # Extract features
    print("\nExtracting features...")
    feature_vectors = []
    for i, audio in enumerate(all_features):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_features)} samples...")
        features = extract_features(audio, TARGET_SR)
        feature_vectors.append(features)
    
    print(f"\nFeature extraction complete!")
    print(f"Feature vector shape: {feature_vectors[0].shape}")
    
    # Print class distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        percentage = count / len(all_labels) * 100
        print(f"  {CLASS_LABELS[label]}: {count} ({percentage:.1f}%)")
    
    # Cache processed data (only if no augmentation)
    if not apply_augmentation:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_file, features=feature_vectors, labels=all_labels)
        print(f"\nCached processed data to {cache_file}")
    
    return feature_vectors, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Speech Emotion Recognition datasets")
    parser.add_argument("--crema-dir", required=True, help="CREMA-D dataset directory")
    parser.add_argument("--tess-dir", required=True, help="TESS dataset directory")
    parser.add_argument("--savee-dir", required=True, help="SAVEE dataset directory")
    parser.add_argument("--cache-dir", default="data/processed", help="Cache directory")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    
    args = parser.parse_args()
    
    features, labels = load_combined_datasets(
        crema_dir=args.crema_dir,
        tess_dir=args.tess_dir,
        savee_dir=args.savee_dir,
        apply_augmentation=args.augment,
        cache_dir=args.cache_dir
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Total samples: {len(features)}")
    print(f"Feature dimension: {features[0].shape[0]}")

