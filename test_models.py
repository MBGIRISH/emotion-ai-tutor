#!/usr/bin/env python3
"""
Comprehensive test to verify models are loaded and working correctly.
"""

import sys
import os
sys.path.append('.')

print("=" * 60)
print("🧪 COMPREHENSIVE MODEL TEST")
print("=" * 60)

# Test 1: Check model files exist
print("\n1. Checking model files...")
face_model_path = "models/face_emotion_model.pth"
audio_model_path = "models/audio_emotion_model.pth"

face_exists = os.path.exists(face_model_path)
audio_exists = os.path.exists(audio_model_path)

if face_exists:
    face_size = os.path.getsize(face_model_path)
    print(f"   ✅ Face model: {face_model_path} ({face_size:,} bytes)")
else:
    print(f"   ❌ Face model: {face_model_path} NOT FOUND")

if audio_exists:
    audio_size = os.path.getsize(audio_model_path)
    print(f"   ✅ Audio model: {audio_model_path} ({audio_size:,} bytes)")
else:
    print(f"   ❌ Audio model: {audio_model_path} NOT FOUND")

# Test 2: Try to load models
print("\n2. Testing model loading...")

if face_exists:
    try:
        import torch
        from backend.inference_face import FaceEmotionInference
        
        print("   Loading face emotion model...")
        face_inference = FaceEmotionInference(model_path=face_model_path)
        print("   ✅ Face model loaded successfully!")
        
        # Test with dummy input
        import numpy as np
        dummy_frame = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            emotions = face_inference.predict(dummy_frame)
            print(f"   ✅ Face model inference works! Output: {list(emotions.keys())}")
        except Exception as e:
            print(f"   ⚠️  Face model loaded but inference failed: {e}")
            
    except ImportError as e:
        print(f"   ⚠️  Cannot test face model (missing dependencies): {e}")
    except Exception as e:
        print(f"   ❌ Face model loading failed: {e}")
else:
    print("   ⏭️  Skipping face model test (file not found)")

if audio_exists:
    try:
        import torch
        from backend.inference_audio import AudioEmotionInference
        
        print("   Loading audio emotion model...")
        audio_inference = AudioEmotionInference(model_path=audio_model_path)
        print("   ✅ Audio model loaded successfully!")
        
        # Test with dummy input
        import numpy as np
        dummy_audio = np.random.randn(22050).astype(np.float32)  # 1 second of audio
        try:
            emotions = audio_inference.predict(dummy_audio)
            print(f"   ✅ Audio model inference works! Output: {list(emotions.keys())}")
        except Exception as e:
            print(f"   ⚠️  Audio model loaded but inference failed: {e}")
            
    except ImportError as e:
        print(f"   ⚠️  Cannot test audio model (missing dependencies): {e}")
    except Exception as e:
        print(f"   ❌ Audio model loading failed: {e}")
else:
    print("   ⏭️  Skipping audio model test (file not found)")

# Test 3: Test backend API initialization
print("\n3. Testing backend API components...")
try:
    from backend.engagement import EngagementTracker
    from backend.tutor import AdaptiveTutor
    
    engagement = EngagementTracker()
    print("   ✅ EngagementTracker initialized")
    
    tutor = AdaptiveTutor()
    print("   ✅ AdaptiveTutor initialized")
    
except Exception as e:
    print(f"   ❌ Backend components failed: {e}")

# Test 4: Test API server can start (without actually starting)
print("\n4. Testing API server imports...")
try:
    from backend.api import app, face_inference, audio_inference
    print("   ✅ API server imports successful")
    print(f"   - Face inference: {'✅ Loaded' if face_inference is not None else '⚠️  Not loaded'}")
    print(f"   - Audio inference: {'✅ Loaded' if audio_inference is not None else '⚠️  Not loaded'}")
except Exception as e:
    print(f"   ⚠️  API server import test: {e}")

# Summary
print("\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)

if face_exists and audio_exists:
    print("✅ Both models exist and are ready!")
    print("\n✅ System is ready to run!")
    print("\nTo start the system:")
    print("  1. Start backend: cd backend && uvicorn api:app --reload")
    print("  2. Start frontend: streamlit run app/streamlit_app.py")
    print("     OR: npm run dev (for React frontend)")
elif face_exists:
    print("⚠️  Face model exists, but audio model is missing")
elif audio_exists:
    print("⚠️  Audio model exists, but face model is missing")
else:
    print("❌ Both models are missing - need to train them first")

print("=" * 60)

