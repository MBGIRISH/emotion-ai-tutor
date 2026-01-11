# Emotion-Aware AI Tutor

A production-ready multimodal AI tutoring system that adapts to student emotions in real-time using face and voice emotion detection, engagement tracking, and LLM-powered adaptive responses.

## 🎯 System Overview

This system combines:
- **Real-time face emotion detection** using FER-2013 trained models
- **Real-time voice emotion detection** using RAVDESS trained models
- **Engagement & confusion tracking** via MediaPipe (blink rate, gaze direction, head pose)
- **Adaptive LLM tutoring** that responds to student emotional state
- **Dual frontend options**: React web app and Streamlit dashboard for real-time visualization and interaction

## 📁 Project Structure

```
emotion-aware-ai-tutor/
 ├── README.md
 ├── requirements.txt
 ├── package.json                # React frontend dependencies
 ├── data/
 │    ├── fer2013/               # FER-2013 dataset (upload here)
 │    ├── ravdess/               # RAVDESS dataset (upload here)
 │    ├── processed/             # Preprocessed data cache
 │    └── instructions.md        # Dataset upload instructions
 ├── models/
 │    ├── face_emotion_model.pth # ⚠️ Needs training
 │    ├── audio_emotion_model.pth # ✅ Trained
 │    └── README.md
 ├── backend/
 │    ├── api.py                 # FastAPI server
 │    ├── inference_face.py      # Real-time face inference
 │    ├── inference_audio.py     # Real-time audio inference
 │    ├── engagement.py          # Engagement & confusion logic
 │    ├── tutor.py               # LLM adaptation engine
 │    └── utils/
 │          ├── face_detector.py
 │          ├── audio_utils.py
 │          └── logger.py
 ├── app/                        # Streamlit frontend
 │    ├── streamlit_app.py       # Main dashboard
 │    └── components/
 │          ├── emotion_meter.py
 │          ├── voice_gauge.py
 │          ├── engagement_bar.py
 │          └── tutor_chatbox.py
 ├── components/                 # React frontend components
 │    ├── MetricsPanel.tsx
 │    ├── TutorChat.tsx
 │    └── WebcamFeed.tsx
 ├── App.tsx                      # React main app
 ├── notebooks/
 │    ├── train_face_emotion.ipynb
 │    ├── train_audio_emotion.ipynb
 │    ├── evaluate_models.ipynb
 │    └── demo_predictions.ipynb
 └── utils/
      ├── preprocessing_face.py
      ├── preprocessing_audio.py
      └── common.py
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

**FER-2013 Dataset:**
- Download FER-2013 from: https://www.kaggle.com/datasets/msambare/fer2013
- Place files in `data/fer2013/`:
  - Option A: `train.csv` and `test.csv` files
  - Option B: Folder structure with `train/` and `test/` subdirectories
- See `data/instructions.md` for detailed instructions

**RAVDESS Dataset:**
- Download RAVDESS from: https://zenodo.org/record/1188976
- Extract the complete dataset to `data/ravdess/`
- Should contain audio files organized by actor, emotion, and statement
- See `data/instructions.md` for detailed instructions

### 3. Data Preprocessing

```bash
# Preprocess FER-2013 data
python utils/preprocessing_face.py

# Preprocess RAVDESS data
python utils/preprocessing_audio.py
```

### 4. Model Training

**Train Face Emotion Model:**
```bash
# Open Jupyter notebook
jupyter notebook notebooks/train_face_emotion.ipynb
# Run all cells to train and save model to models/face_emotion_model.pth
```

**Train Audio Emotion Model:**
```bash
jupyter notebook notebooks/train_audio_emotion.ipynb
# Run all cells to train and save model to models/audio_emotion_model.pth
```

**Evaluate Models:**
```bash
jupyter notebook notebooks/evaluate_models.ipynb
```

### 5. Launch the System

**Start FastAPI Backend:**
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Option A: Launch React Frontend (Recommended)**
```bash
# Install Node.js dependencies (first time only)
npm install

# Start React development server
npm run dev
```

The React app will open at `http://localhost:3000`

**Option B: Launch Streamlit Dashboard:**
```bash
streamlit run app/streamlit_app.py
```

The Streamlit dashboard will open at `http://localhost:8501`

## 🧠 How It Works

### Face Emotion Detection
- Uses MediaPipe for face detection and landmark extraction
- Runs trained CNN/MobileNetV2 model on detected faces
- Outputs 7 emotion probabilities: Happy, Sad, Neutral, Angry, Fear, Disgust, Surprise

### Voice Emotion Detection
- Captures microphone input in real-time
- Extracts MFCC (Mel-frequency cepstral coefficients) features
- Runs trained LSTM/CNN audio classifier
- Outputs emotion probabilities from RAVDESS classes

### Engagement Score Calculation

The engagement score (0-100) combines multiple signals:

1. **Blink Rate** (via Eye Aspect Ratio from MediaPipe)
   - Normal: 15-20 blinks/min
   - High (>30/min): Possible confusion or fatigue
   - Low (<10/min): Possible disengagement

2. **Gaze Direction** (head pose estimation)
   - Looking at screen: +engagement
   - Looking away: -engagement
   - Pitch/yaw/roll angles indicate attention

3. **Emotion Trends**
   - Positive emotions (happy, surprise): +engagement
   - Negative emotions (sad, angry, fear): -engagement
   - Neutral: baseline

4. **Confusion Detection**
   - High blink rate + negative emotions + gaze away = confusion
   - Triggers adaptive tutoring responses

**Formula:**
```
engagement = (
    0.3 * (1 - normalized_blink_deviation) +
    0.3 * gaze_attention_score +
    0.4 * emotion_positive_score
) * 100
```

### LLM Tutor Adaptation

The `tutor.py` module uses emotion and engagement data to:

1. **Detect Confusion**: High confusion triggers simplified explanations
2. **Motivate**: Low engagement triggers encouraging messages
3. **Adapt Pace**: Adjusts explanation complexity based on emotional state
4. **Provide Hints**: Offers contextual hints when student is stuck

**Example Adaptive Responses:**
- Confusion detected → "Let me break this down into simpler steps..."
- Low engagement → "I see you might be finding this challenging. Let's try a different approach!"
- Positive emotions → "Great progress! Let's continue building on this concept."

## 📊 Frontend Features

### React Web App (Primary)
- **Modern UI**: Beautiful, responsive interface built with React and TypeScript
- **Webcam Feed**: Real-time video preview with face detection overlay
- **Metrics Panel**: Live emotion probabilities, engagement scores, and confusion levels
- **Tutor Chat**: Interactive chat with Gemini 2.5 Flash-powered adaptive tutor
- **Session Analytics**: Historical emotion and engagement trends with charts
- **API Key Management**: Integrated Google AI Studio API key selector

### Streamlit Dashboard (Alternative)
- **Webcam Feed**: Real-time video preview with face detection overlay
- **Emotion Bar Graph**: Live emotion probabilities for 7 face emotions
- **Voice Emotion Meter**: Real-time audio emotion visualization
- **Engagement Score Bar**: 0-100 engagement level with color coding
- **Confusion Alerts**: Pop-up notifications when confusion is detected
- **Tutor Chatbox**: Interactive chat with adaptive LLM tutor
- **Session Analytics**: Historical emotion and engagement trends

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# LLM API Keys (choose one)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Model Paths
FACE_MODEL_PATH=models/face_emotion_model.pth
AUDIO_MODEL_PATH=models/audio_emotion_model.pth

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Settings
STREAMLIT_PORT=8501
```

## 🧪 Testing

```bash
# Test face inference
python backend/inference_face.py

# Test audio inference
python backend/inference_audio.py

# Test engagement calculation
python -c "from backend.engagement import compute_engagement; print(compute_engagement(...))"
```

## 📝 Development

### Adding New Features

- **New emotion classes**: Modify model architecture in training notebooks
- **New engagement metrics**: Extend `backend/engagement.py`
- **Custom LLM prompts**: Modify `backend/tutor.py`

### Code Style

```bash
# Format code
black backend/ app/ utils/

# Lint code
flake8 backend/ app/ utils/
```

## ⚠️ Current Status

**Model Training Status:**
- ✅ **Audio Emotion Model**: Trained and available (`models/audio_emotion_model.pth`)
- ⚠️ **Face Emotion Model**: **NOT TRAINED** - needs to be trained before use

**To train the face emotion model:**
```bash
# Option 1: Use automated training script
python train_models_fixed.py

# Option 2: Use Jupyter notebook
jupyter notebook notebooks/train_face_emotion.ipynb
# Run all cells - model will save to models/face_emotion_model.pth
```

**Import Path Issues:**
- ✅ Fixed: All backend imports now use `backend.utils.*` consistently

## 🐛 Troubleshooting

**Webcam not detected:**
- Check camera permissions
- Try different camera index in `inference_face.py`

**Microphone not working:**
- Check system audio permissions
- Install `portaudio` (macOS: `brew install portaudio`)

**Model not found:**
- Ensure models are trained and saved to `models/` directory
- Check model paths in `.env` file
- **Face model must be trained** - see training instructions above

**Import errors:**
- All imports have been fixed to use `backend.utils.*` paths
- If you see import errors, ensure you're running from the project root

**React frontend not starting:**
- Ensure Node.js is installed (`node --version`)
- Run `npm install` to install dependencies
- Check that port 3000 is available

**CUDA/GPU issues:**
- Models default to CPU; modify device in inference scripts for GPU

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.
