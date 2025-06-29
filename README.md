# AI Dance Generator 🕺

A cutting-edge AI-powered dance generation platform that analyzes text emotions and creates personalized 3D character dances in real-time.

## 🌟 Features

- **Real-time Emotion Analysis**: Advanced RoBERTa-based multi-label emotion classification
- **3D Character Animation**: Dynamic dance animations based on detected emotions
- **Multi-User Support**: Each user gets their own isolated experience
- **Responsive Design**: Optimized for all devices with futuristic UI
- **Intensity Prediction**: SVR model for emotion intensity scoring

## 🧠 Machine Learning Models

### RoBERTa Emotion Classification Model

Our primary emotion detection system uses a fine-tuned RoBERTa model trained on the GoEmotions dataset, achieving state-of-the-art performance in multi-label emotion classification.

#### Model Architecture
- **Base Model**: `roberta-base` (125M parameters)
- **Task**: Multi-label classification (6 emotion categories)
- **Training Data**: GoEmotions dataset with 6 macro-emotion categories
- **Training Duration**: 4 epochs (~1.5 hours)

#### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 86.61% | Overall classification accuracy |
| **Hamming Loss** | 0.134 | Multi-label classification error rate |
| **Jaccard Score** | 0.498 | Intersection over union for multi-label predictions |
| **Training Loss** | 0.261 | Final training loss |
| **Validation Loss** | 0.319 | Final validation loss |

#### Emotion Categories & Performance

| Emotion Category | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| **Positive Affect Joy** | 0.74 | 0.76 | 0.75 | 12,380 |
| **Sadness Low Arousal** | 0.59 | 0.41 | 0.48 | 2,859 |
| **Anger High Arousal** | 0.57 | 0.45 | 0.50 | 5,182 |
| **Fear Anxiety** | 0.55 | 0.37 | 0.44 | 709 |
| **Surprise Epistemic** | 0.58 | 0.34 | 0.43 | 4,348 |

#### Training Efficiency
- **Samples per Second**: ~486
- **Steps per Second**: ~30
- **Runtime per Epoch**: ~65 seconds
- **Total Training Time**: 1 hour 29 minutes

### SVR Intensity Model

Complementary Support Vector Regression model for predicting emotion intensity scores.

## 🎭 Emotion-to-Dance Mapping

The system intelligently maps detected emotions to appropriate dance animations:

- **Positive Affect Joy** → Happy dance animation
- **Sadness Low Arousal** → Sad dance animation  
- **Anger High Arousal** → Angry dance animation
- **Fear Anxiety** → Fearful dance animation
- **Surprise Epistemic** → Surprised dance animation
- **Neutral** → Normal/idle animation

## 🏗️ Technical Architecture

### Frontend
- **React.js** with Three.js for 3D rendering
- **Modern CSS** with glassmorphism effects
- **Responsive design** optimized for all devices
- **Real-time animations** with smooth transitions

### Backend
- **Flask API** for model inference
- **Transformers library** for RoBERTa model
- **Scikit-learn** for SVR intensity prediction
- **CORS enabled** for cross-origin requests

### Multi-User Support
- **Client-side state management** ensures user isolation
- **Independent API calls** per user session
- **No shared data** between users
- **Real-time processing** for each request

## 🚀 Live Demo

Experience the AI Dance Generator in action! Simply type how you're feeling and watch as your emotions transform into beautiful 3D dance animations.

## 🔬 Model Training Details

The RoBERTa model was fine-tuned using:
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Optimizer**: AdamW
- **Loss Function**: Binary Cross-Entropy with Logits
- **Evaluation Strategy**: Per-epoch validation

## 📊 Model Robustness

- **Threshold-based prediction**: Ensures at least one emotion is always detected
- **Fallback mechanism**: Uses highest probability emotion when none exceed threshold
- **Multi-label support**: Can detect multiple emotions simultaneously
- **Priority-based selection**: Intelligent emotion selection for animation

## 🎯 Use Cases

- **Entertainment**: Interactive dance experiences
- **Emotion Recognition**: Real-time sentiment analysis
- **Creative Expression**: Visual representation of emotions
- **Educational**: Understanding emotion-text relationships

## 🔧 Technologies Used

- **Frontend**: React, Three.js, CSS3
- **Backend**: Flask, Python
- **ML**: Transformers, PyTorch, Scikit-learn
- **3D Models**: GLB format with Mixamo animations
- **Deployment**: Public hosting with responsive design

---

*Powered by state-of-the-art AI models and cutting-edge web technologies.* 