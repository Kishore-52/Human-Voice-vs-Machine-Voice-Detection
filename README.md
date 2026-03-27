# 🎧 Human vs Machine Voice Classification System

> 🚀 An Intelligent Offline AI Solution to Detect Human vs AI-Generated Voices

---

## 🌟 Overview

With the rapid rise of AI-generated voices and deepfake audio, identifying whether a voice is real or synthetic has become a critical challenge.

This project presents a **highly accurate, fully offline AI system** that classifies audio samples as:

- 🧑 Human Voice  
- 🤖 Machine Generated Voice  

The system combines **Deep Learning and Machine Learning techniques** to deliver fast, reliable, and interpretable results.

---

## 🎯 Key Highlights

- 🔍 Accurate classification of human vs AI-generated voices  
- ⚡ Real-time prediction using a simple web interface  
- 🧠 Hybrid model approach (CNN + ML models)  
- 📊 Confidence-based intelligent decision system  
- 🔒 Fully offline (no internet required)  
- 📈 High performance with optimized feature extraction  

---

## 📊 Dataset

- 📁 **FoR – Fake or Real Dataset**
- 🎵 ~198,000 audio samples  
- 🧑 Human Voices: ~111,000  
- 🤖 Machine Voices: ~87,000  
- ⏱️ Audio Duration: 2 seconds  
- 🎧 Format: WAV  

---

## 🧪 Methodology

### 🔊 1. Audio Preprocessing
- Resampling audio to standard frequency  
- Noise handling and normalization  

### 🎼 2. Feature Extraction
- **MFCC (Mel-Frequency Cepstral Coefficients)**  
- **Mel Spectrograms**  

These features capture **speech patterns and frequency characteristics** effectively.

---

### 🧠 3. Model Training

We trained and compared multiple models:

| Model | Description |
|------|------------|
| CNN | Deep Learning model for spectrogram image analysis |
| Random Forest | Ensemble learning method |
| SVM | Classification using decision boundaries |

---

## 📈 Model Performance

| Model           | Accuracy |
|----------------|----------|
| 🥇 CNN          | **99.9%** |
| 🥈 Random Forest| 94.8%    |
| 🥉 SVM          | 89.0%    |

👉 CNN performs best due to its ability to learn spatial patterns from spectrogram data.

---

## 🧠 Decision Intelligence

To improve reliability, we introduce a **confidence-based decision system**:

| Confidence Score | Decision |
|------------------|--------|
| ≥ 0.85 | ✅ High Confidence |
| 0.65 – 0.85 | ⚠️ Needs Review |
| < 0.65 | ❌ Uncertain |

---

## 🎥 Demo Video

▶️ Watch Full Demo:  
https://youtu.be/tdazqQU-XXI?si=E_t0Zdq3z_zrcYAb

---

## 🖥️ Application Features

- 📂 Upload audio file  
- 📊 View waveform & spectrogram  
- 🔍 Get prediction (Human / Machine)  
- 📈 Confidence score display  
- 🧠 Intelligent decision output  

---

## ⚙️ Tech Stack

- 🐍 Python  
- 🤖 TensorFlow / Keras  
- 📊 Scikit-learn  
- 🎧 Librosa  
- 🌐 Flask  

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install -r requirements.txt
