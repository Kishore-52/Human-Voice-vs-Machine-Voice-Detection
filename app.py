import os
import uuid
import json
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
import joblib

from db import init_db, log_prediction, get_history
from utils import extract_features, get_decision, SR

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Initialize DB
init_db()

# Load models if they exist
try:
    cnn_model = load_model('models/cnn_model.h5')
    rf_model = joblib.load('models/rf_model.joblib')
    svm_model = joblib.load('models/svm_model.joblib')
except Exception as e:
    print(f"Warning: Models not found in 'models/'. Code: {e}")
    cnn_model, rf_model, svm_model = None, None, None


def plot_visualizations(audio_path, uuid_str):
    """
    Plots waveform and spectrogram for the results page.
    Saves them temporarily.
    """
    # Create static uploads structure safely relative to run dir
    save_dir = os.path.join(app.root_path, 'static', 'uploads')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    y, sr = librosa.load(audio_path, sr=SR)
    
    # Light theme plots
    plt.style.use('default')
    
    # 1. Plot Waveform
    plt.figure(figsize=(10, 3), facecolor='none')
    time = np.linspace(0, len(y)/sr, num=len(y))
    plt.plot(time, y, color='#8b5cf6', alpha=0.9) # Purple highlight
    plt.title('Audio Waveform', color='#0f172a')
    plt.xlabel('Time (s)', color='#64748b')
    plt.ylabel('Amplitude', color='#64748b')
    plt.grid(True, linestyle='--', alpha=0.2, color='#64748b')
    plt.tick_params(colors='#64748b')
    
    wave_path = os.path.join(save_dir, f'{uuid_str}_wave.png')
    plt.tight_layout()
    plt.savefig(wave_path, transparent=True)
    plt.close()

    # 2. Plot Spectrogram
    plt.figure(figsize=(10, 3), facecolor='none')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    cb = plt.colorbar(format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color='#64748b')
    cb.outline.set_edgecolor('#64748b')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#64748b')
    
    plt.title('Log-Frequency Spectrogram', color='#0f172a')
    plt.xlabel('Time', color='#64748b')
    plt.ylabel('Hz', color='#64748b')
    plt.tick_params(colors='#64748b')
    
    spec_path = os.path.join(save_dir, f'{uuid_str}_spec.png')
    plt.tight_layout()
    plt.savefig(spec_path, transparent=True)
    plt.close()

    return f'uploads/{uuid_str}_wave.png', f'uploads/{uuid_str}_spec.png'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return redirect(request.url)
        
    file = request.files['audio_file']
    model_choice = request.form.get('model_choice', 'cnn')
    
    if file.filename == '':
        return redirect(url_for('index'))
        
    if file:
        file_id = str(uuid.uuid4())
        save_dir = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        file_path = os.path.join(save_dir, f'{file_id}.wav')
        file.save(file_path)

        # Generate plots
        wave_url, spec_url = plot_visualizations(file_path, file_id)

        # Feature Extraction
        mfcc_mean, multi_ch_img = extract_features(file_path, augment=False, return_all=True)

        if mfcc_mean is None or multi_ch_img is None:
            return "Error parsing audio. Please provide a valid .wav file."

        # Predict
        confidence = 0.0
        label_pred = 0  # 0 = Real, 1 = Fake
        model_name = ""

        if model_choice == 'cnn' and cnn_model:
            pred = cnn_model.predict(np.expand_dims(multi_ch_img, axis=0))[0][0]
            # Since output is sigmoid, pred is prob of class 1
            confidence = float(pred) if pred > 0.5 else float(1.0 - pred)
            label_pred = 1 if pred > 0.5 else 0
            model_name = "CNN (Deep Learning)"
            
        elif model_choice == 'rf' and rf_model:
            probs = rf_model.predict_proba([mfcc_mean])[0]
            label_pred = int(np.argmax(probs))
            confidence = float(probs[label_pred])
            model_name = "Random Forest"
            
        elif model_choice == 'svm' and svm_model:
            probs = svm_model.predict_proba([mfcc_mean])[0]
            label_pred = int(np.argmax(probs))
            confidence = float(probs[label_pred])
            model_name = "SVM"
            
        else:
            return "Selected model not loaded or invalid choice."

        # Interpret Results
        prediction_text = 'Machine Voice (AI-Generated)' if label_pred == 1 else 'Human Voice'
        decision = get_decision(confidence)

        # Log to Database
        log_prediction(file.filename, model_name, prediction_text, confidence, decision)

        # Return info securely via session to results page
        session['result'] = {
            'filename': file.filename,
            'prediction': prediction_text,
            'confidence': f"{confidence * 100:.2f}",
            'decision': decision,
            'model': model_name,
            'wave_url': wave_url,
            'spec_url': spec_url
        }

        return redirect(url_for('results'))


@app.route('/results', methods=['GET'])
def results():
    result = session.get('result')
    if not result:
        return redirect(url_for('index'))
    return render_template('results.html', result=result)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    metrics = {}
    metrics_path = os.path.join(app.root_path, 'models', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    # Process confusion matrices if they exist
    for m in metrics:
        if 'confusion_matrix' in metrics[m]:
            cm = metrics[m]['confusion_matrix']
            # Compute class-specific stats for better UI
            metrics[m]['true_neg'] = cm[0][0]
            metrics[m]['false_pos'] = cm[0][1]
            metrics[m]['false_neg'] = cm[1][0]
            metrics[m]['true_pos'] = cm[1][1]

    return render_template('dashboard.html', metrics=metrics)


@app.route('/history', methods=['GET'])
def history():
    records = get_history()
    return render_template('history.html', history=records)


if __name__ == '__main__':
    # Build models directory mapping ensuring things work smoothly offline
    app.run(debug=True, host='0.0.0.0', port=5000)
