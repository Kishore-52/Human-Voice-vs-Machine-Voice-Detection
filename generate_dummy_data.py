import os
import numpy as np
import scipy.io.wavfile as wav

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_tone(freq, duration, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    # Add noise
    noise = np.random.normal(0, 0.05, tone.shape)
    audio = tone + noise
    # Normalize to int16
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    return audio

def main():
    sr = 16000
    duration = 2.0
    
    # Create dataset directories
    real_dir = os.path.join('dataset', 'real')
    fake_dir = os.path.join('dataset', 'fake')
    
    ensure_dir(real_dir)
    ensure_dir(fake_dir)
    
    # Generate 50 'real' samples (simulated by low freq tones + noise)
    print("Generating real dummy samples...")
    for i in range(50):
        audio = generate_tone(np.random.uniform(200, 400), duration, sr)
        wav.write(os.path.join(real_dir, f'real_{i}.wav'), sr, audio)
        
    # Generate 50 'fake' samples (simulated by high freq tones + noise)
    print("Generating fake dummy samples...")
    for i in range(50):
        audio = generate_tone(np.random.uniform(800, 1000), duration, sr)
        wav.write(os.path.join(fake_dir, f'fake_{i}.wav'), sr, audio)

    # Subdirs for models and static
    ensure_dir('models')
    ensure_dir('static/uploads')
    
    print("Dummy dataset and models/static directories created successfully.")

if __name__ == "__main__":
    main()
