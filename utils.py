import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Target sample rate
SR = 16000
DURATION = 2.0  # seconds
SAMPLES = int(SR * DURATION)

# For exactly 128 frames: hop_length = SAMPLES / 128 = 250
HOP_LENGTH = 250
N_MELS = 128
N_MFCC = 40

def augment_audio(y):
    """Apply noise addition and time shifting."""
    y_aug = y.copy()
    
    # Randomly add noise
    if np.random.rand() > 0.5:
        noise = np.random.randn(len(y_aug))
        y_aug = y_aug + 0.005 * noise
        
    # Randomly shift time
    if np.random.rand() > 0.5:
        shift = np.random.randint(SR // 10) # max shift 100ms
        direction = np.random.choice(['right', 'left'])
        if direction == 'right':
            y_aug = np.roll(y_aug, shift)
        else:
            y_aug = np.roll(y_aug, -shift)
            
    return y_aug

def extract_features(file_path, augment=False, return_all=True):
    """
    Extract MFCC and Mel Spectrogram.
    Returns:
    - mfcc_features: 1D array of mean MFCC for RF and SVM
    - multi_channel_img: (128, 128, 2) array for CNN
    """
    try:
        y, sr = librosa.load(file_path, sr=SR)
        
        # Pad or truncate to ensure exactly SAMPLES length
        if len(y) < SAMPLES:
            y = librosa.util.pad_center(y, size=SAMPLES)
        else:
            y = y[:SAMPLES]
            
        if augment:
            y = augment_audio(y)
            
        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=N_MFCC)
        
        # Ensure exact shape 128x128 for mel
        mel_spec_fixed = librosa.util.fix_length(mel_spec_db, size=128, axis=1)
        
        # Ensure exact shape 40x128 for mfcc
        mfcc_fixed = librosa.util.fix_length(mfcc, size=128, axis=1)
        
        # Normalization
        mel_spec_fixed = (mel_spec_fixed - np.min(mel_spec_fixed)) / (np.max(mel_spec_fixed) - np.min(mel_spec_fixed) + 1e-6)
        mfcc_fixed = (mfcc_fixed - np.min(mfcc_fixed)) / (np.max(mfcc_fixed) - np.min(mfcc_fixed) + 1e-6)
        
        # 1D features for ML models (mean over time)
        mfcc_mean = np.mean(mfcc_fixed, axis=1) # Shape (40,)
        
        # Prepare multi-channel image for CNN
        # Pad MFCC with zeros to reach (128, 128)
        mfcc_padded = np.zeros((128, 128))
        mfcc_padded[:N_MFCC, :] = mfcc_fixed
        
        # Combine (128, 128, 2)
        multi_channel_img = np.stack([mel_spec_fixed, mfcc_padded], axis=-1)
        
        if return_all:
            return mfcc_mean, multi_channel_img
        else:
            return multi_channel_img
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def get_decision(confidence):
    """Return logic for intelligent decision system."""
    if confidence >= 0.85:
        return 'High Confidence ✅'
    elif 0.65 <= confidence < 0.85:
        return 'Needs Review ❓'
    else:
        return 'Uncertain ⚠'
