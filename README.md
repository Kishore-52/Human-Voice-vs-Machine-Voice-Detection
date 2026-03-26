<<<<<<< HEAD
# Human vs Machine Voice Classification System (PRO VERSION)

An advanced, production-ready, fully **OFFLINE** AI system to detect whether an audio snippet is a Human Voice or an AI-generated Machine Voice. Designed with strict adherence to local processing and maximum code modularity.

## Architecture

* **Database**: Local SQLite tracking inference history.
* **Pre-processing (`utils.py`)**: `librosa` extracting 40 MFCCs and a 128x128 2-Channel matrix combining Mel Spectrogram and padded MFCC arrays. Additive noise and time-shift augmentations implemented.
* **Models**: 
  1. **CNN (Deep Learning)**: Evaluates multi-channel (Spectrogram + MFCC) inputs. 
  2. **Random Forest (ML)**: Classifies 1D MFCC inputs.
  3. **SVM (ML)**: RBF Kernel acting on MFCC inputs.
* **Frontend (`app.py` / `templates`)**: Flask framework serving a premium, responsive glassmorphism UI dynamically with drag-and-drop input routing handling confidence computations.

## Initial Setup (Operating Offline)

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Supply Dataset**
Place `.wav` files into the respective folders:
* `dataset/real/`
* `dataset/fake/`

*Note: If you do not have the FoR-2seconds dataset right now, you can generate dummy diagnostic audio files by running:* 
```bash
python generate_dummy_data.py
```

3. **Train Models and Evaluate**
```bash
python train.py
```
This routine splits the loaded dataset, extracts static shapes (128x128 arrays via zero-padding framing adjustments), saves `h5`/`joblib` bundles into `models/`, and generates confusion matrix & accuracy JSONs alongside visual bars to `static/`.

4. **Launch Application**
```bash
python app.py
```
Open `http://localhost:5000` in your web browser. 

## Intelligent Decision Flow

Once analyzed via `/predict`, the system returns the respective Model's confidence probability and filters through intelligent boundaries:
* **High Confidence ✅**: ≥ 0.85 
* **Needs Review ❓**: 0.65 – 0.85 
* **Uncertain ⚠**: < 0.65 

Graphs showing log-mel spectrogram and local waveform amplitudes are displayed alongside the inference result cleanly.
=======
# Human-Voice-vs-Machine-Voice-Detection
>>>>>>> 4249909040f9b15541e6b354c2bc85a0bca49c40
