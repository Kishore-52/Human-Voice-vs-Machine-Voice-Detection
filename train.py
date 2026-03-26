import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

from utils import extract_features


class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=16, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))
        
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        inds = self.indices[start_idx:end_idx]
        batch_x = self.X[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data():
    X_cnn = []
    X_ml = []
    y = []

    # Map labels: 0 = Human (real), 1 = Machine (fake)
    real_paths = glob.glob('dataset/real/*.wav')
    fake_paths = glob.glob('dataset/fake/*.wav')

    # Ensure dataset is somewhat balanced / loaded
    print(f"Loading {len(real_paths)} real samples and {len(fake_paths)} fake samples...")
    for path in real_paths:
        try:
            mfcc_mean, multi_ch_img = extract_features(path, augment=False, return_all=True)
            if mfcc_mean is not None and multi_ch_img is not None:
                X_ml.append(mfcc_mean)
                X_cnn.append(multi_ch_img)
                y.append(0)
        except Exception:
            pass
            
    for path in fake_paths:
        try:
            mfcc_mean, multi_ch_img = extract_features(path, augment=False, return_all=True)
            if mfcc_mean is not None and multi_ch_img is not None:
                X_ml.append(mfcc_mean)
                X_cnn.append(multi_ch_img)
                y.append(1)
        except Exception:
            pass

    return np.array(X_cnn), np.array(X_ml), np.array(y)


def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'confusion_matrix': cm}


def main():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('static'):
        os.makedirs('static')
        
    X_cnn, X_ml, y = load_data()
    if len(y) == 0:
        print("No data found! Please ensure 'dataset/real/' and 'dataset/fake/' have wav files.")
        return

    # Train/test split for CNN and ML
    X_cnn_train, X_cnn_test, X_ml_train, X_ml_test, y_train, y_test = train_test_split(
        X_cnn, X_ml, y, test_size=0.2, random_state=42, stratify=y
    )
    
    metrics_summary = {}

    # === 1. CNN Model ===
    print("\n--- Training CNN ---")
    cnn_model = build_cnn(X_cnn_train.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mc = ModelCheckpoint('models/cnn_model.h5', monitor='val_loss', save_best_only=True)
    
    train_gen = CNNDataGenerator(X_cnn_train, y_train, batch_size=16, shuffle=True)
    val_gen = CNNDataGenerator(X_cnn_test, y_test, batch_size=16, shuffle=False)
    
    cnn_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[es, mc],
        verbose=1
    )
    
    # Predict and evaluate CNN
    cnn_prob = cnn_model.predict(val_gen).ravel()
    cnn_pred = (cnn_prob > 0.5).astype(int)
    metrics_summary['CNN'] = calculate_metrics(y_test, cnn_pred)
    
    # Save the final in case best was not hit in a way h5 accepts implicitly
    if not os.path.exists('models/cnn_model.h5'):
        cnn_model.save('models/cnn_model.h5')

    # === 2. Random Forest Model ===
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_ml_train, y_train)
    rf_pred = rf.predict(X_ml_test)
    joblib.dump(rf, 'models/rf_model.joblib')
    metrics_summary['Random Forest'] = calculate_metrics(y_test, rf_pred)

    # === 3. SVM Model ===
    print("\n--- Training SVM ---")
    svm_clf = SVC(probability=True, random_state=42)
    svm_clf.fit(X_ml_train, y_train)
    svm_pred = svm_clf.predict(X_ml_test)
    joblib.dump(svm_clf, 'models/svm_model.joblib')
    metrics_summary['SVM'] = calculate_metrics(y_test, svm_pred)

    # === Save Metrics ===
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # === Plot Accuracy Comparison Bar Chart ===
    models_names = list(metrics_summary.keys())
    acc_scores = [metrics_summary[m]['accuracy'] * 100 for m in models_names]

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5), facecolor='none')
    colors = ['#8b5cf6', '#0ea5e9', '#10b981']
    bars = plt.bar(models_names, acc_scores, color=colors, alpha=0.9, edgecolor='none')
    plt.ylabel('Accuracy (%)', color='#94a3b8')
    plt.title('Model Accuracy Comparison', color='#f8fafc')
    plt.ylim(0, 100)
    
    plt.tick_params(colors='#94a3b8')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#94a3b8')
    plt.gca().spines['bottom'].set_color('#94a3b8')
    
    # Add labels on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 1, 
            f'{yval:.1f}%', 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            color='#f8fafc'
        )
    
    # Save to static
    plt.savefig('static/model_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print("\n=== Training Complete ===")
    print("Metrics saved to models/metrics.json")
    print("Chart saved to static/model_comparison.png")


if __name__ == "__main__":
    main()
