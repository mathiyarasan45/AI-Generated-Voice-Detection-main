import os
import sys
import glob
import time
import random
import tempfile
import subprocess
import joblib
import librosa
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
import imageio_ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------
# Reproducibility & Configuration
# ---------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_BASE_DIR = r"C:\Users\asus\Downloads\FoR_dataset\for-2seconds"
MODEL_SAVE_PATH = "voice_model_v5.pkl"
SAMPLE_RATE = 16000

try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


def mp3_degrade(y, sr=16000, bitrate="96k"):
    """
    Simulates MP3 lossy encode/decode compression using FFmpeg.
    """
    tmp_in_path = None
    tmp_mp3_path = None
    tmp_out_path = None
    try:
        y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            tmp_in_path = f_in.name
            wavfile.write(tmp_in_path, sr, y_int16)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
            tmp_mp3_path = f_mp3.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            tmp_out_path = f_out.name

        cmd1 = [FFMPEG_EXE, "-y", "-i", tmp_in_path, "-b:a", bitrate, tmp_mp3_path]
        subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        cmd2 = [FFMPEG_EXE, "-y", "-i", tmp_mp3_path, "-ar", str(sr), "-ac", "1", tmp_out_path]
        subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        y_deg, _ = librosa.load(tmp_out_path, sr=sr, mono=True)
        if len(y_deg) > len(y):
            y_deg = y_deg[:len(y)]
        elif len(y_deg) < len(y):
            y_deg = np.pad(y_deg, (0, len(y) - len(y_deg)))
        return y_deg
    except Exception:
        return y
    finally:
        for p in [tmp_in_path, tmp_mp3_path, tmp_out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def augment_audio_waveform(y, sr=16000, seed=None):
    """
    Applies audio degradation/compression simulation to 1D waveform `y`
    to force model to learn acoustic characteristics instead of codec shortcuts.

    APPLIED EQUALLY TO BOTH REAL AND FAKE TRAINING AUDIO SAMPLES.
    VALIDATION AND TESTING SETS ARE KEPT COMPLETELY UNTOUCHED.

    Augmentations:
      1. MP3 Codec Compression (50% prob):
         Simulates lossy MP3 compression across various bitrates (32k to 192k).
      2. Low-pass Butterworth Filter (50% prob):
         Simulates high-frequency codec roll-off (6.5 kHz - 7.8 kHz).
      3. Bandwidth Resampling Jitter (40% prob):
         Simulates telephone/narrowband codec (downsamples to 8k/11k then back to 16k).
      4. Additive Gaussian Noise (40% prob):
         Subtle background noise (SNR 30 - 45 dB).
      5. Amplitude Scaling Jitter (50% prob):
         Random volume scaling (0.8x - 1.0x).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    y_aug = y.copy()

    # 1. MP3 Codec Compression Simulation (50% prob)
    if rng.rand() < 0.5:
        bitrate = rng.choice(["32k", "64k", "96k", "128k", "160k", "192k"])
        y_aug = mp3_degrade(y_aug, sr=sr, bitrate=bitrate)

    # 2. Low-pass filter (50% prob)
    if rng.rand() < 0.5:
        cutoff = rng.uniform(6500, 7800)
        nyquist = sr / 2.0
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='low', analog=False)
        y_aug = scipy.signal.filtfilt(b, a, y_aug)

    # 3. Bandwidth Resampling Jitter (40% prob)
    if rng.rand() < 0.4:
        target_sr = rng.choice([8000, 11025, 12000])
        y_down = librosa.resample(y_aug, orig_sr=sr, target_sr=target_sr)
        y_aug = librosa.resample(y_down, orig_sr=target_sr, target_sr=sr)
        if len(y_aug) > len(y):
            y_aug = y_aug[:len(y)]
        elif len(y_aug) < len(y):
            y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))

    # 4. Additive Gaussian Noise (40% prob)
    if rng.rand() < 0.4:
        snr_db = rng.uniform(30, 45)
        signal_power = np.mean(y_aug ** 2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise = rng.normal(0, np.sqrt(noise_power), size=y_aug.shape)
            y_aug = y_aug + noise

    # 5. Amplitude Scaling Jitter (50% prob)
    if rng.rand() < 0.5:
        scale = rng.uniform(0.8, 1.0)
        y_aug = y_aug * scale

    return y_aug


def extract_file_features(file_path, is_training=False, file_idx=0):
    """
    Extracts 90 acoustic features purely from the peak-normalized audio waveform array.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        if len(y) == 0:
            return None

        # 1. Peak Audio Normalization
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max

        # 2. Waveform Codec-Degradation Augmentation (Only during training)
        if is_training:
            seed = (RANDOM_SEED + file_idx * 17) % (2**31 - 1)
            y = augment_audio_waveform(y, sr=sr, seed=seed)

        # 3. MFCC (20) & Delta MFCC (20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc, axis=1)
        delta_std = np.std(delta_mfcc, axis=1)

        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))

        # 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))

        # 6. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        # 7. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # 8. Speaker-Independent Relative Pitch Variation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 1:
            f0_mean_val = float(np.mean(pitch_values))
            f0_std_val = float(np.std(pitch_values))
            f0_normalized_std = float(f0_std_val / (f0_mean_val + 1e-6))
            f0_delta_std = float(np.std(np.diff(pitch_values)))
        else:
            f0_normalized_std = 0.0
            f0_delta_std = 0.0

        feat_vector = np.hstack([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            rolloff_mean, rolloff_std,
            zcr_mean, zcr_std,
            f0_normalized_std, f0_delta_std
        ])

        if np.isnan(feat_vector).any() or np.isinf(feat_vector).any():
            feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feat_vector
    except Exception:
        return None


def _extract_worker(task):
    file_path, label, is_training, idx = task
    feat = extract_file_features(file_path, is_training=is_training, file_idx=idx)
    return feat, label, file_path


def load_dataset_split(split_name, is_training=False):
    split_dir = os.path.join(DATASET_BASE_DIR, split_name)
    real_dir = os.path.join(split_dir, "real")
    fake_dir = os.path.join(split_dir, "fake")

    real_files = sorted(glob.glob(os.path.join(real_dir, "*.wav")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.wav")))

    file_tasks = []
    idx = 0
    for f in real_files:
        file_tasks.append((f, 0, is_training, idx))
        idx += 1
    for f in fake_files:
        file_tasks.append((f, 1, is_training, idx))
        idx += 1

    total_files = len(file_tasks)

    print(f"\n--- Loading [{split_name.upper()}] Split (Augmentation: {is_training}) ---")
    print(f"Total files found: {total_files} (Real: {len(real_files)}, Fake: {len(fake_files)})")

    X, y = [], []
    skipped_count = 0
    start_time = time.time()

    workers = max(1, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_worker, task) for task in file_tasks]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            feat, label, file_path = future.result()
            if feat is not None:
                X.append(feat)
                y.append(label)
            else:
                skipped_count += 1

            if completed % 2000 == 0 or completed == total_files:
                elapsed = time.time() - start_time
                print(f"  Processed {completed}/{total_files} files ({completed/total_files*100:.1f}%) in {elapsed:.1f}s...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    elapsed = time.time() - start_time

    print(f"  Finished [{split_name.upper()}] extraction in {elapsed:.2f}s!")
    print(f"  Successfully extracted: {len(X)} samples | Skipped / Unreadable: {skipped_count}")

    return X, y, skipped_count


def evaluate_model(model, X, y, split_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    mean_prob_real = float(np.mean(y_proba[y == 0]))
    mean_prob_fake = float(np.mean(y_proba[y == 1]))

    print(f"\n==================================================")
    print(f"       EVALUATION REPORT: {split_name.upper()} SPLIT")
    print(f"==================================================")
    print(f" Total Samples Evaluated: {len(y)}")
    print(f" Accuracy:                {acc * 100:.2f}% ({acc:.4f})")
    print(f" Precision (Fake=1):      {prec * 100:.2f}% ({prec:.4f})")
    print(f" Recall (Fake=1):         {rec * 100:.2f}% ({rec:.4f})")
    print(f" F1-Score (Fake=1):       {f1 * 100:.2f}% ({f1:.4f})")
    print(f" ROC-AUC Score:           {auc:.4f}")
    print(f" Mean AI Prob (Real=0):   {mean_prob_real:.4f}")
    print(f" Mean AI Prob (Fake=1):   {mean_prob_fake:.4f}")
    print(f"\n Confusion Matrix:")
    print(f"                     Predicted Real (0)   Predicted Fake (1)")
    print(f" Actual Real (0)            {cm[0][0]:<18} {cm[0][1]}")
    print(f" Actual Fake (1)            {cm[1][0]:<18} {cm[1][1]}")
    print(f"==================================================\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "mean_prob_real": mean_prob_real,
        "mean_prob_fake": mean_prob_fake
    }


def main():
    print("==================================================")
    print("  AI VOICE DETECTION - MODEL TRAINING (v5)")
    print("==================================================")
    print(f"Dataset Path: {DATASET_BASE_DIR}")
    print(f"Random Seed:  {RANDOM_SEED}")
    print(f"Output Model: {MODEL_SAVE_PATH}")

    # 1. Load Training Data WITH Codec Augmentation & Peak Normalization
    X_train, y_train, train_skipped = load_dataset_split("training", is_training=True)

    # Internal Codec-Balanced Train/Validation Split (80/20) from Training Data
    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(len(X_train))
    val_size = int(len(X_train) * 0.2)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    X_tr, y_tr = X_train[train_indices], y_train[train_indices]
    X_val_internal, y_val_internal = X_train[val_indices], y_train[val_indices]

    # 2. Build Classifier Pipeline
    print("\n--- Training Model V5 ---")
    print("Classifier: RandomForestClassifier (100 trees, balanced weights)")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    print(f"Model V5 trained successfully on {len(X_tr)} samples in {fit_time:.2f} seconds!")

    # Evaluate on Internal Codec-Augmented Validation Split
    print("\n--- Evaluating on Internal Validation Split ---")
    internal_val_metrics = evaluate_model(model, X_val_internal, y_val_internal, "internal_validation")

    # 3. Evaluate on Official Validation Split (UNTOUCHED / UNAUGMENTED)
    X_val, y_val, val_skipped = load_dataset_split("validation", is_training=False)
    val_metrics = evaluate_model(model, X_val, y_val, "official_validation")

    # 4. Evaluate on Untouched Original FoR Testing Split
    X_test, y_test, test_skipped = load_dataset_split("testing", is_training=False)
    test_metrics = evaluate_model(model, X_test, y_test, "testing")

    # 5. Save Newly Trained Model as voice_model_v5.pkl
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saved newly trained model to: {MODEL_SAVE_PATH}")

    # Retrain on 100% of augmented training set and save final model
    print("\nRefitting model on 100% of augmented training data for final distribution...")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Final model refitted & saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
