import numpy as np
import librosa

def extract_features(audio, sr=16000):
    """
    Extracts 90 acoustic features matching the voice_model_v5 pipeline.
    """
    if len(audio) == 0:
        return None

    # 1. Peak Normalization
    ymax = np.max(np.abs(audio))
    if ymax > 0:
        audio = audio / ymax

    # 2. MFCC 20 mean + std (40)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 3. Delta MFCC 20 mean + std (40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    # 4. Spectral Centroid (2)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # 5. Spectral Bandwidth (2)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    # 6. Spectral Rolloff (2)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # 7. Zero Crossing Rate (2)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 8. Speaker-Independent Relative Pitch Variation (2)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
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
