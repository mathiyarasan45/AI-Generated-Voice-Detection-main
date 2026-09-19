import numpy as np
import librosa

def extract_features(audio, sr=16000):
    """
    Extracts 90 acoustic features matching the voice_model_v5 pipeline.
    Optimized for high-speed inference by computing the STFT magnitude matrix once.
    """
    if len(audio) == 0:
        return None

    # 1. Peak Normalization
    ymax = np.max(np.abs(audio))
    if ymax > 0:
        audio = audio / ymax

    # 2. Single STFT Computation (n_fft=2048, hop_length=512)
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(stft)

    # 3. MFCC 20 mean + std (40)
    mel_spectrogram = librosa.feature.melspectrogram(S=S**2, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 4. Delta MFCC 20 mean + std (40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    # 5. Spectral Centroid (2)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # 6. Spectral Bandwidth (2)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    # 7. Spectral Rolloff (2)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # 8. Zero Crossing Rate (2)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 9. Speaker-Independent Relative Pitch Variation (2)
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
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
