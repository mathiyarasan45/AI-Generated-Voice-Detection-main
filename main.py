import os
import base64
import io
import tempfile
import joblib
import librosa
import numpy as np
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# =========================
# App Initialization
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API Key (use env var in production)
# =========================
API_KEY = os.getenv("API_KEY", "test_key_123")

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/style.css")
def serve_css():
    return FileResponse("style.css")

@app.get("/app.js")
def serve_js():
    return FileResponse("app.js")

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# Load ML Model Safely
# =========================
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("voice_model.pkl")
    return _model

# =========================
# Feature Extraction
# =========================
def extract_features(audio, sr):
    features = {}

    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[pitches > 0]

    features["pitch_mean"] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    features["pitch_std"] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    for i, val in enumerate(mfcc_means):
        features[f"mfcc_{i+1}"] = float(val)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))

    # Energy variation
    rms = librosa.feature.rms(y=audio)
    features["rms_std"] = float(np.std(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features["zcr_mean"] = float(np.mean(zcr))

    return features

# =========================
# Rule-Based Detection (Fallback)
# =========================
def rule_based_detection(features):
    score = 0
    reasons = []

    if features["pitch_std"] < 50:
        score += 1
        reasons.append("Unnaturally stable pitch detected")

    if features["spectral_centroid_mean"] > 3000:
        score += 1
        reasons.append("Overly smooth spectral characteristics")

    if features["rms_std"] < 0.01:
        score += 1
        reasons.append("Low energy variation typical of synthetic speech")

    if score >= 2:
        return "AI_GENERATED", 0.65, "; ".join(reasons)

    return "HUMAN", 0.55, "Natural human-like speech dynamics observed"

# =========================
# ML Detection
# =========================
def ml_detection(features):
    try:
        model = get_model()
    except Exception as e:
        print("Model load failed:", e)
        return None

    vector = np.array(list(features.values())).reshape(1, -1)
    return model.predict_proba(vector)[0][1]

# =========================
# Final Decision Logic
# =========================
def final_decision(features):
    prob_ai = ml_detection(features)

    if prob_ai is not None:
        if prob_ai >= 0.75:
            return "AI_GENERATED", round(prob_ai, 2), "ML model detected synthetic voice patterns"
        elif prob_ai <= 0.25:
            return "HUMAN", round(1 - prob_ai, 2), "ML model detected natural human speech patterns"
    print("ML prob_ai:", prob_ai)
    return rule_based_detection(features)


# =========================
# Audio Helper Logic
# =========================
def load_audio_bytes(audio_bytes: bytes):
    # Step 1: Direct librosa loading (for WAV/MP3)
    try:
        return librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as primary_err:
        print(f"Direct audio loading failed ({primary_err}). Invoking FFmpeg conversion...")

    # Step 2: Fallback to FFmpeg decoding using temporary seekable disk files (for AAC / M4A / OGG / WEBM)
    tmp_in_path = None
    tmp_out_path = None
    try:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_out_path = tmp_out.name

        cmd = [
            ffmpeg_exe, "-y",
            "-i", tmp_in_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            tmp_out_path
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode != 0 or not os.path.exists(tmp_out_path) or os.path.getsize(tmp_out_path) == 0:
            err_details = proc.stderr.decode('utf-8', errors='ignore') if proc.stderr else "Empty FFmpeg output"
            raise ValueError(f"FFmpeg decoding failed: {err_details}")

        return librosa.load(tmp_out_path, sr=16000, mono=True)
    except Exception as fallback_err:
        raise ValueError(f"Audio format decoding failed: {str(fallback_err)}")
    finally:
        if tmp_in_path and os.path.exists(tmp_in_path):
            try:
                os.remove(tmp_in_path)
            except Exception:
                pass
        if tmp_out_path and os.path.exists(tmp_out_path):
            try:
                os.remove(tmp_out_path)
            except Exception:
                pass


# =========================
# API Endpoint
# =========================
@app.post("/api/voice-detection")
def voice_detection(payload: dict, x_api_key: str = Header(None)):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if "audioBase64" not in payload:
        raise HTTPException(status_code=400, detail="audioBase64 missing")

    try:
        # Decode Base64 audio (strip data URL prefix if present)
        audio_b64 = payload["audioBase64"]
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(audio_b64)

        # Load audio (WAV / MP3 / AAC / M4A / OGG / WEBM)
        audio, sr = load_audio_bytes(audio_bytes)


        if len(audio) < sr:
            raise ValueError("Audio too short")

        # Feature extraction
        features = extract_features(audio, sr)

        # Final decision
        classification, confidence, explanation = final_decision(features)

        return {
            "status": "success",
            "language": payload.get("language"),
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
