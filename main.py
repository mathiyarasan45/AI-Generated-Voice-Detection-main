import os
import base64
import io
import tempfile
import joblib
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from features import extract_features

# =========================
# App Initialization & Cache
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "test_key_123")

try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = "voice_model_v5.pkl"
        if not os.path.exists(model_path):
            alt_path = os.path.join(os.path.dirname(__file__), "voice_model_v5.pkl")
            if os.path.exists(alt_path):
                model_path = alt_path
        _model = joblib.load(model_path)
    return _model

@app.on_event("startup")
def startup_event():
    """Pre-load model into memory during server startup to avoid first-request latency."""
    get_model()

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
# High-Speed Audio Loader
# =========================
def load_audio_bytes(audio_bytes: bytes):
    """
    Decodes audio bytes (WAV, MP3, AAC, M4A, OGG, WEBM, MPEG) reliably and fast into 16kHz mono.
    """
    # 1. Fast Path: Direct in-memory WAV loading (Zero disk I/O, zero subprocesses)
    try:
        buf = io.BytesIO(audio_bytes)
        sr, y_raw = wavfile.read(buf)
        if sr == 16000:
            if y_raw.ndim > 1:
                y_raw = y_raw.mean(axis=1)
            if y_raw.dtype == np.int16:
                y = (y_raw / 32768.0).astype(np.float32)
            elif y_raw.dtype == np.int32:
                y = (y_raw / 2147483648.0).astype(np.float32)
            elif y_raw.dtype == np.float32:
                y = y_raw
            else:
                y = y_raw.astype(np.float32)
            return y, 16000
    except Exception:
        pass

    # 2. Optimized FFmpeg path with early 10-second truncation (-t 10.05)
    tmp_in_path = None
    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_out_path = tmp_out.name

        cmd = [
            FFMPEG_EXE, "-y",
            "-t", "10.05",
            "-i", tmp_in_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            tmp_out_path
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode == 0 and os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 0:
            sr, y_raw = wavfile.read(tmp_out_path)
            if y_raw.dtype == np.int16:
                y = (y_raw / 32768.0).astype(np.float32)
            else:
                y = y_raw.astype(np.float32)
            return y, 16000

        # Fallback to librosa.load if FFmpeg output parsing fails
        return librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as err:
        raise ValueError(f"Audio format decoding failed: {str(err)}")
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
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if "audioBase64" not in payload:
        raise HTTPException(status_code=400, detail="audioBase64 missing")

    try:
        audio_b64 = payload["audioBase64"]
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(audio_b64)

        # High-speed decoding
        audio, sr = load_audio_bytes(audio_bytes)

        # Duration validation (minimum 1 second)
        if len(audio) < sr:
            raise ValueError("Audio file too short. Minimum duration is 1 second.")

        # Hard limit to 10 seconds
        max_samples = 10 * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract 90 features (optimized single STFT)
        feat_vector = extract_features(audio, sr)
        if feat_vector is None:
            raise ValueError("Failed to extract features from audio.")

        # Predict using voice_model_v5
        model = get_model()
        vector_2d = feat_vector.reshape(1, -1)
        prob_ai = float(model.predict_proba(vector_2d)[0][1])

        if prob_ai >= 0.5:
            classification = "AI_GENERATED"
            confidence = round(prob_ai, 2)
            explanation = f"AI model detected synthetic voice characteristics (AI probability: {prob_ai:.1%})"
        else:
            classification = "HUMAN"
            confidence = round(1.0 - prob_ai, 2)
            explanation = f"AI model detected natural human speech dynamics (Human probability: {(1.0 - prob_ai):.1%})"

        return {
            "status": "success",
            "language": payload.get("language"),
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
