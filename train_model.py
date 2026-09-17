import os
import joblib
import librosa
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from features import extract_features

X, y = [], []

def load_folder(folder, label):
    for file in os.listdir(folder):
        if file.endswith((".wav", ".mp3", ".flac")):
            path = os.path.join(folder, file)
            audio, sr = librosa.load(path, sr=None, mono=True)
            feats = extract_features(audio, sr)
            X.append(list(feats.values()))
            y.append(label)

load_folder("data/human", 0)
load_folder("data/ai", 1)

print("Human:", y.count(0))
print("AI:", y.count(1))

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

model.fit(X, y)

joblib.dump(model, "voice_model.pkl")
print("Model trained successfully")
