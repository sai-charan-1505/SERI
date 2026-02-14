import warnings
warnings.filterwarnings("ignore")


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import Response


import numpy as np
import librosa
import onnxruntime as ort
import pickle
import tempfile
import os


SR = 16000
N_MELS = 64
MAX_LEN = 300

app = FastAPI(title="SERI Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return Response()


session = ort.InferenceSession("emotion_cnn_model.onnx")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def speech_music_gate(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) < sr * 0.3:
        return "unknown"

    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    if flatness < 0.25 and zcr < 0.1:
        return "speech"
    else:
        return "music"


def extract_logmel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256
    )

    logmel = librosa.power_to_db(mel)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    if logmel.shape[1] < MAX_LEN:
        pad = MAX_LEN - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)))
    else:
        logmel = logmel[:, :MAX_LEN]

    return logmel.astype(np.float32)


def predict_emotion_from_file(path):
    gate = speech_music_gate(path)

    if gate != "speech":
        return {
            "gate": gate,
            "emotion": None,
            "confidence": None
        }

    x = extract_logmel(path)
    x = x[np.newaxis, ..., np.newaxis]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})

    probs = outputs[0][0]
    cls = int(np.argmax(probs))

    emotion = str(le.inverse_transform([cls])[0])
    confidence = float(np.max(probs))

    return {
        "gate": "speech",
        "emotion": emotion,
        "confidence": confidence
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_emotion_from_file(tmp_path)
    finally:
        os.remove(tmp_path)

    return JSONResponse(result)


@app.get("/")
def root():
    return {"status": "SERI API running (ONNX version)"}
