import os
import tempfile
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

# ===============================
# Parameters
# ===============================
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 174

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# ===============================
# Model paths
# ===============================
EMOTION_MODEL_PATH = "emotion_model.h5"
SPEAKER_MODEL_PATH = "speaker_identifier_model.h5"
ENCODER_PATH = "speaker_identifier_encoder.pkl"

# ===============================
# Load models once
# ===============================
if not os.path.exists(EMOTION_MODEL_PATH):
    st.error(f"Emotion model not found at {EMOTION_MODEL_PATH}")
else:
    emotion_model = load_model(EMOTION_MODEL_PATH)

if not os.path.exists(SPEAKER_MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.warning("Speaker model or encoder not found. Speaker prediction will not work.")
    speaker_model = None
    le = None
else:
    speaker_model = load_model(SPEAKER_MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

# ===============================
# Feature extraction
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = SAMPLE_RATE * DURATION
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = librosa.util.fix_length(y, size=target_length)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    # Pad/truncate
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

# ===============================
# Predictions
# ===============================
def predict_emotion_file(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = emotion_model.predict(mfcc, verbose=0)
    idx = int(np.argmax(predictions))
    confidence = float(predictions[0][idx])
    return EMOTION_LABELS[idx], confidence

def predict_speaker_file(file_path):
    if speaker_model is None or le is None:
        raise FileNotFoundError("Speaker model or encoder not loaded.")
    
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target_length = SAMPLE_RATE * DURATION
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = librosa.util.fix_length(y, size=target_length)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T

    # Pad/truncate to model input
    max_len = speaker_model.input_shape[1]
    if features.shape[0] < max_len:
        features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    X = np.expand_dims(features, axis=0)
    preds = speaker_model.predict(X, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    speaker_name = le.classes_[idx]
    return speaker_name, confidence

# ===============================
# File handling
# ===============================
def save_uploaded_file(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name

# ===============================
# Streamlit GUI
# ===============================
st.title("Speech Emotion & Speaker Detection")
st.markdown("Upload a WAV file to predict emotion and/or speaker.")

uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

if uploaded is not None:
    file_path = save_uploaded_file(uploaded)
    try:
        st.audio(file_path)
    except:
        st.write("(Unable to play audio preview)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Emotion"):
            try:
                emotion, conf = predict_emotion_file(file_path)
                st.success(f"Emotion: {emotion} (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Emotion prediction failed: {e}")

    with col2:
        if st.button("Predict Speaker"):
            try:
                speaker, conf = predict_speaker_file(file_path)
                st.success(f"Speaker: {speaker} (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Speaker prediction failed: {e}")

st.markdown("---")
st.markdown(
    "**Notes:** Place `emotion_model.h5`, `speaker_identifier_model.h5`, and `speaker_identifier_encoder.pkl` in the same folder as this script."
)


