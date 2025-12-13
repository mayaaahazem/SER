import sys
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model

# ===============================
# Parameters (must match training)
# ===============================
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174
RECORD_SECONDS = 4

MODEL_PATH = "models/speaker_identifier_model.h5"
ENCODER_PATH = "models/speaker_identifier_encoder.pkl"
TEMP_AUDIO = "temp_record.wav"

# ===============================
# Load model & encoder
# ===============================
model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# ===============================
# Feature Extraction
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    ).T

    # Pad / truncate
    if mfcc.shape[0] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    # Speaker-level feature
    return mfcc.mean(axis=0)

# ===============================
# Prediction
# ===============================
def predict_speaker(file_path):
    features = extract_mfcc(file_path)
    features = np.expand_dims(features, axis=0)

    preds = model.predict(features)
    idx = np.argmax(preds)
    confidence = preds[0][idx]

    speaker = encoder.inverse_transform([idx])[0]
    return speaker, confidence

# ===============================
# Microphone Recording
# ===============================
def record_audio():
    print("🎤 Recording... Speak now")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )
    sd.wait()
    sf.write(TEMP_AUDIO, audio, SAMPLE_RATE)
    return TEMP_AUDIO

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    print("\nChoose input method:")
    print("1 - Audio file (.wav)")
    print("2 - Microphone")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        if len(sys.argv) != 2:
            print("Usage: python test_speaker_identifier.py path/to/audio.wav")
            sys.exit(1)
        audio_path = sys.argv[1]

    elif choice == "2":
        audio_path = record_audio()

    else:
        print("❌ Invalid choice")
        sys.exit(1)

    print("\n🔍 Identifying speaker...")
    speaker, confidence = predict_speaker(audio_path)

    print("\n🎯 Speaker Identification Result")
    print(f"Speaker    : {speaker}")
    print(f"Confidence : {confidence:.2f}")
