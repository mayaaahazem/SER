import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

MODEL_PATH = "models/emotion_model.h5"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

def predict_emotion(file_path):
    model = load_model(MODEL_PATH)

    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    predictions = model.predict(mfcc)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    return EMOTION_LABELS[predicted_index], confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict_one_audio.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"\n Predicting emotion for file: {audio_path}")
    emotion, confidence = predict_emotion(audio_path)

    print("\n Prediction Result")
    print(f"Emotion    : {emotion}")
    print(f"Confidence : {confidence:.2f}")
