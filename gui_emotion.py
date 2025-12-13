import tkinter as tk
from tkinter import filedialog, messagebox
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
model = load_model(MODEL_PATH)


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(mfcc)
    idx = np.argmax(predictions)
    confidence = predictions[0][idx]
    return EMOTION_LABELS[idx], confidence


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)

def run_prediction():
    file_path = entry_file.get()
    if not file_path:
        messagebox.showwarning("Input Error", "Please select a .wav file!")
        return
    try:
        emotion, confidence = predict_emotion(file_path)
        label_result.config(text=f"Emotion: {emotion}\nConfidence: {confidence:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Speech Emotion Detection")
root.geometry("400x200")

tk.Label(root, text="Select Audio File (.wav):").pack(pady=10)
entry_file = tk.Entry(root, width=50)
entry_file.pack(padx=10)
tk.Button(root, text="Browse", command=browse_file).pack(pady=5)
tk.Button(root, text="Predict Emotion", command=run_prediction).pack(pady=10)
label_result = tk.Label(root, text="", font=("Helvetica", 14))
label_result.pack(pady=10)

root.mainloop()
