import os
import tempfile
import streamlit as st
import soundfile as sf

from predict_saved_file import predict_emotion

import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

SPEAKER_MODEL_PATH = "models/speaker_identifier_model.h5"
ENCODER_PATH = "models/speaker_identifier_encoder.pkl"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40


def predict_speaker(file_path):
	if not os.path.exists(SPEAKER_MODEL_PATH) or not os.path.exists(ENCODER_PATH):
		raise FileNotFoundError("Speaker model or encoder not found. Train the speaker model first and place files in `models/`.")

	model = load_model(SPEAKER_MODEL_PATH)
	with open(ENCODER_PATH, "rb") as f:
		le = pickle.load(f)

	# Load audio and ensure fixed duration
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

	# pad or trim to model expected input length
	max_len = model.input_shape[1]
	if features.shape[0] < max_len:
		features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant')
	else:
		features = features[:max_len, :]

	X = np.expand_dims(features, axis=0)
	preds = model.predict(X, verbose=0)[0]
	pred_idx = int(np.argmax(preds))
	confidence = float(preds[pred_idx])
	speaker_name = le.classes_[pred_idx]

	return speaker_name, confidence


def save_uploaded_file(uploaded) -> str:
	suffix = os.path.splitext(uploaded.name)[1]
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(uploaded.getbuffer())
		return tmp.name


def main():
	st.title("Audio Emotion & Speaker Inference")

	st.markdown("Upload a WAV file and choose whether to run emotion or speaker prediction.")

	uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

	if uploaded is not None:
		tmp_path = save_uploaded_file(uploaded)
		try:
			st.audio(tmp_path)
		except Exception:
			st.write("(Unable to play audio preview in this environment)")

		col1, col2 = st.columns(2)

		with col1:
			if st.button("Predict Emotion"):
				try:
					emotion, conf = predict_emotion(tmp_path)
					st.success(f"Emotion: {emotion} (confidence: {conf:.2f})")
				except Exception as e:
					st.error(f"Emotion prediction failed: {e}")

		with col2:
			if st.button("Predict Speaker"):
				try:
					speaker, conf = predict_speaker(tmp_path)
					st.success(f"Speaker: {speaker} (confidence: {conf:.2f})")
				except FileNotFoundError as e:
					st.warning(str(e))
				except Exception as e:
					st.error(f"Speaker prediction failed: {e}")

	st.markdown("---")
	st.markdown("**Notes:** Make sure the models are available in the `models/` folder: `emotion_model.h5`, `speaker_identifier_model.h5`, and the speaker encoder pickle file.")


if __name__ == "__main__":
	main()

