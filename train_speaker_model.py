import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# Parameters
# ===============================
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174

RAW_RAVDESS_DIR = "data/raw"
RAW_MAYA_DIR = "data/maya"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "speaker_identifier_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "speaker_identifier_encoder.pkl")

# ===============================
# Preprocessing + Feature Extraction
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Remove silence
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
# Load Dataset
# ===============================
X, y = [], []

# -------- RAVDESS Speakers --------
for actor in sorted(os.listdir(RAW_RAVDESS_DIR)):
    actor_path = os.path.join(RAW_RAVDESS_DIR, actor)
    if not os.path.isdir(actor_path):
        continue

    speaker_label = actor.replace("Actor_", "").lstrip("0")  # Actor 08 → 8

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            X.append(extract_mfcc(file_path))
            y.append(f"actor {speaker_label}")

# -------- Maya Speaker --------
for file in os.listdir(RAW_MAYA_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(RAW_MAYA_DIR, file)
        X.append(extract_mfcc(file_path))
        y.append("maya")

X = np.array(X)
y = np.array(y)

print("📊 Total samples:", X.shape[0])
print("🧑 Speakers:", sorted(set(y)))

# ===============================
# Encode Labels
# ===============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)

# ===============================
# Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ===============================
# Model
# ===============================
model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# Train
# ===============================
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    verbose=1
)

# ===============================
# Save
# ===============================
model.save(MODEL_PATH)

print("\n✅ Speaker Identification Training Complete")
print("➕ Maya added as a new speaker")
print("💾 Model saved to:", MODEL_PATH)
print("💾 Encoder saved to:", ENCODER_PATH)
