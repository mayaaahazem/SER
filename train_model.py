import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical


FEATURES_PATH = "data/features"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(FEATURES_PATH, "X.npy"))
y = np.load(os.path.join(FEATURES_PATH, "y.npy"))

print(" Loaded features")
print("X shape:", X.shape)
print("y shape:", y.shape)


NUM_CLASSES = 8
y_cat = to_categorical(y, num_classes=NUM_CLASSES)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data split completed")
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        input_shape=X_train.shape[1:]
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(128),

    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\n Test Accuracy: {accuracy:.4f}\n")

print(" Classification Report:")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=[
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]
))


model.save(MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")
