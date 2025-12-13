import os
import numpy as np
import librosa
from tqdm import tqdm


CLEAN_DATA_PATH = "data/cleaned_data"
FEATURES_PATH = "data/features"


SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174   


emotion_map = {
    "01": 0,  
    "02": 1,  
    "03": 2,  
    "04": 3,  
    "05": 4,  
    "06": 5,  
    "07": 6,  
    "08": 7   
}

def extract_mfcc(file_path):
    """Extract MFCC features from one audio file"""
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


def extract_and_save_features():
    os.makedirs(FEATURES_PATH, exist_ok=True)

    X = []
    y = []

    files = [f for f in os.listdir(CLEAN_DATA_PATH) if f.endswith(".wav")]

    for file in tqdm(files, desc="Extracting MFCC features"):
        file_path = os.path.join(CLEAN_DATA_PATH, file)

    
        emotion_code = file.split("-")[2]
        label = emotion_map[emotion_code]

        mfcc = extract_mfcc(file_path)

        X.append(mfcc)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

   
    np.save(os.path.join(FEATURES_PATH, "X.npy"), X)
    np.save(os.path.join(FEATURES_PATH, "y.npy"), y)

    print(" Feature extraction completed")
    print(f"X shape: {X.shape}")
    print(f" y shape: {y.shape}")
    print(" Saved to data/features/")

if __name__ == "__main__":
    extract_and_save_features()
