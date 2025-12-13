import os
import librosa
import soundfile as sf

RAW_DATA_PATH = "data/raw"
CLEAN_DATA_PATH = "data/cleaned_data"

def preprocess_dataset():
    os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
    count = 0

    for root, _, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(CLEAN_DATA_PATH, file)

               
                y, sr = librosa.load(input_file, sr=22050)

                
                y, _ = librosa.effects.trim(y)

                
                y = librosa.util.normalize(y)

                
                sf.write(output_file, y, sr)
                count += 1

    print(f" Cleaned {count} audio files saved to data/cleaned_data")

if __name__ == "__main__":
    preprocess_dataset()
