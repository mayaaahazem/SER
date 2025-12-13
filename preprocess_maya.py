import os
import librosa
import soundfile as sf


INPUT_DIR = "data/maya"               
OUTPUT_DIR = "data/maya_preprocessed" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050  
DURATION = 3       


def preprocess_file(file_path, save_path):
  
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
   
    target_length = SAMPLE_RATE * DURATION
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = librosa.util.fix_length(y, size=target_length)
    
    
    if max(abs(y)) > 0:
        y = y / max(abs(y))
    
   
    sf.write(save_path, y, SAMPLE_RATE)
    print(f" Saved preprocessed: {save_path}")


if __name__ == "__main__":
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
    
    for f in files:
        in_path = os.path.join(INPUT_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, f)
        preprocess_file(in_path, out_path)

    print(f"\n Preprocessing done. Preprocessed files saved to: {OUTPUT_DIR}")
