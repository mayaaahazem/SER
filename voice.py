import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os


SAVE_DIR = "data/maya"  
os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLE_RATE = 22050  
DURATION = 3  


def record_audio(filename):
    print(f" Recording: {filename} ({DURATION} sec)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print(f" Saved: {filename}\n")


if __name__ == "__main__":
    num_records = int(input("How many recordings do you want to make? "))
    for i in range(1, num_records + 1):
        filename = os.path.join(SAVE_DIR, f"maya_{i:02d}.wav")
        input(f"Press Enter to start recording {i}/{num_records}...")
        record_audio(filename)
