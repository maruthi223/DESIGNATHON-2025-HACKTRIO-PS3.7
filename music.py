import json
import numpy as np
import librosa
import os
import soundfile as sf
from pathlib import Path
from keras.models import load_model


DATA_ROOT = Path("C:/Users/kalya/Downloads/model_audio/openmic-2018")
MODEL_DIR = DATA_ROOT / "models"
CLASS_MAP_PATH = DATA_ROOT / "class-map.json"

sr = 22050
n_fft = 1024
hop_length = 512
n_mfcc = 40
clip_duration = 10.0  
samples_per_clip = int(sr * clip_duration)


with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

instruments = list(class_map.keys())

models = {}
for instrument in instruments:
    model_path = MODEL_DIR / f"{instrument}_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file for {instrument} not found at {model_path}")
    models[instrument] = load_model(str(model_path))
    print(f"Loaded model for {instrument}")


def extract_mfcc(audio_segment, sr, n_mfcc, n_fft, hop_length):
    """
    Extract MFCC features from a given audio segment.
    Returns a (time_steps, n_mfcc, 1) array.
    """
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T.astype(np.float32)  
    mfcc = np.expand_dims(mfcc, axis=-1) 
    return mfcc

input_audio_path = DATA_ROOT /"sample-2.ogg"  
audio, _ = librosa.load(input_audio_path, sr=sr, mono=True)
audio_length = len(audio)
print(f"Loaded input audio of {audio_length/sr:.2f} seconds")

separated_audio = {instrument: np.zeros_like(audio) for instrument in instruments}

num_segments = int(np.ceil(audio_length / samples_per_clip))
print(f"Processing audio in {num_segments} segments...")

for seg in range(num_segments):
    start_sample = seg * samples_per_clip
    end_sample = min((seg + 1) * samples_per_clip, audio_length)
    segment = audio[start_sample:end_sample]
    
    if len(segment) < samples_per_clip:
        segment = np.pad(segment, (0, samples_per_clip - len(segment)), mode='constant')
    
    mfcc_features = extract_mfcc(segment, sr, n_mfcc, n_fft, hop_length)
    mfcc_input = np.expand_dims(mfcc_features, axis=0) 
    
    for instrument, model in models.items():
        prob = model.predict(mfcc_input)[0, 0]
        print(f"Segment {seg}, {instrument}: probability {prob:.2f}")
        threshold = 0.5 
        if prob > threshold:
            separated_audio[instrument][start_sample:end_sample] = segment[:end_sample - start_sample]


output_dir = Path("separated_instruments")
output_dir.mkdir(exist_ok=True)

for instrument, audio_out in separated_audio.items():
    output_file = output_dir / f"{instrument}_separated.wav"
    sf.write(str(output_file), audio_out, sr)
    print(f"Saved separated audio for {instrument} to {output_file}")
