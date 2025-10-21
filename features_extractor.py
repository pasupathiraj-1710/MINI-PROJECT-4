import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=60)  # First 60 seconds
    features = {}
    features['tempo'] = float(librosa.beat.tempo(y=y, sr=sr)[0])
    features['duration_ms'] = float(len(y) / sr * 1000)
    features['energy'] = float(np.mean(librosa.feature.rms(y=y)))
    # Danceability better estimated from onset envelope than MFCC mean
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['danceability'] = float(np.mean(onset_env))
    # Loudness from RMS converted to decibels
    rms = librosa.feature.rms(y=y)
    features['loudness'] = float(np.mean(librosa.amplitude_to_db(rms)))
    features['speechiness'] = float(np.mean(librosa.effects.harmonic(y)))
    features['acousticness'] = float(np.mean(librosa.effects.percussive(y)))
    features['instrumentalness'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features['liveness'] = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
    features['valence'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return pd.DataFrame([features])
