import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy as np

df = pd.read_csv('single_genre_artists.csv')

audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo', 'duration_ms']

df_audio = df[audio_features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_audio)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=3000)
kmeans.fit(df_scaled)

joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans_model.joblib")

labels, counts = np.unique(kmeans.labels_, return_counts=True)
print("Training samples per cluster:", dict(zip(labels, counts)))
