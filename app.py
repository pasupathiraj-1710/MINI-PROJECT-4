import streamlit as st
import pandas as pd
import joblib
import os
import features_extractor

st.title(" MP3 Music Genre Predictor")

folder_path = st.text_input("Enter MP3 Folder Path:")

if folder_path and os.path.isdir(folder_path):
    mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
    if not mp3_files:
        st.warning("No MP3 files found in this folder.")
    else:
        scaler = joblib.load("scaler.joblib")
        kmeans = joblib.load("kmeans_model.joblib")

        audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness',
                          'valence', 'tempo', 'duration_ms']

        cluster_labels = {
            0: 'Instrumental / Acoustic',
            1: 'Mainstream / Party',
            2: 'Happy / Dance',
            3: 'Vocal / Speech-heavy',
            4: 'Chill / Relaxing'
        }

        for mp3_file in mp3_files:
            st.subheader(f"File: {mp3_file}")
            full_path = os.path.join(folder_path, mp3_file)

            df_song = features_extractor.extract_features(full_path)

            st.dataframe(df_song)
            st.bar_chart(df_song.T)

            for col in audio_features:
                if col not in df_song.columns:
                    df_song[col] = 0  # Fill missing

            df_song = df_song[audio_features]

            df_scaled = scaler.transform(df_song)

            cluster = kmeans.predict(df_scaled)[0]

            st.success(f"Predicted Genre: {cluster_labels[cluster]}")

else:
    st.info("Paste a folder path containing your MP3 files above.")



### ---------------------------------------F:\music folder--------------------------------------------###