# Amazon Music Clustering

> **Project:** Amazon Music Clustering
> **Author:** Pasupathi Raj
---

## 📘 Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation & Goals](#motivation--goals)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Environment & Requirements](#environment--requirements)
6. [How to Run](#how-to-run)
7. [Methodology](#methodology)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
9. [Feature Engineering](#feature-engineering)
10. [Clustering Models & Techniques](#clustering-models--techniques)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Results & Visualizations](#results--visualizations)
13. [Streamlit Dashboard (Optional)](#streamlit-dashboard-optional)
14. [Conclusions & Insights](#conclusions--insights)
15. [Limitations & Future Work](#limitations--future-work)
16. [Skills & Tools Learned](#skills--tools-learned)

---

## 🧾 Project Overview

This project performs **unsupervised clustering** on **Amazon Music data** to discover hidden patterns among songs, users, or genres. The insights can be applied for **music recommendations**, **personalized playlists**, or **marketing segmentation**.

---

## 🎯 Motivation & Goals

* Identify natural groupings among tracks or users based on listening behavior and metadata.
* Visualize clusters using dimensionality reduction.
* Compare clustering algorithms to evaluate cluster quality.
* Develop an interactive **Streamlit dashboard** for exploring results.

---

## 📊 Dataset

**Example Files:**

* `listens.csv` — user listening data (user_id, track_id, timestamp, play_count)
* `tracks.csv` — metadata (artist, genre, duration, tempo, acousticness, energy, valence, danceability)
* `users.csv` — optional user data (country, subscription_type)

> Ensure no personal data is shared publicly.

---

## 📂 Project Structure

```
amazon-music-clustering/
├── data/
│   ├── raw/                # raw datasets
│   ├── processed/          # cleaned/processed data
├── notebooks/              # Jupyter notebooks for EDA & experiments
├── src/                    # source scripts
│   ├── data_processing.py
│   ├── eda.py
│   ├── clustering.py
│   ├── evaluate.py
│   └── utils.py
├── app/                    # Streamlit app files
│   └── streamlit_app.py
├── models/                 # saved models (.pkl)
├── results/                # charts and summaries
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment & Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**Example Requirements:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
umap-learn
scipy
jupyterlab
streamlit
joblib
```

---

## ▶️ How to Run

1. Place datasets in `data/raw/`
2. Preprocess data:

```bash
python src/data_processing.py
```

3. Run clustering:

```bash
python src/clustering.py
```

4. Evaluate results:

```bash
python src/evaluate.py
```

5. Launch dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Methodology

1. **Data Cleaning:** Handle missing values, duplicates, and normalization.
2. **Feature Engineering:** Generate user & track-level features (e.g., tempo, acousticness).
3. **Dimensionality Reduction:** Use **PCA**, **UMAP**, or **t-SNE** for visualization.
4. **Clustering:** Apply **KMeans**, **DBSCAN**, **Agglomerative**, and **Gaussian Mixture Models**.
5. **Evaluation:** Assess using internal metrics and cluster visualization.

---

## 🔍 Exploratory Data Analysis (EDA)

* Track and genre distributions.
* Correlation heatmaps between audio features.
* User activity distribution.
* Listening behavior trends across genres.

---

## 🧱 Feature Engineering

* Normalization via **StandardScaler** or **MinMaxScaler**.
* One-hot encoding for genres and categorical fields.
* Aggregations (per-user and per-track).
* Optional embedding features using **OpenL3** or **SBERT**.

---

## 🤖 Clustering Models & Techniques

* **KMeans** / **MiniBatchKMeans**
* **DBSCAN**
* **Agglomerative Clustering**
* **Gaussian Mixture Models**

**Hyperparameter Tuning:**

* Use **Elbow Method** and **Silhouette Analysis** for KMeans.
* Tune `eps` and `min_samples` for DBSCAN.

---

## 📈 Evaluation Metrics

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

**Qualitative checks:** Cluster genre composition, top tracks, and user segments.

---

## 📊 Results & Visualizations

* PCA & UMAP scatter plots of clusters
* Silhouette plots for comparison
* Cluster summaries: dominant genres, feature averages

Final visual results saved in `/results/`.

---

## 💻 Streamlit Dashboard (Optional)

Interactive web dashboard for exploring clusters:

* Select model and hyperparameters
* Visualize clusters (PCA/UMAP)
* View top tracks per cluster
* Export results to CSV

Run:

```bash
streamlit run app/streamlit_app.py
```

---

## 🧩 Conclusions & Insights

* Identified distinct clusters based on genre, tempo, and energy levels.
* Cluster patterns suggest strong relationships between user preferences and audio mood metrics.
* Insights can enhance playlist recommendations and marketing targeting.

---

## 🚧 Limitations & Future Work

* Improve categorical embeddings for artist/genre data.
* Use pretrained audio embeddings for more content-aware clustering.
* Integrate sequential patterns from user session data.
* Expand dashboard analytics and interactivity.

---

## 🧠 Skills & Tools Learned

### Data Analysis & Preprocessing

* Data wrangling with **Pandas**, **NumPy**
* Cleaning and scaling with **StandardScaler**
* Encoding categorical variables

### Exploratory Data Analysis (EDA)

* Visualization with **Matplotlib**, **Seaborn**, **Plotly**
* Feature correlation and trend analysis

### Feature Engineering

* Creation of custom numeric and categorical features
* Dimensionality reduction (**PCA**, **UMAP**, **t-SNE**)

### Machine Learning & Clustering

* Hands-on with **KMeans**, **DBSCAN**, **Agglomerative**, **GMM**
* Model tuning and evaluation using clustering metrics

### Visualization & Insights

* Cluster visualization and interpretation
* Deriving actionable insights from grouped clusters

### Dashboard Development

* Built a **Streamlit** app for real-time cluster visualization
* Interactive UI for exploring results

### Version Control & Documentation

* Used **Git/GitHub** for versioning and collaboration
* Created detailed documentation and reproducible pipelines

### Tools & Libraries

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Plotly` · `UMAP-learn` · `Streamlit` · `Joblib` · `Git` · `VS Code`

---
