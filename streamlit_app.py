import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
from sktime.clustering.spatio_temporal import STDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator

# Judul Aplikasi
st.title("ST-DBSCAN Hotspot Clustering")

# 1. Upload data CSV
eps2 = st.selectbox(
    "1. Pilih Epsilon2 (temporal) dalam hari:", 
    [3, 7, 30],
    index=0
)
uploaded_file = st.file_uploader(
    "2. Upload file CSV hotspot (harus memuat kolom latitude, longitude, acq_date)", 
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    required = ["latitude", "longitude", "acq_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
        st.stop()

    # Drop missing
    df = df.dropna(subset=required)

    # Konversi acq_date ke datetime
    try:
        df["acq_date"] = pd.to_datetime(df["acq_date"])
    except Exception as e:
        st.error(f"Gagal konversi kolom acq_date: {e}")
        st.stop()

    # Buat kolom timestamp (nilai numerik dalam hari sejak epoch)
    df["timestamp"] = ((df["acq_date"] - pd.Timestamp("1970-01-01"))
                        / pd.Timedelta("1D")).astype(int)

    # Hitung MinPts
    n = df.shape[0]
    minpts = max(1, round(math.log(n)))

    # Hitung Epsilon1 (spatial+temporal) menggunakan k-distance
    X_kdist = np.column_stack([
        df["latitude"].values,
        df["longitude"].values,
        df["timestamp"].values
    ])
    dist_mat = cdist(X_kdist, X_kdist, metric="euclidean")
    def avg_k_distances(mat, K):
        idx = np.argsort(mat, axis=1)[:, 1:K+1]
        dists = np.array([[mat[i, j] for j in idx[i]] for i in range(len(mat))])
        return dists.mean(axis=1)
    avg_dist = np.sort(avg_k_distances(dist_mat, minpts))
    pts = np.arange(1, len(avg_dist) + 1)
    knee = KneeLocator(pts, avg_dist, curve='convex', direction='increasing').knee
    eps1 = float(avg_dist[knee-1]) if knee else float(avg_dist.mean())
    st.write(f"🔵 ε1 spatial+temporal = {eps1:.4f} (knee at {knee if knee else 'mean'})")

    # Run ST-DBSCAN
    coords = df[["longitude", "latitude"]].values
    ts = df["timestamp"].values
    index = pd.MultiIndex.from_arrays([df.index, ts], names=["event_id", "timestamp"])
    df_fit = pd.DataFrame(coords, columns=["x","y"], index=index)

    clustering = STDBSCAN(
        eps1=eps1,
        eps2=eps2,
        min_samples=minpts,
        metric="euclidean",
        n_jobs=-1
    )
    clustering.fit(df_fit)
    df["cluster"] = clustering.labels_

    # Evaluasi tanpa noise
    mask = df["cluster"] != -1
    if mask.sum() > 0:
        X_eval = np.column_stack([
            df.loc[mask, "latitude"].values,
            df.loc[mask, "longitude"].values,
            df.loc[mask, "timestamp"].values
        ])
        sil = silhouette_score(X_eval, df.loc[mask, "cluster"].values)
        dbi = davies_bouldin_score(X_eval, df.loc[mask, "cluster"].values)
        st.write(f"🏷️ Jumlah cluster: {len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)}")
        st.write(f"📃 Silhouette Coefficient: {sil:.4f}")
        st.write(f"🔍 Davies–Bouldin Index: {dbi:.4f}")
    else:
        st.warning("Semua titik dianggap noise, tidak ada cluster untuk dievaluasi.")

    # Tampilkan dan download
    st.subheader("🔎 Tabel Hasil Clustering")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh CSV", data=csv, file_name='hasil.csv', mime='text/csv')