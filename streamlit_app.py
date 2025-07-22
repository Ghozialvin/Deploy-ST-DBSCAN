import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
from sktime.clustering.spatio_temporal import STDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
from shapely.geometry import Point

# Judul Aplikasi
st.title("ST-DBSCAN Hotspot Clustering")

# 1. Upload data CSV
uploaded_file = st.file_uploader(
    "1. Upload file CSV hotspot (harus memuat kolom latitude, longitude, acq_date)", 
    type="csv"
)

# 2. Pilih parameter Epsilon2 (jarak temporal dalam hari)
eps2 = st.selectbox(
    "2. Pilih Epsilon2 (temporal) dalam hari:", 
    [3, 7, 30]
)

if uploaded_file is not None:
    # 3. Baca data
    df = pd.read_csv(uploaded_file)
    # Normalisasi nama kolom: hapus spasi dan lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Cek kolom wajib
    required = ["latitude", "longitude", "acq_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
    else:
        # 4. Drop missing pada kolom wajib
        df = df.dropna(subset=required)

        # 5. Preprocessing tanggal
        try:
            df["acq_date"] = pd.to_datetime(df["acq_date"])  # asumsi format YYYY-MM-DD atau serupa
            df["acq_date"] = df["acq_date"].dt.strftime("%Y%m%d").astype(int)
        except Exception as e:
            st.error(f"Gagal konversi kolom acq_date: {e}")
            st.stop()

        # 6. Hitung MinPts (log(n))
        n = df.shape[0]
        minpts = max(1, round(math.log(n)))

        # 7. Hitung Epsilon1 melalui K-Distance dan KneeLocator
        X = df[["latitude", "longitude", "acq_date"]].values
        distance_matrix = cdist(X, X, "euclidean")
        def avg_k_distances(mat, K):
            idx = np.argsort(mat, axis=1)[:, 1:K+1]
            dists = np.array([[mat[i, j] for j in idx[i]] for i in range(len(mat))])
            return dists.mean(axis=1)
        avg_dist = np.sort(avg_k_distances(distance_matrix, minpts))
        pts = np.arange(1, len(avg_dist) + 1)
        knee = KneeLocator(pts, avg_dist, curve='convex', direction='increasing').knee
        if knee:
            eps1 = float(avg_dist[knee-1])
        else:
            eps1 = float(avg_dist.mean())
        st.write(f"🔵 DITEMUKAN ε1 = {eps1:.4f} pada titik ke-{knee if knee else 'mean'}")

        # 8. Jalankan ST-DBSCAN
        coords = df[["longitude", "latitude"]].values
        timestamps = df["acq_date"].values
        index = pd.MultiIndex.from_arrays(
            [df.index, timestamps], names=["event_id", "timestamp"]
        )
        fit_df = pd.DataFrame(coords, columns=["x", "y"], index=index)

        clustering = STDBSCAN(
            eps1=eps1,
            eps2=eps2,
            min_samples=minpts,
            metric="euclidean",
            n_jobs=-1
        )
        clustering.fit(fit_df)
        labels = clustering.labels_
        df["cluster"] = labels

        # 9. Evaluasi (tanpa noise)
        mask = labels != -1
        if mask.sum() > 0:
            sil = silhouette_score(X[mask], labels[mask])
            dbi = davies_bouldin_score(X[mask], labels[mask])
            st.write(f"❗ Jumlah cluster: {len(set(labels)) - (1 if -1 in labels else 0)}")
            st.write(f"📃 Silhouette Coefficient: {sil:.4f}")
            st.write(f"🔍 Davies–Bouldin Index: {dbi:.4f}")
        else:
            st.warning("Semua titik dianggap noise, tidak ada cluster untuk dievaluasi.")

        # 10. Tampilkan hasil tabel
        st.subheader("🔎 Tabel Hasil Clustering")
        st.dataframe(df)

        # 11. Unduh hasil sebagai CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Cluster sebagai CSV",
            data=csv,
            file_name='hasil_clustering.csv',
            mime='text/csv'
        )