import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
from sktime.clustering.spatio_temporal import STDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from shapely.geometry import Point
    
# Judul Aplikasi
st.title("ST-DBSCAN Hotspot Clustering")

# 1. Upload data CSV
uploaded_file = st.file_uploader("1. Upload file CSV hotspot (harus memuat kolom latitude, longitude, acq_date)", type="csv")

# 2. Pilih parameter Epsilon2 (jarak temporal dalam hari)
eps2 = st.selectbox("2. Pilih Epsilon2 (temporal) dalam hari:", [3, 7, 30])

if uploaded_file is not None:
    # 3. Baca data
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    # 4. Preprocessing tanggal
    df["acq_date"] = pd.to_datetime(df["acq_date"])  # asumsi format YYYY-MM-DD
    df["acq_date"] = df["acq_date"].dt.strftime("%Y%m%d").astype(int)

    # 5. Hitung MinPts (log(n))
    n = df.shape[0]
    minpts = round(math.log(n))

    # 6. Hitung Epsilon1 melalui k-distance dan KneeLocator
    X = df[["latitude", "longitude", "acq_date"]].values
    distance_matrix = cdist(X, X, "euclidean")
    # hitung rata-rata jarak K tetangga terdekat
    def avg_k_distances(mat, K):
        idx = np.argsort(mat, axis=1)[:, 1:K+1]
        dists = np.array([[mat[i, j] for j in idx[i]] for i in range(len(mat))])
        return dists.mean(axis=1)
    avg_d = np.sort(avg_k_distances(distance_matrix, minpts))
    pts = np.arange(1, len(avg_d) + 1)
    from kneed import KneeLocator
    knee = KneeLocator(pts, avg_d, curve='convex', direction='increasing').knee
    eps1 = float(avg_d[knee-1]) if knee else float(avg_d.mean())

    st.write(f"🔵 DITEMUKAN ε1 = {eps1:.4f} pada titik ke-{knee} (atau menggunakan rata-rata jika tidak terdeteksi)")

    # 7. Jalankan ST-DBSCAN
    coords = df[["longitude", "latitude"]].values
    timestamps = df["acq_date"].values
    index = pd.MultiIndex.from_arrays([df.index, timestamps], names=["event_id", "timestamp"])
    fit_df = pd.DataFrame(coords, columns=["x","y"], index=index)

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

    # 8. Evaluasi (tanpa noise)
    mask = labels != -1
    if mask.sum() > 0:
        sil = silhouette_score(X[mask], labels[mask])
        dbi = davies_bouldin_score(X[mask], labels[mask])
        st.write(f"🏷️ Jumlah cluster: {len(set(labels)) - (1 if -1 in labels else 0)}")
        st.write(f"✂️ Silhouette Coefficient: {sil:.4f}")
        st.write(f"🔍 Davies–Bouldin Index: {dbi:.4f}")
    else:
        st.warning("Semua titik dianggap noise, tidak ada cluster untuk dievaluasi.")

    # 9. Tampilkan hasil
    st.subheader("🔎 Tabel Hasil Clustering")
    st.dataframe(df)

    # 10. Unduh hasil sebagai CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Hasil Cluster sebagai CSV",
        data=csv,
        file_name='hasil_clustering.csv',
        mime='text/csv'
    )