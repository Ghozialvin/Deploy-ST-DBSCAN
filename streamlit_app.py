import streamlit as st
import pandas as pd
import numpy as np
import math
import geopandas as gpd
from scipy.spatial.distance import cdist
from sktime.clustering.spatio_temporal import STDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
from shapely.geometry import Point

# Judul Aplikasi
st.title("ST-DBSCAN Hotspot Clustering dengan Boundary Filter Lokal")

# ===== Parameter =====
# 1. Pilih Epsilon2 (temporal)
eps2 = st.selectbox(
    "Pilih Epsilon2 (temporal) dalam hari:", [3, 7, 30]
)

# 2. Upload CSV hotspot
df_file = st.file_uploader(
    "Upload file CSV hotspot (harus memuat kolom latitude, longitude, acq_date):", 
    type="csv"
)

if df_file is not None:
    # Baca CSV dan normalisasi nama kolom
    df = pd.read_csv(df_file)
    df.columns = df.columns.str.strip().str.lower()
    required = ["latitude", "longitude", "acq_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
        st.stop()
    df = df.dropna(subset=required)

    # ===== Filter menggunakan shapefile lokal =====
    try:
        shapefile_path = "./Data_shapefile/gambutsumsel.shp"
        gdf_boundary = gpd.read_file(shapefile_path)
        boundary_union = gdf_boundary.unary_union
        df["geometry"] = df.apply(lambda r: Point(r.longitude, r.latitude), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_boundary.crs)
        df = gdf[gdf.within(boundary_union)].drop(columns=["geometry"])
        st.write(f"🔹 Titik dalam boundary: {df.shape[0]} dari total {len(df) + 0}")
    except Exception as e:
        st.warning(f"Gagal membaca shapefile lokal: {e}. Proses clustering dilanjutkan tanpa filter.")

    # ===== Preprocessing tanggal =====
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"])
    df["timestamp"] = ((df["acq_date"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")).astype(int)

    # ===== Hitung parameter ST-DBSCAN =====
    n = len(df)
    minpts = max(1, round(math.log(n)))
    X_k = np.column_stack([df.latitude.values, df.longitude.values, df.timestamp.values])
    dist_mat = cdist(X_k, X_k, metric="euclidean")
    def avg_k(mat, K):
        idx = np.argsort(mat, axis=1)[:, 1:K+1]
        return np.array([mat[i, idx[i]].mean() for i in range(mat.shape[0])])
    avg_dist = np.sort(avg_k(dist_mat, minpts))
    pts = np.arange(1, len(avg_dist) + 1)
    knee = KneeLocator(pts, avg_dist, curve='convex', direction='increasing').knee
    eps1 = float(avg_dist[knee-1]) if knee else float(avg_dist.mean())
    st.write(f"🔵 ε1 spatial+temporal = {eps1:.4f} (knee @ {knee if knee else 'mean'})")

    # ===== Fit ST-DBSCAN =====
    idx = pd.MultiIndex.from_arrays([df.index, df.timestamp], names=["event_id", "timestamp"])
    fit_df = pd.DataFrame(df[["longitude", "latitude"]].values, columns=["x", "y"], index=idx)
    model = STDBSCAN(eps1=eps1, eps2=eps2, min_samples=minpts, metric="euclidean", n_jobs=-1)
    model.fit(fit_df)
    df["cluster"] = model.labels_

    # ===== Evaluasi =====
    unique_clusters = set(df.cluster)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    mask = df.cluster != -1
    if n_clusters > 1 and mask.sum() > 1:
        X_eval = np.column_stack([df.latitude[mask], df.longitude[mask], df.timestamp[mask]])
        sil = silhouette_score(X_eval, df.cluster[mask])
        dbi = davies_bouldin_score(X_eval, df.cluster[mask])
        st.write(f"🏷️ Jumlah cluster (excl. noise): {n_clusters}")
        st.write(f"📃 Silhouette Coefficient: {sil:.4f}")
        st.write(f"🔍 Davies–Bouldin Index: {dbi:.4f}")
    else:
        st.warning("Tidak cukup cluster (minimal 2) untuk evaluasi Silhouette/DB Index.")

    # ===== Tampilkan & Unduh =====
    st.subheader("Hasil Clustering")
    st.dataframe(df)
    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Hasil CSV", data=csv_out, file_name="hasil_clustering.csv", mime='text/csv')
