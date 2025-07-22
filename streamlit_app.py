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
import zipfile
import io

# Judul Aplikasi
st.title("ST-DBSCAN Hotspot Clustering dengan Boundary Filter")

# Sidebar Deployment Guide
st.sidebar.header("Panduan Deployment")
st.sidebar.markdown(
    """
- `python -m venv venv` dan aktifkan environment
- Install dependencies di `requirements.txt`:
  ```
  pandas
  numpy
  scipy
  scikit-learn
  sktime
  kneed
  geopandas
  shapely
  streamlit
  ```
- Jalankan lokal:
  ```bash
  streamlit run streamlit_app.py
  ```
- Deploy ke Streamlit Cloud via GitHub
"""
)

# 1. Pilih parameter temporal (Epsilon2)
eps2 = st.selectbox("1. Pilih Epsilon2 (temporal) dalam hari:", [3, 7, 30])

# 2. Upload shapefile boundary (zip berisi .shp, .shx, etc.)
shp_zip = st.file_uploader(
    "2. Upload ZIP Shapefile boundary (opsional untuk filter):", 
    type="zip"
)

# 3. Upload CSV hotspot
csv_file = st.file_uploader(
    "3. Upload file CSV hotspot (kolom latitude, longitude, acq_date):", 
    type="csv"
)

if csv_file is not None:
    # Baca CSV dan normalisasi kolom
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    required = ["latitude", "longitude", "acq_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
        st.stop()
    df = df.dropna(subset=required)

    # Baca boundary jika ada
    if shp_zip is not None:
        with zipfile.ZipFile(io.BytesIO(shp_zip.read())) as z:
            z.extractall("./boundary_shp")
        gdf_boundary = gpd.read_file("./boundary_shp")
        boundary_union = gdf_boundary.unary_union
        df["geometry"] = df.apply(
            lambda r: Point(r.longitude, r.latitude), axis=1
        )
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_boundary.crs)
        df = gdf[gdf.within(boundary_union)].drop(columns=["geometry"])
        st.write(f"🔹 Titik dalam boundary: {df.shape[0]} dari total")

    # Preprocessing acq_date
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    # Kolom timestamp (hari sejak epoch)
    df["timestamp"] = ((df["acq_date"] - pd.Timestamp("1970-01-01"))
                       / pd.Timedelta("1D")).astype(int)

    # Hitung MinPts
    n = len(df)
    minpts = max(1, round(math.log(n)))

    # Hitung ε1 (spatial+temporal)
    X_k = np.column_stack([df.latitude, df.longitude, df.timestamp])
    D = cdist(X_k, X_k, 'euclidean')
    def avg_k(mat, K):
        idx = np.argsort(mat, axis=1)[:, 1:K+1]
        return np.mean([mat[i, idx[i]] for i in range(len(mat))], axis=1)
    avg_dist = np.sort(avg_k(D, minpts))
    pts = np.arange(1, len(avg_dist)+1)
    knee = KneeLocator(pts, avg_dist, curve='convex', direction='increasing').knee
    eps1 = float(avg_dist[knee-1]) if knee else float(avg_dist.mean())
    st.write(f"🔵 ε1 = {eps1:.4f} (knee @ {knee if knee else 'mean'})")

    # Fit ST-DBSCAN
    idx = pd.MultiIndex.from_arrays([df.index, df.timestamp], names=["id","time"])
    fit_df = pd.DataFrame(df[["longitude","latitude"]].values,
                          columns=["x","y"], index=idx)
    model = STDBSCAN(eps1=eps1, eps2=eps2, min_samples=minpts,
                     metric='euclidean', n_jobs=-1)
    model.fit(fit_df)
    df["cluster"] = model.labels_

    # Evaluasi (exclude noise)
    mask = df.cluster != -1
    if mask.sum()>0:
        X_eval = np.column_stack([df.latitude[mask], df.longitude[mask], df.timestamp[mask]])
        sil = silhouette_score(X_eval, df.cluster[mask])
        dbi = davies_bouldin_score(X_eval, df.cluster[mask])
        st.write(f"🏷️ Cluster terbentuk: {len(set(df.cluster)) - (1 if -1 in df.cluster.values else 0)}")
        st.write(f"📃 Silhouette: {sil:.4f}")
        st.write(f"🔍 Davies-Bouldin: {dbi:.4f}")
    else:
        st.warning("Semua titik dianggap noise")

    # Tampilkan data
    st.subheader("Hasil Clustering")
    st.dataframe(df)

    # Download
    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh hasil CSV", data=csv_out,
                       file_name="hasil_clustering.csv")
