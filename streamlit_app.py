import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import folium 
from streamlit_folium import st_folium 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sktime.clustering.spatio_temporal import STDBSCAN
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from shapely.geometry import Point

# Streamlit App Configuration
st.set_page_config(page_title="ST-DBSCAN Streamlit App", layout="wide")
st.title(">>> Aplikasi Clustering Spatio-Temporal Hotspot Dilahan Gambut Sumatera Selatan Menggunakan Algoritma ST-DBSCAN Dengan Optimasi Parameter <<<")
st.write("📍Aplikasi clustering ini merupakan hasil penelitian yang dilakukan oleh Ghozi Alvin Karim, Sebagai Tugas Akhir Sarjana Sains Data Institut Teknologi Sumatera (ITERA).")
st.write("Dibuat oleh : Ghozi Alvin Karim")
st.write("Nim : 121450123")
st.write("Program Studi : Sains Data")

# --- 1. Upload Data (CSV) ---
st.sidebar.header("1. Unggah Data Hotspot")
csv_file = st.sidebar.file_uploader("❗Unggah data hotspot format .csv❗", type=["csv"])

# --- 2. Shapefile (Fixed Path) ---
st.sidebar.header("2. Shapefile")
# Sesuaikan dengan struktur folder proyek Anda
SHAPEFILE_PATH = "Data_shapefile/gambutsumsel.shp"
st.sidebar.markdown(f"**Shapefile:** `{SHAPEFILE_PATH}`")

if csv_file:
    # Load CSV
    with st.spinner("Memuat data CSV..."):
        df = pd.read_csv(csv_file)
        df['acq_date'] = pd.to_datetime(df['acq_date'])
    st.success("Data CSV berhasil dimuat 🎉.")

    # Load Shapefile
    try:
        gdf_boundary = gpd.read_file(SHAPEFILE_PATH)
        st.success("Shapefile berhasil dimuat 🎉.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat shapefile: {e}")
        st.stop()

    # --- 3. Spatial Cleaning ---
    st.header("1️⃣ Pembersihan Koordinat Hotspot Dengan Lahan Gambut Sumatera Selatan")
    SHAPEFILE_GAMBUT_PATH = "Data_shapefile/gambutsumsel.shp"
    SHAPEFILE_SUMSEL_PATH = "Sumatera_Selatan/sumselkabupatenshp.shp" # <-- Ganti dengan path Anda

    # Muat data CSV jika belum ada
    if 'df' not in locals():
        st.warning("Silakan unggah file CSV terlebih dahulu.")
        st.stop()
    try:
        gdf_gambut = gpd.read_file(SHAPEFILE_GAMBUT_PATH).to_crs("EPSG:4326")
        gdf_sumsel = gpd.read_file(SHAPEFILE_SUMSEL_PATH).to_crs("EPSG:4326")
        st.success("Shapefile lahan gambut dan batas wilayah berhasil dimuat 🎉.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat shapefile: {e}")
        st.stop()
    df['geometry'] = df.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_gambut.crs)
    boundary_union = gdf_gambut.unary_union
    gdf_clean = gdf[gdf.within(boundary_union)].copy()

    st.write(f"📌 Dimensi data sebelum dibersihkan : {len(gdf)}")
    st.write(f"📌 Dimensi data Setelah dibersihkan : {len(gdf_clean)}")

    centroid = gdf_sumsel.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=7, tiles="OpenStreetMap")
    folium.GeoJson(
        gdf_sumsel,
        name="Batas Kabupaten Sumsel",
        style_function=lambda f: {
            "fillColor": "#FFFADC",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.2
        }
    ).add_to(m)
    folium.GeoJson(
        gdf_gambut,
        name="Lahan Gambut",
        style_function=lambda f: {
            "fillColor": "#B6F500",
            "color": "#006400",
            "weight": 1,
            "fillOpacity": 0.4 # Opacity sedikit dinaikkan agar lebih terlihat
        }
    ).add_to(m)
    for idx, row in gdf_clean.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f"Tanggal: {row['acq_date'].strftime('%Y-%m-%d')}"
        ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=500)

    # --- 4. Preprocessing ---
    st.header("2️⃣ Prapemrosesan Data (Preprocessing)")
    drop_cols = ["brightness","scan","track","satellite","instrument",
                 "version","bright_t31","frp","daynight","type","confidence","acq_time","geometry"]
    data = gdf_clean.drop(columns=[c for c in drop_cols if c in gdf_clean.columns])
    data['acq_date'] = data['acq_date'].dt.strftime("%Y%m%d").astype(int)
    st.write("📝 Pemilihan fitur (feature selection):", data.columns.tolist())
    st.dataframe(data.head())

    # --- 5. Parameter Estimation ---
    st.header(" 3️⃣ Pemilihan Parameter ST-DBSCAN")
    hotspot = data.copy()
    n_samples = len(hotspot)
    minpts = max(2, round(math.log(n_samples)))
    st.write(f"⚙️ Parameter MinPts : {minpts}")

    # Compute k-distance plot
    X = hotspot[['latitude','longitude','acq_date']].values
    dist_mat = cdist(X, X, 'euclidean')
    def avg_k_dist(K):
        idx = np.argsort(dist_mat, axis=1)[:,1:K+1]
        d = np.array([[dist_mat[i,j] for j in idx[i]] for i in range(len(X))])
        return np.sort(d.mean(axis=1))
    k_dist = avg_k_dist(minpts)
    points = np.arange(1, len(k_dist)+1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(points, k_dist, marker='.')
    knee = KneeLocator(points, k_dist, curve='convex', direction='increasing').knee
    if knee:
        eps1 = k_dist[knee-1]
        ax.axhline(eps1, linestyle='--', label=f"eps1={eps1:.3f}")
        st.write(f"⚙️ Parameter Epsilon1 Terdeteksi Pada Indeks {knee}: {eps1:.3f}")
    ax.set_xlabel('Data Points (sorted)')
    ax.set_ylabel(f'Avg distance to {minpts} NN')
    ax.legend()
    st.pyplot(fig,use_container_width=True)

    # User adjusts eps1 & eps2
    eps1_slider = st.sidebar.slider("Parameter Epsilon 1", float(k_dist.min()), float(k_dist.max()), float(eps1))
    eps2_slider = st.sidebar.number_input("Parameter Epsilon 2 (Hari)", min_value=1, max_value=30, value=3)

    # --- 6. ST-DBSCAN Clustering ---
    st.header("4️⃣ Clustering ST-DBSCAN ")
    df_idx = pd.DataFrame(
        hotspot[['longitude','latitude']].values,
        columns=['x','y'],
        index=pd.MultiIndex.from_arrays([hotspot.index, hotspot['acq_date']], names=['event_id','timestamp'])
    )
    clusterer = STDBSCAN(eps1=eps1_slider, eps2=eps2_slider, min_samples=minpts, metric='euclidean', n_jobs=-1)
    clusterer.fit(df_idx)
    hotspot['cluster'] = clusterer.labels_

    st.write("📊 Total Clusters:", len(set(clusterer.labels_)) - ( -1 in clusterer.labels_))
    st.write("🚫 Total Noise :", sum(clusterer.labels_ == -1))
    counts = hotspot['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    st.dataframe(counts)

    fig2, ax2 = plt.subplots()
    ax2.scatter(hotspot['longitude'], hotspot['latitude'], c=hotspot['cluster'], cmap='tab20', s=10)
    ax2.set_title('Cluster Assignments')
    st.pyplot(fig2)

    # --- 7. Evaluation ---
    st.header("5️⃣ Evaluasi Hasil Clustering")
    mask = hotspot['cluster'] != -1
    if mask.sum() > 0:
        X_eval = np.column_stack([
            hotspot.loc[mask,'longitude'],
            hotspot.loc[mask,'latitude'],
            hotspot.loc[mask,'acq_date']
        ])
        y_eval = hotspot.loc[mask,'cluster']
        sil = silhouette_score(X_eval, y_eval)
        db = davies_bouldin_score(X_eval, y_eval)
        st.write(f"🔎 Silhouette Coefficient : {sil:.4f}")
        st.write(f"🔎 Davies-Bouldin Index : {db:.4f}")
    else:
        st.write("Tidak ada klaster yang perlu dievaluasi (semua noise).")

else:
    st.info("Silakan upload CSV hotspot untuk memulai aplikasi.")
