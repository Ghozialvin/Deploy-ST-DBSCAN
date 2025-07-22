import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sktime.clustering.spatio_temporal import STDBSCAN
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from shapely.geometry import Point

# Streamlit App Configuration
st.set_page_config(page_title="ST-DBSCAN Streamlit App", layout="wide")
st.title("Clustering Spatio-Temporal Hotspot di Lahan Gambut Sumatera Selatan")

# --- 1. Upload Data (CSV) ---
st.sidebar.header("1. Unggah Data Hotspot")
csv_file = st.sidebar.file_uploader("Unggah file CSV hotspot", type=["csv"])

# --- 2. Shapefile (Fixed Path) ---
st.sidebar.header("2. Shapefile")
SHAPEFILE_PATH = "Data_shapefile/gambutsumsel.shp"
st.sidebar.markdown(f"**Shapefile path:** `{SHAPEFILE_PATH}`")

if csv_file:
    # Load CSV
    with st.spinner("Memuat data hotspot CSV..."):
        df = pd.read_csv(csv_file)
        df['acq_date'] = pd.to_datetime(df['acq_date'])
    st.success("Data CSV berhasil dimuat")

    # Load shapefile
    try:
        gdf_boundary = gpd.read_file(SHAPEFILE_PATH)
        st.success("Shapefile berhasil dimuat")
    except Exception as e:
        st.error(f"Gagal memuat shapefile: {e}")
        st.stop()

    # --- 3. Spatial Cleaning ---
    st.header("1️⃣ Pembersihan Koordinat Hotspot")
    df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_boundary.crs)
    boundary_union = gdf_boundary.unary_union
    gdf_clean = gdf[gdf.within(boundary_union)].copy()
    st.write(f"📌 Data sebelum dibersihkan: {len(gdf)} titik")
    st.write(f"📌 Data setelah dibersihkan: {len(gdf_clean)} titik")

    # Prepare for pydeck
    view_state = pdk.ViewState(
        latitude=gdf_clean['latitude'].mean(),
        longitude=gdf_clean['longitude'].mean(),
        zoom=6,
        pitch=0
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=gdf_clean,
        get_position='[longitude, latitude]',
        get_radius=500,
        get_color='[200, 30, 0, 160]',
        pickable=False
    )
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=view_state,
        layers=[layer]
    )
    st.pydeck_chart(deck)

    # --- 4. Preprocessing ---
    st.header("2️⃣ Prapemrosesan Data")
    drop_cols = ["brightness","scan","track","satellite","instrument",
                 "version","bright_t31","frp","daynight","type","confidence","acq_time","geometry"]
    data = gdf_clean.drop(columns=[c for c in drop_cols if c in gdf_clean.columns])
    data['acq_date'] = data['acq_date'].dt.strftime("%Y%m%d").astype(int)
    st.write("📝 Fitur terpilih:", data.columns.tolist())
    st.dataframe(data.head())

    # --- 5. Parameter Estimation ---
    st.header("3️⃣ Pemilihan Parameter ST-DBSCAN")
    hotspot = data.copy()
    n_samples = len(hotspot)
    minpts = max(2, round(math.log(n_samples)))
    st.write(f"⚙️ MinPts (min_samples): {minpts}")

    # k-distance
    X = hotspot[['latitude','longitude','acq_date']].values
    dist_mat = cdist(X, X, 'euclidean')
    def avg_k_dist(K):
        idx = np.argsort(dist_mat, axis=1)[:,1:K+1]
        d = np.array([[dist_mat[i,j] for j in idx[i]] for i in range(len(X))])
        return np.sort(d.mean(axis=1))
    k_dist = avg_k_dist(minpts)
    points = np.arange(1, len(k_dist)+1)

    fig, ax = plt.subplots()
    ax.plot(points, k_dist, marker='.')
    knee = KneeLocator(points, k_dist, curve='convex', direction='increasing').knee
    if knee:
        eps1 = k_dist[knee-1]
        ax.axhline(eps1, linestyle='--', label=f"eps1={eps1:.3f}")
        st.write(f"⚙️ Epsilon1 terdeteksi pada indeks {knee}: {eps1:.3f}")
    ax.set_xlabel('Data points (sorted)')
    ax.set_ylabel(f'Jarak rata-rata ke {minpts} tetangga')
    ax.legend()
    st.pyplot(fig)

    eps1_slider = st.sidebar.slider("Epsilon1 (spasial)", float(k_dist.min()), float(k_dist.max()), float(eps1))
    eps2_slider = st.sidebar.number_input("Epsilon2 (temporal, hari)", min_value=1, max_value=30, value=3)

    # --- 6. ST-DBSCAN Clustering ---
    st.header("4️⃣ Clustering ST-DBSCAN")
    df_idx = pd.DataFrame(
        hotspot[['longitude','latitude']].values,
        columns=['x','y'],
        index=pd.MultiIndex.from_arrays([hotspot.index, hotspot['acq_date']], names=['event_id','timestamp'])
    )
    clusterer = STDBSCAN(eps1=eps1_slider, eps2=eps2_slider, min_samples=minpts, metric='euclidean', n_jobs=-1)
    clusterer.fit(df_idx)
    hotspot['cluster'] = clusterer.labels_

    st.write(f"📊 Total cluster: {len(set(clusterer.labels_)) - ( -1 in clusterer.labels_)}")
    st.write(f"🚫 Noise points: {sum(clusterer.labels_ == -1)}")
    counts = hotspot['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    st.dataframe(counts)

    fig2, ax2 = plt.subplots()
    ax2.scatter(hotspot['longitude'], hotspot['latitude'], c=hotspot['cluster'], cmap='tab20', s=10)
    ax2.set_title('Hasil Clustering')
    st.pyplot(fig2)

    # --- 7. Evaluation ---
    st.header("5️⃣ Evaluasi Hasil")
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
        st.write(f"🔍 Silhouette Coefficient: {sil:.4f}")
        st.write(f"🔍 Davies-Bouldin Index: {db:.4f}")
    else:
        st.write("Tidak ada cluster untuk dievaluasi (semua noise).")

else:
    st.info("Silakan unggah file CSV hotspot untuk memulai aplikasi.")