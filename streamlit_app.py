import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sktime.clustering.spatio_temporal import STDBSCAN
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from shapely.geometry import Point

st.set_page_config(page_title="ST-DBSCAN Streamlit App", layout="wide")
st.title("Spatio-Temporal DBSCAN Clustering Application")

# --- 1. Upload Data ---
st.sidebar.header("1. Upload Hotspot CSV Data")
csv_file = st.sidebar.file_uploader("Upload hotspot CSV data", type=['csv'])

# --- Shapefile Path Input ---
st.sidebar.header("Shapefile Configuration")
shp_path = st.sidebar.text_input("Masukkan path folder shapefile (tanpa .shp)", "/content/drive/MyDrive/Berkas TA/Data/Gambut/gambutsumsel")

if csv_file:
    # Load CSV
    with st.spinner("Loading CSV data..."):
        df = pd.read_csv(csv_file)
        df['acq_date'] = pd.to_datetime(df['acq_date'])
    st.success("CSV loaded")

    # Load shapefile by filepath
    try:
        shapefile = f"{shp_path}.shp"
        gdf_boundary = gpd.read_file(shapefile)
        st.success(f"Shapefile loaded from {shapefile}")
    except Exception as e:
        st.error(f"Gagal memuat shapefile: {e}")
        st.stop()

    # --- 2. Cleaning by Boundary ---
    st.header("2. Spatial Cleaning")
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_boundary.crs)
    union_poly = gdf_boundary.unary_union
    gdf_clean = gdf[gdf.within(union_poly)].copy()
    st.write(f"Records before cleaning: {len(gdf)}, after cleaning: {len(gdf_clean)}")
    st.map(gdf_clean[['latitude', 'longitude']])

    # --- 3. Preprocessing ---
    st.header("3. Preprocessing")
    drop_cols = ["brightness","scan","track","satellite","instrument",
                 "version","bright_t31","frp","daynight","type","confidence","acq_time","geometry"]
    data = gdf_clean.drop(columns=[c for c in drop_cols if c in gdf_clean.columns])
    data['acq_date'] = data['acq_date'].dt.strftime("%Y%m%d").astype(int)
    st.write("Selected features:", data.columns.tolist())
    st.dataframe(data.head())

    # --- 4. Parameter Selection ---
    st.header("4. Parameter Estimation")
    hotspot = data.copy()
    n_samples = len(hotspot)
    minpts = max(2, round(math.log(n_samples)))
    st.write(f"Estimated MinPts: {minpts}")

    # Compute k-distance
    X = hotspot[['latitude','longitude','acq_date']].values
    dist_mat = cdist(X, X, 'euclidean')
    def avg_k_distances(K):
        nn_idx = np.argsort(dist_mat, axis=1)[:,1:K+1]
        dists = np.array([[dist_mat[i,j] for j in nn_idx[i]] for i in range(len(X))])
        return np.sort(dists.mean(axis=1))
    k_dist = avg_k_distances(minpts)
    points = np.arange(1, len(k_dist)+1)

    fig, ax = plt.subplots()
    ax.plot(points, k_dist, marker='.')
    knee = KneeLocator(points, k_dist, curve='convex', direction='increasing').knee
    if knee:
        eps1 = k_dist[knee-1]
        ax.axhline(eps1, linestyle='--', label=f"eps1={eps1:.3f}")
        st.write(f"Detected eps1 at index {knee}: {eps1:.3f}")
    ax.set_xlabel('Data Points (sorted)')
    ax.set_ylabel(f'Avg distance to {minpts} NN')
    ax.legend()
    st.pyplot(fig)

    eps1_slider = st.sidebar.slider("Spatial eps1", float(k_dist.min()), float(k_dist.max()), float(eps1))
    eps2_slider = st.sidebar.number_input("Temporal eps2 (days)", min_value=1, max_value=30, value=3)

    # --- 5. ST-DBSCAN Clustering ---
    st.header("5. ST-DBSCAN Clustering")
    df_idx = pd.DataFrame(
        hotspot[['longitude','latitude']].values,
        columns=['x','y'],
        index=pd.MultiIndex.from_arrays([hotspot.index, hotspot['acq_date']], names=['event_id','timestamp'])
    )
    clusterer = STDBSCAN(eps1=eps1_slider, eps2=eps2_slider, min_samples=minpts, metric='euclidean', n_jobs=-1)
    clusterer.fit(df_idx)
    hotspot['cluster'] = clusterer.labels_

    st.write("Total clusters:", len(set(clusterer.labels_)) - ( -1 in clusterer.labels_))
    st.write("Noise points:", sum(clusterer.labels_ == -1))
    counts = hotspot['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    st.dataframe(counts)

    fig2, ax2 = plt.subplots()
    sc = ax2.scatter(hotspot['longitude'], hotspot['latitude'], c=hotspot['cluster'], cmap='tab20', s=10)
    ax2.set_title('Cluster assignments')
    st.pyplot(fig2)

    # --- 6. Evaluation ---
    st.header("6. Evaluation")
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
        st.write(f"Silhouette Coefficient: {sil:.4f}")
        st.write(f"Davies-Bouldin Index: {db:.4f}")
    else:
        st.write("No clusters to evaluate (all noise).")
else:
    st.info("Upload hotspot CSV untuk memulai.")