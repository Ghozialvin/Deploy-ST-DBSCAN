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

# Streamlit configuration
st.set_page_config(page_title="ST-DBSCAN Streamlit App", layout="wide")
st.title("Application Clustering Spatio-Temporal Hotspot Dilahan Gambut Sumatera Selatan Menggunakan Algoritma ST-DBSCAN Dengan Optimasi Parameter")

# Sidebar: upload hotspot data
st.sidebar.header("1. Unggah Data Hotspot")
csv_file = st.sidebar.file_uploader("Unggah file CSV hotspot", type=["csv"])

# Fixed shapefile path
st.sidebar.header("2. Shapefile")
SHAPEFILE_PATH = "Data_shapefile/gambutsumsel.shp"
st.sidebar.markdown(f"**Shapefile path:** `{SHAPEFILE_PATH}`")

if not csv_file:
    st.info("Silakan unggah file CSV hotspot untuk memulai aplikasi.")
    st.stop()

# 1. Load CSV dan shapefile
with st.spinner("Memuat data..."):
    df = pd.read_csv(csv_file)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    gdf_boundary = gpd.read_file(SHAPEFILE_PATH)
st.success("Data berhasil dimuat")

# 2. Spatial Cleaning
st.header("1️⃣ Pembersihan Koordinat Hotspot Diwilayah Lahan Gambut")
df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_boundary.crs)
boundary_union = gdf_boundary.unary_union
gdf_clean = gdf[gdf.within(boundary_union)].copy()
st.write(f"📌 Dimensi data sebelum dibersihkan : {len(gdf)}")
st.write(f"📌 Dimensi data Setelah dibersihkan : {len(gdf_clean)}")

# --- Folium Map with OSM tiles ---
center = [gdf_clean['latitude'].mean(), gdf_clean['longitude'].mean()]
m = folium.Map(location=center, zoom_start=6, tiles='OpenStreetMap')
# Tambahkan boundary shapefile
geo_json = gdf_boundary.to_crs(epsg=4326)
folium.GeoJson(geo_json, name='Boundary', style_function=lambda x: {'color':'black','weight':2,'fill':False}).add_to(m)
# Titik hotspot
for _, row in gdf_clean.iterrows():
    folium.CircleMarker(location=(row['latitude'], row['longitude']),
                        radius=3,
                        color='red',
                        fill=True, fill_opacity=0.7).add_to(m)

st.subheader("Peta Hotspot dengan Latar OpenStreetMap")
st_folium(m, width=700, height=500)

# 3. Preprocessing
st.header("2️⃣ Prapemrosesan Data (Preprocessing)")
drop_cols = ["brightness","scan","track","satellite","instrument",
             "version","bright_t31","frp","daynight","type","confidence","acq_time","geometry"]
data = gdf_clean.drop(columns=[c for c in drop_cols if c in gdf_clean.columns])
data['acq_date'] = data['acq_date'].dt.strftime("%Y%m%d").astype(int)
st.write("📝 Pemilihan fitur (feature selection) :", data.columns.tolist())
st.dataframe(data.head())

# 4. Parameter Estimation
st.header("3️⃣ Pemilihan Parameter ST-DBSCAN")
hotspot = data.copy()
n_samples = len(hotspot)
minpts = max(2, round(math.log(n_samples)))
st.write(f"⚙️ MinPts: {minpts}")
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
    st.write(f"⚙️ Parameter Epsilon1 terdeteksi di indeks {knee}: {eps1:.3f}")
ax.set_xlabel('Data points (sorted)')
ax.set_ylabel(f'Avg distance to {minpts} NN')
ax.legend()
st.pyplot(fig)

eps1_slider = st.sidebar.slider("Parameter Epsilon1 (spasial)", float(k_dist.min()), float(k_dist.max()), float(eps1))
eps2_slider = st.sidebar.number_input("Parameter Epsilon2 (temporal, hari)", min_value=1, max_value=30, value=3)

# 5. ST-DBSCAN Clustering
st.header("4️⃣ Clustering ST-DBSCAN")
df_idx = pd.DataFrame(hotspot[['longitude','latitude']].values, columns=['x','y'],
                      index=pd.MultiIndex.from_arrays([hotspot.index, hotspot['acq_date']], names=['id','timestamp']))
clusterer = STDBSCAN(eps1=eps1_slider, eps2=eps2_slider, min_samples=minpts, metric='euclidean', n_jobs=-1)
clusterer.fit(df_idx)
hotspot['cluster'] = clusterer.labels_
st.write(f"📊 Total cluster: {len(set(clusterer.labels_)) - ( -1 in clusterer.labels_)}")
st.write(f"🚫 Total Noise: {sum(clusterer.labels_ == -1)}")
counts = hotspot['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
st.dataframe(counts)

# Peta klaster
m2 = folium.Map(location=center, zoom_start=6, tiles='OpenStreetMap')
# Boundary
folium.GeoJson(geo_json, style_function=lambda x: {'color':'black','weight':2,'fill':False}).add_to(m2)
# Warna per cluster
colors = ['gray','blue','red','green','orange','purple']
for _, row in hotspot.iterrows():
    c = row['cluster']
    fc = 'gray' if c==-1 else colors[c % len(colors)]
    folium.CircleMarker(location=(row['latitude'], row['longitude']), radius=3,
                        color=fc, fill=True, fill_opacity=0.7).add_to(m2)
# Legend (manual)
legend_html = '''<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background:white; padding:10px;">
<b>Legenda Cluster</b><br>
<svg width="12" height="12"><rect width="12" height="12" style="fill:gray"/></svg> Noise<br>
<svg width="12" height="12"><rect width="12" height="12" style="fill:blue"/></svg> Cluster 0<br>
<svg width="12" height="12"><rect width="12" height="12" style="fill:red"/></svg> Cluster 1<br>
<svg width="12" height="12"><rect width="12" height="12" style="fill:green"/></svg> Cluster 2<br>
</div>''' 
m2.get_root().html.add_child(folium.Element(legend_html))
st.subheader("Peta Hasil Clustering")
st_folium(m2, width=700, height=500)

# 6. Evaluasi
st.header("5️⃣ Evaluasi Hasil Clustering")
mask = hotspot['cluster'] != -1
if mask.sum()>0:
    X_eval = np.column_stack([hotspot.loc[mask,'longitude'], hotspot.loc[mask,'latitude'], hotspot.loc[mask,'acq_date']])
    y_eval = hotspot.loc[mask,'cluster']
    sil = silhouette_score(X_eval, y_eval)
    db = davies_bouldin_score(X_eval, y_eval)
    st.write(f"🔍 Silhouette Coefficient: {sil:.4f}")
    st.write(f"🔍 Davies-Bouldin Index: {db:.4f}")
else:
    st.write("Tidak ada cluster untuk dievaluasi.")