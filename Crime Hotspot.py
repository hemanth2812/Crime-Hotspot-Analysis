import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
sns.set_theme(style="whitegrid")
np.random.seed(42)

# Hotspot 1 (Downtown/Central Area)
h1_lat = np.random.normal(loc=34.05, scale=0.01, size=150)
h1_lon = np.random.normal(loc=-118.25, scale=0.01, size=150)

# Hotspot 2 (Suburb A)
h2_lat = np.random.normal(loc=34.15, scale=0.005, size=80)
h2_lon = np.random.normal(loc=-118.40, scale=0.005, size=80)

# Hotspot 3 (Suburb B, less dense)
h3_lat = np.random.normal(loc=33.98, scale=0.015, size=50)
h3_lon = np.random.normal(loc=-118.10, scale=0.015, size=50)

# Noise (Scattered incidents)
noise_lat = np.random.uniform(low=33.95, high=34.20, size=20)
noise_lon = np.random.uniform(low=-118.45, high=-118.05, size=20)

df = pd.DataFrame({
    'Latitude': np.concatenate([h1_lat, h2_lat, h3_lat, noise_lat]),
    'Longitude': np.concatenate([h1_lon, h2_lon, h3_lon, noise_lon]),
})

print("Synthetic Crime Data Generated (First 5 rows):")
print(df.head())
print(f"\nTotal number of crime incidents: {len(df)}")

X = df[['Latitude', 'Longitude']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data successfully scaled and ready for clustering.")
print(f"Scaled feature shape: {X_scaled.shape}")
dbscan = DBSCAN(eps=0.15, min_samples=10)

df['Cluster'] = dbscan.fit_predict(X_scaled)

print("DBSCAN Clustering Complete.")
print(f"Unique Cluster Labels found: {df['Cluster'].unique()}")

# Calculate the number of clusters (excluding noise, label -1)
n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0)
n_noise = list(df['Cluster']).count(-1)

print(f"\nNumber of identified Hotspots (Clusters): {n_clusters}")
print(f"Number of Noise Points (Unclustered Incidents): {n_noise}")

core_samples = df[df['Cluster'] != -1]
noise_samples = df[df['Cluster'] == -1]


plt.figure(figsize=(10, 8))
palette = sns.color_palette("Spectral", n_colors=df['Cluster'].nunique() - (1 if n_noise > 0 else 0))
cluster_colors = {i: palette[i] for i in range(len(palette))}
cluster_colors[-1] = 'k'

if not noise_samples.empty:
    plt.scatter(
        noise_samples['Longitude'],
        noise_samples['Latitude'],
        c='black',
        s=20,
        marker='x',
        label=f'Noise (n={n_noise})',
        alpha=0.6
    )

# Plot Core Samples (Hotspots)
for cluster_id in sorted(core_samples['Cluster'].unique()):
    cluster_data = core_samples[core_samples['Cluster'] == cluster_id]
    plt.scatter(
        cluster_data['Longitude'],
        cluster_data['Latitude'],
        c=cluster_colors[cluster_id],
        s=100, # Larger marker for clusters
        marker='o',
        label=f'Hotspot {cluster_id} (n={len(cluster_data)})',
        alpha=0.8
    )

plt.title(f'DBSCAN Clustering for Crime Hotspot Analysis (Hotspots: {n_clusters})', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='best', title="Legend")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("\n--- Summary of Hotspots ---")
print(core_samples.groupby('Cluster').size().sort_values(ascending=False).rename('Incident Count'))