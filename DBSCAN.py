
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sqlalchemy import create_engine



engine = create_engine("mysql+mysqlconnector://root:root@localhost/heartdb")

df = pd.read_sql("SELECT * FROM lifestyle_data ", engine)
# Load dataset
#df = pd.read_csv("ss.csv")

# Columns to exclude
exclude_columns = ['id', 'name', 'height','weight',  'email', 'diet_preference', 'profession']
X = df.drop(columns=[col for col in exclude_columns if col in df.columns], errors='ignore')
X = X.select_dtypes(include=['number'])

# Fill missing values
X = X.fillna(X.mean())

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Nearest Neighbors for eps detection
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 3])

# Detect best eps using kneed
# Detect best eps using kneed
kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')

if kneedle.knee is None:
    # If no clear knee detected, fallback to 90th percentile
    best_eps = np.percentile(distances, 90)
    print("⚠️ No clear knee detected. Using 90th percentile as fallback.")
else:
    # Use the knee point as best_eps
    best_eps = distances[kneedle.knee]

# Plot the distance graph
plt.figure(figsize=(8, 5))
plt.plot(distances, label='4-NN distances')

# Only plot the line if a valid knee is detected
if kneedle.knee is not None:
    plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f'Elbow at index {kneedle.knee}')

plt.title("k-NN Distance Plot with Elbow Point")
plt.xlabel("Points sorted by distance")
plt.ylabel("Distance to 4th nearest neighbor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"🔥 Best eps detected: {best_eps:.3f}")

# Apply DBSCAN with best eps
dbscan = DBSCAN(eps=best_eps, min_samples=4)
df['cluster'] = dbscan.fit_predict(X_scaled)

# Analyze each cluster’s feature mean values
risk_order = {}

unique_clusters = df['cluster'].unique()

for cluster_id in unique_clusters:
    if cluster_id == -1:
        continue  # Skip noise for now

    cluster_data = X[df['cluster'] == cluster_id]
    avg_risk_score = cluster_data.mean().mean()  # or pick specific columns to average
    risk_order[cluster_id] = avg_risk_score

# Sort clusters by average risk score
sorted_clusters = sorted(risk_order.items(), key=lambda x: x[1])
cluster_labels = {}

# Assign risk labels
for i, (cluster_id, _) in enumerate(sorted_clusters):
    if i == 0:
        cluster_labels[cluster_id] = "Low Risk"
    elif i == len(sorted_clusters) - 1:
        cluster_labels[cluster_id] = "High Risk"
    else:
        cluster_labels[cluster_id] = "Moderate Risk"

# Handle noise separately
cluster_labels[-1] = "Anomaly / Needs Review"

# Apply risk labels
df['risk_label'] = df['cluster'].map(cluster_labels)




# Show the full dataframe output
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns (if needed)

print(df[['id', 'name', 'cluster', 'risk_label']])

# Visualize with PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
risk_color_map = {
    'Low Risk': 'green',
    'Moderate Risk': 'orange',
    'High Risk': 'red',
    'Anomaly / Needs Review': 'gray'
}
colors = df['risk_label'].map(risk_color_map)

plt.figure(figsize=(10, 6))
for label in df['risk_label'].unique():
    idx = df['risk_label'] == label
    plt.scatter(components[idx, 0], components[idx, 1], c=risk_color_map[label], label=label, s=50, alpha=0.7, edgecolors='k')

plt.title("DBSCAN Clustering (Colored by Risk Label)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Risk Levels")
plt.grid(True)
plt.tight_layout()
plt.show()


# to store clustered data in new table
df.to_sql("clustered_lifestyle_data", engine, if_exists='replace', index=False)

# Export full dataframe with risk labels
df.to_csv("labeled_risk_clusters.csv", index=False)

print("\nCluster distribution:")
print(df['cluster'].value_counts())

