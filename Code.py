# Task 8: K-Means clustering on Mall_Customers dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows:")
print(df.head())

# 2. Select features for clustering
# Common choice: Annual Income and Spending Score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Optional: scale features (not strictly needed here but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method to find optimal K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 4. Choose k based on elbow (for Mall dataset, 5 is commonly used)
optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df["Cluster"] = cluster_labels

print("\nCluster counts:")
print(df["Cluster"].value_counts())

# 5. Silhouette Score
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score for k={optimal_k}: {sil_score:.4f}")

# 6. Visualize clusters (2D) for the two chosen features
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Cluster"],
    palette="tab10",
    s=60
)
plt.title(f"K-Means Clusters (k={optimal_k}) - Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# 7. Optional: Use more features and visualize with PCA
features_full = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X_full = df[features_full]

scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

kmeans_full = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
labels_full = kmeans_full.fit_predict(X_full_scaled)

df["Cluster_full"] = labels_full

# PCA to reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Cluster_full"],
    palette="tab10",
    s=60
)
plt.title(f"K-Means Clusters (k={optimal_k}) - PCA of Age, Income, Spending")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# 8. Print a quick summary of clusters (mean values)
cluster_summary = df.groupby("Cluster_full")[features_full].mean()
print("\nCluster summary (means of features):")
print(cluster_summary)
